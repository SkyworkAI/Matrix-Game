#include "RaceSession.h"
#include "Renderer.h"
#include "VideoEncoder.h"
#include <SDL.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <csignal>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>

using Json=nlohmann::json;
namespace {
struct Channel {
    int fd;std::atomic<bool> stopped=false;
    std::mutex readMutex,writeMutex;std::deque<Json> commands;
    std::jthread reader;
    explicit Channel(int socket):fd(socket),reader([this]{
        try{while(!stopped){uint32_t sizes[2];readExact(sizes,sizeof(sizes));auto length=ntohl(sizes[0]);
            if(length>65536||ntohl(sizes[1])!=0)throw std::runtime_error("Invalid IPC command size");
            std::string text(length,'\0');readExact(text.data(),length);Json command=Json::parse(text);
            std::lock_guard lock(readMutex);if(commands.size()>=512)throw std::runtime_error("IPC command queue overflow");commands.push_back(std::move(command));
        }}catch(const std::exception& e){if(!stopped)std::cerr<<"IPC: "<<e.what()<<'\n';stopped=true;}
    }){}
    ~Channel(){stop();reader.join();close(fd);}
    void stop(){stopped=true;shutdown(fd,SHUT_RDWR);}
    void readExact(void* data,size_t n){auto* p=static_cast<char*>(data);while(n){auto count=recv(fd,p,n,0);if(count<=0){if(count<0&&errno==EINTR)continue;throw std::runtime_error("Gateway disconnected");}p+=count;n-=size_t(count);}}
    void send(Json meta,const std::vector<uint8_t>& bytes={}){
        auto text=meta.dump();uint32_t sizes[]={htonl(uint32_t(text.size())),htonl(uint32_t(bytes.size()))};
        std::lock_guard lock(writeMutex);
        auto write=[&](const void* data,size_t n){const auto* p=static_cast<const char*>(data);while(n){auto count=::send(fd,p,n,0);if(count<=0){if(count<0&&errno==EINTR)continue;throw std::runtime_error("IPC write failed");}p+=count;n-=size_t(count);}};
        write(sizes,sizeof(sizes));write(text.data(),text.size());if(!bytes.empty())write(bytes.data(),bytes.size());
    }
    std::deque<Json> take(){std::lock_guard lock(readMutex);std::deque<Json> result;result.swap(commands);return result;}
};
struct Job {Json meta;std::vector<uint8_t> rgba;bool forceKey=false;};
class Stream {
    Channel& channel;std::mutex mutex;std::condition_variable cv;
    std::optional<Job> pending;bool stopping=false;std::atomic<bool> keyRequested=true;
    std::vector<std::vector<uint8_t>> reusable;
    uint64_t generation=0;int credits=0,creditLimit=0;
    std::atomic<int> bitrateKbps=8000;
    std::jthread worker;
public:
    std::atomic<uint64_t> encoded=0,dropped=0;
    Stream(Channel& c,int w,int h,int fps,std::string codec,int threads):channel(c),worker([this,w,h,fps,codec,threads]{
        try{VideoEncoder encoder(w,h,fps,codec,threads);for(;;){Job job;bool forceKey=false;
            {std::unique_lock lock(mutex);cv.wait(lock,[&]{return stopping||(pending.has_value()&&credits>0);});if(stopping)return;job=std::move(*pending);pending.reset();--credits;
                job.meta["media_generation"]=generation;forceKey=job.forceKey||keyRequested.exchange(false);}
            int bitrate=bitrateKbps.load();encoder.setBitrate(bitrate);job.meta["bitrate_kbps"]=bitrate;
            double start=now();auto output=encoder.encode(job.rgba,job.meta.at("timestamp"),forceKey);
            job.meta["key"]=output.key;job.meta["codec"]=codec;job.meta["encode_ms"]=(now()-start)*1000;job.meta["encoded"]=++encoded;job.meta["dropped_before_encode"]=dropped.load();
            // swscale has copied the pixels into the encoder's own YUV frame.
            // Return ownership only now, never while an encode can read them.
            {std::lock_guard lock(mutex);if(reusable.size()<2)reusable.push_back(std::move(job.rgba));}
            channel.send(job.meta,output.bytes);
        }}catch(const std::exception& e){std::cerr<<"Encoder: "<<e.what()<<'\n';channel.stop();}
    }){}
    ~Stream(){{std::lock_guard lock(mutex);stopping=true;}cv.notify_one();worker.join();}
    void requestKey(){keyRequested=true;}
    void setBitrate(int kbps){bitrateKbps=kbps;}
    void open(uint64_t id,int capacity){std::lock_guard lock(mutex);generation=id;creditLimit=credits=capacity;keyRequested=true;
        if(pending){if(reusable.size()<2)reusable.push_back(std::move(pending->rgba));pending.reset();++dropped;}cv.notify_one();}
    void grant(uint64_t id,int count){std::lock_guard lock(mutex);if(id==generation){credits=std::min(creditLimit,credits+count);cv.notify_one();}}
    void close(uint64_t id){std::lock_guard lock(mutex);if(id==generation)credits=creditLimit=0;}
    std::vector<uint8_t> takeBuffer(){std::lock_guard lock(mutex);if(reusable.empty())return {};auto buffer=std::move(reusable.back());reusable.pop_back();return buffer;}
    void submit(Job job){std::lock_guard lock(mutex);if(pending){++dropped;if(reusable.size()<2)reusable.push_back(std::move(pending->rgba));}pending=std::move(job);cv.notify_one();}
};
struct View {int id;size_t slot;Camera camera;HudState hud;std::vector<GateVisual> gates;Json meta;};
// One aggregate per second, outside the video packets. In particular, snapshot
// lock waits and frame pacing are not included in a view's render_submit_ms.
class RenderProfile {
    struct Metric {double sum=0,max=0;void add(double value){sum+=value;max=std::max(max,value);}};
    std::array<Metric,6> metrics{};
    std::mutex mutex;Json latest;
    double started=0,previous=0;size_t samples=0,lastViews=0;
public:
    void record(double begin,size_t views,double readback,double wait,double snapshot,double render,double work){
        if(views!=lastViews){started=previous=begin;samples=0;metrics={};lastViews=views;}
        const std::array values{(begin-previous)*1000,readback,wait,snapshot,render,work};previous=begin;
        if(!samples)started=begin;
        for(size_t i=0;i<values.size();++i)metrics[i].add(values[i]);
        ++samples;double duration=begin-started;
        if(duration<1)return;
        Json report={{"wall_time",begin},{"views",views},{"frames",samples},{"seconds",duration},{"fps",views?(samples-1)/duration:0}};
        const std::array names{"interval_ms","readback_ms","snapshot_wait_ms","snapshot_ms","render_submit_ms","work_ms"};
        for(size_t i=0;i<names.size();++i)report[names[i]]={{"mean",metrics[i].sum/samples},{"max",metrics[i].max}};
        {std::lock_guard lock(mutex);latest=std::move(report);}
        samples=0;metrics={};
    }
    Json report(){std::lock_guard lock(mutex);return latest;}
};
}
int main(int argc,char** argv){
    std::signal(SIGPIPE,SIG_IGN);
    try{
        std::filesystem::path root=".";int fd=-1,width=1920,height=1080,fps=60,traffic=64,pedestrians=160,threads=2;std::string codec="h264";
        for(int i=1;i<argc;++i){std::string arg=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("Missing option value");return std::string(argv[i]);};
            if(arg=="--data-dir")root=value();else if(arg=="--ipc-fd")fd=std::stoi(value());else if(arg=="--width")width=std::stoi(value());else if(arg=="--height")height=std::stoi(value());
            else if(arg=="--fps")fps=std::stoi(value());else if(arg=="--codec")codec=value();else if(arg=="--traffic")traffic=std::stoi(value());else if(arg=="--pedestrians")pedestrians=std::stoi(value());else if(arg=="--encoder-threads")threads=std::stoi(value());else throw std::runtime_error("Unknown argument: "+arg);
        }
        if(fd<0||width<640||width>3840||height<360||height>2160||width%2||height%2||fps<1||fps>60||traffic<0||traffic>100||pedestrians<0||pedestrians>200||threads<1||threads>8||(codec!="h264"&&codec!="jpeg"))throw std::runtime_error("Invalid server configuration");
        Channel channel(fd);
        const char* eglOption=std::getenv("CITYSAMPLE_EGL");bool headless=eglOption&&std::string(eglOption)=="1";
        if(SDL_Init((headless?0:SDL_INIT_VIDEO)|SDL_INIT_TIMER)!=0)throw std::runtime_error(SDL_GetError());
        struct Quit {~Quit(){SDL_Quit();}} quit;
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,1);SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,24);
#ifdef __APPLE__
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
#endif
        std::unique_ptr<SDL_Window,decltype(&SDL_DestroyWindow)> window(headless?nullptr:SDL_CreateWindow("CitySample race server",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,width,height,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN),SDL_DestroyWindow);
        if(!window&&!headless)throw std::runtime_error(SDL_GetError());
        Renderer renderer(window.get(),width,height);World world(root/"assets/bbox-assets.bin",traffic,pedestrians);
        auto routes=loadRaceRoutes(root/"assets/race-routes.json",world);RaceSession race(world,std::move(routes));world.upload(renderer);
        std::mutex worldMutex;RenderProfile renderProfile;std::array<std::unique_ptr<Stream>,World::MaxPlayers> streams;
        for(auto& stream:streams)stream=std::make_unique<Stream>(channel,width,height,fps,codec,threads);
        std::atomic<int> capturePlayer=0;std::string captureRequest;
        channel.send({{"type","hello"},{"renderer",renderer.adapterName},{"width",width},{"height",height},{"fps_target",fps},{"codec",codec},{"physics_hz",500},{"state",race.state()}});
        std::jthread simulation([&](std::stop_token stop){
            try{double deadline=now(),nextState=0;while(!stop.stop_requested()&&!channel.stopped){
                for(auto command:channel.take()){
                    Json reply={{"type","reply"},{"request",command.value("request",std::string{})}};
                    try{
                        std::lock_guard lock(worldMutex);auto type=command.value("type",std::string{});
                        if(type=="keyframe"||type=="video_config"||type=="video_start"||type=="video_credit"||type=="video_stop"){
                            int id=command.at("player_id");const RaceSession::Player* player=nullptr;
                            for(const auto& p:race.players())if(p&&p->id==id&&p->connected)player=&*p;
                            if(!player)throw std::runtime_error("Unknown video player");
                            auto& stream=streams[player->slot];
                            if(type=="keyframe")stream->requestKey();
                            else if(type=="video_config"){int bitrate=command.at("bitrate_kbps");if(bitrate<300||bitrate>8000)throw std::runtime_error("Invalid video bitrate");stream->setBitrate(bitrate);}
                            else{uint64_t generation=command.at("generation");
                                if(type=="video_stop")stream->close(generation);
                                else{int count=command.at(type=="video_start"?"capacity":"count");if(count<1||count>64)throw std::runtime_error("Invalid video credit");
                                    if(type=="video_start")stream->open(generation,count);else stream->grant(generation,count);}}
                            reply["result"]={{"ok",true}};
                        }
                        else if(type=="capture"){
                            if(capturePlayer!=0)throw std::runtime_error("Another debug frame is pending");
                            int id=command.at("player_id");bool found=false;for(const auto& p:race.players())if(p&&p->id==id&&p->connected)found=true;
                            if(!found)throw std::runtime_error("Unknown debug player");captureRequest=command.at("request");capturePlayer=id;reply["result"]={{"ok",true}};
                        }else reply["result"]=race.command(command);
                    }catch(const std::exception& e){reply["error"]=e.what();}
                    if(!command.value("no_reply",false))channel.send(reply);
                }
                Json state;
                {std::lock_guard lock(worldMutex);race.tick(.002);if(now()>=nextState){state=race.state();state["process_cpu_seconds"]=double(std::clock())/CLOCKS_PER_SEC;state["wall_time"]=now();nextState=now()+.1;
                    const std::array names{"npc_steering","physics","npc_recycling"};
                    for(size_t i=0;i<names.size();++i){const auto& t=world.workTimings[i];state["simulation_performance"][names[i]]={{"calls",t.calls},{"total_ms",t.totalMs},{"max_ms",t.maxMs}};}
                }}
                if(!state.is_null()){state["render_performance"]=renderProfile.report();channel.send(std::move(state));}
                deadline+=.002;double delay=deadline-now();if(delay>0)std::this_thread::sleep_for(std::chrono::duration<double>(delay));else if(delay<-.1)deadline=now();
            }}catch(const std::exception& e){std::cerr<<"Simulation: "<<e.what()<<'\n';channel.stop();}
        });
        try{uint64_t frameId=0;double nextRender=now();std::vector<View> pendingViews;while(!channel.stopped){
            const double begin=now();
            SDL_Event event;while(SDL_PollEvent(&event))if(event.type==SDL_QUIT)channel.stop();
            // Read the previous snapshot after the GPU has had a full frame
            // interval to complete it. Retain its original camera/tick/input
            // metadata; never label these pixels with the current world state.
            for(auto& view:pendingViews){double start=now();auto pixels=renderer.finishReadback(view.slot,streams[view.slot]->takeBuffer());view.meta["readback_ms"]=(now()-start)*1000;
                streams[view.slot]->submit({view.meta,std::move(pixels),false});}
            pendingViews.clear();
            const double readbackDone=now();double snapshotBegin=0;
            std::vector<DrawObject> objects;std::vector<View> views;RouteGuide guide;std::string capture;int captureId=0;
            {std::lock_guard lock(worldMutex);snapshotBegin=now();objects=world.drawObjects();guide=race.route().guide;
                captureId=capturePlayer.load();capture=captureRequest;
                for(const auto& p:race.players())if(p&&p->connected&&(p->visible||p->id==captureId)){
                    View view;view.id=p->id;view.slot=p->slot;view.camera=world.playerCamera(p->slot,p->camera,p->lookBack);auto pose=world.playerPose(p->slot);auto& hud=view.hud;
                    hud.visible=true;hud.race=true;hud.playerNumber=int(p->slot+1);hud.speed=pose.speed;hud.heading=pose.heading;hud.location=position(pose.body);hud.elapsed=p->progress.elapsed;hud.penalty=p->progress.penalty;hud.finished=p->progress.finished;
                    hud.nextCheckpoint=int(p->progress.next);hud.totalCheckpoints=int(race.route().checkpoints.size());hud.notice=p->notice;hud.racePhase=race.phase();std::transform(hud.racePhase.begin(),hud.racePhase.end(),hud.racePhase.begin(),[](unsigned char c){return char(std::toupper(c));});
                    hud.roads=&world.mapLines();hud.mapCenter=world.mapCenter();hud.mapHalfSize=world.mapHalfSize();
                    hud.navigation=navigateRoute(race.route(),p->progress,hud.location,pose.heading,pose.speed);
                    if(!p->progress.finished){const auto& gate=race.route().checkpoints.at(p->progress.next);GateVisual visual{gate.center,gate.forward,gate.halfWidth,gate.height};if(p->progress.next+1==race.route().checkpoints.size())visual.color={.35f,.95f,.63f,.95f};view.gates.push_back(visual);}
                    view.meta={{"type","frame"},{"player_id",p->id},{"race_id",race.raceId()},{"epoch",race.epoch()},{"frame_id",++frameId},{"simulation_tick",race.simulationTick()},{"phase",race.phase()},{"width",width},{"height",height},{"timestamp",int64_t(now()*1000000)},{"input_seq",p->inputSequence},{"camera",{{"eye",{view.camera.eye.x,view.camera.eye.y,view.camera.eye.z}},{"target",{view.camera.target.x,view.camera.target.y,view.camera.target.z}},{"horizontal_fov",view.camera.horizontalFov}}}};
                    view.meta["applied_input_seq"]=p->appliedInputSequence;views.push_back(std::move(view));
                }
            }
            const double snapshotDone=now();
            for(auto& view:views){double start=now();view.hud.routeGuide=&guide;bool debug=view.id==captureId;
                auto frame=renderer.renderLayers(objects,view.camera,view.hud,view.gates,debug,int(view.slot));view.meta["render_submit_ms"]=(now()-start)*1000;
                if(debug){for(const auto& layer:{"scene","ui"}){auto meta=view.meta;meta["type"]="capture";meta["request"]=capture;meta["layer"]=layer;meta["alpha"]=std::string(layer)=="ui"?"premultiplied":"opaque";channel.send(meta,Renderer::png(std::string(layer)=="ui"?frame.ui:frame.scene,width,height));}capturePlayer=0;}
            }
            const double renderDone=now();
            renderProfile.record(begin,views.size(),(readbackDone-begin)*1000,(snapshotBegin-readbackDone)*1000,(snapshotDone-snapshotBegin)*1000,(renderDone-snapshotDone)*1000,(renderDone-begin)*1000);
            bool idle=views.empty();pendingViews=std::move(views);
            nextRender+=1./fps;double delay=nextRender-now();if(idle)delay=.03;if(delay>0)std::this_thread::sleep_for(std::chrono::duration<double>(delay));else nextRender=now();
        }}catch(...){channel.stop();simulation.request_stop();simulation.join();throw;}
        channel.stop();simulation.request_stop();simulation.join();return 0;
    }catch(const std::exception& e){std::cerr<<"CitySampleRaceServer: "<<e.what()<<'\n';return 1;}
}
