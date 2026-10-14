#include "World.h"
#include "Renderer.h"
#include "CaptionStream.h"
#include "ReferenceStream.h"
#include "KeyboardSteering.h"
#include "LocalRaceControls.h"
#include "HandlingMenu.h"
#include "ReferencePresentation.h"
#include "ReferenceCaptureProfile.h"
#include <SDL.h>
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <csignal>
#include <cstring>
#include <iostream>

using Json=nlohmann::json;
namespace {
// Transport and sampling overlap until this bounded queue is genuinely full.
struct Channel {
    int fd;std::vector<uint8_t> incoming;std::deque<std::vector<uint8_t>> outgoing;size_t offset=0;
    explicit Channel(int descriptor):fd(descriptor){}
    ~Channel(){close(fd);}
    bool busy()const{return !outgoing.empty();}
    bool full()const{return outgoing.size()>=2;}
    void send(const Json& meta,const std::vector<uint8_t>& bytes={}){
        if(full())throw std::runtime_error("IPC output queue full");
        auto text=meta.dump();uint32_t sizes[]={htonl(uint32_t(text.size())),htonl(uint32_t(bytes.size()))};
        std::vector<uint8_t> packet(8+text.size()+bytes.size());std::memcpy(packet.data(),sizes,8);
        std::memcpy(packet.data()+8,text.data(),text.size());
        if(!bytes.empty())std::memcpy(packet.data()+8+text.size(),bytes.data(),bytes.size());
        outgoing.push_back(std::move(packet));
    }
    void flush(){
        double deadline=now()+.001;
        while(busy()){
            auto& packet=outgoing.front();
            auto count=::send(fd,packet.data()+offset,packet.size()-offset,MSG_DONTWAIT);
            if(count<0&&(errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR))return;
            if(count<=0)throw std::runtime_error("Reference consumer disconnected");
            offset+=size_t(count);if(offset==packet.size()){outgoing.pop_front();offset=0;}
            if(now()>=deadline)return;
        }
    }
    void wait(){
        // A partially sent RGB chunk must resume when writable, not after
        // another fixed sleep for every socket-buffer-sized fragment.
        pollfd event{fd,short(POLLIN|(busy()?POLLOUT:0)),0};
        if(::poll(&event,1,1)<0&&errno!=EINTR)
            throw std::runtime_error("Reference IPC poll failed");
    }
    std::vector<Json> receive(){
        uint8_t bytes[8192];auto n=recv(fd,bytes,sizeof(bytes),MSG_DONTWAIT);
        if(n==0)throw std::runtime_error("Reference consumer disconnected");
        if(n<0&&errno!=EAGAIN&&errno!=EWOULDBLOCK&&errno!=EINTR)throw std::runtime_error("IPC read failed");
        if(n>0)incoming.insert(incoming.end(),bytes,bytes+n);
        if(incoming.size()>65536)throw std::runtime_error("IPC command buffer overflow");
        std::vector<Json> result;
        while(incoming.size()>=8){uint32_t sizes[2];std::memcpy(sizes,incoming.data(),8);size_t length=ntohl(sizes[0]);
            if(!length||length>16384||ntohl(sizes[1]))throw std::runtime_error("Invalid reference command envelope");
            if(incoming.size()<8+length)break;
            result.push_back(Json::parse(incoming.begin()+8,incoming.begin()+8+length));
            incoming.erase(incoming.begin(),incoming.begin()+8+length);
        }return result;
    }
};
}

int main(int argc,char** argv){
    std::signal(SIGPIPE,SIG_IGN);
    try{
        std::filesystem::path root=".";int fd=-1,width=864,height=480,fps=24,frames=4089,blockLatents=3;
        PlayerHandling handling;std::string mode="drive",route="downtown",renderProfile="dataset";int stripeOverride=-1;
        for(int i=1;i<argc;++i){std::string arg=argv[i];if(++i>=argc)throw std::runtime_error("Missing option value");std::string value=argv[i];
            if(arg=="--data-dir")root=value;else if(arg=="--ipc-fd")fd=std::stoi(value);else if(arg=="--width")width=std::stoi(value);
            else if(arg=="--height")height=std::stoi(value);else if(arg=="--fps")fps=std::stoi(value);else if(arg=="--frames")frames=std::stoi(value);
            else if(arg=="--block-latents")blockLatents=std::stoi(value);
            else if(arg=="--render-profile")renderProfile=value;
            else if(arg=="--building-stripes")stripeOverride=std::stoi(value);
            else if(arg=="--mode")mode=value;else if(arg=="--route")route=value;
            else if(arg=="--front-drive")handling.frontDrive=std::stof(value);
            else if(arg=="--front-grip")handling.frontGrip=std::stof(value);
            else if(arg=="--rear-grip")handling.rearGrip=std::stof(value);
            else throw std::runtime_error("Unknown reference option");}
        for(int row=0;row<3;++row)if(!std::isfinite(handling.value(row))||handling.value(row)<handling.minimum(row)||handling.value(row)>handling.maximum(row))
            throw std::runtime_error("Invalid handling setting");
        if(mode!="drive"&&mode!="time_trial"&&mode!="pursuit")throw std::runtime_error("Invalid game mode");
        if(mode=="pursuit")route="avenue";
        if(renderProfile!="dataset"&&renderProfile!="legacy")throw std::runtime_error("Invalid reference render profile");
        if(stripeOverride< -1||stripeOverride>1)throw std::runtime_error("Building stripes must be 0 or 1");
        if(blockLatents!=1&&blockLatents!=3)throw std::runtime_error("Reference block latents must be 1 or 3");
        const int firstChunk=1+4*(blockLatents-1),chunkFrames=4*blockLatents;
        if(fd<0||width!=864||height!=480||fps<1||fps>60||frames<firstChunk||(frames-firstChunk)%chunkFrames)
            throw std::runtime_error("Reference server requires 864x480, 1-60 FPS and whole latent blocks");
        const int physicsHz=reference::capturePhysicsHz(fps),stepsPerFrame=physicsHz/fps;
        const int renderWidth=renderProfile=="dataset"?reference::capture::RenderWidth:width;
        const int renderHeight=renderProfile=="dataset"?reference::capture::RenderHeight:height;
        const bool stripes=stripeOverride<0?renderProfile=="dataset":stripeOverride!=0;
        if(stripes&&!std::filesystem::exists(root/"assets/reference-building-stripes.bin"))
            throw std::runtime_error("Building stripes require reference-building-stripes.bin");
        const double dt=1./physicsHz;
        Channel channel(fd);
        const bool headless=std::getenv("CITYSAMPLE_EGL")&&std::string(std::getenv("CITYSAMPLE_EGL"))=="1";
        if(SDL_Init((headless?0:SDL_INIT_VIDEO)|SDL_INIT_TIMER))throw std::runtime_error(SDL_GetError());
        struct Quit{~Quit(){SDL_Quit();}}quit;
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef __APPLE__
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
#endif
        std::unique_ptr<SDL_Window,decltype(&SDL_DestroyWindow)> window(headless?nullptr:SDL_CreateWindow("Reference server",0,0,renderWidth,renderHeight,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN),SDL_DestroyWindow);
        if(!headless&&!window)throw std::runtime_error(SDL_GetError());
        Renderer renderer(window.get(),renderWidth,renderHeight);renderer.buildingStripes=stripes;World world(root/"assets/bbox-assets.bin",16,40);
        if(renderProfile=="dataset")renderer.outlineWidth=reference::capture::OutlineWidth;
        world.enableCaptions(root);world.upload(renderer);world.setPlayerSteeringLimit(false);
        world.setPlayerLateralGrip(handling.frontGrip,handling.rearGrip);world.setPlayerFrontDrive(handling.frontDrive);world.reset();
        std::optional<RaceSession> race;std::optional<LocalRaceControls> raceControls;
        if(mode!="drive"){
            race.emplace(world,loadRaceRoutes(root/"assets/race-routes.json",world),mode=="pursuit");
            race->setAutomaticRecovery(false);
            auto joined=race->command({{"type","join"},{"name","Player 1"}});
            std::vector<int> players{joined.at("player_id").get<int>()};
            if(mode=="pursuit")players.push_back(race->command({{"type","join"},{"name","Suspect"}}).at("player_id").get<int>());
            raceControls.emplace(*race,players);
            raceControls->start(route);
        }
        captions::Stream captioner(root/"assets/caption-catalog.json");
        reference::SimulationClock clock(physicsHz);reference::InputQueue inputs;KeyboardSteering steering;
        uint64_t frameId=0,chunkId=0,inputId=0;const uint64_t lead=blockLatents==1?chunkFrames:chunkFrames+3;
        reference::ModelCredits credits(lead,firstChunk,chunkFrames);
        std::vector<captions::Frame> snapshots;std::vector<uint8_t> pixels;Json telemetry=Json::array();
        std::vector<CollisionEvent> impacts;uint64_t checkpoints=0;size_t previousCheckpoint=0;
        bool started=false,running=true;double firstCapture=0,lastCapture=0,lastControl=now(),blockedSeconds=0,previous=now();
        latency::Controls controls;controls.camera=1;
        channel.send({{"type","hello"},{"schema_version",1},{"width",width},{"height",height},{"fps",fps},
            {"render_profile",renderProfile},{"render_width",renderWidth},{"render_height",renderHeight},{"building_stripes",stripes},
            {"outline_width",renderer.outlineWidth},
            {"resize",renderProfile=="dataset"?"antialiased_bilinear_864x486_center_crop_864x480":"none"},
            {"camera_mode",1},{"camera_profile","dataset_chase_raised_v2"},{"ground_profile","road_contrast_junction_v1"},
            {"building_stripe_profile",stripes?"authored_front_geometry_v1":"none"},
            {"renderer",renderer.adapterName},{"physics_hz",physicsHz},{"input_capacity",lead},{"credit_mode","model_admission"},
            {"read_reservation",blockLatents==1},
            {"pursuit_map",mode=="pursuit"?reference::presentation_detail::makeMap(world.mapLines(),world.mapCenter(),nullptr,0,0,
                std::max(100.f,world.mapHalfSize())*1.04f):Json(nullptr)},
            {"first_chunk",firstChunk},{"chunk",chunkFrames},{"game_options",{{"mode",mode},{"route",route},
                {"front_drive",handling.frontDrive},{"front_grip",handling.frontGrip},{"rear_grip",handling.rearGrip}}}});
        clock.reset(now());
        while(running){
            for(const auto& command:channel.receive()){
                auto type=command.at("type").get<std::string>();
                if(type=="stop"){running=false;break;}
                if(type=="start"){
                    if(started)throw std::runtime_error("Reference generation already started");started=true;clock.reset(now());
                }else if(type=="begin"){
                    credits.begin(command.at("frame_end"),frameId);
                }else if(type=="request"){
                    uint64_t end=command.at("frame_end");
                    if(blockLatents!=1||end>uint64_t(frames))throw std::runtime_error("Invalid input reservation");
                    credits.request(end,frameId);
                }else if(type=="ack"){
                    credits.acknowledge(command.at("frame_end"));
                }else if(type=="controls"){
                    auto value=command.at("drive").get<std::array<float,4>>();
                    for(auto v:value)if(!std::isfinite(v)||std::abs(v)>1)throw std::runtime_error("Invalid controls");
                    if(value[2]<0||value[3]<0)throw std::runtime_error("Negative brake input");
                    controls.drive={value[0],value[1],value[2],value[3]};controls.keyboardSteering=command.value("keyboard",true);
                    inputId=command.value("sequence",inputId+1);
                    inputs.push({controls,inputId,now()},clock.inputTime(0));lastControl=now();
                }else if(type=="race_action"){
                    if(!raceControls)throw std::runtime_error("Race action outside race");
                    auto action=command.at("action").get<std::string>();
                    if(action=="pause")raceControls->pause();
                    else if(action=="resume")raceControls->resume();
                    else if(action=="recover")raceControls->recover();
                    else if(action=="retry")raceControls->start(route);
                    else throw std::runtime_error("Unknown race action");
                }else if(type!="start")throw std::runtime_error("Unknown reference command");
            }
            if(!running)break;
            if(now()-lastControl>1.0&&(controls.drive.throttle||controls.drive.steer||controls.drive.brake||controls.drive.handbrake)){
                controls.drive={};inputs.push({controls,++inputId,now()},clock.inputTime(0));
            }
            channel.flush();
            uint64_t allowed=credits.allowed(uint64_t(frames));
            bool blocked=!started||frameId>=allowed||channel.full();double current=now();
            if(blocked&&started&&frameId<uint64_t(frames))blockedSeconds+=current-previous;previous=current;
            clock.observe(current,blocked);
            int steps=0;
            while(!blocked&&clock.due()&&steps++<64){
                auto command=inputs.sample(clock.time());
                auto drive=steering.apply(command.controls.drive,dt,command.controls.keyboardSteering);
                if(race){
                    raceControls->drive(drive,1,false);race->tick(dt);
                    const auto& collisions=race->collisions();impacts.insert(impacts.end(),collisions.begin(),collisions.end());
                    auto next=race->players()[0]->progress.next;
                    if(next>previousCheckpoint)checkpoints+=next-previousCheckpoint;
                    previousCheckpoint=next;
                }else{
                    world.update(float(dt),drive);auto collisions=world.takeCollisions();
                    impacts.insert(impacts.end(),collisions.begin(),collisions.end());
                }
                clock.step();
                if(clock.tick()%uint64_t(stepsPerFrame))continue;
                ++frameId;double captureStarted=now();auto camera=world.camera(1,false);auto pose=world.player();const auto& objects=world.drawObjects();
                renderer.render(objects,camera,{});auto rgba=renderer.readBackResized(width,height);
                auto visible=renderer.captionVisibility(objects,camera,432,float(width)/height);const auto& entities=world.captionEntities();
                uint64_t area=0;for(const auto& item:visible)area+=item.pixels;
                captions::Frame snapshot{frameId,clock.time(),camera,pose.speed,{}};
                for(const auto& item:visible)if(item.token)snapshot.visible.push_back({entities.at(item.token),double(item.pixels)/area,item.x,item.y});
                snapshots.push_back(std::move(snapshot));pixels.insert(pixels.end(),rgba.begin(),rgba.end());
                lastCapture=now();if(snapshots.size()==1)firstCapture=lastCapture;
                reference::PresentationFrameInfo presentation;
                presentation.frame=frameId-1;presentation.inputId=command.id;presentation.inputReceivedMonotonic=command.time;
                presentation.captureMonotonic=lastCapture;presentation.width=width;presentation.height=height;presentation.checkpoint=checkpoints;
                telemetry.push_back(reference::makeGameFrame(world,camera,pose,presentation,race?&*race:nullptr,impacts));
                telemetry.back()["simulation_time"]=clock.time();
                telemetry.back()["input_sequence"]=command.id;
                telemetry.back()["capture_ms"]=(now()-captureStarted)*1000;
                impacts.clear();
                const size_t count=chunkId?chunkFrames:firstChunk;
                if(snapshots.size()==count){
                    auto caption=captioner.chunk(1,++chunkId,snapshots);caption["video_frame_start"]=frameId-count;caption["video_frame_end"]=frameId-1;
                    channel.send({{"type","chunk"},{"epoch",1},{"chunk_id",chunkId},{"frame_start",frameId-count+1},{"frame_end",frameId},
                        {"count",count},{"width",width},{"height",height},{"caption",caption},{"telemetry",telemetry},
                        {"capture_first_monotonic",firstCapture},{"capture_last_monotonic",lastCapture},{"blocked_s",blockedSeconds}},pixels);
                    snapshots.clear();pixels.clear();telemetry=Json::array();
                    credits.capturedReady(frameId);allowed=credits.allowed(uint64_t(frames));
                }
                if(frameId>=allowed||channel.full())blocked=true;
            }
            if(blocked)clock.freeze();SDL_PumpEvents();channel.wait();
        }
        return 0;
    }catch(const std::exception& error){std::cerr<<"Reference server: "<<error.what()<<'\n';return 1;}
}
