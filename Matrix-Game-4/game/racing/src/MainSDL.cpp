#include "GameRuntime.h"
#include "CameraTelemetry.h"
#include "RaceSession.h"
#include "LocalRaceControls.h"
#include "ViewRecorder.h"
#include "OfflineRecorder.h"
#include "ReferencePresentation.h"
#include "ReferenceCaptureProfile.h"
#include "LatencyDemo.h"
#include "ReferenceDemo.h"
#include "VehicleAudio.h"
#include "KeyboardSteering.h"
#include <SDL.h>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <map>
#include <chrono>
#include <thread>

namespace {
struct SdlLifetime {SdlLifetime(){if(SDL_Init(SDL_INIT_VIDEO|SDL_INIT_GAMECONTROLLER|SDL_INIT_TIMER)!=0)throw std::runtime_error(SDL_GetError());}~SdlLifetime(){SDL_Quit();}};
struct WindowDelete {void operator()(SDL_Window* p)const{SDL_DestroyWindow(p);}};
class Controller {
    SDL_GameController* handle=nullptr;
public:
    ~Controller(){if(handle)SDL_GameControllerClose(handle);}
    void refresh(){if(handle&&!SDL_GameControllerGetAttached(handle)){SDL_GameControllerClose(handle);handle=nullptr;}
        if(!handle)for(int i=0;i<SDL_NumJoysticks();++i)if(SDL_IsGameController(i)){handle=SDL_GameControllerOpen(i);if(handle)break;}}
    Input read(){Input in;if(!handle)return in;float x=std::clamp(SDL_GameControllerGetAxis(handle,SDL_CONTROLLER_AXIS_LEFTX)/32767.f,-1.f,1.f);
        in.steer=std::abs(x)>.12f?std::copysign((std::abs(x)-.12f)/.88f,x):0;
        float t=(SDL_GameControllerGetAxis(handle,SDL_CONTROLLER_AXIS_TRIGGERRIGHT)-SDL_GameControllerGetAxis(handle,SDL_CONTROLLER_AXIS_TRIGGERLEFT))/32767.f;
        in.throttle=std::abs(t)>.05f?t:0;in.handbrake=SDL_GameControllerGetButton(handle,SDL_CONTROLLER_BUTTON_A)?1.f:0.f;return in;}
};
std::string trim(std::string value){auto a=value.find_first_not_of(" \t\r\n"),b=value.find_last_not_of(" \t\r\n");return a==std::string::npos?"":value.substr(a,b-a+1);}
std::map<std::string,std::string> readIni(const std::filesystem::path& path){
    std::map<std::string,std::string> values;std::ifstream file(path);std::string line,section;
    while(std::getline(file,line)){line=trim(line);if(line.empty()||line[0]==';'||line[0]=='#')continue;
        if(line.front()=='['&&line.back()==']'){section=line.substr(1,line.size()-2);continue;}
        auto equals=line.find('=');if(equals!=std::string::npos)values[section+"."+trim(line.substr(0,equals))]=trim(line.substr(equals+1));}return values;
}
int integer(const std::string& value){size_t used=0;int v=std::stoi(value,&used);if(used!=value.size())throw std::runtime_error("Invalid integer: "+value);return v;}
float real(const std::string& value){size_t used=0;float v=std::stof(value,&used);if(used!=value.size()||!std::isfinite(v))throw std::runtime_error("Invalid number: "+value);return v;}
struct Launch {Options opt;std::filesystem::path root,offlineDirectory,offlineScript;double duration=0,fps=0,menuDelay=0;float audioVolume=.6f;int players=1;bool hidden=false,help=false,race=false,chase=false,latencyDemo=false,referenceDemo=false,offlineAuto=false;std::string route,record;LatencyDemoOptions latency;ReferenceDemoOptions reference;};
Launch parse(int argc,char** argv){
    Launch launch;for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--help"||arg=="-h"){launch.help=true;return launch;}
        if(arg=="--data-dir"){if(++i>=argc)throw std::runtime_error("Missing --data-dir value");launch.root=argv[i];}}
    if(launch.root.empty()){char* base=SDL_GetBasePath();if(base){launch.root=base;SDL_free(base);}else launch.root=std::filesystem::current_path();
        while(!std::filesystem::exists(launch.root/"assets/bbox-assets.bin")){auto p=launch.root.parent_path();if(p==launch.root){launch.root=std::filesystem::current_path();break;}launch.root=p;}}
    launch.root=std::filesystem::absolute(launch.root);
    if(!std::filesystem::exists(launch.root/"assets/bbox-assets.bin"))throw std::runtime_error("Cannot find assets; pass --data-dir <directory containing assets>");
    auto ini=readIni(launch.root/"settings.ini");auto& o=launch.opt;
    auto get=[&](const char* key,int fallback){auto i=ini.find(key);return i==ini.end()?fallback:integer(i->second);};
    o.width=get("Display.Width",1920);o.height=get("Display.Height",1080);o.traffic=get("World.Traffic",64);o.pedestrians=get("World.Pedestrians",160);
    o.vsync=get("Display.VSync",0)!=0;o.noHud=get("Display.HUD",0)==0;o.physicsHz=float(get("Simulation.PhysicsHz",500));
    for(int i=1;i<argc;++i){std::string arg=argv[i];auto next=[&](){if(++i>=argc)throw std::runtime_error("Missing value for "+arg);return std::string(argv[i]);};
        if(arg=="--data-dir"){next();}else if(arg=="--width")o.width=integer(next());else if(arg=="--height")o.height=integer(next());
        else if(arg=="--traffic")o.traffic=integer(next());else if(arg=="--pedestrians")o.pedestrians=integer(next());
        else if(arg=="--physics-hz")o.physicsHz=real(next());else if(arg=="--duration")launch.duration=real(next());
        else if(arg=="--player-steering-limit"){auto value=next();if(value!="on"&&value!="off")throw std::runtime_error("--player-steering-limit must be on or off");o.playerSteeringLimit=value=="on";}
        else if(arg=="--player-front-grip")o.playerFrontGrip=real(next());
        else if(arg=="--player-rear-grip")o.playerRearGrip=real(next());
        else if(arg=="--player-front-drive")o.playerFrontDrive=real(next());
        else if(arg=="--audio-volume")launch.audioVolume=real(next());
        else if(arg=="--benchmark"){o.benchmark=real(next());o.autodrive=true;o.noAudio=true;}
        else if(arg=="--tag"){o.tag=next();if(o.tag.empty()||o.tag.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")!=std::string::npos)throw std::runtime_error("Invalid benchmark tag");}
        else if(arg=="--self-test")o.selfTest=true;else if(arg=="--visual-test")o.visualTest=true;else if(arg=="--map-test")o.mapTest=true;
        else if(arg=="--underpass-test")o.underpassTest=true;else if(arg=="--ramp-ground-test")o.rampGroundTest=true;else if(arg=="--all-ramps-test")o.allRampsTest=true;
        else if(arg=="--autodrive")o.autodrive=true;else if(arg=="--no-audio")o.noAudio=true;else if(arg=="--no-hud")o.noHud=true;else if(arg=="--hud")o.noHud=false;
        else if(arg=="--race")launch.race=true;else if(arg=="--route")launch.route=next();
        else if(arg=="--pursuit"){launch.chase=true;launch.race=true;}
        else if(arg=="--fps")launch.fps=real(next());
        else if(arg=="--menu-delay")launch.menuDelay=real(next());
        else if(arg=="--players")launch.players=integer(next());
        else if(arg=="--record")launch.record=next();
        else if(arg=="--offline-record")launch.offlineDirectory=next();
        else if(arg=="--offline-auto-record")launch.offlineAuto=true;
        else if(arg=="--offline-input-script")launch.offlineScript=next();
        else if(arg=="--latency-demo"){launch.latencyDemo=true;launch.latency.mode=latency::parseMode(next());}
        else if(arg=="--latency-chunk-frames")launch.latency.chunkFrames=integer(next());
        else if(arg=="--latency-reference-fps")launch.latency.referenceFps=real(next());
        else if(arg=="--latency-inference-ms")launch.latency.inferenceMs=real(next());
        else if(arg=="--latency-fixed-ms")launch.latency.fixedMs=real(next());
        else if(arg=="--latency-input-script")launch.latency.inputScript=next();
        else if(arg=="--latency-screenshot")launch.latency.screenshot=next();
        else if(arg=="--reference-stream")launch.referenceDemo=true;
        else if(arg=="--reference-fps")launch.reference.stream.fps=real(next());
        else if(arg=="--reference-chunk-frames")launch.reference.stream.chunk=integer(next());
        else if(arg=="--reference-first-chunk-frames")launch.reference.stream.firstChunk=integer(next());
        else if(arg=="--reference-captions")launch.reference.captions=true;
        else if(arg=="--reference-record")launch.reference.record=true;
        else if(arg=="--reference-caption-width")launch.reference.captionWidth=integer(next());
        else if(arg=="--reference-input-capacity")launch.reference.stream.inputCapacity=integer(next());
        else if(arg=="--reference-output-capacity")launch.reference.stream.outputCapacity=integer(next());
        else if(arg=="--reference-inference-ms")launch.reference.stream.inferenceMs=real(next());
        else if(arg=="--reference-jitter-ms")launch.reference.stream.jitterMs=real(next());
        else if(arg=="--reference-seed"){int seed=integer(next());if(seed<0)throw std::runtime_error("Negative reference seed");launch.reference.stream.seed=unsigned(seed);}
        else if(arg=="--reference-input-script")launch.reference.inputScript=next();
        else if(arg=="--reference-screenshot")launch.reference.screenshot=next();
        else if(arg=="--vsync")o.vsync=true;else if(arg=="--hidden")launch.hidden=true;else throw std::runtime_error("Unknown option: "+arg);}
    if(o.width<640||o.width>1920||o.height<360||o.height>1080||o.traffic<0||o.traffic>100||o.pedestrians<0||o.pedestrians>200||o.physicsHz<100||o.physicsHz>1000||o.benchmark<0||o.benchmark>600||launch.duration<0||launch.duration>36000||launch.fps<0||launch.fps>240||launch.menuDelay<0||launch.menuDelay>120||launch.players<1||launch.players>5)throw std::runtime_error("Settings outside supported range");
    if(o.playerFrontGrip<=0||o.playerFrontGrip>3||o.playerRearGrip<=0||o.playerRearGrip>3)throw std::runtime_error("Player lateral grip scales must be in (0, 3]");
    if(o.playerFrontDrive<0||o.playerFrontDrive>1)throw std::runtime_error("--player-front-drive must be in [0, 1]");
    if(launch.audioVolume<0||launch.audioVolume>1)throw std::runtime_error("--audio-volume must be in [0, 1]");
    if(launch.chase&&(launch.players!=1||o.autodrive))throw std::runtime_error("Pursuit is manual single-player only");
    if(!launch.record.empty()&&launch.record.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")!=std::string::npos)throw std::runtime_error("Invalid recording name");
    if(launch.latency.chunkFrames<1||launch.latency.chunkFrames>120||launch.latency.referenceFps<1||launch.latency.referenceFps>120||
        launch.latency.inferenceMs<0||launch.latency.inferenceMs>5000||launch.latency.fixedMs<0||launch.latency.fixedMs>5000)
        throw std::runtime_error("Latency settings outside supported range");
    if(launch.latencyDemo&&(launch.race||launch.players!=1||!launch.record.empty()||o.autodrive||o.benchmark>0||launch.fps>0))
        throw std::runtime_error("Latency demo is manual single-player only; use --latency-reference-fps, not --fps");
    launch.latency.duration=launch.duration;
    if(launch.referenceDemo){
        if(launch.reference.captionWidth<64||launch.reference.captionWidth>2048)throw std::runtime_error("Caption width must be 64..2048");
        if(launch.reference.record&&std::abs(launch.reference.stream.fps-std::round(launch.reference.stream.fps))>1e-6)throw std::runtime_error("Reference recording needs integer FPS");
        if(launch.latencyDemo||launch.race||launch.players!=1||!launch.record.empty()||o.autodrive||o.benchmark>0||launch.fps>0)
            throw std::runtime_error("Reference streaming is a separate single-player mode; use --reference-fps");
        launch.reference.stream.physicsHz=o.physicsHz;launch.reference.stream.validate();launch.reference.duration=launch.duration;
    }
    if(!launch.offlineDirectory.empty()){
        if(!launch.race||launch.players!=1||launch.referenceDemo||launch.latencyDemo||!launch.record.empty()||o.benchmark>0||o.autodrive)
            throw std::runtime_error("Offline recording requires a manual single-player race");
        if(launch.fps==0)launch.fps=48;
        if(launch.fps<10||launch.fps>120||launch.fps!=std::round(launch.fps)||o.physicsHz!=std::round(o.physicsHz)||int(o.physicsHz)%int(launch.fps))
            throw std::runtime_error("Offline FPS must be an integer (10..120) dividing physics Hz exactly");
        launch.offlineDirectory=std::filesystem::absolute(launch.offlineDirectory);
    }else if(launch.offlineAuto||!launch.offlineScript.empty())throw std::runtime_error("Offline options require --offline-record DIR");
    return launch;
}
}
int main(int argc,char** argv){
    try{
        const auto launch=parse(argc,argv);const auto& opt=launch.opt;const auto& root=launch.root;
        if(launch.help){std::cout<<"CitySampleBBoxGame (SDL2 / OpenGL)\n"
            "--data-dir PATH  --width N --height N --traffic N --pedestrians N\n"
            "--physics-hz N --vsync --hud --no-hud --no-audio --autodrive\n"
            "--audio-volume N (0-1, default 0.6; normal SDL mode)\n"
            "--player-steering-limit on|off (default on; traffic AI unchanged)\n"
            "--player-front-grip N --player-rear-grip N (lateral only, defaults 1, range (0, 3])\n"
            "--player-front-drive N (0=RWD, 0.5=default AWD, 1=FWD; nominal split with existing LSD)\n"
            "--benchmark SECONDS --tag NAME --duration SECONDS --hidden\n"
            "--race --route ID (downtown|avenue|skyway; manual by default, --autodrive for demo)\n"
            "--pursuit (local pursuit, nearby suspect, full-city map; supports offline recording)\n"
            "--fps N (pace the frame loop, e.g. 60)\n"
            "--menu-delay SECONDS (show route select, then auto-start)\n"
            "--players N (1-5 race windows; P1 manual, other players AI; needs --race)\n"
            "--record NAME (record every window to recordings/NAME-pN.h264)\n"
            "--offline-record DIR (F9 starts/stops synchronized RGB/UI/audio/input takes; single-player race)\n"
            "--offline-auto-record (begin once racing); --offline-input-script PATH (test events JSON)\n"
            "--latency-demo normal|pulse|fixed (single-player experiment; F5/F6/F7 switch)\n"
            "--latency-chunk-frames N --latency-reference-fps N --latency-inference-ms N --latency-fixed-ms N\n"
            "--latency-input-script PATH --latency-screenshot PATH (scripted validation)\n"
            "--reference-stream --reference-fps N --reference-chunk-frames N (RGB mock streaming)\n"
            "--reference-input-capacity N --reference-output-capacity N (output includes in-flight reservation)\n"
            "--reference-inference-ms N --reference-jitter-ms N --reference-seed N\n"
            "--reference-input-script PATH --reference-screenshot PATH\n"
            "--reference-captions --reference-caption-width N --reference-record (aligned captions and optional RGB H264)\n"
            "--reference-first-chunk-frames N (0=same as subsequent chunks)\n"
            "--self-test --visual-test --map-test --underpass-test --ramp-ground-test --all-ramps-test\n"
            "WASD/arrows: drive; Space: handbrake; C: camera; Q: rear view; R: reset;\n"
            "Race: R recovers (+5 s); Enter selects menu items; Esc pauses/resumes.\n"
            "Pursuit: R restarts the mission; camera switching is disabled.\n"
            "Tab: HUD; Esc: pause; F8: timing; F11: fullscreen; F12: PNG screenshot.\n";return 0;}
        if(opt.selfTest)return selfTest(root)?0:2;
        SdlLifetime sdl;
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
#ifdef __APPLE__
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,4);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,1);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
#endif
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,1);SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,24);
        const int viewCount=launch.race?launch.players:1;
        const uint32_t windowFlags=SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE|SDL_WINDOW_ALLOW_HIGHDPI|(launch.hidden?SDL_WINDOW_HIDDEN:SDL_WINDOW_SHOWN);
        std::vector<std::unique_ptr<SDL_Window,WindowDelete>> window(viewCount);
        std::vector<std::unique_ptr<Renderer>> renderers(viewCount);
        for(int i=0;i<viewCount;++i){
            window[i].reset(SDL_CreateWindow((viewCount>1?"CitySample BBox Drive - P"+std::to_string(i+1):std::string("CitySample BBox Drive")).c_str(),SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,opt.width,opt.height,windowFlags));
            if(!window[i])throw std::runtime_error(SDL_GetError());SDL_SetWindowMinimumSize(window[i].get(),640,360);
            int w=0,h=0;SDL_GL_GetDrawableSize(window[i].get(),&w,&h);renderers[i]=std::make_unique<Renderer>(window[i].get(),w,h);
            if(!renderers[i]->adapterName.empty())std::cout<<"Renderer "<<i+1<<": "<<renderers[i]->adapterName<<'\n';
        }
        Renderer& renderer=*renderers[0];
        const bool offline=!launch.offlineDirectory.empty();
        if(offline){
            renderer.resize(reference::capture::RenderWidth,reference::capture::RenderHeight);
            renderer.outlineWidth=reference::capture::OutlineWidth;
            renderer.hudScale=reference::capture::HudScale;
        }
        World world(root/"assets/bbox-assets.bin",opt.traffic,opt.pedestrians,opt.visualTest);
        if(launch.referenceDemo&&launch.reference.captions)world.enableCaptions(root);
        std::vector<Renderer*> all;for(auto& r:renderers)all.push_back(r.get());
        world.uploadAll(all);
        if(auto result=runSceneTests(root,opt,renderer,world))return *result;
        world.setPlayerSteeringLimit(opt.autodrive||opt.playerSteeringLimit);
        world.setPlayerLateralGrip(opt.autodrive?1:opt.playerFrontGrip,opt.autodrive?1:opt.playerRearGrip);
        world.setPlayerFrontDrive(opt.autodrive?.5f:opt.playerFrontDrive);
        std::cout<<"Player steering speed limit: "<<((opt.autodrive||opt.playerSteeringLimit)?"on":"off")<<" (traffic AI unchanged; autodrive keeps the limit).\n";
        std::cout<<"Player lateral grip: front "<<(opt.autodrive?1:opt.playerFrontGrip)<<", rear "<<(opt.autodrive?1:opt.playerRearGrip)<<" (traffic AI and autodrive keep original grip).\n";
        std::cout<<"Player nominal front drive share: "<<(opt.autodrive?.5f:opt.playerFrontDrive)<<" (0=RWD, 1=FWD; traffic AI and autodrive keep 0.5).\n";
        if(launch.latencyDemo)return runLatencyDemo(window[0].get(),renderer,world,opt,root,launch.latency);
        if(launch.referenceDemo)return runReferenceDemo(window[0].get(),renderer,world,opt,root,launch.reference);
        std::optional<RaceSession> race;std::vector<std::string> routeIds,routeLabels;
        PlayerHandling handling{opt.autodrive?.5f:opt.playerFrontDrive,opt.autodrive?1:opt.playerFrontGrip,opt.autodrive?1:opt.playerRearGrip};
        bool customizing=false;int draggedSlider=-1;
        auto applyHandling=[&](){world.setPlayerLateralGrip(handling.frontGrip,handling.rearGrip);world.setPlayerFrontDrive(handling.frontDrive);};
        std::optional<LocalRaceControls> localRace;
        KeyboardSteering steering;
        std::vector<std::vector<GateVisual>> gates;std::vector<std::string> playerNames;
        std::vector<int> playerIds;std::vector<size_t> playerSlots,playerNearest;
        std::vector<uint64_t> playerSeq;std::vector<V3> playerLastPos;std::vector<double> playerStuck;std::vector<int> playerRecover;
        size_t raceSlot=0;double raceFinish=0,resultDisplaySeconds=0;
        std::unique_ptr<OfflineRecorder> offlineRecorder;
        uint64_t offlineTick=0,offlineFrame=0;
        bool recordToggle=false,autoRecordUsed=false;
        std::array<uint8_t,SDL_NUM_SCANCODES> offlineKeys{};
        nlohmann::json offlineEvents=nlohmann::json::array();size_t offlineEventIndex=0;
        if(!launch.offlineScript.empty()){
            std::ifstream file(launch.offlineScript);if(!file)throw std::runtime_error("Cannot open offline input script");file>>offlineEvents;
            if(!offlineEvents.is_array())throw std::runtime_error("Offline input script must be a JSON array");
            uint64_t previousFrame=0;
            for(const auto& e:offlineEvents){const auto frame=e.at("frame").get<uint64_t>();
                if(frame<previousFrame||SDL_GetScancodeFromName(e.at("key").get<std::string>().c_str())==SDL_SCANCODE_UNKNOWN||!e.at("down").is_boolean())
                    throw std::runtime_error("Invalid or unsorted offline input script");previousFrame=frame;}
        }
        auto finishTake=[&](){if(offlineRecorder){const auto path=offlineRecorder->directory();offlineRecorder->finish();
            std::cout<<"Offline take saved: "<<path<<" ("<<offlineRecorder->frames()<<" frames)\n"<<std::flush;offlineRecorder.reset();
            SDL_SetWindowTitle(window.front().get(),"CitySample BBox Drive");}};
        uint64_t audioFrameId=0,audioEpoch=0;
        size_t raceStartIndex=0;bool raceStarted=false;
        auto raceLobby=[&](){return race&&race->phase()=="lobby";};
        auto startRace=[&](size_t index){
            finishTake();
            steering.reset();
            customizing=false;draggedSlider=-1;
            raceStartIndex=index;raceStarted=true;raceFinish=0;resultDisplaySeconds=0;++audioEpoch;
            std::fill(playerNearest.begin(),playerNearest.end(),0);
            std::fill(playerStuck.begin(),playerStuck.end(),0);
            std::fill(playerLastPos.begin(),playerLastPos.end(),V3{});
            localRace->start(routeIds.at(index));};
        auto stepBots=[&](){
            if(launch.chase)return;
            for(size_t j=0;j<playerIds.size();++j){
                if(j==0&&!opt.autodrive)continue;
                const auto& p=race->players()[playerSlots[j]];if(!p)continue;
                int id=playerIds[j];auto& nearest=playerNearest[j];auto& seq=playerSeq[j];
                const auto& progress=p->progress;auto pose=world.playerPose(playerSlots[j]);V3 here=position(pose.body);
                Input in;
                const auto& routePath=race->route().path;V3 finishDir=(routePath.back()-routePath[routePath.size()>2?routePath.size()-3:0]).normalized();
                float beyond=dot(here-routePath.back(),finishDir);
                const bool pullOver=progress.finished&&beyond<14.f+8.f*float(j)&&playerIds.size()>1;
                if(!progress.finished||pullOver){const auto& path=routePath;float best=1e9f;size_t begin=nearest>2?nearest-2:0,end=std::min(path.size(),nearest+25);
                    for(size_t i=begin;i<end;++i){float d=(path[i]-here).length();if(d<best){best=d;nearest=i;}}
                    float speed=pose.speed/3.6f,lookAhead=std::clamp(7.5f+std::abs(speed)*.4f,8.f,19.f);size_t target=nearest;
                    while(target+1<path.size()&&(path[target]-here).length()<lookAhead)++target;
                    V3 aim=path[target];
                    if(pullOver)aim=path.back()+finishDir*(14.f+8.f*float(j))+cross({0,1,0},finishDir).normalized()*5.f;
                    V3 delta=aim-here;float angle=std::remainder(std::atan2(delta.x,delta.z)-pose.heading,2*Pi);
                    float desired=pullOver?6.f:std::abs(angle)>.8f?3.f:std::abs(angle)>.22f?6.f:11.f;
                    float gap=world.obstacleAhead(playerSlots[j],16.f,.3f);if(!pullOver&&gap<16.f)desired=std::min(desired,std::max(0.f,(gap-3.5f)*.85f));
                    if(!pullOver&&progress.next==0&&progress.elapsed<7)desired=std::min(desired,4.f+2.2f*float(j));
                    in.throttle=speed<desired?.55f:0.f;in.steer=std::clamp(angle*1.7f,-1.f,1.f);in.brake=speed>desired+.7f?.6f:0.f;}
                else in={0,0,1,1};
                race->command({{"type","input"},{"player_id",id},{"seq",++seq},{"throttle",in.throttle},{"steer",in.steer},{"brake",in.brake},{"handbrake",in.handbrake},{"camera",0},{"race_id",race->raceId()},{"epoch",race->epoch()}});
            }};
        auto unstick=[&](size_t index,double dt){
            if(launch.chase)return;
            if(!launch.race||race->phase()!="racing"||index>=playerIds.size())return;
            if(index==0&&!opt.autodrive)return;
            const auto& p=race->players()[playerSlots[index]];if(!p||p->progress.finished||!p->recovery.empty()){playerStuck[index]=0;return;}
            V3 here=position(world.playerPose(playerSlots[index]).body);
            if((here-playerLastPos[index]).length()>1.f||playerLastPos[index].length()==0){playerLastPos[index]=here;playerStuck[index]=0;return;}
            playerStuck[index]+=dt;if(playerStuck[index]<5)return;
            playerStuck[index]=0;race->command({{"type","recover"},{"player_id",playerIds[index]},{"request_id","auto-"+std::to_string(playerRecover[index]++)},{"race_id",race->raceId()},{"epoch",race->epoch()}});
        };
        auto raceMenu=[&]()->std::vector<std::string>{
            if(!race)return {};
            if(raceLobby()){
                if(customizing)return {"FRONT / REAR DRIVE","FRONT GRIP","REAR GRIP","BACK"};
                auto items=routeLabels;items.push_back("CUSTOMIZE");items.push_back("EXIT");return items;}
            if(race->phase()=="paused"){
                if(launch.chase)return {"RESUME","RESTART PURSUIT","SELECT MISSION","EXIT"};
                return {"RESUME","RECOVER  +5 S","RESTART RACE","SELECT ROUTE","EXIT"};
            }
            if(race->phase()=="results"){
                if(launch.chase&&resultDisplaySeconds<3)return {};
                return {"RETRY",launch.chase?"SELECT MISSION":"SELECT ROUTE","EXIT"};
            }
            return {};
        };
        auto fillView=[&](size_t view,const CarPose& player,HudState& hud,Camera& camera,const std::vector<GateVisual>*& viewGates){
            const auto& p=*race->players()[playerSlots[view]];
            hud.race=true;hud.playerNumber=int(playerSlots[view]+1);hud.elapsed=p.progress.elapsed;hud.penalty=p.progress.penalty;hud.finished=p.progress.finished;
            hud.nextCheckpoint=int(p.progress.next);hud.totalCheckpoints=int(race->route().checkpoints.size());hud.notice=p.notice;
            hud.racePhase=race->phase();std::transform(hud.racePhase.begin(),hud.racePhase.end(),hud.racePhase.begin(),[](unsigned char c){return char(std::toupper(c));});
            hud.cameraMode=p.camera;hud.routeGuide=&race->route().guide;
            hud.navigation=navigateRoute(race->route(),p.progress,hud.location,player.heading,player.speed);
            if(race->phase()=="countdown")hud.notice="STARTING "+std::to_string(std::max(1,int(std::ceil(race->countdownSeconds()))));
            hud.menu=raceMenu();
            if(launch.chase){
                const auto chase=race->pursuitState();hud.pursuit=true;hud.routeGuide=nullptr;hud.navigation={};hud.notice.clear();
                hud.suspectPosition={chase.at("position")[0],chase.at("position")[1],chase.at("position")[2]};
                hud.suspectHeading=chase.at("heading");hud.suspectDistance=chase.at("distance_m");
                hud.captureSeconds=chase.at("capture_held_s");hud.captureEligible=chase.at("capture_eligible");
                hud.racePhase=race->phase()=="racing"?"PURSUIT":race->phase()=="results"?(p.progress.finished?"CAPTURED":"ENDED"):hud.racePhase;
                if(race->phase()=="countdown")hud.notice="STARTING "+std::to_string(std::max(1,int(std::ceil(race->countdownSeconds()))));
            }
            if(raceLobby()){
                hud.menuTitle=customizing?"CUSTOMIZE":launch.chase?"CITY PURSUIT":"SELECT ROUTE";hud.paused=false;
                if(customizing)hud.handling=handling;
            }
            else if(race->phase()=="paused"){
                hud.menuTitle="PAUSED";
            }else if(race->phase()=="results"){
                hud.menuTitle=launch.chase?(p.progress.finished?"SUSPECT CAPTURED":"PURSUIT ENDED"):(p.progress.finished?"FINISHED":"RACE ENDED");
                std::ostringstream score;score<<std::fixed<<std::setprecision(2)<<"TOTAL "<<p.progress.total()<<" S";
                if(!launch.chase)score<<"  +"<<p.progress.penalty<<" PENALTY";
                hud.menuDetail=score.str();
            }
            auto& viewGatesStorage=gates[view];viewGatesStorage.clear();
            if(!launch.chase&&!p.progress.finished){const auto& gate=race->route().checkpoints.at(p.progress.next);GateVisual visual{gate.center,gate.forward,gate.halfWidth,gate.height};
                if(p.progress.next+1==race->route().checkpoints.size())visual.color={.35f,.95f,.63f,.95f};viewGatesStorage.push_back(visual);}
            viewGates=&viewGatesStorage;camera=world.playerCamera(playerSlots[view],p.camera,p.lookBack);};
        if(launch.race){
            auto routes=loadRaceRoutes(root/"assets/race-routes.json",world);
            if(launch.chase)std::erase_if(routes,[](const auto& r){return r.id!="avenue";});
            for(const auto& r:routes){std::string id=r.id;std::transform(id.begin(),id.end(),id.begin(),[](unsigned char c){return char(std::toupper(c));});
                std::ostringstream label;label<<id<<"  "<<std::fixed<<std::setprecision(1)<<r.length/1000<<" KM  "<<r.checkpoints.size()<<" GATES";
                routeIds.push_back(r.id);routeLabels.push_back(launch.chase?"START PURSUIT":label.str());}
            race.emplace(world,std::move(routes),launch.chase);
            const size_t rosterSize=launch.chase?2:size_t(launch.players);
            gates.assign(size_t(viewCount),{});playerNearest.assign(rosterSize,0);playerSeq.assign(rosterSize,0);
            playerLastPos.assign(rosterSize,{});playerStuck.assign(rosterSize,0);playerRecover.assign(rosterSize,0);
            for(size_t i=0;i<rosterSize;++i){playerNames.push_back(launch.chase&&i==1?"Suspect":viewCount>1?"Player "+std::to_string(i+1):"Player 1");
                auto joined=race->command({{"type","join"},{"name",playerNames.back()}});playerIds.push_back(joined.at("player_id"));playerSlots.push_back(joined.at("slot").get<size_t>());}
            raceSlot=playerSlots[0];
            localRace.emplace(*race,playerIds);
            for(size_t i=0;i<playerSlots.size();++i)std::cout<<(launch.chase&&i==1?"Suspect AI":"View "+std::to_string(i+1))<<" -> player "<<playerIds[i]<<" slot "<<playerSlots[i]<<'\n';
            for(int id:playerIds)race->command({{"type","focus"},{"value",true},{"player_id",id}});
            if(launch.chase){
                race->command({{"type","select_route"},{"route_id","avenue"},{"player_id",playerIds[0]}});
                std::cout<<"Pursuit initial distance: "<<race->pursuitState().at("initial_distance_m")<<" m\n";
            }
            if(!launch.route.empty()){
                auto it=std::find(routeIds.begin(),routeIds.end(),launch.route);
                if(it==routeIds.end())throw std::runtime_error("Unknown route: "+launch.route);
                raceStartIndex=size_t(it-routeIds.begin());
                if(launch.menuDelay<=0){startRace(raceStartIndex);raceStarted=true;}
            }
        }
        VehicleAudio audio(root/"assets/audio",!opt.noAudio,launch.audioVolume);
        Controller controller;controller.refresh();CameraTelemetry telemetry;
        bool running=true,paused=false,active=true,fullscreen=false,hudVisible=!opt.noHud,performance=opt.benchmark>0,capture=false,resetRequested=false;
        const bool benchmark=opt.benchmark>0;int mode=1,selection=0;uint64_t captureId=0;
        auto setPaused=[&](bool value){
            if(value)finishTake();
            if(value)steering.reset();
            if(localRace){if(value)localRace->pause();else localRace->resume();paused=race->phase()=="paused";}
            else paused=value;
            selection=0;
        };
        auto selectRoutes=[&](){finishTake();localRace->lobby();steering.reset();paused=false;customizing=false;draggedSlider=-1;selection=int(raceStartIndex);raceFinish=0;++audioEpoch;};
        auto closeHandling=[&](){customizing=false;draggedSlider=-1;selection=int(routeIds.size());};
        auto command=[&](){
            if(!race){if(selection==0)paused=false;else if(selection==1){resetRequested=true;paused=false;}else running=false;return;}
            if(raceLobby()){
                if(customizing){if(selection==3)closeHandling();}
                else if(size_t(selection)<routeIds.size()){startRace(size_t(selection));selection=0;}
                else if(size_t(selection)==routeIds.size()){
                    for(int row=0;row<3;++row)handling.set(row,handling.value(row));
                    applyHandling();customizing=true;selection=0;
                }else running=false;
            }else if(race->phase()=="paused"){
                if(launch.chase){
                    if(selection==0)setPaused(false);
                    else if(selection==1){startRace(raceStartIndex);paused=false;selection=0;}
                    else if(selection==2)selectRoutes();else running=false;
                }else if(selection==0)setPaused(false);
                else if(selection==1){finishTake();if(localRace->recover()){++audioEpoch;steering.reset();}setPaused(false);}
                else if(selection==2){startRace(raceStartIndex);paused=false;selection=0;}
                else if(selection==3)selectRoutes();else running=false;
            }else if(race->phase()=="results"){
                if(selection==0){startRace(raceStartIndex);selection=0;}
                else if(selection==1)selectRoutes();else running=false;
            }
        };
        auto events=[&](){bool focusChanged=false;SDL_Event event;while(SDL_PollEvent(&event)){
            if(offline&&(event.type==SDL_KEYDOWN||event.type==SDL_KEYUP)&&!event.key.repeat){
                offlineKeys.at(size_t(event.key.keysym.scancode))=event.type==SDL_KEYDOWN;
                if(offlineRecorder)offlineRecorder->input({{"event","key"},{"key",SDL_GetScancodeName(event.key.keysym.scancode)},
                    {"down",event.type==SDL_KEYDOWN},{"sdl_timestamp_ms",event.key.timestamp},{"received_monotonic_s",now()},{"physics_tick",offlineTick}});
            }
            if(offlineRecorder&&(event.type==SDL_CONTROLLERAXISMOTION||event.type==SDL_CONTROLLERBUTTONDOWN||event.type==SDL_CONTROLLERBUTTONUP)){
                nlohmann::json e={{"event","controller"},{"received_monotonic_s",now()},{"physics_tick",offlineTick}};
                if(event.type==SDL_CONTROLLERAXISMOTION)e.update({{"axis",event.caxis.axis},{"value",event.caxis.value}});
                else e.update({{"button",event.cbutton.button},{"down",event.type==SDL_CONTROLLERBUTTONDOWN}});
                offlineRecorder->input(e);
            }
            if(event.type==SDL_QUIT)running=false;
            if(event.type==SDL_CONTROLLERDEVICEADDED||event.type==SDL_CONTROLLERDEVICEREMOVED)controller.refresh();
            if(event.type==SDL_WINDOWEVENT){
                if(event.window.event==SDL_WINDOWEVENT_CLOSE)running=false;
                if(event.window.event==SDL_WINDOWEVENT_FOCUS_LOST||event.window.event==SDL_WINDOWEVENT_FOCUS_GAINED)focusChanged=true;
            }
            if(customizing&&event.type==SDL_KEYDOWN&&(event.key.keysym.sym==SDLK_LEFT||event.key.keysym.sym==SDLK_RIGHT)){
                if(selection<3){handling.adjust(selection,event.key.keysym.sym==SDLK_RIGHT?1:-1);applyHandling();}
            }
            if(event.type==SDL_KEYDOWN&&!event.key.repeat){auto key=event.key.keysym.sym;
                if(key==SDLK_ESCAPE){
                    if(customizing)closeHandling();
                    else if(raceLobby())running=false;
                    else if(race&&race->phase()=="results")selectRoutes();
                    else setPaused(!paused);
                }
                if(key==SDLK_r)resetRequested=true;if(key==SDLK_c&&!offline&&!launch.chase)mode=(mode+1)%3;if(key==SDLK_TAB)hudVisible=!hudVisible;
                if(key==SDLK_F9&&offline)recordToggle=true;
                if(key==SDLK_F8)performance=!performance;if(key==SDLK_F12)capture=true;
                if(key==SDLK_F4&&(event.key.keysym.mod&KMOD_ALT))running=false;
                if(key==SDLK_F11){bool wanted=!fullscreen;for(auto& view:window)SDL_SetWindowFullscreen(view.get(),wanted?SDL_WINDOW_FULLSCREEN_DESKTOP:0);fullscreen=wanted;}
                if(race&&!raceMenu().empty()){int count=int(raceMenu().size());
                    if(key==SDLK_UP||key==SDLK_w)selection=(selection+count-1)%count;if(key==SDLK_DOWN||key==SDLK_s)selection=(selection+1)%count;
                    if(key==SDLK_RETURN)command();}
                else if(paused&&hudVisible){if(key==SDLK_UP||key==SDLK_w)selection=(selection+2)%3;if(key==SDLK_DOWN||key==SDLK_s)selection=(selection+1)%3;if(key==SDLK_RETURN)command();}}
            if(((race&&!raceMenu().empty())||(paused&&hudVisible))&&(event.type==SDL_MOUSEMOTION||event.type==SDL_MOUSEBUTTONUP||event.type==SDL_MOUSEBUTTONDOWN)){
                if((event.type==SDL_MOUSEMOTION?event.motion.windowID:event.button.windowID)!=SDL_GetWindowID(window.front().get()))continue;
                const int hudWidth=offline?reference::capture::HudWidth:renderer.width;
                const int hudHeight=offline?reference::capture::HudHeight:renderer.height;
                int ww,wh;SDL_GetWindowSize(window.front().get(),&ww,&wh);float x=float(event.type==SDL_MOUSEMOTION?event.motion.x:event.button.x)*hudWidth/std::max(ww,1),y=float(event.type==SDL_MOUSEMOTION?event.motion.y:event.button.y)*hudHeight/std::max(wh,1);
                if(offline)y=(float(event.type==SDL_MOUSEMOTION?event.motion.y:event.button.y)/std::max(wh,1)*480+3)*hudHeight/486;
                if(race){
                    RaceMenuLayout layout(hudWidth,hudHeight,customizing);
                    int row=layout.hit(x,y,int(raceMenu().size()));
                    if(draggedSlider>=0){
                        if(event.type==SDL_MOUSEMOTION||(event.type==SDL_MOUSEBUTTONUP&&event.button.button==SDL_BUTTON_LEFT)){
                            handling.slide(draggedSlider,layout.fraction(x));applyHandling();
                            if(event.type==SDL_MOUSEBUTTONUP)draggedSlider=-1;
                        }
                    }else if(row>=0){
                        selection=row;
                        if(event.type==SDL_MOUSEBUTTONDOWN&&event.button.button==SDL_BUTTON_LEFT&&layout.hitTrack(x,y,row)){
                            draggedSlider=row;handling.slide(row,layout.fraction(x));applyHandling();
                        }else if(event.type==SDL_MOUSEBUTTONUP&&event.button.button==SDL_BUTTON_LEFT)command();
                    }
                }else {
                    float mx=hudWidth*.5f-160,my=hudHeight*.5f-59;
                    for(int i=0;i<3;++i)if(x>=mx&&x<mx+305&&y>=my+i*48&&y<my+i*48+38){selection=i;if(event.type==SDL_MOUSEBUTTONUP&&event.button.button==SDL_BUTTON_LEFT){command();break;}}
                }
            }
        }
            if(focusChanged){auto* focused=SDL_GetKeyboardFocus();active=std::any_of(window.begin(),window.end(),[&](const auto& w){return w.get()==focused;});
                if(!active){draggedSlider=-1;steering.reset();offlineKeys.fill(0);if(offline)finishTake();}
                if(!active&&!benchmark&&!launch.hidden&&!(race&&opt.autodrive))setPaused(true);}
        };
        std::vector<Sample> samples;double start=now(),previous=start,lastFrame=start,accumulator=0,cpuAverage=0,frameAverage=3,previousCpu=0,previousPresent=0;
        bool benchmarkCaptured=false,previousScreenshot=false;
        std::vector<std::unique_ptr<ViewRecorder>> recorders;
        std::vector<int64_t> recordStamp(viewCount,0);
        uint64_t recordedFrames=0;
        uint64_t frameCount=0;
        if(!launch.record.empty()){
            int bitrate=launch.players>1?6000:12000;
            if(!launch.race)throw std::runtime_error("--record requires --race");
            for(int i=0;i<viewCount;++i)recorders.push_back(std::make_unique<ViewRecorder>(root/"recordings"/(launch.record+(viewCount>1?"-p"+std::to_string(i+1):"")+".h264"),opt.width,opt.height,int(launch.fps>0?launch.fps:60),bitrate,1));
            std::cout<<"Recording "<<recorders.size()<<" view"<<(recorders.size()>1?"s":"")<<" at "<<opt.width<<"x"<<opt.height<<'\n';
        }
        while(running){
            if(offline){
                renderer.resize(reference::capture::RenderWidth,reference::capture::RenderHeight);
                while(offlineEventIndex<offlineEvents.size()&&offlineEvents[offlineEventIndex].at("frame").get<uint64_t>()<=offlineFrame){
                    const auto& e=offlineEvents[offlineEventIndex++];SDL_Event key{};
                    key.type=e.at("down").get<bool>()?SDL_KEYDOWN:SDL_KEYUP;key.key.timestamp=SDL_GetTicks();
                    key.key.keysym.scancode=SDL_GetScancodeFromName(e.at("key").get<std::string>().c_str());
                    key.key.keysym.sym=SDL_GetKeyFromScancode(key.key.keysym.scancode);SDL_PushEvent(&key);
                }
            }
            const double messageStart=now();events();const double messages=(now()-messageStart)*1000;if(!running)break;
            const double resizeStart=now();int cw=0,ch=0;SDL_GL_GetDrawableSize(window.front().get(),&cw,&ch);
            if(!offline)for(size_t i=0;i<renderers.size();++i){int vw=0,vh=0;SDL_GL_GetDrawableSize(window[i].get(),&vw,&vh);if(vw!=renderers[i]->width||vh!=renderers[i]->height){renderers[i]->makeCurrent();renderers[i]->resize(vw,vh);}}
            double resize=(now()-resizeStart)*1000;
            if(cw<=0||ch<=0||(SDL_GetWindowFlags(window.front().get())&SDL_WINDOW_MINIMIZED)){
                if(offline)finishTake();
                steering.reset();
                VehicleAudioFrame silent;silent.frame=++audioFrameId;silent.epoch=audioEpoch;silent.muted=true;audio.present(silent);
                SDL_Delay(20);previous=lastFrame=now();if(launch.duration>0&&now()-start>=launch.duration)break;continue;}
            double wait=0;const double frameStart=now(),interval=(frameStart-lastFrame)*1000;lastFrame=frameStart;
            for(size_t i=0;i<renderers.size();++i){renderers[i]->makeCurrent();wait+=renderers[i]->waitForFrame(false);}
            events();if(!running)break;
            if(race&&!raceStarted&&raceLobby()&&!customizing&&launch.menuDelay>0&&now()-start>=launch.menuDelay){startRace(raceStartIndex);raceStarted=true;}
            double dt=offline?1.0/launch.fps:std::min(frameStart-previous,.05);previous=frameStart;Input input;bool keyboardSteering=true;
            if(resetRequested){
                finishTake();
                if(launch.chase){startRace(raceStartIndex);paused=false;selection=0;}
                else if(localRace){if(localRace->recover()){++audioEpoch;steering.reset();}}
                else {world.reset();++audioEpoch;steering.reset();}
                accumulator=0;resetRequested=false;
            }
            VehicleAudioFrame audioFrame;audioFrame.frame=++audioFrameId;audioFrame.epoch=audioEpoch;
            const Uint8* keys=offline?offlineKeys.data():SDL_GetKeyboardState(nullptr);
            const bool drivingActive=active||(offline&&launch.hidden&&!launch.offlineScript.empty());
            const bool rearView=!offline&&!launch.chase&&active&&keys[SDL_SCANCODE_Q];
            if(drivingActive&&!paused&&(!race||race->phase()=="racing")){input.throttle=float(keys[SDL_SCANCODE_W]||keys[SDL_SCANCODE_UP])-float(keys[SDL_SCANCODE_S]||keys[SDL_SCANCODE_DOWN]);input.steer=float(keys[SDL_SCANCODE_D]||keys[SDL_SCANCODE_RIGHT])-float(keys[SDL_SCANCODE_A]||keys[SDL_SCANCODE_LEFT]);input.handbrake=keys[SDL_SCANCODE_SPACE]?1.f:0;
                auto pad=controller.read();if(std::abs(pad.throttle)>.01f)input.throttle=pad.throttle;
                if(std::abs(pad.steer)>.01f){input.steer=pad.steer;keyboardSteering=false;}input.handbrake=std::max(input.handbrake,pad.handbrake);}
            if(localRace&&race->phase()!="racing"){
                steering.reset();if(!opt.autodrive)localRace->drive({},mode,rearView);
            }
            nlohmann::json appliedInputs=nlohmann::json::array();
            auto inputJson=[](const Input& value){return nlohmann::json{{"throttle",value.throttle},{"steer",value.steer},{"brake",value.brake},{"handbrake",value.handbrake}};};
            const double physicsStart=now();int updates=0;
            if(!paused){accumulator+=dt;double step=1.0/opt.physicsHz;
                const int offlineSteps=offline?int(opt.physicsHz)/int(launch.fps):0;
                while(offline?updates<offlineSteps:(accumulator>=step&&updates<25)){
                if(race){
                    if(race->phase()=="racing"){
                        if(!opt.autodrive){const auto applied=steering.apply(input,step,keyboardSteering);localRace->drive(applied,mode,rearView);
                            if(offline){auto row=inputJson(applied);row["tick"]=offlineTick+1;appliedInputs.push_back(row);}}
                        stepBots();
                    }
                    const size_t previousGate=race->players()[raceSlot]->progress.next;
                    const bool wasFinished=race->players()[raceSlot]->progress.finished;
                    race->tick(step);
                    const size_t currentGate=race->players()[raceSlot]->progress.next;
                    if(currentGate>previousGate)audioFrame.checkpoint=currentGate;
                    if(launch.chase&&!wasFinished&&race->players()[raceSlot]->progress.finished)audioFrame.checkpoint=1;
                    const uint64_t entity=world.playerCar(raceSlot)+1;
                    for(const auto& event:race->collisions())if(event.entityA==entity||event.entityB==entity)audioFrame.addImpact(event.id,event.speed);
                }
                else {
                    world.update(float(step),opt.autodrive?input:steering.apply(input,step,keyboardSteering),opt.autodrive);
                    // Physics clears events at the next substep, not the next display frame.
                    for(const auto& event:world.takeCollisions())if(event.entityA==1||event.entityB==1)audioFrame.addImpact(event.id,event.speed);
                }
                for(size_t i=0;i<playerIds.size();++i)unstick(i,step);
                accumulator-=step;++updates;if(offline)++offlineTick;}if(offline||updates==25)accumulator=0;}else accumulator=0;
            const double physicsMs=(now()-physicsStart)*1000;auto player=race?world.playerPose(raceSlot):world.player();
            audioFrame.rpm=player.engineRPM;audioFrame.throttle=player.engineThrottle;audioFrame.rearLateralSlipSpeed=player.rearLateralSlipSpeed;
            audioFrame.groundedRearWheels=player.groundedRearWheels;audioFrame.muted=paused||(race&&race->phase()!="racing");
            audioFrame.frontLateralSlipSpeed=player.frontLateralSlipSpeed;audioFrame.groundedFrontWheels=player.groundedFrontWheels;
            const auto& objects=world.drawObjects();Camera camera=world.camera(mode,rearView);HudState hud;Camera primary;
            if(offline&&launch.offlineAuto&&!autoRecordUsed&&race->phase()=="racing"){recordToggle=true;autoRecordUsed=true;}
            if(recordToggle){
                recordToggle=false;
                if(offlineRecorder)finishTake();
                else if(race->phase()=="racing"){
                    std::filesystem::create_directories(launch.offlineDirectory);std::filesystem::path take;
                    for(int n=1;;++n){std::ostringstream name;name<<"take-"<<std::setfill('0')<<std::setw(3)<<n;
                        take=launch.offlineDirectory/name.str();if(!std::filesystem::exists(take))break;}
                    nlohmann::json metadata={{"mode",launch.chase?"pursuit":"time_trial"},{"route",race->route().id},{"physics_hz",opt.physicsHz},{"camera_mode",1},
                        {"camera_profile","dataset_chase_raised_v2"},{"render_profile","dataset"},{"ground_profile","road_contrast_junction_v1"},
                        {"render_width",reference::capture::RenderWidth},{"render_height",reference::capture::RenderHeight},
                        {"hud_layout_width",reference::capture::HudWidth},{"hud_layout_height",reference::capture::HudHeight},{"hud_render_scale",renderer.hudScale},
                        {"resize","antialiased_bilinear_864x486_center_crop_864x480"},{"building_stripes",renderer.buildingStripes},
                        {"outline_width",renderer.outlineWidth},{"traffic",opt.traffic},{"pedestrians",opt.pedestrians},
                        {"handling",{{"front_drive",handling.frontDrive},{"front_grip",handling.frontGrip},{"rear_grip",handling.rearGrip},{"steering_limit",opt.playerSteeringLimit},
                            {"forward_speed_limit_kmh",World::PlayerMaxSpeedKmh},{"speed_limit_throttle_taper_kmh",5}}},
                        {"start_monotonic_s",now()},{"start_simulation_s",world.simulationTime},{"start_tick",offlineTick},
                        {"prompt","Daytime city, cloudy sky. Center: player-controlled modern four-door sedan with a curved roof and a separate rear trunk, seen from behind in the foreground. Around: urban streets and multi-story buildings."}};
                    if(launch.chase){
                        metadata["pursuit"]=race->pursuitState();
                        const auto center=world.mapCenter();const float range=std::max(100.f,world.mapHalfSize())*1.04f;
                        metadata["map"]=reference::presentation_detail::makeMap(world.mapLines(),center,nullptr,0,0,range,world.mapLines().size());
                        metadata["map"].update({{"center",{center.x,center.y,center.z}},{"half_range_m",range},{"north_up",true}});
                    }
                    offlineRecorder=std::make_unique<OfflineRecorder>(take,root/"assets/audio",864,480,int(launch.fps),opt.noAudio?0:launch.audioVolume,metadata);
                    auto heldKeys=nlohmann::json::array();for(size_t key=0;key<offlineKeys.size();++key)
                        if(offlineKeys[key])heldKeys.push_back(SDL_GetScancodeName(SDL_Scancode(key)));
                    offlineRecorder->input({{"event","initial_state"},{"input",inputJson(input)},{"held_keys",heldKeys},{"physics_tick",offlineTick}});
                    std::cout<<"Offline recording started: "<<take<<"; F9 stops.\n"<<std::flush;
                    SDL_SetWindowTitle(window.front().get(),"CitySample BBox Drive - REC");
                }else std::cout<<"Start recording with F9 after the countdown.\n"<<std::flush;
            }
            std::vector<std::vector<uint8_t>> viewPixels(renderers.size());
            const bool recording=!recorders.empty();
            double present=0;bool newGpu=false;
            const bool grab=capture||(benchmark&&!benchmarkCaptured&&frameStart-start>(race?12:2));
            for(size_t view=0;view<renderers.size();++view){
                ++frameCount;
                auto& target=*renderers[view];
                const auto& pose=race?world.playerPose(playerSlots[view]):player;
                hud={};hud.speed=pose.speed;hud.location=position(pose.body);hud.heading=pose.heading;hud.visible=hudVisible||bool(race);hud.performance=performance;hud.paused=paused;hud.pauseSelection=selection;hud.cameraMode=mode;hud.roads=&world.mapLines();hud.mapCenter=world.mapCenter();hud.mapHalfSize=world.mapHalfSize();hud.cpuMs=cpuAverage;hud.gpuMs=target.gpuMs();hud.frameMs=frameAverage;
                const std::vector<GateVisual>* viewGates=nullptr;
                if(race)fillView(view,pose,hud,camera,viewGates);
                // Collect the frame this view handed to the GPU last iteration.
                // It must be taken before the next draw overwrites the target.
                if(recording){target.makeCurrent();
                    if(target.readbackPending(view))recorders[view]->submit(target.finishReadback(view,recorders[view]->takeReusable()),recordStamp[view]);}
                target.makeCurrent();
                if(offline){
                    auto layers=target.renderLayers(objects,camera,hud,viewGates?*viewGates:std::vector<GateVisual>{},bool(offlineRecorder),-1,864,480);
                    if(offlineRecorder){
                        reference::PresentationFrameInfo info;info.frame=offlineRecorder->frames();info.captureMonotonic=frameStart;
                        info.muted=audioFrame.muted;info.checkpoint=audioFrame.checkpoint;
                        nlohmann::json impacts=nlohmann::json::array();for(size_t k=0;k<audioFrame.impactCount;++k)impacts.push_back({{"id",audioFrame.impacts[k].id},{"speed",audioFrame.impacts[k].speed}});
                        auto state=reference::makeGameFrame(world,camera,pose,info,&*race,impacts);
                        state.update({{"physics_tick",offlineTick},{"physics_updates",updates},{"input_raw",inputJson(input)},{"input_applied",appliedInputs},
                            {"keyboard_steering",keyboardSteering},{"camera",{{"eye",{camera.eye.x,camera.eye.y,camera.eye.z}},
                            {"target",{camera.target.x,camera.target.y,camera.target.z}},{"horizontal_fov",camera.horizontalFov}}},
                            {"wall_s",frameStart-start},{"audio_epoch",audioFrame.epoch}});
                        auto appliedEvent=nlohmann::json{{"event","applied"},{"physics_tick",offlineTick},{"input",inputJson(input)},{"substeps",appliedInputs}};
                        offlineRecorder->input(appliedEvent);
                        offlineRecorder->submit(std::move(layers.scene),std::move(layers.ui),std::move(state),audioFrame);
                    }
                    target.resize(864,480);target.displayPixels(layers.display,{});
                }else target.render(objects,camera,hud,viewGates?*viewGates:std::vector<GateVisual>{});
                if(recordedFrames==2)std::cout<<"view "<<view+1<<" P"<<hud.playerNumber<<" eye "<<int(camera.eye.x)<<","<<int(camera.eye.z)<<" hud.x "<<int(hud.location.x)<<","<<int(hud.location.z)<<'\n';
                if(view==0){primary=camera;newGpu=target.newGpuSample;}
                if(grab)viewPixels[view]=target.readBack();
                if(recording){target.beginReadback(view);recordStamp[view]=int64_t((frameStart-start)*1000000);}
                present+=target.present(opt.vsync);
                if(view==0)audio.present(audioFrame);
            }
            const int cameraMode=race?race->players()[raceSlot]->camera:mode;
            const bool lookBack=race?race->players()[raceSlot]->lookBack:rearView;
            uint32_t flags=(paused?CameraTelemetry::Paused:0u)|(active?CameraTelemetry::Active:0u)|(lookBack?CameraTelemetry::LookBack:0u);telemetry.publish(primary,renderer.width,renderer.height,cameraMode,flags,frameStart-start,world.simulationTime);
            bool screenshot=false;
            if(grab){for(size_t view=0;view<renderers.size();++view){std::string suffix=renderers.size()>1?"-p"+std::to_string(view+1):"";
                renderers[view]->makeCurrent();
                renderers[view]->savePng(root/"captures"/((benchmark?opt.tag:"drive-"+std::to_string(uint64_t(frameStart*1000))+"-"+std::to_string(captureId))+suffix+".png"),viewPixels[view]);}
                ++captureId;capture=false;benchmarkCaptured=true;screenshot=true;}
            const double cpu=(now()-frameStart)*1000;cpuAverage=cpuAverage*.97+cpu*.03;frameAverage=frameAverage*.97+interval*.03;
            const double unaccounted=std::max(0.0,interval-previousCpu-previousPresent-wait-messages-resize);
            if(benchmark&&frameStart-start>4&&!screenshot&&!previousScreenshot)samples.push_back({interval,cpu,newGpu?renderer.gpuMs():-1,physicsMs,present,wait,messages,resize,unaccounted,updates});
            previousCpu=cpu;previousPresent=present;previousScreenshot=screenshot;
            if(race&&race->phase()=="results")resultDisplaySeconds+=dt;
            if(offlineRecorder&&race->phase()=="results"&&(!launch.chase||resultDisplaySeconds>=3))finishTake();
            if(offline)++offlineFrame;
            if((benchmark&&frameStart-start>opt.benchmark+4)||(launch.duration>0&&frameStart-start>=launch.duration))running=false;
            if(race&&race->phase()=="results"&&raceFinish==0){raceFinish=frameStart;selection=0;}
            if(opt.autodrive&&raceFinish>0&&frameStart-raceFinish>6)running=false;
            if(paused&&!benchmark)SDL_Delay(12);
            if(launch.fps>0){double budget=1.0/launch.fps,spent=now()-frameStart;
                if(spent<budget){double left=budget-spent;if(left>.0015)std::this_thread::sleep_for(std::chrono::duration<double>(left-.0012));while(now()-frameStart<budget)std::this_thread::yield();}}
        }
        finishTake();
        if(race){
            // Drain the readback submitted on the last iteration.
            for(size_t view=0;view<renderers.size();++view)
                if(view<recorders.size()&&renderers[view]->readbackPending(view)){renderers[view]->makeCurrent();
                    recorders[view]->submit(renderers[view]->finishReadback(view,recorders[view]->takeReusable()),recordStamp[view]);}
            uint64_t encoded=0,dropped=0;for(auto& recorder:recorders){encoded+=recorder->encoded();dropped+=recorder->dropped();}
            for(auto& recorder:recorders)recorder->finish();
            if(!recorders.empty()){
                double span=recorders.front()->spanSeconds();
                std::cout<<"Recorded "<<encoded<<" frames per view over "<<span<<" s ("
                    <<(span>0?double(recorders.front()->encoded())/span:0)<<" fps), dropped "<<dropped<<'\n';
            }
            auto state=race->state();int finished=0;std::string summary;
            for(const auto& m:state.at("players")){
                bool done=m.at("finished").get<bool>();finished+=done?1:0;
                summary+=" P"+std::to_string(m.at("slot").get<int>()+1)+" "+std::to_string(m.at("next").get<size_t>())+"/"+std::to_string(race->route().checkpoints.size())+(done?"F ":" ")
                    +std::to_string(int(m.at("total").get<double>()))+"s";
            }
            std::cout<<"Race "<<race->route().id<<" ("<<playerIds.size()<<" players): "<<race->phase()<<", "<<finished<<"/"<<playerIds.size()<<" finished;"<<summary<<'\n';
        }
        if(benchmark)report(root,opt,renderer,world,samples);return 0;
    }catch(const std::exception& error){std::cerr<<"CitySampleBBoxGame: "<<error.what()<<'\n';return 1;}
}
