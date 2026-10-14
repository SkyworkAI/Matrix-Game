#include "Map.h"
#include "Recording.h"
#include "OfflineDemo.h"
#include <SDL.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <cstdlib>

namespace {
#ifdef __APPLE__
constexpr const char* Platform="macOS";
constexpr const char* RuntimeLog="runtime-macos.log";
#else
constexpr const char* Platform="Linux";
constexpr const char* RuntimeLog="runtime-linux.log";
#endif
struct Options {int width=1280,height=720;uint32_t seed=1729;float duration=10,coverProbability=.5f;bool record=false,demo=false,map=false,vsync=false,fullscreen=false,manual=false,offline=false;DemoOptions offlineOptions;std::string test,view="all",scene="castle";std::filesystem::path output,capture,benchmark;};
Options parse(int argc,char** argv){Options o;for(int i=1;i<argc;++i){std::string arg=argv[i];auto value=[&]{if(++i>=argc)throw std::runtime_error("Missing option value");return std::string(argv[i]);};
    if(arg=="--width")o.width=std::stoi(value());else if(arg=="--height")o.height=std::stoi(value());else if(arg=="--seed")o.seed=uint32_t(std::stoul(value()));else if(arg=="--duration")o.duration=std::stof(value());else if(arg=="--cover-probability")o.coverProbability=std::stof(value());else if(arg=="--recording-dir")o.output=value();else if(arg=="--capture")o.capture=value();else if(arg=="--benchmark")o.benchmark=value();else if(arg=="--view")o.view=value();else if(arg=="--record")o.record=true;else if(arg=="--map")o.map=true;else if(arg=="--vsync")o.vsync=true;else if(arg=="--fullscreen")o.fullscreen=true;
    else if(arg=="--scene")o.scene=value();
    else if(arg=="--manual")o.manual=true;
    else if(arg=="--offline-demo")o.offline=true;
    else if(arg=="--offline-test"){o.offline=true;o.offlineOptions.test=true;}
    else if(arg=="--offline-auto-record")o.offlineOptions.automaticRecord=true;
    else if(arg=="--offline-dir")o.offlineOptions.output=value();
    else if(arg=="--audio-volume")o.offlineOptions.volume=std::stof(value());
    else if(arg=="--no-audio")o.offlineOptions.noAudio=true;
    else if(arg=="--demo-record"){o.demo=true;o.record=true;o.manual=true;}
    else if(arg=="--scene-test"||arg=="--map-surface-test"||arg=="--loadout-test"||arg=="--menu-test"||arg=="--demo-rules-test")o.test=arg;
    else if(arg=="--validate"||arg=="--pose-test"||arg=="--event-test"||arg=="--cover-test"||arg=="--fire-test"||arg=="--enemy-fire-test"||arg=="--trail-test"||arg=="--ue-pose-test"||arg=="--visual-test"||arg=="--combat-test")o.test=arg;
    else throw std::runtime_error("Unknown option: "+arg);}
    if(o.width<360||o.width>7680||o.height<240||o.height>4320||o.duration<=0||o.duration>3600||!std::isfinite(o.duration)||!std::isfinite(o.coverProbability)||o.coverProbability<0||o.coverProbability>1)throw std::runtime_error("Invalid window/duration/probability");if(o.map&&(o.record||o.test=="--combat-test"))throw std::runtime_error("Recording requires encounter mode");return o;}
struct View {V3 pivot;float distance=200,yaw=-.6f,pitch=-.68f,speed=20;V3 forward()const{return {std::sin(yaw)*std::cos(pitch),std::sin(pitch),std::cos(yaw)*std::cos(pitch)};}V3 right()const{return {std::cos(yaw),0,-std::sin(yaw)};}Camera camera()const{return {pivot-forward()*distance,pivot,90};}
    void fit(const MapData& map,float aspect,bool castle,bool top){const auto& bounds=map.hasNavigationBounds?map.navigationBounds:(castle?map.castleBounds:map.bounds);pivot=bounds.center();yaw=top?0.f:-.6f;pitch=top?-Pi*.5f+.0001f:-.68f;distance=1;V3 f=forward(),r=right(),u=cross(f,r);for(const auto& object:map.objects)if((!map.hasNavigationBounds||(object.bounds.max.x>=bounds.min.x&&object.bounds.min.x<=bounds.max.x&&object.bounds.max.z>=bounds.min.z&&object.bounds.min.z<=bounds.max.z))&&(!castle||(!object.tree&&!object.terrain&&!object.road)))for(auto p:object.mesh.vertices){V3 d=p-pivot;float z=dot(d,f);distance=std::max({distance,std::abs(dot(d,r))-z,std::abs(dot(d,u))*aspect-z});}distance=distance*1.12f+1;}
};
int key(SDL_Keycode code){if(code>=SDLK_a&&code<=SDLK_z)return 'A'+int(code-SDLK_a);if(code>=SDLK_0&&code<=SDLK_9)return int(code);if(code==SDLK_LSHIFT||code==SDLK_RSHIFT)return 16;if(code==SDLK_SPACE)return 32;return -1;}
std::filesystem::path appRoot(){if(const char* data=std::getenv("MATRIX_GAME_DATA_DIR");data&&*data)return std::filesystem::absolute(data);char* base=SDL_GetBasePath();if(!base)throw std::runtime_error("Cannot locate executable: "+std::string(SDL_GetError()));std::filesystem::path path(base);SDL_free(base);return path.lexically_normal().parent_path().parent_path();}
void saveJson(const std::filesystem::path& path,const Json& data){if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());std::ofstream file(path);if(!file)throw std::runtime_error("Cannot write "+path.string());file<<data.dump(2);}
CombatInput demoInput(double seconds){
    CombatInput input;double phase=std::fmod(seconds,12.0);
    input.forward=phase<6?.7f:0.f;input.right=phase>=6&&phase<10?.65f:0.f;
    input.lookX=float((int(seconds/10)%2?-.23:.23)/(.002*120));
    input.fire=std::fmod(seconds,6.0)<4.0;input.weapon=int(seconds/8)%3;
    return input;
}
}
int main(int argc,char** argv){SDL_Window* window=nullptr;std::filesystem::path root;
try{auto opt=parse(argc,argv);root=appRoot();const auto scene=SceneConfig::load(root,opt.scene);const auto reportRoot=scene.outputDirectory(root,"captures");std::filesystem::create_directories(reportRoot);
    if(opt.offline){opt.offlineOptions.width=opt.width;opt.offlineOptions.height=opt.height;opt.offlineOptions.seed=opt.seed;opt.offlineOptions.scene=opt.scene;return runOfflineDemo(root,opt.offlineOptions);}
    MapData map(scene);saveJson(reportRoot/"validation.json",{{"passed",true},{"objects",map.objects.size()},{"primitives",map.primitiveCount},{"triangles",map.triangleCount},{"trees",map.treeCount},{"terrainObjects",map.terrainCount},{"roadObjects",map.roadCount},{"platform",Platform}});if(opt.test=="--validate")return 0;
    if(SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER))throw std::runtime_error(SDL_GetError());
#ifdef __APPLE__
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,4);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,1);SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
#else
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
#endif
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,1);SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,24);
    bool automated=!opt.test.empty()||opt.record||!opt.capture.empty()||!opt.benchmark.empty();window=SDL_CreateWindow(scene.name.c_str(),SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,opt.width,opt.height,SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE|(opt.fullscreen?SDL_WINDOW_FULLSCREEN_DESKTOP:0)|(automated?SDL_WINDOW_HIDDEN:SDL_WINDOW_SHOWN));if(!window)throw std::runtime_error(SDL_GetError());
    {
    Renderer renderer(window,opt.width,opt.height);std::vector<DrawObject> mapDraws;
    if(opt.test=="--visual-test"||!opt.capture.empty())mapDraws=map.upload(renderer);
    View view;view.fit(map,float(opt.width)/opt.height,opt.view=="castle",opt.view=="top");std::ofstream runtime(root/RuntimeLog);runtime<<"GPU: "<<renderer.adapterName<<"\nObjects: "<<map.objects.size()<<"\nPrimitives: "<<map.primitiveCount<<'\n';runtime.flush();
    if(opt.test=="--visual-test"){Json views=Json::array();for(const auto* mode:{"all","castle","top"}){view.fit(map,float(renderer.width)/renderer.height,std::string(mode)=="castle",std::string(mode)=="top");renderer.render(mapDraws,view.camera(),{});renderer.screenshot(reportRoot/(std::string(mode)+"-linux.png"));views.push_back({{"name",mode},{"visibleObjects",renderer.visibleCount}});}saveJson(reportRoot/"visual-tests-linux.json",{{"passed",true},{"views",views}});}
    else if(!opt.capture.empty()){renderer.render(mapDraws,view.camera(),{});if(!renderer.screenshot(std::filesystem::absolute(opt.capture)))throw std::runtime_error("Screenshot failed");}
    else{
        Encounter encounter(root,map,renderer,opt.seed);
        mapDraws=encounter.mapDraws(); // Reuse uploaded meshes when switching to the map viewer.
        if(opt.test=="--scene-test")saveJson(reportRoot/"scene-tests.json",encounter.sceneTest());
        else if(opt.test=="--map-surface-test")saveJson(reportRoot/"map-surface-tests.json",encounter.mapSurfaceTest());
        else if(opt.test=="--pose-test")saveJson(reportRoot/"pose-tests.json",encounter.poseTest());
        else if(opt.test=="--loadout-test")saveJson(reportRoot/"loadout-tests.json",encounter.loadoutTest());
        else if(opt.test=="--demo-rules-test")saveJson(reportRoot/"demo-rules-tests.json",encounter.demoRulesTest());
        else if(opt.test=="--event-test")saveJson(reportRoot/"event-tests.json",encounter.eventTest());
        else if(opt.test=="--cover-test")saveJson(reportRoot/"cover-tests.json",encounter.coverTest());
        else if(opt.test=="--fire-test")saveJson(reportRoot/"fire-tests.json",encounter.firingTest());
        else if(opt.test=="--enemy-fire-test")saveJson(reportRoot/"enemy-fire-tests.json",encounter.enemyFiringTest());
        else if(opt.test=="--trail-test")saveJson(reportRoot/"trail-tests.json",encounter.trailTest());
        else if(opt.test=="--ue-pose-test")saveJson(reportRoot/"ue-pose-comparison.json",encounter.uePoseTest());
        else{
            if(opt.test=="--combat-test")saveJson(reportRoot/"combat-tests.json",encounter.selfTest());
            std::unique_ptr<Recording> recording;auto directory=opt.output.empty()?scene.outputDirectory(root,"recordings"):std::filesystem::absolute(opt.output);
            auto start=[&]{encounter.setCoverProbability(opt.coverProbability);encounter.prepareClip();encounter.takeEvents();recording=std::make_unique<Recording>(directory,root/"assets",opt.seed,scene.assets);encounter.setRecording(true);};
            auto stop=[&]{if(recording){recording->captureFinalEvents(renderer,encounter);recording->finish();recording.reset();encounter.setRecording(false);}};
            if(opt.record)start();std::array<bool,256> keys{};bool running=true,manual=opt.manual,paused=false,mapMode=opt.map,focused=true,fullscreen=opt.fullscreen;encounter.setAutopilot(!manual);CombatInput input;int recordTicks=0,totalFrames=0,completedClips=0;double accumulator=0,simulated=0,previous=now(),begin=previous,fpsBegin=previous;int fpsFrames=0;HudState hud;hud.showFps=true;std::vector<double> frameTimes;
            std::ofstream demoInputs;if(opt.demo){demoInputs.open(directory/"demo-inputs.jsonl");if(!demoInputs)throw std::runtime_error("Cannot write demo input trace");}
            int menuSelection=0,menuTestFrame=0;
            auto chooseMenu=[&](int selection){
                if(!encounter.playerAlive()){
                    if(selection==1){running=false;return;}if(selection!=0)return;
                    if(!encounter.retry())throw std::runtime_error("Cannot retry encounter");
                    keys.fill(false);input={};accumulator=0;paused=false;manual=true;
                    SDL_SetRelativeMouseMode(!automated?SDL_TRUE:SDL_FALSE);return;
                }
                if(selection<0||selection>=int(CombatUi::MenuLabels.size()))return;
                keys.fill(false);input={};accumulator=0;
                if(selection==4){running=false;return;}
                if(selection>=1&&selection<=3){if(!manual){manual=true;encounter.setAutopilot(false);}input.weapon=selection-1;}
                paused=false;SDL_SetRelativeMouseMode(manual&&!mapMode&&!automated?SDL_TRUE:SDL_FALSE);
            };
            while(running){
                if(opt.test=="--menu-test"){
                    auto press=[](SDL_Keycode code){SDL_Event e{};e.type=SDL_KEYDOWN;e.key.keysym.sym=code;SDL_PushEvent(&e);e.type=SDL_KEYUP;SDL_PushEvent(&e);};
                    auto click=[&](int item){SDL_Event e{};e.type=SDL_MOUSEBUTTONDOWN;e.button.button=SDL_BUTTON_LEFT;
                        auto r=item>=0?CombatUi::menuItem(renderer.width,renderer.height,item):CombatUi::Rect{0,0,0,0};
                        e.button.x=int(r.x+r.w/2);e.button.y=int(r.y+r.h/2);SDL_PushEvent(&e);e.type=SDL_MOUSEBUTTONUP;SDL_PushEvent(&e);};
                    switch(menuTestFrame){
                    case 0:case 5:case 8:case 12:press(SDLK_ESCAPE);break;
                    case 1:case 2:press(SDLK_DOWN);break;
                    case 3:press(SDLK_RETURN);break;
                    case 4:press(SDLK_3);break;
                    case 6:click(1);break;
                    case 7:press(SDLK_2);break;
                    case 9:click(-1);break;
                    case 10:press(SDLK_t);break;
                    case 11:click(0);break;
                    case 13:click(4);break;
                    }
                }
                SDL_Event event;while(SDL_PollEvent(&event)){
                if(event.type==SDL_QUIT)running=false;
                else if(event.type==SDL_WINDOWEVENT){if(event.window.event==SDL_WINDOWEVENT_SIZE_CHANGED)renderer.resize(event.window.data1,event.window.data2);if(event.window.event==SDL_WINDOWEVENT_FOCUS_LOST){focused=false;keys.fill(false);SDL_SetRelativeMouseMode(SDL_FALSE);}if(event.window.event==SDL_WINDOWEVENT_FOCUS_GAINED)focused=true;}
                else if(event.type==SDL_KEYDOWN||event.type==SDL_KEYUP){int k=key(event.key.keysym.sym);if(k>=0&&k<256)keys[k]=event.type==SDL_KEYDOWN;if(event.type==SDL_KEYDOWN&&!event.key.repeat){auto code=event.key.keysym.sym;
                    if(code==SDLK_ESCAPE){
                        if(paused){if(encounter.playerAlive())chooseMenu(0);}
                        else{paused=true;menuSelection=0;keys.fill(false);input={};accumulator=0;stop();SDL_SetRelativeMouseMode(SDL_FALSE);}
                        continue;
                    }
                    if(code==SDLK_F11){fullscreen=!fullscreen;SDL_SetWindowFullscreen(window,fullscreen?SDL_WINDOW_FULLSCREEN_DESKTOP:0);continue;}
                    if(code==SDLK_F12){renderer.screenshot(reportRoot/("linux-"+std::to_string(uint64_t(now()*1000))+".png"));continue;}
                    if(paused){
                        if(code==SDLK_UP)menuSelection=CombatUi::moveSelection(menuSelection,-1,!encounter.playerAlive());
                        else if(code==SDLK_DOWN)menuSelection=CombatUi::moveSelection(menuSelection,1,!encounter.playerAlive());
                        else if(code==SDLK_RETURN||code==SDLK_KP_ENTER)chooseMenu(menuSelection);
                        else if(encounter.playerAlive()&&code>=SDLK_1&&code<=SDLK_3)chooseMenu(int(code-SDLK_1)+1);
                        continue;
                    }
                    if(code==SDLK_F7){manual=!manual;encounter.setAutopilot(!manual);SDL_SetRelativeMouseMode(SDL_FALSE);}
                    else if(code==SDLK_F8){mapMode=!mapMode;SDL_SetRelativeMouseMode(SDL_FALSE);}
                    else if(code==SDLK_F9){if(recording)stop();else{start();recordTicks=0;}}
                    if(code>=SDLK_1&&code<=SDLK_3&&!mapMode){if(!manual){manual=true;encounter.setAutopilot(false);}input.weapon=int(code-SDLK_1);}
                    if(code==SDLK_SPACE)input.jump=true;if(code==SDLK_LSHIFT||code==SDLK_RSHIFT)input.dash=true;if(code==SDLK_r)input.reload=true;if(code==SDLK_t)input.trap=true;if(code==SDLK_h)input.cover=true;}}
                else if(event.type==SDL_MOUSEBUTTONDOWN||event.type==SDL_MOUSEBUTTONUP){bool down=event.type==SDL_MOUSEBUTTONDOWN;
                    if(paused){if(down&&event.button.button==SDL_BUTTON_LEFT&&!mapMode)chooseMenu(CombatUi::hitMenu(renderer.width,renderer.height,event.button.x,event.button.y,!encounter.playerAlive()));continue;}
                    if(event.button.button==SDL_BUTTON_LEFT)keys[1]=down;if(event.button.button==SDL_BUTTON_RIGHT)keys[2]=down;if(down&&manual&&!mapMode)SDL_SetRelativeMouseMode(SDL_TRUE);}
                else if(event.type==SDL_MOUSEMOTION){
                    if(paused){if(!mapMode){int item=CombatUi::hitMenu(renderer.width,renderer.height,event.motion.x,event.motion.y);if(item>=0)menuSelection=item;}continue;}
                    if(manual&&!mapMode&&SDL_GetRelativeMouseMode()){input.lookX+=event.motion.xrel;input.lookY+=event.motion.yrel;}else if(mapMode&&(event.motion.state&(SDL_BUTTON_LMASK|SDL_BUTTON_RMASK))){view.yaw=std::remainder(view.yaw+event.motion.xrel*.004f,2*Pi);view.pitch=std::clamp(view.pitch-event.motion.yrel*.004f,-Pi*.5f+.0001f,Pi*.5f-.0001f);}}
                else if(event.type==SDL_MOUSEWHEEL&&mapMode)view.distance=std::clamp(view.distance*std::pow(.85f,float(event.wheel.y)),.25f,1200.f);
            }
            if(!running)break;double current=now(),elapsed=std::min(.1,current-previous);previous=current;if(!focused&&!automated&&!recording){SDL_Delay(12);continue;}
            if(!paused){accumulator+=(opt.demo||opt.test=="--combat-test"||opt.test=="--menu-test")?1./24:elapsed;input.forward=float(keys['W'])-float(keys['S']);input.right=float(keys['D'])-float(keys['A']);input.fire=keys[1];input.ads=keys[2];input.shield=keys['F'];input.crouch=keys['C'];
                if(!manual||mapMode||!focused){bool cover=input.cover;input={};input.cover=cover&&!mapMode;}
                const int physicsTicks=int(accumulator/(1./120));
                const float lookX=CameraInput::pixelsPerTick(input.lookX,physicsTicks,1.f/120);
                const float lookY=CameraInput::pixelsPerTick(input.lookY,physicsTicks,1.f/120);
                for(int tick=0;tick<physicsTicks;++tick){
                    input.lookX=lookX;input.lookY=lookY;
                    if(opt.demo){input=demoInput(simulated);demoInputs<<Json{{"simulation_time",simulated},{"forward",input.forward},{"right",input.right},{"look_x",input.lookX},{"fire",input.fire},{"weapon",input.weapon}}.dump()<<'\n';if(!demoInputs)throw std::runtime_error("Demo input trace write failed");}
                    encounter.advance(1.f/120,input);accumulator-=1./120;simulated+=1./120;input.lookX=input.lookY=0;input.jump=input.dash=input.reload=input.trap=input.cover=false;input.weapon=-1;
                    if(!encounter.playerAlive()){if(recording)recording->capture(renderer,encounter);stop();paused=true;menuSelection=0;keys.fill(false);input={};accumulator=0;SDL_SetRelativeMouseMode(SDL_FALSE);if(automated)running=false;break;}
                    if(recording&&++recordTicks==5){recording->capture(renderer,encounter);recordTicks=0;++totalFrames;if(opt.record&&totalFrames>=int(std::round(opt.duration*24))){stop();running=false;break;}if(!opt.demo&&recording->frames()>=240){stop();if(++completedClips%20==0)encounter.recover();start();}}
                    else if(!recording)encounter.takeEvents();if(encounter.recoveryRequested()){if(opt.demo)throw std::runtime_error("Demo requires continuous gameplay without recovery");bool resume=bool(recording);stop();if(encounter.recover()&&resume){start();recordTicks=0;}}
                }if(opt.test=="--combat-test"&&simulated>=opt.duration)running=false;
            }else accumulator=0;
            if(mapMode){V3 motion=view.forward()*(float(keys['W'])-float(keys['S']))+view.right()*(float(keys['D'])-float(keys['A']))+V3{0,float(keys['E'])-float(keys['Q']),0};view.pivot=view.pivot+motion.normalized()*float(elapsed)*view.speed*(keys[16]?4.f:1.f);}
            hud.combat=!mapMode;hud.paused=paused&&!mapMode;hud.dead=!encounter.playerAlive();hud.pauseSelection=menuSelection;if(!mapMode)encounter.hud(hud,0);if(paused)hud.combatAction="PAUSED";hud.recording=bool(recording);hud.recordFrames=recording?recording->frames():0;renderer.render(mapMode?mapDraws:encounter.draws(paused?1.f:float(accumulator*120)),mapMode?view.camera():encounter.camera(0),hud);renderer.present(opt.vsync);
            if(opt.test=="--menu-test"){
                constexpr bool expectedPause[]={true,true,true,false,false,true,false,false,true,true,true,false,true};
                constexpr int expectedWeapon[]={0,0,0,1,2,2,0,1,1,1,1,1,1};
                if(menuTestFrame>=13||paused!=expectedPause[menuTestFrame]||hud.weapon!=expectedWeapon[menuTestFrame])throw std::runtime_error("Menu input regression at frame "+std::to_string(menuTestFrame));
                auto player=encounter.diagnostics().at("characters")[0];
                if(player.at("ammo")!=Json::array({12,30,8})||player.at("action")!="")throw std::runtime_error("Menu input triggered a combat action");
                ++menuTestFrame;
            }
            double end=now();++fpsFrames;if(end-fpsBegin>=.25){hud.fps=fpsFrames/(end-fpsBegin);fpsBegin=end;fpsFrames=0;}if(!opt.benchmark.empty()){if(end-begin>=1)frameTimes.push_back((end-current)*1000);if(end-begin>=6)running=false;}
            }
            stop();
            if(opt.test=="--menu-test"){
                if(menuTestFrame!=13)throw std::runtime_error("Menu input test exited early");
                saveJson(reportRoot/"menu-input-tests.json",{{"passed",true},{"checks",{"keyboard menu selection","mouse menu selection","1/2/3 direct switching","outside click ignored","paused combat input ignored","menu clicks do not fire","resume and exit"}}});
            }
            saveJson(reportRoot/"encounter-last.json",encounter.diagnostics());if(!opt.benchmark.empty()){double sum=0;for(double n:frameTimes)sum+=n;saveJson(opt.benchmark,{{"platform",Platform},{"gpu",renderer.adapterName},{"width",renderer.width},{"height",renderer.height},{"frames",frameTimes.size()},{"averageFps",frameTimes.empty()?0:1000*frameTimes.size()/sum},{"vsync",opt.vsync}});}
        }
    }
    }
    SDL_DestroyWindow(window);SDL_Quit();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(!root.empty()){std::ofstream log(root/RuntimeLog,std::ios::app);log<<"ERROR: "<<e.what()<<'\n';}if(window)SDL_DestroyWindow(window);SDL_Quit();return 1;}}
