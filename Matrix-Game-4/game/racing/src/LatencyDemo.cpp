#include "LatencyDemo.h"
#include "CameraTelemetry.h"
#include "KeyboardSteering.h"
#include <SDL.h>
#include <deque>
#include <iostream>
#include <set>

namespace {
struct Snapshot {
    std::vector<DrawObject> objects;
    Camera camera;
    CarPose player;
    latency::Command command;
    std::vector<latency::Command> appliedInputs;
    double simulationTime=0;
    uint64_t chunk=0;
    int frame=0;
};
struct ScriptEntry {double time;latency::Controls controls;};
std::vector<ScriptEntry> readScript(const std::filesystem::path& path) {
    std::vector<ScriptEntry> result;if(path.empty())return result;
    std::ifstream file(path);if(!file)throw std::runtime_error("Cannot read latency input script");
    std::string line;while(std::getline(file,line)) {
        if(line.empty()||line[0]=='#')continue;
        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
        ScriptEntry entry{};int rear=0;std::string extra;auto& c=entry.controls;
        if(!(row>>entry.time>>c.drive.throttle>>c.drive.steer>>c.drive.brake>>c.drive.handbrake>>c.camera>>rear)||row>>extra)
            throw std::runtime_error("Script rows: seconds,throttle,steer,brake,handbrake,camera,rear");
        if(!std::isfinite(entry.time)||entry.time<0||(!result.empty()&&entry.time<result.back().time)||
            !std::isfinite(c.drive.throttle)||std::abs(c.drive.throttle)>1||!std::isfinite(c.drive.steer)||std::abs(c.drive.steer)>1||
            !std::isfinite(c.drive.brake)||c.drive.brake<0||c.drive.brake>1||!std::isfinite(c.drive.handbrake)||
            c.drive.handbrake<0||c.drive.handbrake>1||c.camera<0||c.camera>2||(rear!=0&&rear!=1))
            throw std::runtime_error("Invalid latency script value");
        c.rear=rear!=0;result.push_back(entry);
    }
    return result;
}
struct Pad {
    SDL_GameController* handle=nullptr;
    ~Pad(){if(handle)SDL_GameControllerClose(handle);}
    void refresh() {
        if(handle&&!SDL_GameControllerGetAttached(handle)){SDL_GameControllerClose(handle);handle=nullptr;}
        if(!handle)for(int i=0;i<SDL_NumJoysticks();++i)if(SDL_IsGameController(i)){handle=SDL_GameControllerOpen(i);if(handle)break;}
    }
    Input read()const {
        Input input;if(!handle)return input;
        const float x=std::clamp(SDL_GameControllerGetAxis(handle,SDL_CONTROLLER_AXIS_LEFTX)/32767.f,-1.f,1.f);
        input.steer=std::abs(x)>.12f?std::copysign((std::abs(x)-.12f)/.88f,x):0;
        const float t=(SDL_GameControllerGetAxis(handle,SDL_CONTROLLER_AXIS_TRIGGERRIGHT)-SDL_GameControllerGetAxis(handle,SDL_CONTROLLER_AXIS_TRIGGERLEFT))/32767.f;
        input.throttle=std::abs(t)>.05f?t:0;
        input.handbrake=SDL_GameControllerGetButton(handle,SDL_CONTROLLER_BUTTON_A)?1.f:0.f;
        return input;
    }
};
}

int runLatencyDemo(SDL_Window* window,Renderer& renderer,World& world,const Options& options,
    const std::filesystem::path& root,const LatencyDemoOptions& demo) {
    using namespace latency;
    auto mode=demo.mode;
    const auto script=readScript(demo.inputScript);
    const auto folder=root/"latency-logs";std::filesystem::create_directories(folder);
    const auto logfile=folder/("session-"+std::to_string(uint64_t(now()*1000000))+".csv");
    std::ofstream log(logfile);log.exceptions(std::ios::badbit|std::ios::failbit);
    log<<"event,mode,epoch,wall_s,active_s,input_id,input_active_s,chunk,frame,simulation_s,command_age_ms,prepare_ms,queued_frames,throttle,steer,brake,handbrake,camera,rear,x,y,z,speed_kmh,player_steering_limit,player_front_grip,player_rear_grip,player_front_drive\n";
    log<<std::fixed<<std::setprecision(6);
    std::cout<<"Latency demo: F5 normal, F6 pulse, F7 fixed; switching modes resets the car.\n"
        <<"WASD/arrows drive; Space handbrake; C camera; Q rear; R reset; Esc pause; F12 screenshot.\n"
        <<"Pulse: "<<demo.chunkFrames<<" frames at "<<demo.referenceFps<<" FPS, "<<demo.inferenceMs<<" ms inference. Fixed input delay: "<<demo.fixedMs<<" ms.\n"
        <<"Audio disabled in this experiment. Log: "<<logfile<<std::endl;
    InputDelay delayed;
    PulseInput pulseInput;
    KeyboardSteering steering;
    PulsePlayback<Snapshot> playback(demo.referenceFps);
    CameraTelemetry telemetry;
    Pad pad;pad.refresh();
    std::array<bool,SDL_NUM_SCANCODES> keys{};
    Controls controls;
    Command latest,applied;
    Snapshot shown;
    uint64_t commandId=0,chunkId=0,epoch=0,displayed=0;
    std::set<uint64_t> presentedInputs;
    const double start=now(),step=1.0/options.physicsHz;
    double previous=start,clock=0,physicsTime=0,pulseTarget=0,worldBase=world.simulationTime,nextDisplay=0;
    double prepareMs=0,lastResponse=-1,titleAt=0,lastLogFlush=0;
    size_t scriptCursor=0;
    bool running=true,paused=false,active=true,minimized=false,fullscreen=false,capture=false,refresh=true;
    const auto stamp=[&](const char* event,const Command& command,uint64_t chunk,int frame,double simulation) {
        const auto p=position(shown.player.body);const auto& c=command.controls;
        log<<event<<','<<name(mode)<<','<<epoch<<','<<now()-start<<','<<clock<<','<<command.id<<','<<command.time<<','
            <<chunk<<','<<frame<<','<<simulation<<','<<(clock-command.time)*1000<<','<<prepareMs<<','<<playback.queuedFrames()<<','
            <<c.drive.throttle<<','<<c.drive.steer<<','<<c.drive.brake<<','<<c.drive.handbrake<<','<<c.camera<<','<<c.rear<<','
            <<p.x<<','<<p.y<<','<<p.z<<','<<shown.player.speed<<','<<options.playerSteeringLimit<<','<<options.playerFrontGrip<<','<<options.playerRearGrip<<','<<options.playerFrontDrive<<'\n';
    };
    const auto snapshot=[&](const Command& command,uint64_t chunk,int frame) {
        Snapshot s;s.objects=world.drawObjects();s.camera=world.camera(command.controls.camera,command.controls.rear);
        s.player=world.player();s.command=command;s.simulationTime=world.simulationTime;s.chunk=chunk;s.frame=frame;return s;
    };
    const auto reset=[&]() {
        world.reset();steering.reset();delayed.reset();pulseInput.reset();playback.reset();controls={};latest={};applied={};keys.fill(false);
        clock=physicsTime=pulseTarget=nextDisplay=0;worldBase=world.simulationTime;previous=now();scriptCursor=0;
        chunkId=0;displayed=0;presentedInputs.clear();lastResponse=-1;prepareMs=0;titleAt=0;
        ++epoch;shown=snapshot({},0,0);stamp("reset",latest,0,0,world.simulationTime);refresh=true;
    };
    const auto input=[&](Controls value,double time) {
        if(value==latest.controls)return;
        latest={value,++commandId,std::max(time,latest.time)};
        if(mode==Mode::Pulse)pulseInput.push(latest);else delayed.push(latest);
        stamp("input",latest,0,0,world.simulationTime);
    };
    const auto readControls=[&]() {
        Controls value=controls;value.keyboardSteering=true;
        value.drive.throttle=float(keys[SDL_SCANCODE_W]||keys[SDL_SCANCODE_UP])-float(keys[SDL_SCANCODE_S]||keys[SDL_SCANCODE_DOWN]);
        value.drive.steer=float(keys[SDL_SCANCODE_D]||keys[SDL_SCANCODE_RIGHT])-float(keys[SDL_SCANCODE_A]||keys[SDL_SCANCODE_LEFT]);
        value.drive.handbrake=keys[SDL_SCANCODE_SPACE]?1.f:0;value.rear=keys[SDL_SCANCODE_Q];
        if(active){const auto p=pad.read();if(std::abs(p.throttle)>.01f)value.drive.throttle=p.throttle;
            if(std::abs(p.steer)>.01f){value.drive.steer=p.steer;value.keyboardSteering=false;}value.drive.handbrake=std::max(value.drive.handbrake,p.handbrake);}
        if(!active)value.drive={};
        return value;
    };
    reset();
    while(running) {
        double wall=now();if(!paused&&!minimized)clock+=wall-previous;previous=wall;
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            if(event.type==SDL_QUIT)running=false;
            if(event.type==SDL_CONTROLLERDEVICEADDED||event.type==SDL_CONTROLLERDEVICEREMOVED)pad.refresh();
            if(event.type==SDL_WINDOWEVENT) {
                if(event.window.event==SDL_WINDOWEVENT_FOCUS_LOST){active=false;keys.fill(false);if(demo.inputScript.empty())paused=true;refresh=true;}
                if(event.window.event==SDL_WINDOWEVENT_FOCUS_GAINED)active=true;
                if(event.window.event==SDL_WINDOWEVENT_MINIMIZED)minimized=true;
                if(event.window.event==SDL_WINDOWEVENT_RESTORED){minimized=false;refresh=true;}
                if(event.window.event==SDL_WINDOWEVENT_EXPOSED||event.window.event==SDL_WINDOWEVENT_SIZE_CHANGED)refresh=true;
            }
            if(event.type==SDL_KEYDOWN||event.type==SDL_KEYUP) {
                keys.at(size_t(event.key.keysym.scancode))=event.type==SDL_KEYDOWN;
                if(event.type==SDL_KEYDOWN&&!event.key.repeat) {
                    const auto key=event.key.keysym.sym;
                    if(key==SDLK_ESCAPE){paused=!paused;stamp(paused?"pause":"resume",latest,0,0,world.simulationTime);refresh=true;}
                    if(key==SDLK_F5||key==SDLK_F6||key==SDLK_F7){mode=key==SDLK_F5?Mode::Normal:key==SDLK_F6?Mode::Pulse:Mode::Fixed;reset();}
                    if(key==SDLK_r)reset();
                    if(key==SDLK_c)controls.camera=(controls.camera+1)%3;
                    if(key==SDLK_F12)capture=true;
                    if(key==SDLK_F11){fullscreen=!fullscreen;SDL_SetWindowFullscreen(window,fullscreen?SDL_WINDOW_FULLSCREEN_DESKTOP:0);}
                    if(key==SDLK_F4&&(event.key.keysym.mod&KMOD_ALT))running=false;
                }
                if(demo.inputScript.empty()) {
                    // SDL timestamps preserve short presses delivered together after a burst.
                    const double age=double(uint32_t(SDL_GetTicks()-event.key.timestamp))/1000;
                    input(readControls(),std::max(0.0,clock-age));
                }
            }
        }
        if(!running)break;
        if(demo.duration>0&&now()-start>=demo.duration)break;
        if(demo.inputScript.empty())input(readControls(),clock);
        else if(!paused)while(scriptCursor<script.size()&&script[scriptCursor].time<=clock) {
            const auto& e=script[scriptCursor++];input(e.controls,e.time);
        }
        int width=0,height=0;SDL_GL_GetDrawableSize(window,&width,&height);
        if(width<=0||height<=0||minimized){SDL_Delay(10);continue;}
        renderer.makeCurrent();if(width!=renderer.width||height!=renderer.height){renderer.resize(width,height);refresh=true;}
        bool newFrame=false;
        if(!paused&&mode==Mode::Pulse) {
            if(auto frame=playback.poll(clock)){shown=std::move(*frame);newFrame=true;}
            // Show a newly ready frame before spending CPU time preparing the next chunk.
            if(!newFrame&&playback.canPrepare()) {
                const uint64_t chunk=++chunkId;const double began=now();
                std::vector<Snapshot> frames;frames.reserve(size_t(demo.chunkFrames));
                for(int i=0;i<demo.chunkFrames;++i) {
                    pulseTarget+=1/demo.referenceFps;
                    std::vector<Command> appliedInputs;
                    while(physicsTime+step<=pulseTarget+1e-9){
                        const auto command=pulseInput.sample(physicsTime,step);
                        if(command.id!=applied.id){
                            appliedInputs.push_back(command);
                            stamp("pulse_apply",command,chunk,i,world.simulationTime);
                        }
                        applied=command;world.update(float(step),steering.apply(applied.controls.drive,step,applied.controls.keyboardSteering));physicsTime+=step;
                    }
                    auto frame=snapshot(applied,chunk,i);frame.appliedInputs=std::move(appliedInputs);
                    frames.push_back(std::move(frame));
                }
                const double finished=now();prepareMs=(finished-began)*1000;
                clock+=finished-previous;previous=finished;
                playback.submit(std::move(frames),clock+demo.inferenceMs/1000);
                stamp("chunk_submit",applied,chunk,0,world.simulationTime);
            }
        } else if(!paused) {
            int steps=0;
            while(physicsTime+step<=clock+1e-9&&steps++<250) {
                physicsTime+=step;
                applied=delayed.sample(physicsTime,mode==Mode::Fixed?demo.fixedMs/1000:0);
                world.update(float(step),steering.apply(applied.controls.drive,step,applied.controls.keyboardSteering));
            }
            if(clock>=nextDisplay) {
                shown=snapshot(applied,0,int(displayed));newFrame=true;nextDisplay=clock+1/demo.referenceFps;
            }
        }
        if(newFrame||refresh||capture) {
            HudState hud;hud.visible=true;hud.speed=shown.player.speed;hud.location=position(shown.player.body);hud.heading=shown.player.heading;
            hud.roads=&world.mapLines();hud.mapCenter=world.mapCenter();hud.mapHalfSize=world.mapHalfSize();
            const auto number=[](double v){std::ostringstream s;s<<std::fixed<<std::setprecision(1)<<v;return s.str();};
            hud.latencyStatus={std::string(name(mode))+(paused?"  PAUSED":""),
                mode==Mode::Pulse?std::to_string(demo.chunkFrames)+" FRAMES  "+number(demo.inferenceMs)+" MS INFERENCE":
                    mode==Mode::Fixed?number(demo.fixedMs)+" MS INPUT DELAY":"NO ADDED DELAY",
                "PREP "+number(prepareMs)+" MS  QUEUED "+std::to_string(playback.queuedFrames()),
                "LAST INPUT TO PRESENT "+(lastResponse<0?std::string("NA"):number(lastResponse)+" MS")};
            renderer.waitForFrame(false);renderer.render(shown.objects,shown.camera,hud);
            if(capture){renderer.screenshot(folder/("capture-"+std::to_string(uint64_t(now()*1000000))+".png"));capture=false;}
            renderer.present(options.vsync);
            const double presented=now();if(!paused)clock+=presented-previous;previous=presented;
            if(newFrame) {
                ++displayed;stamp("present",shown.command,shown.chunk,shown.frame,shown.simulationTime);
                const auto firstPresent=[&](const Command& command){
                    if(command.id&&presentedInputs.insert(command.id).second) {
                        lastResponse=(clock-command.time)*1000;stamp("first_present",command,shown.chunk,shown.frame,shown.simulationTime);
                    }
                };
                for(const auto& command:shown.appliedInputs)firstPresent(command);
                firstPresent(shown.command);
            }
            telemetry.publish(shown.camera,width,height,shown.command.controls.camera,
                (paused?CameraTelemetry::Paused:0u)|(active?CameraTelemetry::Active:0u)|(shown.command.controls.rear?CameraTelemetry::LookBack:0u),
                now()-start,shown.simulationTime);
            refresh=false;
        }
        if(clock>=titleAt) {
            std::ostringstream title;title<<"CitySample - "<<name(mode)<<(paused?" [PAUSED]":"")<<" | "
                <<demo.referenceFps<<" FPS | sim/wall "<<std::fixed<<std::setprecision(2)
                <<(clock>0?(shown.simulationTime-worldBase)/clock:0);
            SDL_SetWindowTitle(window,title.str().c_str());titleAt=clock+.5;
        }
        if(now()-lastLogFlush>1){log.flush();lastLogFlush=now();}
        SDL_Delay(1);
    }
    if(!demo.screenshot.empty())renderer.screenshot(demo.screenshot);
    stamp("end",shown.command,shown.chunk,shown.frame,shown.simulationTime);log.flush();
    std::cout<<"Latency session saved: "<<logfile<<std::endl;
    return 0;
}
