#include "ReferenceDemo.h"
#include "CameraTelemetry.h"
#include "KeyboardSteering.h"
#include "CaptionStream.h"
#include "VideoEncoder.h"
#include <SDL.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <set>

namespace {
using Json=nlohmann::json;
using latency::Command;
using latency::Controls;
struct Frame {
    uint64_t id=0,tick=0;
    double simulationTime=0,capturedAt=0;
    Camera camera;
    CarPose player;
    Command command;
    std::vector<Command> changes;
    std::vector<uint8_t> rgba;
    captions::Frame caption;
    uint64_t videoFrame=0;
    double visibilityMs=0;
};
struct ScriptEntry {double time=0;Controls controls;};
std::vector<ScriptEntry> readScript(const std::filesystem::path& path){
    std::vector<ScriptEntry> result;if(path.empty())return result;
    std::ifstream file(path);if(!file)throw std::runtime_error("Cannot read reference input script");
    std::string line;while(std::getline(file,line)){
        if(line.empty()||line[0]=='#')continue;
        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
        ScriptEntry e;int rear=0;std::string extra;auto& c=e.controls;
        if(!(row>>e.time>>c.drive.throttle>>c.drive.steer>>c.drive.brake>>c.drive.handbrake>>c.camera>>rear)||row>>extra)
            throw std::runtime_error("Script rows: seconds,throttle,steer,brake,handbrake,camera,rear");
        if(!std::isfinite(e.time)||e.time<0||(!result.empty()&&e.time<result.back().time)||
            !std::isfinite(c.drive.throttle)||std::abs(c.drive.throttle)>1||!std::isfinite(c.drive.steer)||std::abs(c.drive.steer)>1||
            !std::isfinite(c.drive.brake)||c.drive.brake<0||c.drive.brake>1||!std::isfinite(c.drive.handbrake)||
            c.drive.handbrake<0||c.drive.handbrake>1||c.camera<0||c.camera>2||(rear!=0&&rear!=1))
            throw std::runtime_error("Invalid reference input script value");
        c.rear=rear!=0;result.push_back(e);
    }return result;
}
struct Controller {
    SDL_GameController* handle=nullptr;
    ~Controller(){if(handle)SDL_GameControllerClose(handle);}
    void refresh(){
        if(handle&&!SDL_GameControllerGetAttached(handle)){SDL_GameControllerClose(handle);handle=nullptr;}
        if(!handle)for(int i=0;i<SDL_NumJoysticks();++i)if(SDL_IsGameController(i)){handle=SDL_GameControllerOpen(i);if(handle)break;}
    }
    Input read()const{
        Input v;if(!handle)return v;
        float x=std::clamp(SDL_GameControllerGetAxis(handle,SDL_CONTROLLER_AXIS_LEFTX)/32767.f,-1.f,1.f);
        v.steer=std::abs(x)>.12f?std::copysign((std::abs(x)-.12f)/.88f,x):0;
        float t=(SDL_GameControllerGetAxis(handle,SDL_CONTROLLER_AXIS_TRIGGERRIGHT)-SDL_GameControllerGetAxis(handle,SDL_CONTROLLER_AXIS_TRIGGERLEFT))/32767.f;
        v.throttle=std::abs(t)>.05f?t:0;v.handbrake=SDL_GameControllerGetButton(handle,SDL_CONTROLLER_BUTTON_A)?1.f:0.f;return v;
    }
};
}

int runReferenceDemo(SDL_Window* window,Renderer& renderer,World& world,const Options& options,
    const std::filesystem::path& root,const ReferenceDemoOptions& demo){
    const auto& config=demo.stream;config.validate();const auto script=readScript(demo.inputScript);
    reference::Pipeline<Frame> pipeline(config);reference::SimulationClock simulation(config.physicsHz);
    reference::InputQueue inputs;Controller controller;controller.refresh();CameraTelemetry telemetry;
    KeyboardSteering steering;
    // Keep model resolution fixed even on Retina screens or after window resizing.
    renderer.resize(options.width,options.height);
    const auto folder=root/"reference-logs";std::filesystem::create_directories(folder);
    const auto stem=folder/("session-"+std::to_string(uint64_t(now()*1000000)));
    std::ofstream log(stem.string()+".jsonl");log.exceptions(std::ios::badbit|std::ios::failbit);
    std::unique_ptr<captions::Stream> captioner;std::ofstream captionLog,visibilityLog,videoFile;
    std::unique_ptr<VideoEncoder> video;uint64_t videoFrames=0;
    if(demo.captions){captioner=std::make_unique<captions::Stream>(root/"assets/caption-catalog.json");
        captionLog.exceptions(std::ios::badbit|std::ios::failbit);captionLog.open(stem.string()+".captions.jsonl");
        visibilityLog.exceptions(std::ios::badbit|std::ios::failbit);visibilityLog.open(stem.string()+".visibility.jsonl");}
    if(demo.record){videoFile.exceptions(std::ios::badbit|std::ios::failbit);videoFile.open(stem.string()+".h264",std::ios::binary);
        video=std::make_unique<VideoEncoder>(options.width,options.height,int(config.fps),"h264",2,"ultrafast");video->setBitrate(8000);}
    double start=now(),previous=start,activeTime=0,lastFlush=0,lastUi=0,blockedSeconds=0,blockedSince=0,maxBlock=0,lastLatency=-1;
    uint64_t epoch=0,inputId=0,produced=0,presented=0,blocks=0,maxInput=0,maxOutput=0,maxFlight=0,maxTotal=0;
    size_t scriptCursor=0;int cameraMode=1;
    bool running=true,paused=false,minimized=false,focused=true,blocked=false,capture=false,refresh=true,fullscreen=false;
    Command latest,applied;Frame shown;HudState displayHud;std::vector<Command> changes;std::set<uint64_t> seenInputs;
    std::vector<double> responseMs,frameAgeMs;
    std::array<bool,SDL_NUM_SCANCODES> keys{};
    const auto syncTime=[&](){double t=now();if(!paused&&!minimized){double dt=t-previous;activeTime+=dt;if(blocked)blockedSeconds+=dt;}previous=t;};
    const auto write=[&](const char* event,Json extra=Json::object()){
        maxInput=std::max<uint64_t>(maxInput,pipeline.inputSize());maxOutput=std::max<uint64_t>(maxOutput,pipeline.outputSize());
        maxFlight=std::max<uint64_t>(maxFlight,pipeline.inFlight());
        maxTotal=std::max<uint64_t>(maxTotal,pipeline.inputSize()+pipeline.outputSize()+pipeline.inFlight());
        Json row={{"event",event},{"epoch",epoch},{"wall_s",now()-start},{"active_s",activeTime},
            {"simulation_s",simulation.time()},{"tick",simulation.tick()},{"input_queue",pipeline.inputSize()},
            {"output_queue",pipeline.outputSize()},{"in_flight",pipeline.inFlight()},{"pending_inputs",inputs.size()}};
        row.update(extra);log<<row.dump()<<'\n';
    };
    const auto commandFields=[](const Command& c){return Json{{"input_id",c.id},{"input_s",c.time},{"throttle",c.controls.drive.throttle},
        {"steer",c.controls.drive.steer},{"brake",c.controls.drive.brake},{"handbrake",c.controls.drive.handbrake},
        {"camera_mode",c.controls.camera},{"rear",c.controls.rear},{"keyboard_steering",c.controls.keyboardSteering}};};
    const auto grab=[&](uint64_t id){
        Frame frame;frame.id=id;frame.tick=simulation.tick();frame.simulationTime=simulation.time();frame.command=applied;
        frame.camera=world.camera(applied.controls.camera,applied.controls.rear);frame.player=world.player();
        const auto& objects=world.drawObjects();renderer.render(objects,frame.camera,{});frame.rgba=renderer.readBack();
        if(captioner&&id){
            const auto begin=now();auto stats=renderer.captionVisibility(objects,frame.camera,demo.captionWidth);
            frame.caption={id,frame.simulationTime,frame.camera,frame.player.speed,{}};
            const auto& entities=world.captionEntities();uint64_t pixels=0;for(const auto& s:stats)pixels+=s.pixels;
            for(const auto& s:stats)if(s.token){const auto& entity=entities.at(s.token);
                frame.caption.visible.push_back({entity,double(s.pixels)/pixels,s.x,s.y});}
            frame.visibilityMs=(now()-begin)*1000;
            auto row=captions::Stream::frameJson(frame.caption);row["epoch"]=epoch;row["visibility_ms"]=frame.visibilityMs;
            if(video)row["video_frame_index"]=videoFrames;
            visibilityLog<<row.dump()<<'\n';
        }
        if(video&&id){frame.videoFrame=videoFrames++;
            auto encoded=video->encode(frame.rgba,int64_t(std::llround(frame.videoFrame/config.fps*1e6)),id==1);
            videoFile.write(reinterpret_cast<const char*>(encoded.bytes.data()),std::streamsize(encoded.bytes.size()));}
        syncTime();frame.capturedAt=activeTime;
        return frame;
    };
    const auto startModel=[&](){
        if(!pipeline.start(activeTime))return;
        const auto& frames=pipeline.activeFrames();
        if(captioner){double begin=now();std::vector<captions::Frame> snapshots;snapshots.reserve(frames.size());
            for(const auto& frame:frames)snapshots.push_back(frame.caption);
            auto prompt=captioner->chunk(epoch,pipeline.chunk(),snapshots);prompt["compose_ms"]=(now()-begin)*1000;
            if(video){prompt["video_frame_start"]=frames.front().videoFrame;prompt["video_frame_end"]=frames.back().videoFrame;}
            captionLog<<prompt.dump()<<'\n';write("prompt",prompt);
        }
        write("model_start",{{"chunk_id",pipeline.chunk()},{"duration_ms",pipeline.duration()*1000},
            {"frame_start",frames.front().id},{"frame_end",frames.back().id},{"frame_count",frames.size()}});
    };
    const auto reset=[&](){
        write("reset_flush",{{"discarded_frames",pipeline.inputSize()+pipeline.outputSize()+pipeline.inFlight()},
            {"discarded_input_edges",inputs.size()+changes.size()}});
        world.reset();steering.reset();pipeline.reset();simulation.reset();inputs.reset();keys.fill(false);latest={};applied={};changes.clear();seenInputs.clear();
        if(captioner)captioner->reset();
        activeTime=0;previous=now();scriptCursor=0;produced=presented=blocks=maxInput=maxOutput=maxFlight=maxTotal=0;
        blockedSeconds=maxBlock=0;blocked=false;lastLatency=-1;cameraMode=1;responseMs.clear();frameAgeMs.clear();++epoch;
        shown=grab(0);simulation.reset(activeTime);refresh=true;
        write("config",{{"fps",config.fps},{"physics_hz",config.physicsHz},{"chunk_frames",config.chunk},
            {"first_chunk_frames",config.firstChunk?config.firstChunk:config.chunk},{"captions",demo.captions},
            {"caption_visibility_width",demo.captionWidth},{"caption_provenance",captioner?captioner->provenance:Json(nullptr)},
            {"recording",demo.record},{"recording_frame_index_base",0},
            {"player_steering_limit",options.playerSteeringLimit},
            {"player_front_grip",options.playerFrontGrip},{"player_rear_grip",options.playerRearGrip},
            {"player_front_drive",options.playerFrontDrive},
            {"keyboard_turn_s",KeyboardSteering::TurnSeconds},{"keyboard_return_s",KeyboardSteering::ReturnSeconds},
            {"input_capacity",config.inputCapacity},{"output_capacity",config.outputCapacity},
            {"inference_ms",config.inferenceMs},{"jitter_ms",config.jitterMs},{"seed",config.seed},
            {"width",options.width},{"height",options.height},{"format","RGBA8"},{"mock",true}});
    };
    const auto accept=[&](Controls value,double time){
        if(value==latest.controls)return;
        latest={value,++inputId,std::max(time,latest.time)};
        double eligible=simulation.inputTime(activeTime-latest.time);inputs.push(latest,eligible);
        auto fields=commandFields(latest);fields["eligible_simulation_s"]=eligible;fields["backpressure"]=blocked;write("input",fields);
    };
    const auto readControls=[&](){
        Controls c;c.camera=cameraMode;c.rear=keys[SDL_SCANCODE_Q];c.keyboardSteering=true;
        c.drive.throttle=float(keys[SDL_SCANCODE_W]||keys[SDL_SCANCODE_UP])-float(keys[SDL_SCANCODE_S]||keys[SDL_SCANCODE_DOWN]);
        c.drive.steer=float(keys[SDL_SCANCODE_D]||keys[SDL_SCANCODE_RIGHT])-float(keys[SDL_SCANCODE_A]||keys[SDL_SCANCODE_LEFT]);
        c.drive.handbrake=keys[SDL_SCANCODE_SPACE]?1.f:0;
        if(focused){auto pad=controller.read();if(std::abs(pad.throttle)>.01f)c.drive.throttle=pad.throttle;
            if(std::abs(pad.steer)>.01f){c.drive.steer=pad.steer;c.keyboardSteering=false;}c.drive.handbrake=std::max(c.drive.handbrake,pad.handbrake);}
        else c.drive={};return c;
    };
    const auto setPause=[&](bool value){
        if(paused==value)return;
        syncTime();paused=value;simulation.freeze();steering.reset();keys.fill(false);
        write(value?"pause":"resume",{{"discarded_input_edges",inputs.size()}});inputs.reset();
        // Manual pause is a control boundary, unlike automatic backpressure.
        Controls neutral;neutral.camera=cameraMode;latest={neutral,++inputId,activeTime};inputs.push(latest,simulation.time());
        write("input_sync",commandFields(latest));refresh=true;
    };
    reset();
    std::cout<<"Reference streaming mock: "<<config.fps<<" FPS, "<<config.chunk<<" frames/chunk, inference "
        <<config.inferenceMs<<" +/- "<<config.jitterMs<<" ms.\nWASD/arrows drive; Space handbrake; C camera; Q rear; R reset; Esc pause.\n"
        <<"Audio disabled. JSONL: "<<stem.string()<<".jsonl"<<std::endl;
    while(running){
        syncTime();if(demo.duration>0&&now()-start>=demo.duration)break;
        if(!paused&&!minimized&&pipeline.complete(activeTime))write("model_done",{{"chunk_id",pipeline.chunk()}});
        std::optional<Frame> next;
        simulation.observe(activeTime,paused||minimized||pipeline.full());
        SDL_Event event;
        while(SDL_PollEvent(&event)){
            if(event.type==SDL_QUIT)running=false;
            if(event.type==SDL_CONTROLLERDEVICEADDED||event.type==SDL_CONTROLLERDEVICEREMOVED)controller.refresh();
            if(event.type==SDL_WINDOWEVENT){
                if(event.window.event==SDL_WINDOWEVENT_FOCUS_LOST){focused=false;if(demo.inputScript.empty())setPause(true);}
                if(event.window.event==SDL_WINDOWEVENT_FOCUS_GAINED)focused=true;
                if(event.window.event==SDL_WINDOWEVENT_MINIMIZED){syncTime();minimized=true;simulation.freeze();}
                if(event.window.event==SDL_WINDOWEVENT_RESTORED){previous=now();minimized=false;refresh=true;}
                if(event.window.event==SDL_WINDOWEVENT_SIZE_CHANGED||event.window.event==SDL_WINDOWEVENT_EXPOSED)refresh=true;
            }
            if(event.type==SDL_KEYDOWN||event.type==SDL_KEYUP){
                keys.at(size_t(event.key.keysym.scancode))=event.type==SDL_KEYDOWN;
                if(event.type==SDL_KEYDOWN&&!event.key.repeat){auto key=event.key.keysym.sym;
                    if(key==SDLK_ESCAPE)setPause(!paused);
                    if(key==SDLK_r){reset();next.reset();}
                    if(key==SDLK_c&&!paused)cameraMode=(cameraMode+1)%3;
                    if(key==SDLK_F12)capture=true;
                    if(key==SDLK_F11){fullscreen=!fullscreen;SDL_SetWindowFullscreen(window,fullscreen?SDL_WINDOW_FULLSCREEN_DESKTOP:0);}
                    if(key==SDLK_F4&&(event.key.keysym.mod&KMOD_ALT))running=false;
                }
                if(!paused&&demo.inputScript.empty()){
                    double age=double(uint32_t(SDL_GetTicks()-event.key.timestamp))/1000;
                    accept(readControls(),std::max(0.0,activeTime-age));
                }
            }
        }
        if(!running)break;
        if(!paused&&!minimized){
            next=pipeline.pop(activeTime);if(next){shown=std::move(*next);refresh=true;}
            startModel();
            simulation.observe(activeTime,pipeline.full());
            if(demo.inputScript.empty())accept(readControls(),activeTime);
            else while(scriptCursor<script.size()&&script[scriptCursor].time<=activeTime){auto e=script[scriptCursor++];accept(e.controls,e.time);}
            int updates=0;
            while(!pipeline.full()&&simulation.due()&&updates++<64){
                auto command=inputs.sample(simulation.time());
                if(command.id!=applied.id){changes.push_back(command);write("input_apply",commandFields(command));}
                applied=command;world.update(float(1/config.physicsHz),steering.apply(applied.controls.drive,1/config.physicsHz,applied.controls.keyboardSteering));simulation.step();
                if(simulation.tick()%uint64_t(config.stepsPerFrame())==0){
                    Frame frame=grab(++produced);frame.changes=std::move(changes);changes.clear();
                    auto position=::position(frame.player.body);
                    Json meta={{"frame_id",frame.id},{"frame_tick",frame.tick},{"frame_simulation_s",frame.simulationTime},
                        {"caption_visibility_ms",frame.visibilityMs},
                        {"capture_s",frame.capturedAt},{"input_id",frame.command.id},{"world_simulation_s",world.simulationTime},
                        {"camera",{{"eye",{frame.camera.eye.x,frame.camera.eye.y,frame.camera.eye.z}},
                        {"target",{frame.camera.target.x,frame.camera.target.y,frame.camera.target.z}},{"fov",frame.camera.horizontalFov}}},
                        {"position",{position.x,position.y,position.z}},{"speed_kmh",frame.player.speed},
                        {"steer_input",frame.command.controls.drive.steer},{"keyboard_steering",frame.command.controls.keyboardSteering},{"wheel_steer_rad",frame.player.steerAngle},
                        {"yaw_rate_rad_s",frame.player.yawRate},{"lateral_speed_m_s",frame.player.lateralSpeed},
                        {"front_slip_rad",frame.player.frontSlipAngle},{"rear_slip_rad",frame.player.rearSlipAngle},
                        {"grounded_wheels",frame.player.groundedWheels},{"body_up_y",frame.player.body.m[1][1]}};
                    pipeline.push(std::move(frame));write("capture",meta);
                    startModel();
                }
            }
        }
        syncTime();const bool full=pipeline.full();
        if(full)simulation.freeze();
        if(full!=blocked){
            if(full){++blocks;blockedSince=activeTime;write("backpressure_start");}
            else{double span=activeTime-blockedSince;maxBlock=std::max(maxBlock,span);write("backpressure_end",{{"duration_s",span}});}
            blocked=full;refresh=true;
        }
        if(!minimized&&(refresh||capture||activeTime-lastUi>=.1)){
            HudState hud;hud.visible=true;hud.speed=shown.player.speed;hud.heading=shown.player.heading;hud.location=position(shown.player.body);
            hud.roads=&world.mapLines();hud.mapCenter=world.mapCenter();hud.mapHalfSize=world.mapHalfSize();
            const auto number=[](double v){std::ostringstream s;s<<std::fixed<<std::setprecision(1)<<v;return s.str();};
            hud.latencyStatus={std::string("REFERENCE MOCK")+(paused?" PAUSED":blocked?" BACKPRESSURE":""),
                "IN "+std::to_string(pipeline.inputSize())+"  MODEL "+std::to_string(pipeline.inFlight())+"  OUT "+std::to_string(pipeline.outputSize()),
                "SIM/WALL "+number(activeTime>0?simulation.time()/activeTime:0)+"  BLOCKED "+number(blockedSeconds)+" S",
                "LAST INPUT "+(lastLatency<0?std::string("NA"):number(lastLatency)+" MS")};
            displayHud=hud;renderer.displayPixels(shown.rgba,hud);
            if(capture){renderer.screenshot(folder/("capture-"+std::to_string(uint64_t(now()*1000000))+".png"));capture=false;}
            renderer.present(options.vsync);syncTime();
            if(next){
                ++presented;double age=(activeTime-shown.capturedAt)*1000;frameAgeMs.push_back(age);
                write("present",{{"frame_id",shown.id},{"frame_tick",shown.tick},{"frame_simulation_s",shown.simulationTime},
                    {"capture_s",shown.capturedAt},{"frame_age_ms",age}});
                for(const auto& c:shown.changes)if(c.id&&seenInputs.insert(c.id).second){
                    lastLatency=(activeTime-c.time)*1000;responseMs.push_back(lastLatency);
                    auto fields=commandFields(c);fields["frame_id"]=shown.id;fields["response_ms"]=lastLatency;write("first_present",fields);
                }
            }
            telemetry.publish(shown.camera,options.width,options.height,shown.command.controls.camera,
                (paused?CameraTelemetry::Paused:0u)|(focused?CameraTelemetry::Active:0u),now()-start,shown.simulationTime);
            lastUi=activeTime;refresh=false;
        }
        if(now()-lastFlush>=1){log.flush();if(captioner){captionLog.flush();visibilityLog.flush();}lastFlush=now();}
        SDL_Delay(1);
    }
    syncTime();if(blocked)maxBlock=std::max(maxBlock,activeTime-blockedSince);
    if(!demo.screenshot.empty()){renderer.displayPixels(shown.rgba,displayHud);renderer.screenshot(demo.screenshot);}
    Json summary={{"epoch",epoch},{"active_s",activeTime},{"simulation_s",simulation.time()},
        {"produced",produced},{"presented",presented},{"backpressure_count",blocks},{"backpressure_s",blockedSeconds},{"max_backpressure_s",maxBlock},
        {"max_input_frames",maxInput},{"max_output_frames",maxOutput},{"max_in_flight",maxFlight},{"max_total_queued_frames",maxTotal},
        {"input_capacity",config.inputCapacity},{"output_capacity_including_in_flight",config.outputCapacity},
        {"remaining_frames",pipeline.inputSize()+pipeline.outputSize()+pipeline.inFlight()},
        {"pending_input_edges",inputs.size()+changes.size()},{"response_count",responseMs.size()},
        {"recorded_frames",videoFrames},
        {"response_p50_ms",percentile(responseMs,.5)},{"response_p95_ms",percentile(responseMs,.95)},
        {"frame_age_p50_ms",percentile(frameAgeMs,.5)},{"frame_age_p95_ms",percentile(frameAgeMs,.95)}};
    write("end",summary);log.flush();std::ofstream report(stem.string()+".json");report<<summary.dump(2)<<'\n';
    std::cout<<"Reference session saved: "<<stem.string()<<".jsonl\n"<<summary.dump(2)<<std::endl;return 0;
}
