#include "OfflineDemo.h"
#include "Encounter.h"
#include "CombatAudio.h"
#include "CombatSoundCues.h"
#include "DemoRecorder.h"
#include <SDL.h>
#include <chrono>
#include <iostream>
#include <set>
#include <thread>

namespace {
constexpr int Fps=48,TicksPerFrame=5;
constexpr double Step=1./120,Speed=2;
Json inputJson(const CombatInput& in){return {{"forward",in.forward},{"right",in.right},{"look_x",in.lookX},{"look_y",in.lookY},{"fire",in.fire},{"ads",in.ads},{"shield",in.shield},{"crouch",in.crouch},{"jump",in.jump},{"dash",in.dash},{"reload",in.reload},{"trap",in.trap},{"cover",in.cover},{"weapon",in.weapon}};}
Json vec(V3 v){return {v.x,v.y,v.z};}
class AudioOutput {
    SDL_AudioDeviceID device=0;
public:
    explicit AudioOutput(bool disabled){
        if(disabled)return;
        if(SDL_InitSubSystem(SDL_INIT_AUDIO)){std::cerr<<"Audio disabled: "<<SDL_GetError()<<'\n';return;}
        SDL_AudioSpec spec{};spec.freq=48000;spec.format=AUDIO_F32SYS;spec.channels=2;spec.samples=512;
        device=SDL_OpenAudioDevice(nullptr,0,&spec,nullptr,0);
        if(!device)std::cerr<<"Audio disabled: "<<SDL_GetError()<<'\n';else SDL_PauseAudioDevice(device,0);
    }
    ~AudioOutput(){if(device)SDL_CloseAudioDevice(device);}
    void clear(){if(device)SDL_ClearQueuedAudio(device);}
    void play(const std::vector<float>& block){if(!device)return;
        if(SDL_GetQueuedAudioSize(device)>48000*2*sizeof(float)/10)clear();
        if(SDL_QueueAudio(device,block.data(),Uint32(block.size()*sizeof(float))))std::cerr<<"Audio queue: "<<SDL_GetError()<<'\n';
    }
};
std::vector<SoundCue> soundCues(const std::vector<CombatEvent>& events,const Encounter& game,double start){
    return combatSoundCues(events,game.camera(0),game.playerToken(0),start,1000,Speed);
}
}

int runOfflineDemo(const std::filesystem::path& root,const DemoOptions& options){
    const auto scene=SceneConfig::load(root,options.scene);
    const auto title=scene.name+" - 48 FPS / 2x";
    if(SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER))throw std::runtime_error(SDL_GetError());
    struct Quit{~Quit(){SDL_Quit();}}quit;
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef __APPLE__
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
#endif
    std::unique_ptr<SDL_Window,decltype(&SDL_DestroyWindow)> window(SDL_CreateWindow(title.c_str(),SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,options.width,options.height,SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE|(options.test?SDL_WINDOW_HIDDEN:SDL_WINDOW_SHOWN)),SDL_DestroyWindow);
    if(!window)throw std::runtime_error(SDL_GetError());
    Renderer renderer(window.get(),options.width,options.height);MapData map(scene);Encounter game(root,map,renderer,options.seed);game.setAutopilot(false);game.takeEvents();
    CombatAudio audio(options.volume);AudioOutput output(options.noAudio||options.test);
    std::unique_ptr<DemoRecorder> recorder;
    auto folder=options.output.empty()?scene.outputDirectory(root,"offline-recordings"):std::filesystem::absolute(options.output);
    const auto stamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto session=folder/("session-"+std::to_string(stamp));int take=0;
    std::set<SDL_Scancode> keys;CombatInput input;float dx=0,dy=0;bool fire=false,firePulse=false,ads=false,paused=false,dead=false,running=true,focused=true;int selection=0;
    uint64_t frames=0;double origin=now(),next=origin;
    auto rawState=[&]{Json held=Json::array();for(auto k:keys)held.push_back(SDL_GetScancodeName(k));return Json{{"keys",held},{"fire",fire},{"ads",ads}};};
    auto clearInput=[&]{keys.clear();input={};dx=dy=0;fire=firePulse=ads=false;};
    auto finish=[&](const std::string& reason){if(recorder){recorder->finish(reason);std::cout<<"Saved "<<recorder->directory()<<" ("<<recorder->frames()<<" frames)"<<std::endl;recorder.reset();SDL_SetWindowTitle(window.get(),title.c_str());}};
    auto start=[&]{if(recorder||paused||dead)return;
        recorder=std::make_unique<DemoRecorder>(session/("take-"+std::to_string(++take)),1280,720,Fps,Json{{"game",scene.id},{"scene",scene.id},{"map_profile",(scene.assets/"map.json").string()},{"seed",options.seed},{"speed",Speed},{"physics_hz",120},{"physics_steps_per_frame",TicksPerFrame},{"simulation_start",game.seconds()},{"simulation_step_start",game.step()},{"prompt",scene.prompt},{"prompt_verified",!scene.prompt.empty()},{"camera_distance",scene.cameraDistance},{"camera_up",scene.cameraUp},{"outline_width",scene.outlineWidth},{"fade_small_edges",scene.fadeSmallEdges},{"nested_checker",true},{"ground_pattern",scene.groundPattern==SceneConfig::GroundPattern::CastleNested?"castle_nested":"egypt_nested"},{"audio_volume",options.volume},{"reference_preprocessing","1280x720 source; use the model's crop_and_resize to 864x480"}});
        recorder->input({{"type","initial"},{"wall_s",now()-origin},{"state",rawState()}});SDL_SetWindowTitle(window.get(),("REC - "+title).c_str());std::cout<<"Recording "<<recorder->directory()<<std::endl;
    };
    auto pause=[&](const std::string& reason){finish(reason);paused=true;selection=0;clearInput();if(reason!="death")output.clear();audio.reset();SDL_SetRelativeMouseMode(SDL_FALSE);};
    auto choose=[&](int item){
        if(dead){if(item==1){running=false;return;}if(item!=0)return;if(!game.retry())throw std::runtime_error("Cannot restart encounter");dead=false;}
        else{if(item==4){running=false;return;}if(item<0||item>3)return;if(item>0)input.weapon=item-1;}
        paused=false;keys.clear();dx=dy=0;fire=firePulse=ads=false;output.clear();audio.reset();game.takeEvents();next=now();
        if(!options.test)SDL_SetRelativeMouseMode(SDL_TRUE);
    };
    if(options.automaticRecord||options.test)start();
    while(running){
        if(options.test){
            auto press=[](SDL_Keycode key){SDL_Event event{};event.type=SDL_KEYDOWN;event.key.keysym.sym=key;event.key.keysym.scancode=SDL_GetScancodeFromKey(key);SDL_PushEvent(&event);event.type=SDL_KEYUP;SDL_PushEvent(&event);};
            if(frames==0){keys.insert(SDL_SCANCODE_W);fire=true;}
            if(frames==72)press(SDLK_2);if(frames==144)press(SDLK_3);
            if(frames==480)press(SDLK_ESCAPE);if(frames==481)press(SDLK_RETURN);if(frames==482)press(SDLK_F9);
            if(frames==530)press(SDLK_F9);if(frames==532)press(SDLK_F9);if(frames==580)running=false;
            dx+=1.5f;
        }
        SDL_Event event;while(SDL_PollEvent(&event)){
            Json recorded;
            if(event.type==SDL_KEYDOWN||event.type==SDL_KEYUP)recorded={{"type","key"},{"key",SDL_GetScancodeName(event.key.keysym.scancode)},{"down",event.type==SDL_KEYDOWN},{"repeat",bool(event.key.repeat)}};
            else if(event.type==SDL_MOUSEBUTTONDOWN||event.type==SDL_MOUSEBUTTONUP)recorded={{"type","button"},{"button",event.button.button},{"down",event.type==SDL_MOUSEBUTTONDOWN}};
            else if(event.type==SDL_MOUSEMOTION)recorded={{"type","mouse"},{"dx",event.motion.xrel},{"dy",event.motion.yrel}};
            if(recorder&&!recorded.is_null()){recorded["wall_s"]=now()-origin;recorded["sdl_ms"]=event.common.timestamp;recorder->input(recorded);}
            if(event.type==SDL_QUIT){running=false;continue;}
            if(event.type==SDL_WINDOWEVENT){
                if(event.window.event==SDL_WINDOWEVENT_SIZE_CHANGED)renderer.resize(event.window.data1,event.window.data2);
                if(!options.test&&(event.window.event==SDL_WINDOWEVENT_FOCUS_LOST||event.window.event==SDL_WINDOWEVENT_MINIMIZED)){focused=false;pause("focus_lost");}
                if(event.window.event==SDL_WINDOWEVENT_FOCUS_GAINED)focused=true;
            }
            if(event.type==SDL_KEYUP)keys.erase(event.key.keysym.scancode);
            if(event.type==SDL_KEYDOWN&&!event.key.repeat){const auto code=event.key.keysym.sym;
                if(code==SDLK_ESCAPE){if(!paused)pause("menu");else if(!dead)choose(0);continue;}
                if(paused){if(code==SDLK_UP)selection=CombatUi::moveSelection(selection,-1,dead);else if(code==SDLK_DOWN)selection=CombatUi::moveSelection(selection,1,dead);else if(code==SDLK_RETURN||code==SDLK_KP_ENTER)choose(selection);continue;}
                if(code==SDLK_F9){if(recorder)finish("manual");else start();continue;}
                keys.insert(event.key.keysym.scancode);
                if(code>=SDLK_1&&code<=SDLK_3)input.weapon=int(code-SDLK_1);
                if(code==SDLK_SPACE)input.jump=true;if(code==SDLK_LSHIFT||code==SDLK_RSHIFT)input.dash=true;
                if(code==SDLK_r)input.reload=true;if(code==SDLK_t)input.trap=true;if(code==SDLK_h)input.cover=true;
            }
            if(event.type==SDL_MOUSEBUTTONDOWN||event.type==SDL_MOUSEBUTTONUP){bool down=event.type==SDL_MOUSEBUTTONDOWN;
                if(paused){if(down&&event.button.button==SDL_BUTTON_LEFT)choose(CombatUi::hitMenu(renderer.width,renderer.height,event.button.x,event.button.y,dead));continue;}
                if(down&&!options.test)SDL_SetRelativeMouseMode(SDL_TRUE);
                if(event.button.button==SDL_BUTTON_LEFT){firePulse|=down&&!fire;fire=down;}if(event.button.button==SDL_BUTTON_RIGHT)ads=down;
            }
            if(event.type==SDL_MOUSEMOTION&&!paused&&SDL_GetRelativeMouseMode()){dx+=event.motion.xrel;dy+=event.motion.yrel;}
        }
        if(!running)break;
        HudState hud;hud.combat=true;hud.paused=paused;hud.dead=dead;hud.pauseSelection=selection;
        if(!paused&&focused){
            auto held=[&](SDL_Scancode k){return keys.contains(k);};
            input.forward=float(held(SDL_SCANCODE_W)||held(SDL_SCANCODE_UP))-float(held(SDL_SCANCODE_S)||held(SDL_SCANCODE_DOWN));
            input.right=float(held(SDL_SCANCODE_D)||held(SDL_SCANCODE_RIGHT))-float(held(SDL_SCANCODE_A)||held(SDL_SCANCODE_LEFT));
            input.fire=fire||firePulse;input.ads=ads;input.shield=held(SDL_SCANCODE_F);input.crouch=held(SDL_SCANCODE_C);
            input.lookX=CameraInput::pixelsPerTick(dx,TicksPerFrame,Step);input.lookY=CameraInput::pixelsPerTick(dy,TicksPerFrame,Step);dx=dy=0;
            const double begin=game.seconds();const uint64_t firstStep=game.step();Json applied=Json::array();std::vector<CombatEvent> events;
            for(int tick=0;tick<TicksPerFrame&&game.playerAlive();++tick){applied.push_back(inputJson(input));game.advance(Step,input);auto part=game.takeEvents();events.insert(events.end(),part.begin(),part.end());input.jump=input.dash=input.reload=input.trap=input.cover=false;input.weapon=-1;firePulse=false;input.fire=fire;}
            input.lookX=input.lookY=0;dead=!game.playerAlive();game.hud(hud,0);hud.dead=dead;
            auto cues=soundCues(events,game,begin);auto pcm=audio.render(1000,cues);
            if(recorder){
                HudState cleanHud=hud;cleanHud.showHelp=false;cleanHud.recording=false;cleanHud.showFps=false;
                Json eventRows=Json::array();for(const auto& e:events)eventRows.push_back({{"id",e.id},{"step",e.step},{"simulation_s",e.time},{"type",e.type},{"actor",e.actor},{"weapon",e.weapon},{"indices",e.indices},{"tokens",e.tokens},{"position",vec(e.position)}});
                const auto camera=game.camera(0);
                recorder->submit(renderer.captureDemoScene(game.draws(),camera),renderer.captureDemoUi(cleanHud),
                    {{"simulation_s",game.seconds()},{"simulation_start_s",begin},{"physics_step",game.step()},{"physics_step_start",firstStep},{"input_applied",applied},{"raw_input",rawState()},{"health",hud.health},{"dead",dead},{"weapon",hud.weapon},{"ammo",hud.ammo},{"events",eventRows},{"camera",{{"eye",vec(camera.eye)},{"target",vec(camera.target)},{"horizontal_fov",camera.horizontalFov}}}},pcm);
            }
            hud.recording=bool(recorder);hud.recordFrames=recorder?int(recorder->frames()):0;
            renderer.render(game.draws(),game.camera(0),hud);renderer.present(false);output.play(pcm);
            if(dead){pause("death");}
            else if(game.recoveryRequested()){pause("recovery");if(!game.recover())throw std::runtime_error("Cannot recover encounter");}
        }else{game.hud(hud,0);renderer.render(game.draws(),game.camera(0),hud);renderer.present(false);}
        ++frames;
        // No catch-up burst after encoding stalls: preserve every input frame.
        if(!options.test){next+=1./Fps;const double current=now();if(current<next)std::this_thread::sleep_for(std::chrono::duration<double>(next-current));else next=current;}
    }
    finish("exit");SDL_SetRelativeMouseMode(SDL_FALSE);return 0;
}
