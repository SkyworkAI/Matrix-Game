#include "Map.h"
#include "Encounter.h"
#include "ReferencePacing.h"
#include "CombatSoundCues.h"
#include <SDL.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <csignal>
#include <cstring>
#include <deque>
#include <iostream>

namespace {
struct Channel {
    int fd;std::vector<uint8_t> incoming;std::deque<std::vector<uint8_t>> outgoing;size_t offset=0;
    explicit Channel(int f):fd(f){}
    ~Channel(){close(fd);}
    void send(const Json& meta,const std::vector<uint8_t>& bytes={}){
        if(outgoing.size()>=2)throw std::runtime_error("IPC output queue full");
        auto text=meta.dump();uint32_t sizes[]={htonl(uint32_t(text.size())),htonl(uint32_t(bytes.size()))};
        std::vector<uint8_t> packet(8+text.size()+bytes.size());std::memcpy(packet.data(),sizes,8);
        std::memcpy(packet.data()+8,text.data(),text.size());
        if(!bytes.empty())std::memcpy(packet.data()+8+text.size(),bytes.data(),bytes.size());outgoing.push_back(std::move(packet));
    }
    void flush(){
        if(outgoing.empty())return;auto& packet=outgoing.front();
        auto n=::send(fd,packet.data()+offset,packet.size()-offset,MSG_DONTWAIT);
        if(n<0&&(errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR))return;
        if(n<=0)throw std::runtime_error("IPC consumer disconnected");offset+=size_t(n);
        if(offset==packet.size()){outgoing.pop_front();offset=0;}
    }
    std::vector<Json> receive(){
        uint8_t data[8192];auto n=recv(fd,data,sizeof(data),MSG_DONTWAIT);
        if(n==0)throw std::runtime_error("IPC consumer disconnected");
        if(n<0&&errno!=EAGAIN&&errno!=EWOULDBLOCK&&errno!=EINTR)throw std::runtime_error("IPC read failed");
        if(n>0)incoming.insert(incoming.end(),data,data+n);
        if(incoming.size()>65536)throw std::runtime_error("IPC input limit exceeded");
        std::vector<Json> result;
        while(incoming.size()>=8){uint32_t sizes[2];std::memcpy(sizes,incoming.data(),8);size_t length=ntohl(sizes[0]);
            if(!length||length>16384||ntohl(sizes[1]))throw std::runtime_error("Invalid IPC command");
            if(incoming.size()<8+length)break;
            result.push_back(Json::parse(incoming.begin()+8,incoming.begin()+8+length));incoming.erase(incoming.begin(),incoming.begin()+8+length);
        }return result;
    }
    void wait(){pollfd p{fd,short(POLLIN|(outgoing.empty()?0:POLLOUT)),0};if(poll(&p,1,1)<0&&errno!=EINTR)throw std::runtime_error("IPC poll failed");}
};
}

int main(int argc,char** argv){
    std::signal(SIGPIPE,SIG_IGN);
    try{
        std::filesystem::path root=".";int fd=-1,fps=8;uint32_t seed=1729;std::string sceneId="castle",captureMode="color";
        bool framePackets=false,continuousPacing=true;
        for(int i=1;i<argc;++i){std::string arg=argv[i];if(++i>=argc)throw std::runtime_error("Missing option value");
            if(arg=="--frame-packets")framePackets=std::stoi(argv[i])!=0;
            else if(arg=="--continuous-pacing")continuousPacing=std::stoi(argv[i])!=0;
            else if(arg=="--capture-mode")captureMode=argv[i];
            else if(arg=="--scene")sceneId=argv[i];else if(arg=="--data-dir")root=argv[i];else if(arg=="--ipc-fd")fd=std::stoi(argv[i]);else if(arg=="--fps")fps=std::stoi(argv[i]);else if(arg=="--seed")seed=uint32_t(std::stoul(argv[i]));else throw std::runtime_error("Unknown reference option");}
        if(fd<0||fps<1||fps>60||120%fps)throw std::runtime_error("FPS must divide the 120 Hz physics rate");
        if(captureMode!="full"&&captureMode!="color"&&captureMode!="verify")throw std::runtime_error("Invalid capture mode");
        const auto scene=SceneConfig::load(root,sceneId);
        const bool headless=std::getenv("CASTLE_EGL")!=nullptr;
        if(SDL_Init((headless?0:SDL_INIT_VIDEO)|SDL_INIT_TIMER))throw std::runtime_error(SDL_GetError());
        struct Quit{~Quit(){SDL_Quit();}}quit;
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef __APPLE__
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
#endif
        std::unique_ptr<SDL_Window,decltype(&SDL_DestroyWindow)> window(headless?nullptr:SDL_CreateWindow("Castle Reference",0,0,1280,720,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN),SDL_DestroyWindow);
        if(!headless&&!window)throw std::runtime_error(SDL_GetError());
        Renderer renderer(window.get(),1280,720);MapData map(scene);Encounter game(root,map,renderer,seed);
        game.setAutopilot(false);game.prepareClip();game.takeEvents();Channel channel(fd);
        channel.send({{"type","hello"},{"schema",1},{"scene",scene.id},{"prompt",scene.prompt},{"prompt_verified",!scene.prompt.empty()},{"width",1280},{"height",720},{"fps",fps},{"first_chunk",1},{"chunk",4},{"trees",map.treeCount},{"renderer",renderer.adapterName},
            {"frame_packets",framePackets},{"continuous_pacing",continuousPacing},{"capture_mode",captureMode},{"audio_schema",1},
            {"mouse_input_schema",2},{"camera_max_degrees_per_second",CameraInput::MaxDegreesPerSecond},
            {"fade_small_edges",scene.fadeSmallEdges},{"road_surface_splits",map.roadSurfaceSplits.size()}});
        CombatInput input;bool firePulse=false;uint64_t frame=0,seq=0,tick=0;int remaining=0;double nextTick=0,lastInput=now();
        CameraInput::NetworkAxis lookX,lookY;
        std::vector<uint8_t> pixels;Json states=Json::array();bool running=true;
        while(running){
            for(const auto& command:channel.receive()){
                std::string type=command.at("type");
                if(type=="stop"){running=false;break;}
                if(type=="request"){
                    if(remaining||!states.empty()||command.at("frame_start").get<uint64_t>()!=frame)throw std::runtime_error("Out-of-order reference request");
                    remaining=frame==0?1:4;
                    nextTick=continuousPacing?resumeReferenceTick(nextTick,now(),fps):now()+1./120;
                }else if(type=="controls"){
                    auto axis=[&](const char* key,float limit){float value=command.value(key,0.f);if(!std::isfinite(value)||std::abs(value)>limit)throw std::runtime_error("Invalid control value");return value;};
                    input.forward=axis("forward",1);input.right=axis("right",1);
                    const float mouseX=axis("look_x",2000),mouseY=axis("look_y",2000);
                    if(command.value("look_reset",false)){lookX.clear();lookY.clear();}
                    else{const double received=now();lookX.add(mouseX,received);lookY.add(mouseY,received);}
                    bool fire=command.value("fire",false);firePulse|=fire&&!input.fire;input.fire=fire;
                    input.ads=command.value("ads",false);input.shield=command.value("shield",false);input.crouch=command.value("crouch",false);
                    for(const auto& action:command.value("actions",Json::array())){
                        if(action=="jump")input.jump=true;else if(action=="dash")input.dash=true;else if(action=="reload")input.reload=true;else if(action=="trap")input.trap=true;else if(action=="cover")input.cover=true;else throw std::runtime_error("Unknown action");}
                    int weapon=command.value("weapon",-1);if(weapon< -1||weapon>2)throw std::runtime_error("Invalid weapon");if(weapon>=0)input.weapon=weapon;
                    seq=command.at("sequence");lastInput=now();
                }else throw std::runtime_error("Unknown IPC command");
            }
            if(!running)break;
            if(now()-lastInput>1){input={};firePulse=false;lookX.clear();lookY.clear();}
            channel.flush();
            if(!remaining||channel.outgoing.size()>=2){if(!continuousPacing)nextTick=now()+1./120;channel.wait();continue;}
            if(now()<nextTick){channel.wait();continue;}
            CombatInput applied=input;applied.fire|=firePulse;
            const double inputTime=now();
            applied.lookX=lookX.consume(1.f/120,inputTime);
            applied.lookY=lookY.consume(1.f/120,inputTime);
            game.advance(1.f/120,applied);++tick;nextTick+=1./120;
            if(game.step()!=tick){
                if(game.eventState().at("Actors")[0].at("Health").get<float>()>0)
                    throw std::runtime_error("Live game physics clock stopped unexpectedly");
                // A terminal partial block is not a valid model input. End the
                // session explicitly instead of inventing timestamps for frozen frames.
                channel.send({{"type","end"},{"reason","defeated"},{"frame_start",frame}});
                while(!channel.outgoing.empty()){channel.flush();channel.wait();}
                break;
            }
            if(now()-nextTick>.1)nextTick=now()+1./120;
            input.lookX=input.lookY=0;input.jump=input.dash=input.reload=input.trap=input.cover=false;input.weapon=-1;firePulse=false;
            if(tick%uint64_t(120/fps))continue;
            const double captureStart=now();
            auto color=captureMode=="full"?renderer.captureScene(game.draws(),game.camera(0),game.catalog(),false).color:
                renderer.captureDemoScene(game.draws(),game.camera(0));
            if(captureMode=="verify"&&color!=renderer.captureScene(game.draws(),game.camera(0),game.catalog(),false).color)
                throw std::runtime_error("Color-only capture differs from full capture");
            const double captureMs=(now()-captureStart)*1000;
            pixels.insert(pixels.end(),color.begin(),color.end());HudState hud;game.hud(hud,0);
            auto actors=game.eventState().at("Actors");if(actors.size()!=3||actors[1].at("Weapon")!=2||actors[2].at("Weapon")!=2)throw std::runtime_error("Unexpected actor/loadout policy");
            auto camera=game.camera(0);
            Json sounds=Json::array();
            for(const auto& cue:combatSoundCues(game.takeEvents(),camera,game.playerToken(0),
                    game.seconds()-1./fps,CombatAudio::SampleRate/fps))
                sounds.push_back({{"sound",int(cue.sound)},{"gain",cue.gain},{"pan",cue.pan},
                    {"offset",double(cue.sample)/CombatAudio::SampleRate-1./fps}});
            states.push_back({{"frame",frame},{"capture_monotonic",now()},{"capture_ms",captureMs},{"simulation_time",game.seconds()},{"input_sequence",seq},
                {"sounds",sounds},
                {"health",hud.health},{"shield",hud.shield},{"ammo",hud.ammo},{"weapon",hud.weapon},{"kills",hud.kills},{"action",hud.combatAction},
                {"player",hud.player},{"automatic",hud.automatic},{"owned",hud.owned},{"raised",hud.raised},
                {"trap_cooldown",hud.trapCooldown},{"cover_durability",hud.coverDurability},{"cover_status",hud.coverStatus},{"cover_hint",hud.coverHint},
                {"camera",{{"eye",{camera.eye.x,camera.eye.y,camera.eye.z}},{"target",{camera.target.x,camera.target.y,camera.target.z}}}}});
            ++frame;--remaining;
            if(framePackets||!remaining){channel.send({{"type",framePackets?"frame":"chunk"},{"frame_start",frame-states.size()},{"count",states.size()},{"states",states}},pixels);states=Json::array();pixels.clear();}
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
