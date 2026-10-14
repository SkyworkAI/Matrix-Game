#include "LocalRaceControls.h"
#include "KeyboardSteering.h"
#include <iostream>

namespace {
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
RaceRoute testRoute(const std::string& id) {
    RaceRoute route;route.id=id;route.name=id;
    for(int i=0;i<5;++i)route.grid.push_back({{4.2f,0,-38.f-i*9},{0,0,1}});
    for(float z:{50.f,100.f})route.checkpoints.push_back({{4.2f,0,z},{0,0,1},5.5f,3.5f,{{{4.2f,0,z-8},{0,0,1}}}});
    return route;
}
void run(const std::filesystem::path& root) {
    World world(root/"assets/bbox-assets.bin",0,0,true);
    world.setPlayerSteeringLimit(false);world.setPlayerLateralGrip(1.15f,.9f);world.setPlayerFrontDrive(.7f);
    RaceSession race(world,{testRoute("first"),testRoute("second")});
    int id=race.command({{"type","join"},{"name","Local"}}).at("player_id");
    LocalRaceControls controls(race,{id});
    const auto& player=*race.players()[0];
    size_t bodies=0;
    for(const auto& object:world.drawObjects())if(object.radius==4&&object.range==380){
        const Color original=rgb(79,116,140),actual=object.instance.color;
        require(actual.r==original.r&&actual.g==original.g&&actual.b==original.b,"race changed original car color");++bodies;
    }
    require(bodies>0,"no player car checked for original color");
    constexpr double step=1./480;
    auto advance=[&](double seconds,Input input=Input{}){
        for(int i=0;i<int(std::ceil(seconds/step));++i){controls.drive(input,1,false);race.tick(step);}
    };

    require(!controls.recover(),"no recovery in lobby");
    controls.start("first");require(race.phase()=="countdown","local start enters countdown");
    controls.drive({1,.5f,0,1},2,true);
    require(player.camera==2&&player.lookBack,"camera and rear view reach race player");
    require(player.input.throttle==0,"countdown ignores driving");
    require(!controls.recover(),"no recovery before start");
    race.tick(.5);controls.pause();double countdown=race.countdownSeconds();
    race.tick(5);require(race.countdownSeconds()==countdown&&race.phase()=="paused","pause freezes countdown");
    require(!controls.recover(),"paused countdown cannot queue recovery");
    controls.resume();advance(3);require(race.phase()=="racing","countdown resumes");
    KeyboardSteering steering;
    for(int i=0;i<144;++i){
        Input filtered=steering.apply({0,1,0,0},step,true);controls.drive(filtered,1,false);race.tick(step);
        require(std::abs(world.playerPose(0).steerAngle-filtered.steer*37*Pi/180)<2e-5f,"local race lost per-substep keyboard steering");
        if(i==0)require(player.input.steer>0&&player.input.steer<.01f,"race keyboard instantly steered full lock");
    }
    require(std::abs(steering.current()-1)<1e-5f,"race keyboard failed to reach full lock");
    steering.reset();controls.drive({},1,false);race.tick(step);

    auto idle=position(world.playerPose(0).body);advance(8);
    auto afterIdle=position(world.playerPose(0).body);
    require(std::hypot(afterIdle.x-idle.x,afterIdle.z-idle.z)<.5,"manual player stays still without throttle");
    require(player.progress.penalty==0,"waiting does not trigger automatic stuck recovery");
    controls.drive({.8f,-.4f,.2f,1},0,true);
    require(player.input.throttle==.8f&&player.input.steer==-.4f&&player.input.brake==.2f&&player.input.handbrake==1,"all manual controls reach physics input");
    advance(3,{1,0,0,0});
    require((position(world.playerPose(0).body)-afterIdle).length()>4,"manual throttle moves tuned car");
    controls.drive({},1,false);require(player.input.throttle==0&&player.input.steer==0,"release clears controls");
    require(player.camera==1&&!player.lookBack,"rear view releases");

    controls.pause();controls.drive({1,1,0,0},1,false);
    auto frozen=race.state();auto frozenPosition=position(world.playerPose(0).body);
    for(int i=0;i<480;++i)race.tick(step);
    require(race.state()==frozen&&race.collisions().empty(),"pause freezes clocks and emits no repeated collisions");
    require((position(world.playerPose(0).body)-frozenPosition).length()<1e-6f,"pause freezes physics");
    controls.resume();require(race.phase()=="racing"&&player.input.throttle==0,"resume clears stale input");
    double penalty=player.progress.penalty;
    require(controls.recover()&&!controls.recover(),"repeated recovery is not queued twice");
    race.tick(step);require(player.progress.penalty==penalty+5&&player.recovery.empty(),"recovery applies exactly five seconds");
    require(player.progress.next==0,"recovery does not skip checkpoints");

    controls.pause();controls.start("second");
    require(race.route().id=="second"&&race.phase()=="countdown","restart from pause can change route");
    require(player.progress.elapsed==0&&player.progress.penalty==0&&player.progress.next==0&&!player.progress.finished,"restart resets score");
    require(player.input.throttle==0,"restart clears held throttle");
    advance(3.1);
    for(size_t gate=0;gate<race.route().checkpoints.size();++gate){
        auto center=race.route().checkpoints[gate].center;
        world.placePlayer(0,center-V3{0,0,3},0);
        for(int i=0;i<2400&&player.progress.next==gate;++i){controls.drive({1,0,0,0},1,false);race.tick(step);}
        require(player.progress.next==gate+1,"manual driving crosses the next gate");
    }
    require(race.phase()=="results"&&player.progress.finished,"manual finish reaches results");
    double total=player.progress.total();advance(8);
    require(race.phase()=="results"&&player.progress.total()==total,"results remain with frozen score");
    require(!controls.recover(),"recovery after finish is ignored");
    controls.start("second");require(race.phase()=="countdown"&&player.progress.total()==0,"retry resets finished race");
    controls.lobby();require(race.phase()=="lobby"&&player.progress.next==0,"return to route selection ends current race");
    controls.start("first");advance(3.1,{1,0,0,0});
    require(race.phase()=="racing"&&player.input.throttle==1,"input sequence remains valid across retries");
}
}
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Pass repository path");run(argv[1]);
    std::cout<<"Local manual driving, camera, countdown, pause, recovery, finish, retry and route switching passed\n";return 0;
}catch(const std::exception& e){std::cerr<<"LocalRaceTests: "<<e.what()<<'\n';return 1;}}
