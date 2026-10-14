#include "RaceSession.h"
#include "CollisionEvents.h"
#include <iostream>

namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void crossingTests(){
    Checkpoint gate{{0,0,10},{0,0,1},4,3.5f,{}};
    require(crossCheckpoint(gate,{0,1,9},{0,1,11}).has_value(),"forward crossing");
    require(!crossCheckpoint(gate,{0,1,11},{0,1,9}),"reverse crossing");
    require(!crossCheckpoint(gate,{0,-7,9},{0,-7,11}),"wrong road level");
    require(!crossCheckpoint(gate,{5,1,9},{5,1,11}),"outside gate");
    RaceRoute route;route.checkpoints={gate,{{0,0,20},{0,0,1},4,3.5f,{}}};RaceProgress p;
    p.advance(route,{0,1,15},{0,1,25},1);require(p.next==0&&!p.finished,"cannot skip earlier gate");
    p.advance(route,{0,1,0},{0,1,30},3);require(p.next==2&&p.finished&&std::abs(p.elapsed-3)<1e-6,"multiple gates and fractional finish time");
    p.advance(route,{0,1,0},{0,1,30},9);require(std::abs(p.elapsed-3)<1e-6,"finish time frozen");
    std::reverse(route.checkpoints.begin(),route.checkpoints.end());p={};p.advance(route,{0,1,0},{0,1,30},1);require(p.next==1&&!p.finished,"intersection order cannot run backwards within a step");
}
void contactsTests(){
    CollisionEvents events;CollisionEvents::Contact a{1,2,7,9},b{1,2,8,9};
    events.touch(a,1,2,4);events.touch(b,1,2,4);require(events.take().size()==1,"subshapes merge");
    for(int i=0;i<1000;++i){events.touch(a,1,2,4);events.step(.002f);}require(events.take().empty(),"continuous scrape counted once");
    events.remove(a);events.step(.3f);events.touch(a,1,2,4);require(events.take().empty(),"remaining subshape keeps contact active");
    events.remove(a);events.remove(b);events.step(.05f);events.touch(a,1,2,4);require(events.take().empty(),"solver jitter does not repeat penalty");
    events.remove(a);events.step(.3f);events.touch(a,1,2,4);require(events.take().size()==1,"separated impact counts again");
}
void navigationTests(){
    RaceRoute route;route.guide.points={{0,0,0},{0,0,60},{80,0,60}};
    route.guide.distances={0,60,140};route.guide.checkpoints={140};
    route.checkpoints={{{80,0,60},{1,0,0},5.5f,3.5f,{}}};RaceProgress progress;
    auto straight=navigateRoute(route,progress,{0,.8f,10},0,36);
    require(straight.valid&&std::abs(straight.targetAngle)<.001f,"navigation follows road before corner instead of pointing through blocks");
    require(std::abs(straight.distanceToGate-130)<.01f&&std::abs(straight.distanceToFinish-130)<.01f,"distance follows road geometry");
    auto turn=navigateRoute(route,progress,{0,.8f,55},0,36);
    require(turn.targetAngle>.5f&&turn.targetAngle<Pi/2,"navigation anticipates right turn");
    require(navigateRoute(route,progress,{0,.8f,10},Pi,0).wrongWay,"wrong-way warning uses route direction");
    require(navigateRoute(route,progress,{30,.8f,10},0,0).offRoute,"off-route warning");
    auto missed=navigateRoute(route,progress,{88,.8f,60},Pi/2,0);
    require(missed.missedCheckpoint&&std::abs(missed.targetAngle)>3,"missed gate directs player back to its approach side");
    require(progress.next==0&&!progress.finished,"navigation never validates a gate");
    progress.finished=true;auto finish=navigateRoute(route,progress,{88,.8f,60},Pi/2,0);
    require(finish.valid&&finish.distanceToFinish==0&&finish.targetAngle==0,"finished player has no further driving instructions");
    route.guide.points={{0,0,0},{0,0,50},{50,10,50},{50,10,0},{0,10,0},{0,10,50}};
    route.guide.distances={0,50,101,151,201,251};route.guide.checkpoints={50,251};
    route.checkpoints={{{0,0,50},{0,0,1},5.5f,3.5f,{}},{{0,10,50},{0,0,1},5.5f,3.5f,{}}};progress={};
    auto elevated=navigateRoute(route,progress,{0,10.8f,25},0,0);
    require(elevated.offRoute&&elevated.along<=50&&elevated.distanceToFinish>200,"stacked future road cannot steal current checkpoint leg");
    progress.next=1;auto second=navigateRoute(route,progress,{0,10.8f,25},0,0);
    require(!second.offRoute&&second.along>200&&second.distanceToFinish<30,"height distinguishes coincident roads after legitimate gate progress");
    route.guide.points={{0,0,0},{0,0,20},{.01f,0,19.98f},{.01f,0,40}};route.guide.distances={0};
    for(size_t i=1;i<route.guide.points.size();++i)route.guide.distances.push_back(route.guide.distances.back()+(route.guide.points[i]-route.guide.points[i-1]).length());
    route.guide.checkpoints={route.guide.distances.back()};route.checkpoints={{{.01f,0,40},{0,0,1},5.5f,3.5f,{}}};progress={};
    require(!navigateRoute(route,progress,{.005f,.8f,19.99f},0,36).wrongWay,"tiny reverse lane seam is not wrong-way driving");
    require(navigateRoute(route,progress,{.005f,.8f,19.99f},Pi,36).wrongWay,"smoothed seam still detects actual wrong-way driving");
}
void recoveryPolicyTests(const std::filesystem::path& root){
    World world(root/"assets/bbox-assets.bin",0,0,true);RaceRoute route;route.id="recovery-policy";route.name=route.id;
    // One participating player; coincident candidate spots let a second body
    // deterministically block every automatic/manual recovery attempt.
    route.grid.assign(World::MaxPlayers,{{4.2f,0,-38},{0,0,1}});
    for(float z:{50.f,100.f,150.f})route.checkpoints.push_back({{4.2f,0,z},{0,0,1},5.5f,3.5f,{{{4.2f,0,z-8},{0,0,1}}}});
    RaceSession race(world,{route});int id=race.command({{"type","join"},{"name","Policy"}}).at("player_id");
    const auto& player=*race.players()[0];constexpr double dt=.002;
    auto cmd=[&](std::string type,nlohmann::json extra=nlohmann::json::object()){
        extra["type"]=type;extra["player_id"]=id;extra["race_id"]=race.raceId();extra["epoch"]=race.epoch();return race.command(extra);
    };
    auto fall=[&]{world.placePlayer(0,{4.2f,-30,0},0);};
    auto below=[&]{return position(world.playerPose(0).body).y< -20;};
    cmd("ready");race.tick(3);require(race.phase()=="racing","policy fixture starts racing");
    fall();race.tick(dt);
    require(!below()&&player.progress.penalty==5&&player.progress.next==0&&player.recovery.empty(),"automatic recovery remains enabled by default");
    require(!player.recoveryAutomatic,"successful recovery clears its automatic origin");

    race.setAutomaticRecovery(false);fall();
    const double elapsed=player.progress.elapsed,simulation=world.simulationTime;
    const uint64_t tick=race.simulationTick();uint64_t seq=0;
    for(int i=0;i<120;++i){
        cmd("input",{{"seq",++seq},{"throttle",.5}});race.tick(dt);
        require(below()&&player.recovery.empty()&&!player.recoveryAutomatic,"disabled policy must never queue or teleport a fallen car");
        require(player.progress.penalty==5&&player.progress.next==0&&!player.progress.finished,"falling without recovery must not add penalties or pass gates");
        require(player.appliedInputSequence==seq&&world.playerControls().throttle==.5f,"disabled recovery must still apply driving inputs");
    }
    require(race.simulationTick()==tick+120&&std::abs(player.progress.elapsed-elapsed-120*dt)<1e-8&&
        std::abs(world.simulationTime-simulation-120*dt)<1e-6,"disabled recovery must still advance physics and race clocks");
    race.setAutomaticRecovery(true);race.tick(dt);
    require(!below()&&player.progress.penalty==10,"reenabling policy restores fall recovery");

    fall();world.setPlayerActive(1,true);world.placePlayer(1,route.grid.front().ground,0);race.tick(dt);
    require(below()&&!player.recovery.empty()&&player.recoveryAutomatic,"blocked automatic recovery stays pending");
    const auto collisions=race.collisions();const auto pendingTick=race.simulationTick();
    race.setAutomaticRecovery(false);
    require(player.recovery.empty()&&!player.recoveryAutomatic&&race.simulationTick()==pendingTick&&race.collisions().size()==collisions.size(),"disabling policy cancels only pending automatic recovery");
    for(size_t i=0;i<collisions.size();++i){const auto& before=collisions[i];const auto& after=race.collisions()[i];
        require(before.id==after.id&&before.entityA==after.entityA&&before.entityB==after.entityB&&before.speed==after.speed,"policy setter must not alter audio collision events");}
    world.setPlayerActive(1,false);race.tick(dt);
    require(below()&&player.progress.penalty==10,"cancelled automatic recovery must not teleport when space opens");

    // The manual request ID may literally be "automatic"; origin is not inferred from text.
    world.setPlayerActive(1,true);world.placePlayer(1,route.grid.front().ground,0);
    cmd("recover",{{"request_id","automatic"}});race.tick(dt);
    require(!player.recovery.empty()&&!player.recoveryAutomatic,"manual recovery remains available with policy disabled");
    race.setAutomaticRecovery(true);race.setAutomaticRecovery(false);
    require(player.recovery=="automatic","disabling policy must preserve queued manual recovery");
    world.setPlayerActive(1,false);race.tick(dt);
    require(!below()&&player.progress.penalty==15&&player.recovery.empty(),"manual recovery completes while automatic recovery is disabled");
    cmd("recover",{{"request_id","automatic"}});race.tick(dt);
    require(player.progress.penalty==15,"manual recovery remains idempotent");

    for(size_t gate=0;gate<route.checkpoints.size();++gate){
        world.placePlayer(0,route.checkpoints[gate].center-V3{0,0,3},0);
        for(int i=0;i<2400&&player.progress.next==gate;++i){cmd("input",{{"seq",++seq},{"throttle",1}});race.tick(dt);}
        require(player.progress.next==gate+1,"disabled recovery must preserve checkpoint progression for cumulative audio");
    }
    require(player.progress.finished&&race.phase()=="results","disabled recovery must preserve finish progression");
    cmd("next_race");cmd("ready");race.tick(3);fall();race.tick(dt);
    require(below()&&player.recovery.empty()&&player.progress.penalty==0,"automatic recovery policy persists across race restart");
}
void sessionTests(const std::filesystem::path& root,bool disableAutomaticRecovery=false){
    World world(root/"assets/bbox-assets.bin",0,0,true);RaceRoute route;route.id="test";route.name="test";
    for(int i=0;i<5;++i)route.grid.push_back({{4.2f,0,-38.f-i*9},{0,0,1}});
    for(float z:{50.f,100.f,150.f})route.checkpoints.push_back({{4.2f,0,z},{0,0,1},5.5f,3.5f,{{{4.2f,0,z-8},{0,0,1}}}});
    RaceSession race(world,{route});
    if(disableAutomaticRecovery)race.setAutomaticRecovery(false);
    int a=race.command({{"type","join"},{"name","A"}}).at("player_id");int b=race.command({{"type","join"},{"name","B"}}).at("player_id");
    auto cmd=[&](int id,std::string type,nlohmann::json extra=nlohmann::json::object()){
        extra["type"]=type;extra["player_id"]=id;extra["race_id"]=race.raceId();extra["epoch"]=race.epoch();return race.command(extra);
    };
    cmd(a,"ready");require(race.phase()=="lobby","one ready is insufficient");cmd(b,"ready");
    for(int i=0;i<1501;++i)race.tick(.002);require(race.phase()=="racing","countdown starts one world");
    uint64_t seq=0;auto initialA=position(world.playerPose(0).body),initialB=position(world.playerPose(1).body);
    bool collisionObserved=false;
    for(int i=0;i<3500;++i){if(i%25==0)cmd(b,"input",{{"seq",++seq},{"throttle",1}});race.tick(.002);
        for(const auto& event:race.collisions())if(event.entityA==world.playerCar(0)+1||event.entityB==world.playerCar(0)+1)collisionObserved=true;}
    require(collisionObserved,"scored contacts remain available to local collision audio");
    auto state=race.state();auto pa=state["players"][0],pb=state["players"][1];
    require((position(world.playerPose(1).body)-initialB).length()>3,"second player moves independently");
    require((position(world.playerPose(0).body)-initialA).length()>.1f,"shared collision moves other player");
    require(pa["penalty"].get<double>()>=3&&pa["penalty"]==pb["penalty"],"mutual contact gives equal penalties");
    auto cameraA=world.playerCamera(0),cameraB=world.playerCamera(1);require((cameraA.eye-cameraB.eye).length()>1,"independent cameras");
    auto oldEpoch=race.epoch();cmd(b,"focus",{{"value",false}});require(race.phase()=="paused","blur pauses room");
    auto frozen=race.state();auto drawBefore=world.drawObjects();for(int i=0;i<1000;++i)race.tick(.002);
    require(race.state()==frozen,"paused simulation and clocks freeze");require(race.collisions().empty(),"pause does not replay collision audio");auto drawAfter=world.drawObjects();
    require(drawBefore.size()==drawAfter.size(),"paused population remains");
    for(size_t i=0;i<drawBefore.size();++i)require((position(drawBefore[i].instance.world)-position(drawAfter[i].instance.world)).length()<1e-6f,"paused world transforms freeze");
    bool rejected=false;try{race.command({{"type","ready"},{"player_id",a},{"race_id",race.raceId()},{"epoch",oldEpoch}});}catch(...){rejected=true;}require(rejected,"stale ready rejected");
    cmd(a,"ready");require(race.phase()=="paused","hidden participant blocks resume");cmd(b,"focus",{{"value",true}});cmd(b,"ready");require(race.phase()=="racing","everyone ready resumes");
    cmd(b,"connected",{{"value",false}});auto reconnectPosition=position(world.playerPose(1).body);require(race.phase()=="paused","disconnect pauses all");
    cmd(b,"connected",{{"value",true}});cmd(b,"focus",{{"value",true}});cmd(a,"ready");cmd(b,"ready");require((position(world.playerPose(1).body)-reconnectPosition).length()<1e-6f,"reconnect preserves car");
    double penalty=race.state()["players"][1]["penalty"];cmd(b,"recover",{{"request_id","one"}});race.tick(.002);
    require(race.state()["players"][1]["penalty"].get<double>()==penalty+5,"one successful recovery penalty");
    cmd(b,"recover",{{"request_id","one"}});race.tick(.002);require(race.state()["players"][1]["penalty"].get<double>()==penalty+5,"recovery retries are idempotent");
    require(race.state()["players"][1]["next"]==0,"teleport cannot pass checkpoints");
}
}
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Pass repository path");crossingTests();contactsTests();navigationTests();recoveryPolicyTests(argv[1]);sessionTests(argv[1]);sessionTests(argv[1],true);
    {World world(std::filesystem::path(argv[1])/"assets/bbox-assets.bin",0,0);auto routes=loadRaceRoutes(std::filesystem::path(argv[1])/"assets/race-routes.json",world);require(routes.size()==3,"three supported map routes");for(const auto& r:routes){
        require((r.guide.points.back()-r.checkpoints.back().center).length()<.01f,"navigation ends exactly at finish gate");
        require(std::is_sorted(r.guide.checkpoints.begin(),r.guide.checkpoints.end()),"navigation gate distances ordered");
        for(size_t i=0;i<r.grid.size();++i){RaceProgress p;auto spawn=r.grid[i];auto nav=navigateRoute(r,p,spawn.ground+V3{0,.8f,0},std::atan2(spawn.forward.x,spawn.forward.z),0);
            require(nav.valid&&!nav.offRoute&&!nav.wrongWay&&!nav.missedCheckpoint&&nav.distanceToGate>0,"all five grid slots have forward guidance");}
        std::cout<<r.id<<": "<<r.length<<" m, "<<r.checkpoints.size()<<" checkpoints\n";
    }}
    std::cout<<"Race rules, navigation, shared physics, pause, reconnect, automatic recovery policy, manual recovery, and map validation passed\n";return 0;
}catch(const std::exception& e){std::cerr<<"RaceTests: "<<e.what()<<'\n';return 1;}}
