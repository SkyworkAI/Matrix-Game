#include "../src/ReferencePresentation.h"
#include <iostream>
#include <limits>
#include <random>

namespace {
using Json=nlohmann::json;
using reference::PresentationFrameInfo;
using reference::makeGameFrame;
using reference::presentation_detail::GateProjector;
using reference::presentation_detail::makeMap;

void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void near(double actual,double expected){require(std::abs(actual-expected)<2e-5,"Unexpected projected coordinate");}
void bounded(const Json& lines,float low,float high,size_t columns=4) {
    require(lines.is_array(),"Segments must be JSON arrays");
    for(const auto& line:lines){
        require(line.is_array()&&line.size()==columns,"Wrong segment width");
        for(size_t i=0;i<4;++i){double value=line.at(i);
            require(std::isfinite(value)&&value>=low&&value<=high,"Non-finite or unclipped segment");}
    }
}
void projectionTests() {
    Camera camera{{0,0,0},{0,0,1},90};GateProjector project(camera,800,400);
    auto line=project.segment({-5,0,10},{5,0,10});require(bool(line),"Front line rejected");
    near((*line)[0],.25);near((*line)[1],.5);near((*line)[2],.75);near((*line)[3],.5);
    line=project.segment({0,-2.5f,10},{0,2.5f,10});require(bool(line),"Vertical line rejected");
    near((*line)[1],.75);near((*line)[3],.25);
    line=GateProjector(camera,400,800).segment({0,0,10},{0,10,10});
    require(bool(line),"Portrait aspect rejected");near((*line)[3],.25);
    camera.horizontalFov=60;
    line=GateProjector(camera,800,400).segment({0,0,10},{float(10*std::tan(Pi/6)),0,10});
    require(bool(line),"Custom FOV rejected");near((*line)[2],1);

    require(!project.segment({-1,0,-10},{1,0,-10}),"Behind-camera geometry mirrored");
    require(!project.segment({0,0,.01f},{.01f,0,.07f}),"Geometry inside near plane emitted");
    require(!project.segment({0,0,2001},{1,0,2100}),"Geometry beyond far plane emitted");
    require(!project.segment({2,-1,1},{2,1,1}),"Offscreen line clamped to border");
    line=project.segment({-1000,0,1},{1000,0,1});require(bool(line),"Frustum-crossing line lost");
    near((*line)[0],0);near((*line)[2],1);
    line=project.segment({-.04f,0,-.08f},{.08f,0,.24f});require(bool(line),"Near-plane crossing lost");
    near((*line)[0],.625);near((*line)[2],2./3);
    const float nan=std::numeric_limits<float>::quiet_NaN();
    require(!project.segment({nan,0,1},{0,1,1}),"NaN input projected");
    require(!GateProjector(camera,0,480).segment({0,0,1},{1,0,1}),"Invalid dimensions accepted");
    require(!GateProjector({{0,0,0},{0,0,0},90},800,400).segment({0,0,1},{1,0,1}),"Degenerate view accepted");
    require(!GateProjector({{0,0,0},{0,1,0},90},800,400).segment({0,0,1},{1,0,1}),"Degenerate up vector accepted");

    camera={{13,8,-45},{19,10,-30},103};
    V3 forward=(camera.target-camera.eye).normalized(),right=cross({0,1,0},forward).normalized(),up=cross(forward,right);
    const float tx=std::tan(camera.horizontalFov*Pi/360),ty=tx*480/864;
    line=GateProjector(camera,864,480).segment(camera.eye+forward*20+right*(10*tx),camera.eye+forward*20+up*(10*ty));
    require(bool(line),"Translated yaw/pitch camera rejected");
    near((*line)[0],.75);near((*line)[1],.5);near((*line)[2],.5);near((*line)[3],.25);
    line=GateProjector({{0,0,0},{0,0,-1},90},800,400).segment({-5,0,-10},{5,0,-10});
    require(bool(line),"Look-back camera rejected");near((*line)[0],.75);near((*line)[2],.25);

    std::mt19937 random(73);std::uniform_real_distribution<float> coordinate(-30,30);
    for(int i=0;i<4000;++i){
        V3 a{coordinate(random),coordinate(random),coordinate(random)},b{coordinate(random),coordinate(random),coordinate(random)};
        auto ab=project.segment(a,b),ba=project.segment(b,a);
        require(bool(ab)==bool(ba),"Projection depends on segment direction");
        if(ab){bounded(Json::array({*ab}),0,1);near((*ab)[0],(*ba)[2]);near((*ab)[1],(*ba)[3]);}
    }
    Checkpoint gate{{0,0,20},{0,0,1},5.5f,3.5f,{}};
    camera={{0,1,0},{0,1,1},90};
    auto segments=reference::presentation_detail::gateSegments(gate,camera,864,480);
    require(segments.size()==22,"Arch must have twenty curved segments and two posts");bounded(segments,0,1);
    gate.center.z=-20;require(reference::presentation_detail::gateSegments(gate,camera,864,480).empty(),"Rear arch mirrored");
    gate.center.z=.04f;gate.forward={1,0,1};
    bounded(reference::presentation_detail::gateSegments(gate,camera,864,480),0,1);
}

void minimapTests() {
    const V3 center{140,10,-200};
    auto map=makeMap({{center+V3{-400,0,0},center+V3{400,0,0},true},
        {center+V3{0,0,0},center+V3{0,0,200},false},
        {center+V3{500,0,-500},center+V3{500,0,500},false}},center);
    require(map["roads"].size()==2,"Map failed to discard offscreen road");
    require(map["roads"][0]==Json::array({-1,0,1,0,1}),"400 m range or freeway flag changed");
    require(map["roads"][1]==Json::array({0,0,0,-1,0}),"North-up map orientation changed");
    require(map["gate"].is_null()&&map["active"].empty()&&map["preview"].empty(),"Empty map has race geometry");
    bounded(map["roads"],-1,1,5);
    std::vector<MapLine> roads(600,{{-1,0,0},{1,0,0},false});
    roads.insert(roads.begin(),100,{{400,0,0},{500,0,0},false});
    require(makeMap(roads,{})["roads"].size()==512,"Road cap counts invisible segments or exceeds 512");

    RouteGuide overlap;
    overlap.points={{0,0,0},{100,0,0},{0,0,0},{0,0,100},{0,20,0}};
    overlap.distances={0,100,200,300,402};overlap.checkpoints={100,200,300,402};
    auto outbound=makeMap({}, {50,0,0},&overlap,0,50),inbound=makeMap({}, {50,0,0},&overlap,1,150);
    require(outbound["active"].size()==1&&inbound["active"].size()==1,"Overlapping legs leaked into active route");
    near(outbound["active"][0][0],-.15);near(outbound["active"][0][2],.25);
    near(inbound["active"][0][0],.15);near(inbound["active"][0][2],-.25);
    near(outbound["preview"][0][0],.25);near(outbound["preview"][0][2],.20);
    near(outbound["preview"][1][0],.15);near(outbound["preview"][1][2],.10);
    require(inbound["gate"]==Json::array({-.25,0}),"Active gate marker follows wrong traversal");

    RouteGuide straight;straight.points={{0,0,0},{250,0,0}};straight.distances={0,250};straight.checkpoints={50,100,150,200,250};
    map=makeMap({}, {},&straight,0,0);
    require(map["preview"].size()==8,"Preview must end after three additional gates");
    near(map["preview"][0][0],.25);near(map["preview"][0][2],.30);
    near(map["preview"].back()[0],.95);near(map["preview"].back()[2],1);
    map=makeMap({}, {},&straight,4,200);
    require(map["preview"].empty()&&map["gate"].is_null(),"Final/off-map gate handling changed");
    require(makeMap({}, {},&straight,5,250)["active"].empty(),"Invalid checkpoint produced a route");

    RouteGuide dense;
    for(int i=0;i<=1000;++i){dense.points.push_back({60*std::sin(i*.02f),0,60*std::cos(i*.02f)});dense.distances.push_back(float(i));}
    dense.checkpoints={200,400,600,800,1000};map=makeMap({}, {},&dense);
    require(map["active"].size()==128&&map["preview"].size()==128,"Route geometry caps not enforced");
    bounded(map["active"],-1,1);bounded(map["preview"],-1,1);
}

RaceRoute testRoute() {
    RaceRoute route;route.id="presentation-test";route.name="Presentation Test";
    for(int i=0;i<5;++i)route.grid.push_back({{4.2f,0,-38.f-i*9},{0,0,1}});
    route.guide.points.push_back({4.2f,0,-38});route.guide.distances.push_back(0);
    for(float z:{50.f,100.f,150.f}){
        route.checkpoints.push_back({{4.2f,0,z},{0,0,1},5.5f,3.5f,{{{4.2f,0,z-8},{0,0,1}}}});
        route.guide.points.push_back({4.2f,0,z});route.guide.distances.push_back(z+38);route.guide.checkpoints.push_back(z+38);
    }
    return route;
}

void metadataTests(const std::filesystem::path& root) {
    // Existing CPU test track and race rules only: no Renderer, SDL, audio device, or GPU.
    World world(root/"assets/bbox-assets.bin",0,0,true);
    world.simulationTime=12.25;
    CarPose pose{};pose.body=transform({4.2f,0,0});pose.speed=-36.5f;pose.heading=.25f;
    pose.engineRPM=4500;pose.engineThrottle=.75f;pose.frontLateralSlipSpeed=2;pose.rearLateralSlipSpeed=4;
    pose.groundedFrontWheels=1;pose.groundedRearWheels=2;
    Camera camera{{4.2f,1,0},{4.2f,1,1},90};
    PresentationFrameInfo info;info.inputId=77;info.inputReceivedMonotonic=100;info.captureMonotonic=100.25;
    info.checkpoint=8;
    std::vector<CollisionEvent> events{{10,1,0,2.5f},{11,2,3,8},{12,1000,1,3.5f}};
    auto frame=makeGameFrame(world,camera,pose,info,nullptr,events);
    require(frame.size()==12&&frame["frame"]==0&&frame["simulation_s"]==12.25,"GameFrame keys, numbering, or simulation time changed");
    require(frame["speed_kmh"]==-36.5f&&frame["heading"]==.25f,"Pose units were converted twice");
    require(frame["position"]==Json::array({pose.body.m[3][0],0,0}),"Position was not taken from captured pose");
    require(frame["input_id"]==77&&frame["input_received_monotonic"]==100&&frame["capture_monotonic"]==100.25,"Caller timing changed");
    auto& audio=frame["audio"];
    require(audio.size()==9&&audio["rpm"]==4500&&audio["throttle"]==.75,"Audio engine snapshot changed");
    require(audio["front_slip"]==2&&audio["rear_slip"]==4&&audio["front_contacts"]==1&&audio["rear_contacts"]==2,"Audio axle state changed");
    require(!audio["muted"].get<bool>()&&audio["checkpoint"]==8,"Caller audio state changed");
    require(audio["impacts"]==Json::array({{{"id",10},{"speed",2.5}},{{"id",12},{"speed",3.5}}}),"Substep impacts filtered incorrectly");
    require(frame["race"].is_null()&&frame["gate"].is_null(),"Free drive has race metadata");
    const Json supplied=Json::array({{{"id",uint64_t(1)<<40},{"speed",5}}});
    require(makeGameFrame(world,camera,pose,info,nullptr,supplied)["audio"]["impacts"]==supplied,"JSON impact IDs altered");
    info.muted=true;require(makeGameFrame(world,camera,pose,info)["audio"]["muted"]==true,"Caller mute ignored");info.muted=false;
    bool rejected=false;try{makeGameFrame(world,camera,pose,info,nullptr,Json::object());}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Non-array impact input accepted");

    RaceSession race(world,{testRoute()});
    require(makeGameFrame(world,camera,pose,info,&race)["race"].is_null(),"Empty player slot dereferenced");
    int id=race.command({{"type","join"},{"name","Presentation"}}).at("player_id");
    const auto& player=*race.players()[0];
    auto command=[&](Json value){value["player_id"]=id;value["race_id"]=race.raceId();value["epoch"]=race.epoch();return race.command(value);};
    frame=makeGameFrame(world,camera,pose,info,&race,events);
    require(frame["race"].size()==10&&frame["race"]["phase"]=="lobby"&&frame["race"]["total"]==3&&frame["race"]["next"]==0,"Race schema changed");
    require(frame["race"]["route_id"]=="presentation-test"&&frame["race"]["route_name"]=="Presentation Test","Route labels lost");
    near(frame["gate"]["distance_m"],50);bounded(frame["gate"]["segments"],0,1);
    require(frame["map"]["gate"]==Json::array({0,-.25}),"Race map is not pose-centred");
    require(frame["audio"]["muted"]==true,"Lobby audio not muted");
    command({{"type","ready"}});race.tick(.5);
    frame=makeGameFrame(world,camera,pose,info,&race);
    require(frame["race"]["phase"]=="countdown"&&frame["race"]["countdown"]==2.5,"Countdown changed");
    command({{"type","pause"}});require(makeGameFrame(world,camera,pose,info,&race)["race"]["phase"]=="paused","Pause lost");
    command({{"type","ready"}});race.tick(3);
    require(makeGameFrame(world,camera,pose,info,&race)["audio"]["muted"]==false,"Racing audio stayed muted");
    command({{"type","recover"},{"request_id","presentation-recover"}});race.tick(1./480);
    frame=makeGameFrame(world,camera,pose,info,&race);
    require(frame["race"]["penalty"]==player.progress.penalty&&frame["race"]["notice"]==player.notice&&!player.notice.empty(),"Race notice or penalty lost");
    const auto state=race.state();const auto collisionCount=race.collisions().size();const double time=world.simulationTime;
    require(makeGameFrame(world,camera,pose,info,&race)==frame&&race.state()==state&&world.simulationTime==time&&race.collisions().size()==collisionCount,"Metadata helper mutated simulation state");

    uint64_t sequence=0;info.checkpoint=0;
    for(size_t gate=0;gate<race.route().checkpoints.size();++gate){
        world.placePlayer(0,race.route().checkpoints[gate].center-V3{0,0,3},0);
        for(int step=0;step<2400&&player.progress.next==gate;++step){
            command({{"type","input"},{"seq",++sequence},{"throttle",1}});
            size_t before=player.progress.next;race.tick(1./480);info.checkpoint+=player.progress.next-before;
        }
        require(player.progress.next==gate+1,"Fixture did not cross gate");
        if(gate==1){
            info.frame=40;frame=makeGameFrame(world,camera,world.playerPose(0),info,&race);
            require(frame["audio"]["checkpoint"]==2&&frame["race"]["next"]==2,"Skipped display frames lost checkpoint cues");
            ++info.frame;require(makeGameFrame(world,camera,world.playerPose(0),info,&race)["audio"]["checkpoint"]==2,"Checkpoint count became a one-frame pulse");
        }
    }
    frame=makeGameFrame(world,camera,world.playerPose(0),info,&race);
    require(frame["race"]["phase"]=="results"&&frame["race"]["finished"]==true&&frame["audio"]["muted"]==true,"Finish state lost");
    require(frame["audio"]["checkpoint"]==3&&frame["gate"].is_null()&&frame["map"]["gate"].is_null()&&frame["map"]["active"].empty()&&frame["map"]["preview"].empty(),"Finish emits stale gates or loses final chime");
    command({{"type","next_race"}});
    require(makeGameFrame(world,camera,pose,info,&race)["audio"]["checkpoint"]==3,"Retry reset stream-cumulative checkpoint count");
    require(Json::parse(frame.dump())==frame,"GameFrame does not round-trip JSON");
}
}

int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Pass repository path for CPU test-track assets");
    projectionTests();minimapTests();metadataTests(argv[1]);
    std::cout<<"Reference presentation: projection, minimap bounds, metadata, race and cumulative audio passed\n";return 0;
}catch(const std::exception& error){std::cerr<<"ReferencePresentationTests: "<<error.what()<<'\n';return 1;}}
