#include "LocalRaceControls.h"
#include "ReferencePresentation.h"
#include <iostream>

namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::array<V3,4> body(V3 p,float yaw=0){const auto m=transform(p,{1,1,1},yaw);
    return {transformPoint({-1,0,-2},m),transformPoint({1,0,-2},m),transformPoint({1,0,2},m),transformPoint({-1,0,2},m)};}
void rules(){
    require(std::abs(pursuitBodyGap(body({}),body({0,0,9}))-5)<1e-5f,"body gap is not bumper distance");
    require(pursuitBodyGap(body({}),body({},Pi/2))==0,"crossed rectangles must overlap");
    require(pursuitBodyGap(body({}),body({0,0,4}))<1e-5f,"touching bodies gap");
    PursuitProgress progress;
    for(int i=0;i<2400;++i)progress.advance(1./480,5,true);
    require(progress.captured,"five seconds at threshold did not capture");
    progress.advance(1,100,false);require(progress.captured,"capture not latched");
    progress={};progress.advance(4,4,true);progress.advance(.01,5.01f,true);
    require(progress.held==0,"leaving radius did not reset");
    progress.advance(4,1,true);progress.advance(.01,1,false);require(progress.held==0,"occlusion did not reset");
    for(int i=0;i<240;++i)progress.advance(1./48,1,true);
    require(progress.captured,"capture changed with timestep");
}
}
int main(int argc,char** argv){try{
    if(argc<2)throw std::runtime_error("Pass root [drive seconds] [report.json] [loop]");rules();
    const std::filesystem::path root=argv[1];const double seconds=argc>2?std::stod(argv[2]):10;
    const int traffic=argc>5?std::stoi(argv[5]):0;
    World world(root/"assets/bbox-assets.bin",traffic,traffic?40:0);auto routes=loadRaceRoutes(root/"assets/race-routes.json",world);
    std::erase_if(routes,[](const auto& r){return r.id!="avenue";});
    RaceSession race(world,std::move(routes),true);
    int id=race.command({{"type","join"},{"name","Driver"}}).at("player_id");
    int suspect=race.command({{"type","join"},{"name","Suspect"}}).at("player_id");
    LocalRaceControls controls(race,{id,suspect});controls.start("avenue");
    const bool loop=argc>4&&std::string(argv[4])=="loop";
    if(loop)world.setPlayerActive(0,false);
    require(race.isPursuit()&&race.phase()=="countdown","pursuit start failed");
    const auto initial=race.pursuitState();
    const float initialDistance=initial.at("initial_distance_m"),routeDistance=initial.at("initial_route_distance_m");
    require(initialDistance>135&&initialDistance<=330,"suspect is not farther away or starts too far");
    require(routeDistance>=270&&routeDistance<=330,"suspect is not ahead in the configured road range");
    require((position(world.playerPose(0).body)-race.route().grid.at(0).ground).length()<3,"player spawn changed");
    require(!initial.at("capture_eligible").get<bool>(),"capture can start at spawn");
    const V3 initialTarget=position(world.playerPose(1).body);
    require(!controls.recover(),"pursuit allowed teleport recovery");
    constexpr double dt=1./480;
    for(int i=0;i<1441;++i)race.tick(dt);
    require(race.phase()=="racing","countdown failed");
    controls.pause();const auto frozen=race.state();for(int i=0;i<480;++i)race.tick(dt);
    require(race.state()==frozen,"pause changed suspect or capture state");controls.resume();
    const auto start=position(world.playerPose(1).body);auto moved=start,previous=start;
    double travel=0,maxJump=0,stationary=0,maxStationary=0,minY=1e9,elapsed=0;bool stalled=false;
    auto samples=nlohmann::json::array();
    for(int i=0;i<int(seconds/dt);++i){
        controls.drive({},1,false);race.tick(dt);auto p=position(world.playerPose(1).body);
        elapsed+=dt;
        const double jump=(p-previous).length();travel+=jump;maxJump=std::max(maxJump,jump);previous=p;minY=std::min(minY,double(p.y));
        if((p-moved).length()>1){moved=p;stationary=0;}else stationary+=dt;
        maxStationary=std::max(maxStationary,stationary);
        if(i%480==0)samples.push_back({{"second",i/480},{"position",{p.x,p.y,p.z}},{"speed",world.playerPose(1).speed}});
        if(race.phase()=="results")break;
        if(stationary>12){stalled=true;break;}
    }
    nlohmann::json report={{"initial",initial},{"final",race.pursuitState()},{"travel_m",travel},{"max_jump_m",maxJump},
        {"max_stationary_s",maxStationary},{"min_y",minY},{"stalled",stalled},{"elapsed",elapsed},{"map_half_size",world.mapHalfSize()},{"samples",samples}};
    if(argc>3){std::ofstream out(argv[3]);out<<report.dump(2);}report.erase("samples");std::cout<<report.dump()<<'\n';
    require(travel>elapsed*2,"suspect did not keep driving");require(maxJump<.3,"suspect teleported");require(!stalled,"suspect stuck");
    if(race.phase()=="results"){
        require(!loop&&race.pursuitState()["captured"]==true&&race.players()[0]->progress.finished,"incorrect capture result");
        const auto result=race.state();for(int i=0;i<1440;++i)race.tick(dt);
        require(result==race.state(),"result tail advanced simulation");
    }
    require(race.pursuitState().at("suspect_id")==initial.at("suspect_id"),"suspect identity changed");
    require(race.players()[0]->progress.next==0&&race.players()[0]->progress.penalty==0,"race rules leaked into pursuit");
    reference::PresentationFrameInfo info;
    auto frame=reference::makeGameFrame(world,world.playerCamera(0,1),world.playerPose(0),info,&race);
    require(frame["gate"].is_null()&&frame["map"]["active"].empty(),"hidden escape route leaked into HUD");
    require(frame["map"]["suspect"].size()==2&&frame["pursuit"]["suspect_id"]==initial["suspect_id"],"recording lost target");
    controls.start("avenue");require(race.pursuitState()["capture_held_s"]==0,"retry kept capture progress");
    require((position(world.playerPose(1).body)-initialTarget).length()<.1f,"retry changed suspect spawn");
    std::cout<<"Pursuit rules, nearby forward spawn, persistent suspect, driving, pause, recording state and retry passed\n";return 0;
}catch(const std::exception& e){std::cerr<<"PursuitTests: "<<e.what()<<'\n';return 1;}}
