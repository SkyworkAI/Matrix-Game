#pragma once
#include "Renderer.h"
#include "World.h"
#include <sstream>
#include <iomanip>
#include <numeric>
#include <optional>

struct Options {
    int width=1920,height=1080,traffic=64,pedestrians=160;
    float physicsHz=500,benchmark=0;
    bool playerSteeringLimit=true;
    float playerFrontGrip=1,playerRearGrip=1;
    float playerFrontDrive=.5f;
    bool selfTest=false,visualTest=false,mapTest=false,underpassTest=false,rampGroundTest=false,allRampsTest=false,autodrive=false,noAudio=false,vsync=false,noHud=true,lowLatency=false;
    std::string tag="latest";
};
struct Sample {double frame,cpu,gpu,physics,present,wait,messages,resize,unaccounted;int physicsSteps;};
inline double percentile(std::vector<double> values,double fraction){if(values.empty())return 0;std::sort(values.begin(),values.end());return values[std::min(values.size()-1,size_t(fraction*(values.size()-1)))];}
inline void report(const std::filesystem::path& root,const Options& opt,const Renderer& renderer,const World& world,const std::vector<Sample>& samples){
    auto folder=root/L"benchmarks";std::filesystem::create_directories(folder);
    std::ofstream csv(folder/(opt.tag+".csv"));csv<<"frame_interval_ms,cpu_work_ms,gpu_ms,physics_ms,present_ms,queue_wait_ms,pre_frame_messages_ms,resize_ms,unaccounted_ms,physics_steps,physics_per_step_ms\n"<<std::fixed<<std::setprecision(6);
    uint64_t physicsSteps=0;double physicsTotal=0;int maxSteps=0;
    for(auto s:samples){
        csv<<s.frame<<','<<s.cpu<<','<<s.gpu<<','<<s.physics<<','<<s.present<<','<<s.wait<<','<<s.messages<<','<<s.resize<<','<<s.unaccounted<<','<<s.physicsSteps<<','<<(s.physicsSteps?s.physics/s.physicsSteps:0)<<'\n';
        physicsSteps+=s.physicsSteps;maxSteps=std::max(maxSteps,s.physicsSteps);if(s.physicsSteps)physicsTotal+=s.physics;
    }
    std::ofstream json(folder/(opt.tag+".json"));json<<std::fixed<<std::setprecision(4);
    json<<"{\n  \"adapter\": \""<<renderer.adapterName<<"\",\n  \"width\": "<<renderer.width<<", \"height\": "<<renderer.height<<",\n  \"samples\": "<<samples.size()<<", \"traffic\": "<<opt.traffic<<", \"pedestrians\": "<<opt.pedestrians<<",\n  \"physics_hz\": "<<opt.physicsHz<<", \"vsync\": "<<(opt.vsync?"true":"false")<<", \"tearing_supported\": "<<(renderer.tearing?"true":"false")<<", \"low_latency_wait\": "<<(opt.lowLatency?"true":"false")<<",\n  \"distance_m\": "<<world.distance<<", \"collisions\": "<<world.collisions()<<",\n  \"static_objects\": "<<world.staticObjectCount()<<",\n";
    // Weight by executed steps; slow frames contain more fixed-rate updates.
    json<<"  \"physics_updates\": {\"steps\": "<<physicsSteps<<", \"simulated_seconds\": "<<physicsSteps/opt.physicsHz<<", \"mean_steps_per_frame\": "<<(samples.empty()?0:double(physicsSteps)/samples.size())<<", \"max_steps_per_frame\": "<<maxSteps<<", \"mean_ms_per_step\": "<<(physicsSteps?physicsTotal/physicsSteps:0)<<"},\n";
    auto metric=[&](const char* name,auto member,bool last=false){std::vector<double> data;for(auto s:samples){double v=s.*member;if(v>=0)data.push_back(v);}
        double sum=std::accumulate(data.begin(),data.end(),0.0);size_t good=size_t(std::count_if(data.begin(),data.end(),[](double v){return v<=3;}));
        json<<"  \""<<name<<"\": {\"count\": "<<data.size()<<", \"mean_ms\": "<<(data.empty()?0:sum/data.size())<<", \"p50_ms\": "<<percentile(data,.5)<<", \"p95_ms\": "<<percentile(data,.95)<<", \"p99_ms\": "<<percentile(data,.99)<<", \"max_ms\": "<<percentile(data,1)<<", \"within_3ms_percent\": "<<(data.empty()?0:100.0*good/data.size())<<"}"<<(last?"\n":",\n");};
    metric("frame_interval",&Sample::frame);metric("cpu_work",&Sample::cpu);metric("gpu",&Sample::gpu);metric("physics",&Sample::physics);metric("present",&Sample::present);metric("queue_wait",&Sample::wait);metric("pre_frame_messages",&Sample::messages);metric("resize",&Sample::resize);metric("unaccounted",&Sample::unaccounted,true);json<<"}\n";
}
inline bool selfTest(const std::filesystem::path& root){
    World world(root/L"assets/bbox-assets.bin",0,0,true);float dt=.002f;
    for(int i=0;i<300;++i)world.update(dt,{});
    V3 start=position(world.player().body);float restHeight=start.y;
    for(int i=0;i<1500;++i)world.update(dt,{1,0,0,0});
    float traveled=(position(world.player().body)-start).length(),speed=world.player().speed;
    for(int i=0;i<750;++i)world.update(dt,{0,0,1,1});
    float stopped=std::abs(world.player().speed);
    world.reset();for(int i=0;i<300;++i)world.update(dt,{});
    V3 reverseStart=position(world.player().body);for(int i=0;i<1250;++i)world.update(dt,{-1,0,0,0});float reversed=world.player().speed;
    float reverseDistance=position(world.player().body).z-reverseStart.z;
    world.reset();for(int i=0;i<350;++i)world.update(dt,{});
    for(int i=0;i<2000;++i)world.update(dt,{.7f,.8f,0,0});
    float turn=std::abs(world.player().heading);bool finite=std::isfinite(world.player().speed)&&std::isfinite(position(world.player().body).y);
    bool interactions=world.collisionFixture();
    auto steering=world.steeringFixture();bool steeringCorrect=steering[0]>.2f&&steering[1]>.02f&&steering[2]<-.2f&&steering[3]<-.02f;
    bool pass=restHeight>.3f&&restHeight<1.3f&&traveled>8&&speed>15&&stopped<3&&reversed< -5&&reverseDistance< -3&&turn>.2f&&finite&&interactions&&steeringCorrect;
    std::filesystem::create_directories(root/L"benchmarks");std::ofstream file(root/L"benchmarks/self-test.json");
    file<<"{\"passed\": "<<(pass?"true":"false")<<", \"rest_height_m\": "<<restHeight<<", \"forward_distance_m\": "<<traveled<<", \"forward_speed_kmh\": "<<speed<<", \"braked_speed_kmh\": "<<stopped<<", \"reverse_speed_kmh\": "<<reversed<<", \"reverse_distance_m\": "<<reverseDistance<<", \"turn_radians\": "<<turn<<", \"finite\": "<<(finite?"true":"false")<<", \"wall_pedestrian_vehicle_collision\": "<<(interactions?"true":"false")<<", \"steering_correct\": "<<(steeringCorrect?"true":"false")<<", \"right_turn_x_m\": "<<steering[0]<<", \"right_turn_heading\": "<<steering[1]<<", \"left_turn_x_m\": "<<steering[2]<<", \"left_turn_heading\": "<<steering[3]<<", \"collision_contacts\": "<<world.collisions()<<"}\n";
    return pass;
}
inline std::optional<int> runSceneTests(const std::filesystem::path& root,const Options& opt,Renderer& renderer,World& world){
        if(opt.visualTest){
            Camera camera;camera.eye={0,0,0};camera.target={0,0,1};HudState hud;hud.visible=false;
            auto fixture=world.outlineFixture();renderer.outlineWidth=0;renderer.render(fixture,camera,hud);
            renderer.screenshot(root/L"captures/outline-baseline.png");
            renderer.outlineWidth=.72f;renderer.render(fixture,camera,hud);renderer.screenshot(root/L"captures/outline-fixed.png");
            fixture=world.nearClipFixture();renderer.outlineWidth=0;renderer.render(fixture,camera,hud);renderer.screenshot(root/L"captures/nearclip-baseline.png");
            renderer.outlineWidth=.72f;renderer.render(fixture,camera,hud);renderer.screenshot(root/L"captures/nearclip-fixed.png");
            return 0;
        }
        if(opt.underpassTest){
            HudState hud;hud.visible=false;world.placeAtUnderpass();
            renderer.render(world.drawObjects(),world.camera(1),hud);renderer.screenshot(root/L"captures/small-city-underpass.png");
            renderer.render(world.drawObjects(),{{173.5f,4,335},{172,0,307},90},hud);renderer.screenshot(root/L"captures/small-city-underpass-overview.png");
            bool passed=world.underpassFixture(root);
            return passed?0:2;
        }
        if(opt.rampGroundTest){
            HudState hud;hud.visible=false;world.placeAtRampSide();
            renderer.render(world.drawObjects(),world.camera(1),hud);renderer.screenshot(root/L"captures/small-city-ramp-side.png");
            renderer.render(world.drawObjects(),{{.2f,3.4f,-114},{-.3f,.7f,-97},90},hud);renderer.screenshot(root/L"captures/small-city-ramp-side-overview.png");
            bool passed=world.rampGroundFixture(root);
            return passed?0:2;
        }
        if(opt.allRampsTest){
            HudState hud;hud.visible=false;
            for(const auto& [name,camera]:world.rampSurveyCameras(root)){
                renderer.render(world.drawObjects(),camera,hud);renderer.screenshot(root/L"captures"/(name+".png"));
            }
            bool passed=world.allRampsFixture(root);
            return passed?0:2;
        }
        if(opt.mapTest){
            HudState hud;hud.visible=false;
            for(int i=0;i<500;++i)world.update(.002f,{});
            auto population=world.nearbyPopulation();V3 street=position(world.player().body);
            renderer.render(world.drawObjects(),world.camera(0),hud);renderer.screenshot(root/L"captures/small-city-street.png");
            auto overview=world.drawObjects();for(auto& object:overview)object.range=10000;
            renderer.render(overview,world.overviewCamera(),hud);renderer.screenshot(root/L"captures/small-city-overview.png");
            auto movingPopulation=world.populationMotionFixture();
            renderer.render(world.drawObjects(),world.camera(0),hud);renderer.screenshot(root/L"captures/small-city-traffic.png");
            world.placeAtFreeway();for(int i=0;i<500;++i)world.update(.002f,{});
            V3 bridge=position(world.player().body);renderer.render(world.drawObjects(),world.camera(0),hud);renderer.screenshot(root/L"captures/small-city-bridge.png");
            std::filesystem::create_directories(root/L"benchmarks");std::ofstream trace(root/L"benchmarks/bridge-driving.csv");
            trace<<"time_s,x,y,z,speed_kmh,heading,grounded_wheels,throttle,steer,brake\n";
            int bridgeSupported=0;
            for(int i=0;i<10000;++i){
                world.update(.002f,{},true);auto pose=world.player();if(pose.groundedWheels>=2)++bridgeSupported;
                if(i%125==0){auto p=position(pose.body);auto in=world.playerControls();trace<<i*.002f<<','<<p.x<<','<<p.y<<','<<p.z<<','<<pose.speed<<','<<pose.heading<<','<<pose.groundedWheels<<','<<in.throttle<<','<<in.steer<<','<<in.brake<<'\n';}
                if(i==2999){renderer.render(world.drawObjects(),world.camera(0),hud);renderer.screenshot(root/L"captures/small-city-bridge-driving.png");}
            }
            V3 driven=position(world.player().body);float bridgeDistance=(driven-bridge).length();
            renderer.render(world.drawObjects(),world.camera(0),hud);renderer.screenshot(root/L"captures/small-city-ramp-driving.png");
            auto afterDriving=world.nearbyPopulation();
            std::vector<std::array<float,4>> groundMisses;auto groundCoverage=world.groundCoverageFixture(&groundMisses);
            bool passed=world.buildingCount()>=600&&world.bridgeDeckCount()>=40&&population[0]>=8&&population[1]>=20&&movingPopulation[0]>=8&&movingPopulation[1]>=20&&street.y>-.5f&&bridge.y>5&&driven.y>-.5f&&bridgeDistance>180&&bridgeSupported>9500&&world.player().groundedWheels>=2&&afterDriving[0]>=8&&afterDriving[1]>=20&&groundCoverage[0]>0&&groundCoverage[1]==0;
            std::filesystem::create_directories(root/L"benchmarks");std::ofstream file(root/L"benchmarks/small-city-tests.json");
            std::ofstream groundFile(root/L"benchmarks/ground-collision-tests.json");groundFile<<"{\"passed\": "<<(groundCoverage[0]>0&&groundCoverage[1]==0?"true":"false")<<", \"samples\": "<<groundCoverage[0]<<", \"missing_ground\": "<<groundCoverage[1]<<", \"misses_x_y_z_actual\": [";
            for(size_t i=0;i<groundMisses.size();++i){auto p=groundMisses[i];groundFile<<(i?",":"")<<'['<<p[0]<<','<<p[1]<<','<<p[2]<<','<<p[3]<<']';}groundFile<<"]}\n";
            file<<"{\"passed\": "<<(passed?"true":"false")<<", \"original_buildings\": "<<world.buildingCount()<<", \"bridge_decks\": "<<world.bridgeDeckCount()<<", \"moving_cars_within_180m\": "<<population[0]<<", \"pedestrians_within_180m\": "<<population[1]<<", \"traffic_cars_moved_3m\": "<<movingPopulation[0]<<", \"pedestrians_walked_1m\": "<<movingPopulation[1]<<", \"street_height_m\": "<<street.y<<", \"bridge_height_m\": "<<bridge.y<<", \"bridge_drive_distance_m\": "<<bridgeDistance<<", \"bridge_final_height_m\": "<<driven.y<<", \"bridge_supported_percent\": "<<bridgeSupported/100.f<<", \"cars_nearby_after_driving\": "<<afterDriving[0]<<", \"pedestrians_nearby_after_driving\": "<<afterDriving[1]<<"}\n";
            return passed?0:2;
        }
    return {};
}
