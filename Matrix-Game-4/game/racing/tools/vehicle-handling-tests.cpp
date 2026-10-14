#include "../src/Physics.h"
#include "../src/KeyboardSteering.h"
#include <iostream>
#include <iomanip>
#include <limits>

struct Reader {
    std::ifstream file;
    explicit Reader(const std::filesystem::path& path):file(path,std::ios::binary){if(!file)throw std::runtime_error("Missing bbox asset");}
    template<class T>T get(){T v{};file.read(reinterpret_cast<char*>(&v),sizeof(v));if(!file)throw std::runtime_error("Truncated bbox asset");return v;}
    template<class T>void list(std::vector<T>& values,size_t size){values.resize(size);file.read(reinterpret_cast<char*>(values.data()),sizeof(T)*size);if(!file)throw std::runtime_error("Truncated bbox geometry");}
};
struct Fixture {CarModel model;MeshData body;};
Fixture load(const std::filesystem::path& path){
    Reader r(path);if(r.get<uint32_t>()!=0x31474242)throw std::runtime_error("Unsupported asset version");
    auto count=r.get<uint32_t>();auto cars=r.get<uint32_t>();r.get<uint32_t>();
    if(count>512||cars<1)throw std::runtime_error("Invalid asset counts");
    std::vector<MeshData> meshes;
    for(uint32_t i=0;i<count;++i){MeshData mesh;auto v=r.get<uint32_t>(),n=r.get<uint32_t>(),e=r.get<uint32_t>();
        if(v>4000000||n>4000000||e>4000000)throw std::runtime_error("Invalid mesh counts");
        r.list(mesh.vertices,v);r.list(mesh.indices,n);r.list(mesh.edges,e);meshes.push_back(std::move(mesh));}
    Fixture f;auto& m=f.model;m.bodyMesh=r.get<uint32_t>();for(auto& id:m.wheelMeshes)id=r.get<uint32_t>();
    for(auto& p:m.wheelPositions)p=r.get<V3>();for(auto& s:m.wheelSizes)s=r.get<std::array<float,2>>();
    m.halfExtent=r.get<V3>();m.bodyCenter=r.get<V3>();f.body=meshes.at(m.bodyMesh);return f;
}
void expectSteering(const CarPose& pose,float input,float previousSpeed,bool limited){
    const float scale=limited?std::clamp(1.f-std::abs(previousSpeed/3.6f)/55.f,.3f,1.f):1.f;
    const float expected=input*37*Pi/180*scale;
    if(!std::isfinite(pose.steerAngle)||std::abs(pose.steerAngle-expected)>2e-5f)
        throw std::runtime_error("Wheel angle does not match steering-limit setting");
}
void verifySteeringIsolation(const Fixture& f,double hz){
    Physics physics;const float dt=float(1/hz);
    physics.addBox({0,-.2f,0},{3000,.2f,3000},0,true);
    for(float x:{0.f,30.f})physics.addCar(f.model,f.body,{x,.9f,0},0);
    physics.optimize();
    for(int i=0;i<int(hz);++i){physics.drive(0,{});physics.drive(1,{});physics.step(dt);}
    for(int i=0;i<int(60*hz)&&std::min(physics.car(0).speed,physics.car(1).speed)<100;++i){
        physics.drive(0,{1,0,0,0});physics.drive(1,{1,0,0,0});physics.step(dt);
    }
    if(std::min(physics.car(0).speed,physics.car(1).speed)<100)throw std::runtime_error("Isolation test failed to reach speed");
    for(bool limited:{false,true,false})for(float steer:{-1.f,-.4f,0.f,.4f,1.f}){
        const float playerSpeed=physics.car(0).speed,trafficSpeed=physics.car(1).speed;
        physics.drive(0,{1,steer,0,0},limited);
        physics.drive(1,{1,.6f,0,0});
        physics.step(dt);
        expectSteering(physics.car(0),steer,playerSpeed,limited);
        expectSteering(physics.car(1),.6f,trafficSpeed,true);
    }
    std::cout<<"Steering toggle and default-limited neighboring car: passed\n";
}
void verifyKeyboardSteering(const Fixture& f,double hz){
    Physics physics;KeyboardSteering steering;const double dt=1/hz;
    physics.addBox({0,-.2f,0},{100,.2f,100},0,true);
    physics.addCar(f.model,f.body,{0,.9f,0},0);physics.optimize();
    for(auto [target,duration]:{std::pair{1.f,.3},std::pair{-1.f,.5},std::pair{0.f,.2}}){
        float before=steering.current();
        // The requested ramp may end between fixed ticks (e.g. 0.3 s at 504 Hz).
        for(int tick=0;tick<int(std::ceil(duration*hz));++tick){
            Input input=steering.apply({0,target,0,0},dt,true);
            if(std::abs(input.steer-before)>float(dt/.2)+1e-5f)throw std::runtime_error("Keyboard wheel angle jumped");
            physics.drive(0,input,false);physics.step(float(dt));expectSteering(physics.car(0),input.steer,0,false);before=input.steer;
        }
        if(std::abs(steering.current()-target)>1e-5f)throw std::runtime_error("Keyboard did not reach target in expected simulation time");
    }
    std::cout<<"Keyboard ramp reaches actual Jolt wheel angles at "<<hz<<" Hz: passed\n";
}
void verifyForwardSpeedLimit(const Fixture& f,double hz){
    const float dt=float(1/hz);
    std::vector<float> baselineSpeed;std::vector<V3> baselineNeighbor,baselineCoast,baselineReverse;
    auto setup=[&](Physics& physics){
        physics.addBox({0,-.2f,0},{3000,.2f,3000},0,true);
        for(float x:{0.f,100.f}){
            const auto car=physics.addCar(f.model,f.body,{x,.9f,0},0);
            physics.setCarLateralGrip(car,1.15f,.9f);physics.setCarFrontDrive(car,.7f);
        }
        physics.optimize();
        for(int i=0;i<int(hz);++i){physics.drive(0,{});physics.drive(1,{});physics.step(dt);}
    };
    // Fresh fixtures preserve identical contact, engine and gearbox state for each A/B test.
    for(bool limited:{false,true}){
        {
            Physics physics;setup(physics);float maxSpeed=0;bool sawTaper=false;
            for(int i=0;i<int(45*hz);++i){
                const float speed=physics.car(0).speed;
                physics.drive(0,{1,0,0,0},false,limited?80:0);physics.drive(1,{1,0,0,0},false);
                const float throttle=physics.car(0).engineThrottle;
                if(limited&&std::abs(throttle-std::clamp((80-speed)/5,0.f,1.f))>1e-5f)
                    throw std::runtime_error("Incorrect speed-limit throttle taper");
                if(speed>75&&speed<80)sawTaper=true;
                if(!limited)baselineSpeed.push_back(speed);
                else if(!sawTaper&&std::abs(speed-baselineSpeed.at(i))>1e-3f)
                    throw std::runtime_error("Speed limit changed low-speed acceleration");
                physics.step(dt);maxSpeed=std::max(maxSpeed,physics.car(0).speed);
                const auto neighbor=physics.car(1).velocity;
                if(!limited)baselineNeighbor.push_back(neighbor);
                else if((neighbor-baselineNeighbor.at(i)).length()>1e-3f)
                    throw std::runtime_error("Player speed limit changed the neighboring car");
            }
            if(limited&&(!sawTaper||physics.car(0).speed<77||maxSpeed>80.1f||physics.car(1).speed<100))
                throw std::runtime_error("Speed limit failed or affected the unrestricted car");
            if(limited)std::cout<<"Player speed limit: peak "<<maxSpeed<<" km/h, unrestricted car "<<physics.car(1).speed<<" km/h\n";
        }
        {
            Physics physics;setup(physics);
            for(int i=0;i<int(60*hz)&&physics.car(0).speed<100;++i){
                physics.drive(0,{1,0,0,0});physics.drive(1,{1,0,0,0});physics.step(dt);
            }
            if(physics.car(0).speed<100)throw std::runtime_error("Overspeed fixture failed to reach speed");
            for(int i=0;i<int(hz/2);++i){
                const auto before=physics.car(0);
                physics.drive(0,{limited?1.f:0.f,.1f,0,0},false,limited?80:0);physics.drive(1,{0,.1f,0,0},false);
                if(physics.car(0).engineThrottle!=0||(physics.car(0).velocity-before.velocity).length()>1e-6f)
                    throw std::runtime_error("Overspeed must cut throttle without clipping velocity");
                physics.step(dt);expectSteering(physics.car(0),.1f,before.speed,false);
                if(!limited)baselineCoast.push_back(physics.car(0).velocity);
                else if((physics.car(0).velocity-baselineCoast.at(i)).length()>1e-3f)
                    throw std::runtime_error("Speed limiter differs from coasting (brake/physics changed)");
            }
        }
        {
            Physics physics;setup(physics);
            for(int i=0;i<int(5*hz);++i){
                physics.drive(0,{-1,0,0,0},false,limited?80:0);physics.drive(1,{-1,0,0,0},false);physics.step(dt);
                if(!limited)baselineReverse.push_back(physics.car(0).velocity);
                else if((physics.car(0).velocity-baselineReverse.at(i)).length()>1e-3f)
                    throw std::runtime_error("Forward speed limiter changed reverse driving");
            }
            if(physics.car(0).speed> -5)throw std::runtime_error("Reverse fixture did not reverse");
        }
    }
    std::cout<<"Throttle taper, low-speed acceleration, overspeed coasting, steering and reverse: passed\n";
}
void verifyGripIsolation(const Fixture& f,double hz){
    std::vector<CarPose> baselineTraffic;float baselinePlayerYaw=0;
    for(bool tuned:{false,true}){
        Physics physics;const float dt=float(1/hz);
        physics.addBox({0,-.2f,0},{3000,.2f,3000},0,true);
        physics.addCar(f.model,f.body,{0,.9f,0},0);
        for(float bad:{0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
            for(bool front:{false,true}){
                bool rejected=false;
                try{physics.setCarLateralGrip(0,front?bad:1,front?1:bad);}catch(const std::runtime_error&){rejected=true;}
                if(!rejected)throw std::runtime_error("Invalid grip accepted");
            }
        }
        physics.setCarLateralGrip(0,tuned?1.15f:1,tuned?.85f:1);
        physics.resetCar(0,{0,.9f,0},0);
        physics.addCar(f.model,f.body,{100,.9f,0},0);
        physics.optimize();
        for(int i=0;i<int(hz);++i){physics.drive(0,{});physics.drive(1,{});physics.step(dt);}
        for(int i=0;i<int(60*hz)&&physics.car(1).speed<100;++i){
            physics.drive(0,{1,0,0,0});physics.drive(1,{1,0,0,0});physics.step(dt);
        }
        if(physics.car(1).speed<100)throw std::runtime_error("Grip isolation test failed to reach speed");
        for(int i=0;i<int(hz);++i){
            physics.drive(0,{1,1,0,0},false);physics.drive(1,{1,1,0,0},false);physics.step(dt);
            auto traffic=physics.car(1);
            if(!tuned)baselineTraffic.push_back(traffic);
            else {
                const auto& before=baselineTraffic.at(size_t(i));
                if((position(traffic.body)-position(before.body)).length()>1e-3f||std::abs(traffic.yawRate-before.yawRate)>1e-4f)
                    throw std::runtime_error("Player grip changed an unmodified neighboring car");
            }
        }
        const float yaw=physics.car(0).yawRate;
        if(!tuned)baselinePlayerYaw=yaw;
        else if(!std::isfinite(yaw)||std::abs(yaw-baselinePlayerYaw)<1e-3f)
            throw std::runtime_error("Player grip setting had no measurable effect");
    }
    std::cout<<"Player-only grip, reset persistence and invalid values: passed\n";
}
void verifyDriveSplit(const Fixture& f,double hz){
    std::array<float,4> baselineTraffic{};
    for(float share:{.5f,0.f,.7f,1.f,.3f}){
        Physics physics;const float dt=float(1/hz);
        physics.addCar(f.model,f.body,{0,100,0},0);
        physics.addCar(f.model,f.body,{100,100,0},0);
        for(float bad:{-.1f,1.1f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
            bool rejected=false;
            try{physics.setCarFrontDrive(0,bad);}catch(const std::runtime_error&){rejected=true;}
            if(!rejected)throw std::runtime_error("Invalid drive split accepted");
        }
        physics.setCarFrontDrive(0,0);physics.setCarFrontDrive(0,1);physics.setCarFrontDrive(0,share);
        physics.resetCar(0,{0,100,0},0);
        bool checkedInitialSplit=false;
        // In free fall an unpowered axle cannot spin up from road contact.
        for(int i=0;i<int(hz);++i){
            physics.drive(0,{1,0,0,0});physics.drive(1,{1,0,0,0});physics.step(dt);
            const auto pose=physics.car(0);const auto& wheels=pose.wheelAngularSpeed;
            const float front=std::abs(wheels[0])+std::abs(wheels[1]),rear=std::abs(wheels[2])+std::abs(wheels[3]);
            if(pose.groundedWheels||!std::isfinite(front+rear))throw std::runtime_error("Invalid drivetrain fixture state");
            if(pose.groundedFrontWheels||pose.groundedRearWheels||pose.frontLateralSlipSpeed!=0||pose.rearLateralSlipSpeed!=0)
                throw std::runtime_error("Airborne wheel spin triggered axle slip audio data");
            if((share==0&&front>1e-6f)||(share==1&&rear>1e-6f))throw std::runtime_error("Unpowered axle received engine torque");
            if(!checkedInitialSplit&&front+rear>1e-3f){
                if(std::abs(front/(front+rear)-share)>1e-3f)throw std::runtime_error("Initial axle split does not match requested share");
                checkedInitialSplit=true;
            }
        }
        if(!checkedInitialSplit)throw std::runtime_error("Powered wheels failed to spin");
        const auto traffic=physics.car(1).wheelAngularSpeed;
        if(share==.5f)baselineTraffic=traffic;
        for(size_t wheel=0;wheel<4;++wheel)if(!std::isfinite(traffic[wheel])||std::abs(traffic[wheel])<1e-3f||std::abs(traffic[wheel]-baselineTraffic[wheel])>1e-4f)
            throw std::runtime_error("Player drive split changed the default AWD neighboring car");
    }
    std::cout<<"Drive split: RWD/FWD isolation, nominal ratios, reconnect/reset and invalid values passed\n";
}
void verifyAxleAudioSlip(const Fixture& f,double hz){
    for(bool drift:{false,true}){
        Physics physics;const float dt=float(1/hz),target=drift?80:40;
        physics.addBox({0,-.2f,0},{3000,.2f,3000},0,true);physics.addCar(f.model,f.body,{0,.9f,0},0);
        physics.setCarLateralGrip(0,1.15f,.9f);physics.setCarFrontDrive(0,.7f);physics.optimize();
        for(int i=0;i<int(hz);++i){physics.drive(0,{});physics.step(dt);}
        for(int i=0;i<int(60*hz)&&physics.car(0).speed<target;++i){physics.drive(0,{1,0,0,0});physics.step(dt);}
        if(physics.car(0).speed<target)throw std::runtime_error("Audio slip fixture failed to reach speed");
        float maxFront=0,maxRear=0,maxFrontAngle=0;
        for(int i=0;i<int((drift?1:.1)*hz);++i){
            physics.drive(0,{0,drift?.1f:1.f,0,0},false);physics.step(dt);const auto pose=physics.car(0);
            if(pose.groundedFrontWheels!=2||pose.groundedRearWheels!=2||!std::isfinite(pose.frontLateralSlipSpeed)||pose.frontLateralSlipSpeed<0||!std::isfinite(pose.rearLateralSlipSpeed)||pose.rearLateralSlipSpeed<0)
                throw std::runtime_error("Invalid axle audio telemetry");
            maxFront=std::max(maxFront,pose.frontLateralSlipSpeed);
            maxRear=std::max(maxRear,pose.rearLateralSlipSpeed);maxFrontAngle=std::max(maxFrontAngle,pose.frontSlipAngle);
        }
        if(!drift&&(maxFrontAngle<20*Pi/180||maxFront<=3||maxRear>=1))throw std::runtime_error("Front/rear audio slip not separated correctly");
        if(drift&&maxRear<=3)throw std::runtime_error("Rear lateral slide missing from audio data");
        std::cout<<"Axle audio slip: "<<(drift?"slide":"initial full steer")<<" max front "<<maxFront<<", rear "<<maxRear<<" m/s, front angle "<<maxFrontAngle*180/Pi<<" deg\n";
    }
    Physics physics;const float dt=float(1/hz);
    physics.addBox({0,-.2f,0},{3000,.2f,3000},0,true);
    // Deliberately reduce longitudinal grip in this fixture to isolate wheel spin.
    physics.addCar(f.model,f.body,{0,.9f,0},0,.01f);physics.optimize();
    for(int i=0;i<int(hz);++i){physics.drive(0,{});physics.step(dt);}
    float maxRollingError=0,maxFrontLateral=0,maxRearLateral=0;
    for(int i=0;i<int(2*hz);++i){
        physics.drive(0,{1,0,0,0});physics.step(dt);const auto pose=physics.car(0);
        if(pose.groundedFrontWheels!=2||pose.groundedRearWheels!=2)throw std::runtime_error("Wheel-spin fixture lost axle contact");
        for(int wheel=0;wheel<4;++wheel)maxRollingError=std::max(maxRollingError,std::abs(pose.wheelAngularSpeed[wheel]*f.model.wheelSizes[wheel][0]-pose.speed/3.6f));
        maxFrontLateral=std::max(maxFrontLateral,pose.frontLateralSlipSpeed);
        maxRearLateral=std::max(maxRearLateral,pose.rearLateralSlipSpeed);
    }
    if(maxRollingError<=3||maxFrontLateral>=1||maxRearLateral>=1)throw std::runtime_error("Longitudinal spin was not isolated from lateral audio slip");
    std::cout<<"Axle audio slip: straight wheel spin "<<maxRollingError<<" m/s, lateral front "<<maxFrontLateral<<", rear "<<maxRearLateral<<" m/s\n";
}
int main(int argc,char** argv){try{
    if(argc<2||argc>9||argc==7)throw std::runtime_error("Usage: VehicleHandlingTests assets/bbox-assets.bin [trace.csv] [physics_hz] [longitudinal_impulse_scale] [on|off] [front_grip rear_grip] [front_drive_share]");
    auto f=load(argv[1]);double hz=argc>3?std::stod(argv[3]):480,dt=1/hz;
    if(!std::isfinite(hz)||hz<100||hz>1000)throw std::runtime_error("Physics Hz outside test range");
    float impulseScale=argc>4?std::stof(argv[4]):10;
    const std::string setting=argc>5?argv[5]:"on";
    if(setting!="on"&&setting!="off")throw std::runtime_error("Steering limit must be on or off");
    const bool limited=setting=="on";
    const float frontGrip=argc>6?std::stof(argv[6]):1,rearGrip=argc>7?std::stof(argv[7]):1;
    const float frontDrive=argc>8?std::stof(argv[8]):.5f;
    verifySteeringIsolation(f,hz);
    verifyKeyboardSteering(f,hz);
    verifyForwardSpeedLimit(f,hz);
    verifyGripIsolation(f,hz);
    verifyDriveSplit(f,hz);
    verifyAxleAudioSlip(f,hz);
    std::cout<<"Steering speed limit: "<<setting<<"; lateral grip front "<<frontGrip<<", rear "<<rearGrip<<"; nominal front drive share "<<frontDrive<<'\n';
    std::ofstream trace;if(argc>2){trace.open(argv[2]);if(!trace)throw std::runtime_error("Cannot create trace");
        trace<<"target_kmh,time_s,phase,input_steer,speed_kmh,wheel_deg,yaw_deg_s,lateral_m_s,up_y,grounded,x,z,front_slip_deg,rear_slip_deg\n";}
    bool stable=true;
    for(float target:{20.f,60.f,100.f,140.f}){
        Physics physics;physics.addBox({0,-.2f,0},{3000,.2f,3000},0,true);physics.addCar(f.model,f.body,{0,.9f,0},0,impulseScale);physics.optimize();
        physics.setCarLateralGrip(0,frontGrip,rearGrip);
        physics.setCarFrontDrive(0,frontDrive);
        for(int i=0;i<int(hz);++i){physics.drive(0,{});physics.step(float(dt));}
        double acceleration=0;
        while(physics.car(0).speed<target&&acceleration<60){physics.drive(0,{1,0,0,0});physics.step(float(dt));acceleration+=dt;}
        const float attained=physics.car(0).speed;float maxSide=0,minUp=1,minSpeed=attained,frontSlip=0,rearSlip=0;double yawAtRelease=0,yawAfterRelease=0;int airborne=0;
        if(attained<target-.1f)throw std::runtime_error("Could not reach requested test speed");
        for(int i=0;i<int(2.4*hz);++i){double t=i*dt;int phase=t<1?0:t<1.7?1:2;float steer=phase==0?1:phase==1?0:-1;
            const float previousSpeed=physics.car(0).speed;
            physics.drive(0,{1,steer,0,0},limited);physics.step(float(dt));auto p=physics.car(0);auto pos=position(p.body);
            expectSteering(p,steer,previousSpeed,limited);
            maxSide=std::max(maxSide,std::abs(p.lateralSpeed));minUp=std::min(minUp,p.body.m[1][1]);minSpeed=std::min(minSpeed,p.speed);
            airborne+=p.groundedWheels<2;if(t<1){yawAtRelease=p.yawRate;frontSlip=p.frontSlipAngle;rearSlip=p.rearSlipAngle;}if(t>=1&&t<1.7)yawAfterRelease=p.yawRate;
            if(trace.is_open()&&i%4==0)trace<<target<<','<<t<<','<<phase<<','<<steer<<','<<p.speed<<','<<p.steerAngle*180/Pi<<','<<p.yawRate*180/Pi<<','<<p.lateralSpeed<<','<<p.body.m[1][1]<<','<<p.groundedWheels<<','<<pos.x<<','<<pos.z<<','<<p.frontSlipAngle*180/Pi<<','<<p.rearSlipAngle*180/Pi<<'\n';
            if(!std::isfinite(p.speed)||!std::isfinite(pos.y))throw std::runtime_error("Non-finite handling state");
            if((phase==1&&std::abs(p.steerAngle)>1e-5f)||(phase!=1&&p.steerAngle*steer<=0))
                throw std::runtime_error("Steering command stuck or wrong wheel direction");
        }
        std::cout<<std::fixed<<std::setprecision(2)<<"target "<<target<<" attained "<<attained<<" min_speed "<<minSpeed
            <<" max_side_m_s "<<maxSide<<" min_up "<<minUp<<" airborne_steps "<<airborne<<" yaw_release "<<yawAtRelease*180/Pi
            <<" yaw_after_0.7s "<<yawAfterRelease*180/Pi<<" front_slip_deg "<<frontSlip*180/Pi<<" rear_slip_deg "<<rearSlip*180/Pi<<'\n';
        if(yawAtRelease<=0||std::abs(yawAfterRelease)>2*Pi/180||minUp<.8f||airborne>int(.1*hz)){
            stable=false;std::cerr<<"Stability/release check FAILED at "<<target<<" km/h\n";
        }
    }
    return stable?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
