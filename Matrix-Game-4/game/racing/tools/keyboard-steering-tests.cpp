#include "../src/KeyboardSteering.h"
#include "../src/ReferenceStream.h"
#include <iostream>
#include <limits>

namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void near(float actual,float expected){require(std::abs(actual-expected)<1e-5f,"Unexpected steering value");}
void profile(double dt){
    KeyboardSteering steering;
    auto advance=[&](float target,double duration){
        for(double time=0;time<duration-1e-10;){double step=std::min(dt,duration-time);steering.step(target,step);time+=step;}
    };
    advance(1,.05);near(steering.current(),1.f/6);
    advance(1,.25);near(steering.current(),1);
    advance(1,.7);near(steering.current(),1);
    advance(-1,.1);near(steering.current(),.5f);
    advance(-1,.1);near(steering.current(),0);
    advance(-1,.3);near(steering.current(),-1);
    advance(0,.1);near(steering.current(),-.5f);
    advance(0,.1);near(steering.current(),0);
    advance(-1,.05);near(steering.current(),-1.f/6);
    advance(0,.2);near(steering.current(),0);
}
}
int main(){try{
    for(double hz:{30.,40.,60.,144.,480.,500.,504.,1000.})profile(1/hz);
    KeyboardSteering steering;
    require(steering.step(1,1./480)>0&&steering.current()<.01f,"First tick instantly steered or had a dead time");
    steering.reset();near(steering.current(),0);
    steering.step(1,.3);near(steering.step(-1,.25),-1.f/6);
    float before=steering.current();near(steering.step(1,0),before);near(steering.step(1,-1),before);
    near(steering.step(std::numeric_limits<float>::quiet_NaN(),.1),before);
    near(steering.step(1,std::numeric_limits<double>::infinity()),before);
    steering.reset();near(steering.step(100,1),1);near(steering.step(-100,1),-1);

    Input raw{.8f,-1,.3f,.6f};steering.reset();auto filtered=steering.apply(raw,.05,true);
    near(filtered.steer,-1.f/6);require(filtered.throttle==raw.throttle&&filtered.brake==raw.brake&&filtered.handbrake==raw.handbrake,"Steering modified other controls");
    raw.steer=.72f;near(steering.apply(raw,.002,false).steer,.72f);
    raw.steer=-.4f;near(steering.apply(raw,.002,false).steer,-.4f);
    near(steering.apply({},.002,true).steer,0); // Releasing the stick leaves no keyboard residual.
    latency::Controls keyboard,analog;keyboard.keyboardSteering=true;
    require(!(keyboard==analog),"Source change at same steering value was lost");

    latency::InputDelay delay;keyboard.drive.steer=1;
    delay.push({keyboard,1,.1});keyboard.drive.steer=0;delay.push({keyboard,2,.15});
    steering.reset();float peak=0;
    for(int tick=1;tick<=480;++tick){double t=double(tick)/480;
        auto c=delay.sample(t,.6);auto value=steering.apply(c.controls.drive,1./480,c.controls.keyboardSteering).steer;
        if(t<.7-1e-9)near(value,0);peak=std::max(peak,value);
    }
    near(peak,1.f/6);near(steering.current(),0);
    latency::PulseInput pulse;keyboard.drive.steer=-1;pulse.push({keyboard,1,1});keyboard.drive.steer=0;pulse.push({keyboard,2,1.05});
    steering.reset();peak=0;
    for(int tick=0;tick<144;++tick){auto c=pulse.sample(double(tick)/480,1./480);
        peak=std::max(peak,std::abs(steering.apply(c.controls.drive,1./480,c.controls.keyboardSteering).steer));}
    near(peak,1.f/6);near(steering.current(),0);

    reference::SimulationClock clock(480);steering.reset();clock.observe(.1,false);
    while(clock.due()){steering.step(1,1./480);clock.step();}
    near(steering.current(),1.f/3);clock.freeze();clock.observe(10,true);
    require(!clock.due(),"Blocked simulation advanced steering time");near(steering.current(),1.f/3);
    clock.observe(20,false);require(!clock.due(),"Resume accumulated blocked steering time");
    clock.observe(20.05,false);while(clock.due()){steering.step(1,1./480);clock.step();}
    near(steering.current(),.5f);
    std::cout<<"Keyboard ramp/release/countersteer, timestep invariance, analog bypass, reset, delayed taps and backpressure passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
