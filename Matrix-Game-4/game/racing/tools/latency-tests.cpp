#include "../src/LatencySimulation.h"
#include <iostream>

void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(){try {
    using namespace latency;
    require(parseMode("pulse")==Mode::Pulse&&parseMode("fixed")==Mode::Fixed,"mode parsing");
    bool rejected=false;try{parseMode("unknown");}catch(...){rejected=true;}require(rejected,"unknown mode rejected");
    InputDelay delay;Controls press;press.drive.throttle=1;
    delay.push({press,1,.1});delay.push({{},2,.2});
    require(delay.sample(.699,.6).id==0,"fixed delay cannot apply early");
    require(delay.sample(.7,.6).controls.drive.throttle==1,"press delayed exactly 600ms");
    require(delay.sample(.799,.6).id==1,"short press retained");
    require(delay.sample(.8,.6).id==2,"release retained");
    require(delay.size()==0,"consumed events removed");
    delay.reset();require(delay.sample(20,0).id==0,"reset flushes delayed controls");
    PulseInput inputs;Controls left=press,right=press;
    left.drive.steer=-1;right.drive.steer=1;
    inputs.push({left,1,.1});inputs.push({press,2,.15});
    inputs.push({right,3,.19});inputs.push({press,4,.23});
    require(inputs.sample(.3,.002).id==1,"completed short left tap is replayed next burst");
    require(inputs.sample(.348,.002).id==1,"left tap preserves 50ms duration");
    require(inputs.sample(.35,.002).id==2,"left release is retained");
    require(inputs.sample(.39,.002).id==3,"right tap is not averaged against left");
    require(inputs.sample(.43,.002).id==4,"right release is retained");
    require(inputs.sample(.6,.002).id==4&&inputs.size()==0,"hold persists without replay duplication");
    inputs.push({{},5,.75});
    require(inputs.sample(.6,.002).id==5,"new release applies without replaying an old hold");
    inputs.reset();inputs.push({left,1,1});inputs.push({right,2,1});inputs.push({{},3,1});
    require(inputs.sample(0,.002).id==1&&inputs.sample(.002,.002).id==2&&inputs.sample(.004,.002).id==3,
        "equal-timestamp edges each receive a physics step");
    inputs.reset();inputs.push({left,1,.1});inputs.push({right,2,.5});inputs.push({{},3,.55});
    require(inputs.sample(0,.002).id==1&&inputs.sample(.298,.002).id==1,"long replay starts in first chunk");
    require(inputs.sample(.3,.002).id==1&&inputs.sample(.4,.002).id==2&&inputs.sample(.45,.002).id==3,
        "replay carries across chunk boundaries without dropping edges");
    inputs.reset();require(inputs.sample(2,.002).id==0&&inputs.size()==0,"reset flushes pulse controls");
    inputs.push({left,1,.1});require(inputs.sample(0,.002).id==1,"press can be committed before release arrives");
    require(inputs.sample(.298,.002).id==1,"unknown future release cannot change a generated chunk");
    inputs.push({{},2,.15});require(inputs.sample(.3,.002).id==2,"late release applies in next chunk without becoming stuck");
    PulsePlayback<int> playback(40);
    playback.submit({0,1,2},.3);
    require(!playback.poll(.299),"no display before model completion");
    require(playback.poll(.3)==0,"first frame at completion");
    playback.submit({3,4},.6);
    require(playback.queuedFrames()==4&&!playback.canPrepare(),"bounded two-chunk pipeline");
    require(!playback.poll(.31),"playback paced");
    require(playback.poll(.325)==1&&playback.poll(.35)==2,"ordered playback");
    require(!playback.poll(.5),"underflow does not reveal future frames");
    require(playback.poll(.6)==3,"next chunk ready");
    require(playback.poll(1.)==4&&!playback.poll(1.),"stall does not skip or burst frames");
    playback.reset();require(playback.queuedFrames()==0&&playback.canPrepare(),"mode change flushes pipeline");
    // For an ideal 300ms preparation cadence and 300ms inference: L = wait + 300ms.
    double total=0;for(int i=0;i<300;++i){double event=(i+.5)/1000,latency=.6-event;
        require(latency>.3&&latency<.6,"pulse input-to-first-frame bound");total+=latency;}
    require(std::abs(total/300-.45)<1e-9,"pulse mean latency");
    std::cout<<"PASS: fixed-delay edges, pulse input replay/duration/order/reset, playback readiness/pacing/capacity, ideal latch bounds\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
