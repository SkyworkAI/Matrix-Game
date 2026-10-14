#include "../src/ReferenceStream.h"
#include <iostream>

void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void runScenario(double inference,double jitter,bool slowDisplay){
    reference::Settings s;s.inferenceMs=inference;s.jitterMs=jitter;
    reference::Pipeline<int> p(s);reference::SimulationClock clock(s.physicsHz);
    int produced=0,consumed=0,blocked=0;bool previousFull=false;
    for(int i=0;i<12000;++i){double t=i*.001;
        p.complete(t);
        if(!slowDisplay||i%100==0)if(auto frame=p.pop(t))require(*frame==++consumed,"output order/no loss");
        p.start(t);clock.observe(t,p.full());
        const auto before=clock.tick();
        while(clock.due()&&!p.full()){
            clock.step();if(clock.tick()%uint64_t(s.stepsPerFrame())==0)p.push(++produced);
        }
        if(previousFull)require(clock.tick()==before,"unblocking must not catch up frozen time");
        if(p.full()){clock.freeze();++blocked;}previousFull=p.full();
        p.start(t);
        require(p.inputSize()<=size_t(s.inputCapacity),"bounded reference queue");
        require(p.outputSize()+p.inFlight()<=size_t(s.outputCapacity),"output reservation includes in-flight chunk");
        require(size_t(produced-consumed)==p.inputSize()+p.inFlight()+p.outputSize(),"frame conservation");
        require(clock.time()<=t+1e-8,"no prediction or future advancement");
    }
    require(produced>60&&consumed>24,"pipeline makes progress");
    if(inference>300||slowDisplay)require(blocked>0&&clock.time()<11,"overload freezes game time");
    else if(jitter==0)require(blocked==0,"fast consumer should not stall producer");
    std::cout<<"scenario "<<inference<<" +/- "<<jitter<<" ms, slow display "<<slowDisplay
        <<": captured "<<produced<<", played "<<consumed<<", blocked ticks "<<blocked<<'\n';
}
int main(){try{
    require(reference::capturePhysicsHz(24)==480&&reference::capturePhysicsHz(30)==480,"existing capture timing unchanged");
    require(reference::capturePhysicsHz(28)==504,"28 FPS uses 18 equal physics steps per frame");
    for(int fps=1;fps<=60;++fps){
        const int hz=reference::capturePhysicsHz(fps);
        require(hz>=480&&hz<480+fps&&hz%fps==0,"bounded integral capture ratio");
        reference::SimulationClock sampling(hz);int frames=0;
        for(int ms=1;ms<=2000;++ms){
            sampling.observe(ms/1000.,false);
            while(sampling.due()){
                sampling.step();
                if(sampling.tick()%uint64_t(hz/fps)==0){
                    ++frames;require(std::abs(sampling.time()-double(frames)/fps)<1e-9,"exact frame simulation timestamps");
                }
            }
        }
        require(frames==2*fps,"configured FPS equals captured FPS");
    }
    for(int invalid:{0,-1,61}){
        bool rejected=false;try{reference::capturePhysicsHz(invalid);}catch(...){rejected=true;}
        require(rejected,"reject invalid native capture FPS");
    }
    reference::ModelCredits credits;
    require(credits.allowed(100)==15,"initial shallow input allowance");
    credits.begin(9,9);require(credits.allowed(100)==24,"first block ownership releases input slots");
    credits.acknowledge(9);require(credits.allowed(100)==24,"display does not double-release slots");
    credits.begin(21,21);require(credits.allowed(100)==36,"12-frame block and 15 waiting frames are separate");
    bool badCredit=false;try{credits.begin(33,33);}catch(...){badCredit=true;}
    require(badCredit,"cannot admit another block before preceding output");
    credits.acknowledge(21);require(credits.allowed(30)==30,"credits respect end of stream");
    reference::Settings s;s.validate();s.physicsHz=500;
    bool rejected=false;try{s.validate();}catch(...){rejected=true;}require(rejected,"reject inexact physical sampling ratio");
    s.physicsHz=480;s.inputCapacity=11;rejected=false;try{s.validate();}catch(...){rejected=true;}require(rejected,"queue must hold a chunk");
    reference::SimulationClock clock(480);clock.observe(.01,false);while(clock.due())clock.step();
    auto before=clock.tick();clock.freeze();clock.observe(10,true);require(!clock.due()&&clock.tick()==before,"blocked time cannot advance physics");
    clock.observe(20,false);require(!clock.due(),"resume has no accumulated debt");
    clock.observe(20.003,false);require(clock.due(),"new wall time resumes simulation");
    reference::InputQueue inputs;latency::Controls press;press.drive.steer=-1;
    inputs.push({press,1,1},clock.time());inputs.push({{},2,1.01},clock.time());
    require(inputs.sample(clock.time()).id==1,"tap during freeze retained");
    require(inputs.sample(clock.time()+1./480).id==2,"release follows tap one step later");
    inputs.reset();require(inputs.size()==0&&inputs.sample(0).id==0,"reset flushes pending inputs");
    s={};reference::Pipeline<int> p(s);
    for(int i=0;i<s.chunk-1;++i)p.push(i);require(!p.start(0),"no inference on partial chunk");
    p.push(11);require(p.start(0)&&!p.complete(.249)&&!p.pop(.249),"no frame before full inference");
    require(p.complete(.25)&&p.pop(.25)==0&&!p.pop(.251),"playback is paced");
    require(p.pop(10)==1&&!p.pop(10),"display stall does not drop or burst frames");
    p.reset();require(!p.complete(20)&&!p.pop(20)&&p.inputSize()==0,"reset cancels old completion");
    s.firstChunk=9;reference::Pipeline<int> variable(s);
    for(int i=1;i<=8;++i)variable.push(i);require(!variable.start(0),"first chunk must also be complete");
    variable.push(9);require(variable.start(0)&&variable.activeFrames().front()==1&&variable.activeFrames().back()==9,"first chunk has actual 9-frame span");
    variable.complete(1);for(int i=10;i<=21;++i)variable.push(i);
    require(variable.start(1)&&variable.activeFrames().size()==12&&variable.activeFrames().front()==10,"subsequent chunk has 12 contiguous frames");
    variable.reset();for(int i=1;i<=9;++i)variable.push(i);require(variable.start(0)&&variable.inFlight()==9,"epoch reset restores first chunk size");
    runScenario(200,0,false);runScenario(250,180,false);runScenario(700,0,false);runScenario(0,0,true);
    reference::ModelCredits small(4,1,4);
    reference::ModelCredits legacyCapacity(24);
    require(legacyCapacity.allowed(100)==24,"legacy queue capacity remains supported");
    require(small.allowed(100)==4,"chunk1 shallow queue");
    for(uint64_t end=1;end<=93;end+=4){
        small.begin(end,end);require(small.allowed(100)==std::min<uint64_t>(100,end+4),"chunk1 admission credit");
        small.acknowledge(end);
    }
    bool badAck=false;try{small.acknowledge(93);}catch(...){badAck=true;}
    require(badAck,"duplicate chunk1 acknowledgement");
    reference::ModelCredits reserved(4,1,4);
    reserved.request(1,0);require(reserved.allowed(100)==4,"request alone grants no extra capture credit");
    bool badRequest=false;try{reserved.request(1,0);}catch(...){badRequest=true;}
    require(badRequest,"only one pending model read");
    reserved.capturedReady(1);require(reserved.allowed(100)==5,"complete requested block transfers ownership");
    badRequest=false;try{reserved.request(5,1);}catch(...){badRequest=true;}
    require(badRequest,"cannot request before preceding output");
    reserved.acknowledge(1);reserved.request(5,3);reserved.capturedReady(4);
    require(reserved.allowed(100)==5,"partial block cannot release pending slots");
    reserved.capturedReady(5);require(reserved.allowed(100)==9,"waiting queue stays exactly four frames");
    reserved.acknowledge(5);reserved.request(9,9);
    require(reserved.allowed(100)==13,"already queued input transfers immediately");
    reserved.acknowledge(9);reserved.begin(13,13);
    require(reserved.allowed(100)==17,"legacy admission still supported");
    std::cout<<"PASS: fixed sampling, backpressure clocks, input edges, chunk readiness, capacity, conservation, reset\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
