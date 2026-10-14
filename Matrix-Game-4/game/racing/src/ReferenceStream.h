#pragma once
#include "LatencySimulation.h"
#include <random>

namespace reference {
// Keep exact capture intervals without lowering the existing physics rate.
inline int capturePhysicsHz(int fps){
    if(fps<1||fps>60)throw std::runtime_error("Native reference FPS must be between 1 and 60");
    return ((480+fps-1)/fps)*fps;
}
// Waiting input credit is released when the model takes ownership, not when
// it later displays a result. One model block and a shallow input queue coexist.
class ModelCredits {
    uint64_t begun=0,displayed=0,requested=0,capacity,first,chunk;
public:
    explicit ModelCredits(uint64_t pending=15,uint64_t firstFrames=9,uint64_t chunkFrames=12)
        :capacity(pending),first(firstFrames),chunk(chunkFrames){
        if(!first||!chunk||first>chunk||capacity<chunk||capacity>2*chunk)
            throw std::runtime_error("Invalid reference input capacity");
    }
    uint64_t allowed(uint64_t total)const{return std::min(total,std::max(first,begun+capacity));}
    void begin(uint64_t end,uint64_t captured){
        if(requested||end!=(begun?begun+chunk:first)||end>captured||begun!=displayed)
            throw std::runtime_error("Invalid model block admission");
        begun=end;
    }
    void request(uint64_t end,uint64_t captured){
        if(requested||end!=(begun?begun+chunk:first)||begun!=displayed)
            throw std::runtime_error("Invalid model input request");
        requested=end;
        capturedReady(captured);
    }
    void capturedReady(uint64_t captured){
        // Ownership transfers only after the entire requested block exists.
        if(requested&&requested<=captured){begun=requested;requested=0;}
    }
    void acknowledge(uint64_t end){
        if(end!=(displayed?displayed+chunk:first)||end!=begun)
            throw std::runtime_error("Invalid output acknowledgement");
        displayed=end;
    }
};
struct Settings {
    int chunk=12,firstChunk=0,inputCapacity=15,outputCapacity=24;
    double fps=40,physicsHz=480,inferenceMs=250,jitterMs=0;
    unsigned seed=1;
    int stepsPerFrame()const{return int(std::llround(physicsHz/fps));}
    void validate()const {
        if(!std::isfinite(fps)||fps<1||fps>120||!std::isfinite(physicsHz)||physicsHz<100||physicsHz>1000||
            std::abs(physicsHz/fps-std::round(physicsHz/fps))>1e-6)
            throw std::runtime_error("Reference FPS must divide physics Hz exactly (e.g. 40 FPS and --physics-hz 480)");
        if(chunk<1||chunk>120||firstChunk<0||firstChunk>120||inputCapacity<std::max(chunk,firstChunk)||inputCapacity>240||outputCapacity<std::max(chunk,firstChunk)||outputCapacity>240||
            !std::isfinite(inferenceMs)||inferenceMs<0||inferenceMs>5000||!std::isfinite(jitterMs)||jitterMs<0||jitterMs>inferenceMs)
            throw std::runtime_error("Invalid reference queue/chunk/inference settings");
    }
};

// Only admitted wall time advances the simulation. Backpressure discards debt,
// not reference frames; unblocking never catches up the frozen wall interval.
class SimulationClock {
    double hz,budget=0,previous=0;
    bool blocked=false;
    uint64_t ticks=0;
public:
    explicit SimulationClock(double frequency):hz(frequency){}
    void reset(double time=0){budget=0;previous=time;blocked=false;ticks=0;}
    void observe(double time,bool full) {
        if(time<previous)throw std::runtime_error("Non-monotonic stream clock");
        if(full||blocked)budget=0;else budget+=time-previous;
        previous=time;blocked=full;
    }
    void freeze(){budget=0;blocked=true;}
    bool due()const{return budget+1e-9>=1/hz;}
    void step(){if(!due())throw std::runtime_error("Simulation advanced ahead of wall time");budget=std::max(0.0,budget-1/hz);++ticks;}
    double time()const{return ticks/hz;}
    uint64_t tick()const{return ticks;}
    double inputTime(double age)const{return std::max(time(),time()+budget-std::max(0.0,age));}
};

struct InputEvent {latency::Command command;double simulationTime=0;};
class InputQueue {
    std::deque<InputEvent> pending;
    latency::Command current;
public:
    void reset(){pending.clear();current={};}
    void push(latency::Command command,double simulationTime) {
        if(pending.size()>=4096)throw std::runtime_error("Input queue overflow: stopping instead of silently dropping controls");
        if(!pending.empty())simulationTime=std::max(simulationTime,pending.back().simulationTime);
        pending.push_back({command,simulationTime});
    }
    latency::Command sample(double simulationTime) {
        // One edge per physics step retains even taps received while time was frozen.
        if(!pending.empty()&&pending.front().simulationTime<=simulationTime+1e-9){current=pending.front().command;pending.pop_front();}
        return current;
    }
    size_t size()const{return pending.size();}
};

// Single-threaded deterministic mock consumer. Timers never block event polling.
// Reserve output slots before starting work, including all in-flight frames.
template<class Frame> class Pipeline {
    Settings config;
    std::deque<Frame> input,output;
    std::vector<Frame> job;
    std::mt19937 random;
    double ready=0,nextDisplay=0,lastDuration=0;
    uint64_t chunkId=0;
public:
    explicit Pipeline(Settings settings):config(settings),random(settings.seed){config.validate();}
    void reset(){input.clear();output.clear();job.clear();ready=nextDisplay=lastDuration=0;chunkId=0;random.seed(config.seed);}
    bool full()const{return input.size()>=size_t(config.inputCapacity);}
    void push(Frame frame){if(full())throw std::runtime_error("Reference queue overflow");input.push_back(std::move(frame));}
    bool complete(double time) {
        if(job.empty()||time+1e-9<ready)return false;
        for(auto& frame:job)output.push_back(std::move(frame));job.clear();return true;
    }
    bool start(double time) {
        const int count=chunkId==0&&config.firstChunk?config.firstChunk:config.chunk;
        if(!job.empty()||input.size()<size_t(count)||output.size()+size_t(count)>size_t(config.outputCapacity))return false;
        for(int i=0;i<count;++i){job.push_back(std::move(input.front()));input.pop_front();}
        lastDuration=(config.inferenceMs+std::uniform_real_distribution<double>(-config.jitterMs,config.jitterMs)(random))/1000;
        ready=time+lastDuration;++chunkId;return true;
    }
    std::optional<Frame> pop(double time) {
        if(output.empty()||time+1e-9<nextDisplay)return {};
        Frame frame=std::move(output.front());output.pop_front();
        const double interval=1/config.fps;
        // Preserve phase across small scheduling jitter; rebase after a full-frame
        // stall instead of dropping frames or draining the queue in one iteration.
        nextDisplay=time-nextDisplay>=interval?time+interval:nextDisplay+interval;
        return frame;
    }
    size_t inputSize()const{return input.size();}
    size_t outputSize()const{return output.size();}
    size_t inFlight()const{return job.size();}
    uint64_t chunk()const{return chunkId;}
    double duration()const{return lastDuration;}
    const std::vector<Frame>& activeFrames()const{return job;}
};
}
