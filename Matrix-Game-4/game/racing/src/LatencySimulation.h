#pragma once
#include "Common.h"
#include <deque>
#include <optional>

namespace latency {
enum class Mode { Normal, Pulse, Fixed };
inline Mode parseMode(const std::string& value) {
    if(value=="normal")return Mode::Normal;
    if(value=="pulse")return Mode::Pulse;
    if(value=="fixed")return Mode::Fixed;
    throw std::runtime_error("Latency mode must be normal, pulse, or fixed");
}
inline const char* name(Mode mode) {
    return mode==Mode::Pulse?"PULSE":mode==Mode::Fixed?"FIXED":"NORMAL";
}
struct Controls {
    Input drive;
    int camera=1;
    bool rear=false;
    bool keyboardSteering=false;
    bool operator==(const Controls& b)const {
        return drive.throttle==b.drive.throttle&&drive.steer==b.drive.steer&&
            drive.brake==b.drive.brake&&drive.handbrake==b.drive.handbrake&&camera==b.camera&&rear==b.rear&&keyboardSteering==b.keyboardSteering;
    }
};
struct Command {Controls controls;uint64_t id=0;double time=0;};

// Preserve input edges, including a press and release during the delay window.
class InputDelay {
    std::deque<Command> pending;
    Command current;
public:
    void reset(){pending.clear();current={};}
    void push(Command command) {
        const double previous=pending.empty()?current.time:pending.back().time;
        if(!std::isfinite(command.time)||command.time<previous)throw std::runtime_error("Non-monotonic input time");
        pending.push_back(command);
    }
    Command sample(double time,double delay) {
        while(!pending.empty()&&pending.front().time<=time-delay+1e-9){current=pending.front();pending.pop_front();}
        return current;
    }
    size_t size()const{return pending.size();}
};

// Replay unseen input edges at the next burst, preserving intervals between them.
// Once caught up, hold the final state; do not replay an already simulated hold.
class PulseInput {
    std::deque<Command> pending;
    Command current;
    double nextSwitch=0;
    bool scheduled=false;
public:
    void reset(){pending.clear();current={};nextSwitch=0;scheduled=false;}
    void push(Command command) {
        const double previous=pending.empty()?current.time:pending.back().time;
        if(!std::isfinite(command.time)||command.time<previous)throw std::runtime_error("Non-monotonic pulse input time");
        pending.push_back(command);
    }
    Command sample(double simulationTime,double step) {
        if(!std::isfinite(step)||step<=0)throw std::runtime_error("Invalid pulse input step");
        if(!scheduled&&!pending.empty()){nextSwitch=simulationTime;scheduled=true;}
        while(scheduled&&simulationTime+1e-9>=nextSwitch) {
            current=pending.front();pending.pop_front();
            scheduled=!pending.empty();
            // Even equal timestamps must give distinct edges one physics substep.
            if(scheduled)nextSwitch+=std::max(step,pending.front().time-current.time);
        }
        return current;
    }
    size_t size()const{return pending.size();}
};

// At most one in-flight chunk and one playing chunk. Never skip a reference frame.
template<class Frame> class PulsePlayback {
    struct Chunk {std::vector<Frame> frames;double ready=0;size_t cursor=0;};
    std::optional<Chunk> pending,playing;
    double interval,nextFrame=0;
public:
    explicit PulsePlayback(double fps):interval(1/fps) {
        if(!std::isfinite(fps)||fps<=0)throw std::runtime_error("Invalid playback FPS");
    }
    void reset(){pending.reset();playing.reset();nextFrame=0;}
    bool canPrepare()const{return !pending;}
    void submit(std::vector<Frame> frames,double ready) {
        if(pending||frames.empty()||!std::isfinite(ready))throw std::runtime_error("Invalid or excess pulse chunk");
        pending=Chunk{std::move(frames),ready,0};
    }
    std::optional<Frame> poll(double time) {
        if(time+1e-9<nextFrame)return {};
        if(!playing||playing->cursor==playing->frames.size()) {
            if(!pending||time+1e-9<pending->ready)return {};
            playing=std::move(pending);pending.reset();
        }
        Frame result=std::move(playing->frames[playing->cursor++]);
        // A late display holds the timeline back instead of rapidly draining the queue.
        nextFrame=time+interval;
        return result;
    }
    size_t queuedFrames()const {
        return (pending?pending->frames.size():0)+(playing?playing->frames.size()-playing->cursor:0);
    }
};
}
