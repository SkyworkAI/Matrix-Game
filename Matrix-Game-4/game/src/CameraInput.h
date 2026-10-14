#pragma once
#include <algorithm>
#include <cmath>
#include <numbers>

namespace CameraInput {
constexpr float RadiansPerPixel=.002f;
constexpr float MaxDegreesPerSecond=90.f;

inline float maxPixels(float seconds){
    return MaxDegreesPerSecond*std::numbers::pi_v<float>/180*std::max(0.f,seconds)/RadiansPerPixel;
}

// Split this batch's mouse delta over its physics ticks, discarding excess.
inline float pixelsPerTick(float pixels,int ticks,float tickSeconds){
    if(ticks<=0||tickSeconds<=0)return 0;
    const float limit=maxPixels(tickSeconds);
    return std::clamp(pixels/ticks,-limit,limit);
}

// Preserve short packet bursts across physics ticks, independent of render FPS.
// Bound both queued displacement and wall-clock age; no catch-up after a stall.
class NetworkAxis {
    float pending=0;
    double expires=0;
public:
    static constexpr float MaxPendingSeconds=.05f;
    void clear(){pending=0;expires=0;}
    void add(float pixels,double now){
        if(now>=expires)clear();
        if(pixels==0)return;
        if(pixels*pending<0)pending=0;
        const float limit=maxPixels(MaxPendingSeconds);
        pending=std::clamp(pending+pixels,-limit,limit);
        expires=now+MaxPendingSeconds;
    }
    float consume(float seconds,double now){
        if(now>=expires){clear();return 0;}
        const float limit=maxPixels(seconds);
        const float applied=std::clamp(pending,-limit,limit);
        pending-=applied;
        return applied;
    }
};
}
