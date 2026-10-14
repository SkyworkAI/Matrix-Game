#pragma once
#include "Common.h"

// Normalized steering, advanced only when the vehicle's simulation advances.
class KeyboardSteering {
public:
    static constexpr double TurnSeconds=.3,ReturnSeconds=.2;
    void reset(){value=0;}
    float current()const{return float(value);}
    float step(float target,double dt){
        if(!std::isfinite(target)||!std::isfinite(dt)||dt<=0)return current();
        target=std::clamp(target,-1.f,1.f);
        if(value*target<0){
            // Account for time on both sides of center even within one substep.
            double returning=std::min(dt,std::abs(value)*ReturnSeconds);
            value=approach(value,0,returning/ReturnSeconds);dt-=returning;
        }
        const double seconds=std::abs(target)<std::abs(value)?ReturnSeconds:TurnSeconds;
        value=approach(value,target,dt/seconds);return current();
    }
    Input apply(Input input,double dt,bool keyboard){
        if(keyboard)input.steer=step(input.steer,dt);
        else reset(); // Analog input bypasses smoothing, including its release.
        return input;
    }
private:
    double value=0;
    static double approach(double from,double to,double distance){
        if(std::abs(to-from)<=distance+1e-12)return to;
        return from+std::copysign(distance,to-from);
    }
};
