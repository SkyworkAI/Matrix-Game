#pragma once
#include "RaceRules.h"
#include "World.h"
#include <nlohmann/json.hpp>

struct PursuitProgress {
    double held=0;
    bool captured=false;
    void advance(double dt,float gap,bool clear) {
        if(captured)return;
        held=clear&&std::isfinite(gap)&&gap<=5.f?std::min(5.,held+dt):0;
        captured=held>=5.-1e-9;
        if(captured)held=5;
    }
};

// Ground-plane distance between oriented car-body footprints, not centre distance.
float pursuitBodyGap(const std::array<V3,4>& a,const std::array<V3,4>& b);

class Pursuit {
public:
    explicit Pursuit(World& world):world(world){}
    void reset(const RaceRoute& route);
    Input drive(double dt);
    void advance(double dt);
    nlohmann::json state()const;
    bool captured()const{return progress.captured;}
private:
    World& world;
    std::vector<V3> points;
    size_t nearest=0;
    PursuitProgress progress;
    float gap=0,centerDistance=0,initialDistance=0,initialRouteDistance=0;
    bool eligible=false;
    double stuck=0,reverse=0;
};
