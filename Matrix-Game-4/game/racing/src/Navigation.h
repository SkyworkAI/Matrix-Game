#pragma once
#include "Common.h"

// Positions on the road surface, ordered from the starting grid to the finish.
// This is display data; only RaceProgress can validate a checkpoint crossing.
struct RouteGuide {
    std::vector<V3> points;
    std::vector<float> distances,checkpoints;
};
struct RouteNavigation {
    bool valid=false,offRoute=false,wrongWay=false,missedCheckpoint=false;
    float along=0,distanceToGate=0,distanceToFinish=0,targetAngle=0;
};
