#pragma once
#include "Common.h"
#include "Navigation.h"
#include <optional>
class World;
struct RaceSpawn {V3 ground,forward;};
struct Checkpoint {
    V3 center,forward;
    float halfWidth=5.5f,height=3.5f;
    std::vector<RaceSpawn> recovery;
};
struct RaceRoute {
    std::string id,name,description;
    std::vector<uint32_t> lanes;
    std::vector<RaceSpawn> grid;
    std::vector<Checkpoint> checkpoints;
    std::vector<V3> path;
    RouteGuide guide;
    float length=0;
};
struct RaceProgress {
    size_t next=0;
    double elapsed=0,penalty=0;
    bool finished=false;
    void advance(const RaceRoute& route,V3 before,V3 after,double dt);
    double total()const{return elapsed+penalty;}
};
std::optional<float> crossCheckpoint(const Checkpoint& gate,V3 before,V3 after);
RouteNavigation navigateRoute(const RaceRoute& route,const RaceProgress& progress,V3 location,float heading,float speed);
std::vector<RaceRoute> loadRaceRoutes(const std::filesystem::path& file,const World& world);
