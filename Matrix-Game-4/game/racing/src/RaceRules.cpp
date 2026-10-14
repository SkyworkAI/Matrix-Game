#include "RaceRules.h"
#include "World.h"
#include <nlohmann/json.hpp>
#include <limits>
#include <set>

namespace {
V3 guidePoint(const RouteGuide& guide,float distance){
    auto end=std::upper_bound(guide.distances.begin(),guide.distances.end(),distance);
    if(end==guide.distances.begin())return guide.points.front();
    if(end==guide.distances.end())return guide.points.back();
    size_t i=size_t(end-guide.distances.begin());
    float t=(distance-guide.distances[i-1])/(guide.distances[i]-guide.distances[i-1]);
    return guide.points[i-1]+(guide.points[i]-guide.points[i-1])*t;
}
void buildGuide(RaceRoute& route,const nlohmann::json& entry,const World& world){
    auto roadDistance=[&](const nlohmann::json& mark){
        float distance=mark.at("along");uint32_t lane=mark.at("lane");
        for(auto current:route.lanes){if(current==lane)return distance;distance+=world.roadLength(current);}
        throw std::runtime_error("Navigation mark outside route");
    };
    float begin=roadDistance(entry.at("grid").front());
    for(const auto& grid:entry.at("grid"))begin=std::min(begin,roadDistance(grid));
    float end=roadDistance(entry.at("checkpoints").back()),offset=0;
    for(auto lane:route.lanes){
        float length=world.roadLength(lane);std::vector<float> marks{0,length};
        for(float at=5;at<length;at+=5)marks.push_back(at);
        for(const char* kind:{"grid","checkpoints"})for(const auto& mark:entry.at(kind))
            if(mark.at("lane").get<uint32_t>()==lane)marks.push_back(mark.at("along"));
        std::sort(marks.begin(),marks.end());marks.erase(std::unique(marks.begin(),marks.end()),marks.end());
        for(float at:marks){
            if(offset+at<begin-.001f||offset+at>end+.001f)continue;
            V3 point=world.roadSample(lane,at).position;
            if(auto y=world.groundSupport(point);y&&std::abs(*y-point.y)<.75f)point.y=*y;
            auto& guide=route.guide;
            float segment=guide.points.empty()?0:(point-guide.points.back()).length();
            if(guide.points.empty()||segment>.001f){
                guide.distances.push_back(guide.distances.empty()?0:guide.distances.back()+segment);
                guide.points.push_back(point);
            }
            for(const auto& gate:entry.at("checkpoints"))
                if(gate.at("lane").get<uint32_t>()==lane&&std::abs(gate.at("along").get<float>()-at)<.0001f)
                    guide.checkpoints.push_back(guide.distances.back());
        }
        offset+=length;
    }
    if(route.guide.points.size()<2||route.guide.checkpoints.size()!=route.checkpoints.size())
        throw std::runtime_error("Incomplete route navigation");
}
}

RouteNavigation navigateRoute(const RaceRoute& route,const RaceProgress& progress,V3 location,float heading,float speed){
    RouteNavigation result;const auto& guide=route.guide;
    if(guide.points.size()<2||guide.distances.size()!=guide.points.size()||guide.checkpoints.size()!=route.checkpoints.size()||route.checkpoints.empty())return result;
    result.valid=true;
    if(progress.finished)return result;
    size_t next=std::min(progress.next,route.checkpoints.size()-1);
    float gateDistance=guide.checkpoints[next];
    // Search only the current leg, plus a short recovery margin. At crossings
    // and stacked roads, a later leg must never steal the player's progress.
    float begin=next?std::max(0.f,guide.checkpoints[next-1]-30):0;
    float best=std::numeric_limits<float>::max();V3 closest{},forward{};
    for(size_t i=1;i<guide.points.size();++i){
        float a=guide.distances[i-1],b=guide.distances[i];
        if(b<begin||a>gateDistance||b-a<.001f)continue;
        V3 delta=guide.points[i]-guide.points[i-1],relative=location-guide.points[i-1];
        // Height matters more than lateral error: do not snap to an underpass.
        V3 weighted=delta;weighted.y*=9;
        float t=std::clamp(dot(relative,weighted)/std::max(.0001f,dot(delta,weighted)),std::max(0.f,(begin-a)/(b-a)),std::min(1.f,(gateDistance-a)/(b-a)));
        V3 point=guide.points[i-1]+delta*t,error=location-point;
        float cost=error.x*error.x+error.z*error.z+9*error.y*error.y;
        if(cost<best){best=cost;closest=point;forward=delta.normalized();result.along=a+(b-a)*t;}
    }
    // Imported lanes can meet with a tiny backwards seam. Use a short route
    // tangent, not that centimetre-scale segment, for the wrong-way test.
    V3 tangent=guidePoint(guide,std::min(gateDistance,result.along+2))-guidePoint(guide,std::max(begin,result.along-2));
    if(tangent.length()>.1f)forward=tangent.normalized();
    const auto& gate=route.checkpoints[next];V3 error=location-closest;
    result.offRoute=std::hypot(error.x,error.z)>12||std::abs(error.y)>3.5f;
    result.missedCheckpoint=gateDistance-result.along<3&&dot(location-gate.center,gate.forward)>.1f&&std::abs(location.y-gate.center.y)<gate.height;
    result.wrongWay=!result.offRoute&&!result.missedCheckpoint&&dot({std::sin(heading),0,std::cos(heading)},forward)<-.35f;
    result.distanceToGate=std::max(0.f,gateDistance-result.along);
    result.distanceToFinish=std::max(0.f,guide.distances.back()-result.along);
    float lookAhead=std::clamp(8+std::abs(speed)*.25f,12.f,28.f);
    float target=result.missedCheckpoint?std::max(begin,gateDistance-10):std::min(gateDistance,result.along+(result.offRoute?8:lookAhead));
    V3 delta=guidePoint(guide,target)-location;
    result.targetAngle=std::remainder(std::atan2(delta.x,delta.z)-heading,2*Pi);
    return result;
}

std::optional<float> crossCheckpoint(const Checkpoint& g,V3 a,V3 b){
    float da=dot(a-g.center,g.forward),db=dot(b-g.center,g.forward);
    if(da>=0||db<0||db-da<1e-6f)return {};
    float t=-da/(db-da);V3 p=a+(b-a)*t,d=p-g.center;
    V3 right=cross({0,1,0},g.forward).normalized();
    if(std::abs(dot(d,right))>g.halfWidth||d.y<-.3f||d.y>g.height)return {};
    return t;
}
void RaceProgress::advance(const RaceRoute& route,V3 a,V3 b,double dt){
    if(finished)return;float last=-1;
    while(next<route.checkpoints.size()){
        auto t=crossCheckpoint(route.checkpoints[next],a,b);
        if(!t||*t<=last)break;last=*t;++next;
        if(next==route.checkpoints.size()){finished=true;elapsed+=dt*(*t);return;}
    }
    elapsed+=dt;
}
std::vector<RaceRoute> loadRaceRoutes(const std::filesystem::path& file,const World& world){
    using nlohmann::json;std::ifstream input(file);if(!input)throw std::runtime_error("Missing race-routes.json");
    json config;input>>config;if(config.at("version")!=1)throw std::runtime_error("Unsupported race route version");
    std::vector<RaceRoute> routes;std::set<std::string> names;
    auto sample=[&](uint32_t lane,float at){
        if(lane>=world.roadCount()||world.roadFlags(lane)&4||!std::isfinite(at)||at<0||at>world.roadLength(lane)+.1f)throw std::runtime_error("Invalid race road reference");
        auto p=world.roadSample(lane,at);auto y=world.groundSupport(p.position);
        if(!y||std::abs(*y-p.position.y)>.5f)throw std::runtime_error("Race route lacks road support: lane "+std::to_string(lane));
        p.position.y=*y;p.forward.y=0;p.forward=p.forward.normalized();return RaceSpawn{p.position,p.forward};
    };
    for(const auto& entry:config.at("routes")){
        RaceRoute r;r.id=entry.at("id");r.name=entry.at("name");r.description=entry.at("description");r.lanes=entry.at("lanes").get<std::vector<uint32_t>>();
        if(r.lanes.empty()||!names.insert(r.id).second)throw std::runtime_error("Empty or duplicate race route");
        for(size_t i=0;i<r.lanes.size();++i){auto lane=r.lanes[i];
            if(i){const auto& links=world.roadLinks(r.lanes[i-1]);if(std::find(links.begin(),links.end(),lane)==links.end())throw std::runtime_error("Disconnected race route");}
            float length=world.roadLength(lane);r.length+=length;
            for(float at=0;at<=length;at+=5)r.path.push_back(world.roadSample(lane,at).position);
            r.path.push_back(world.roadSample(lane,length).position);
        }
        for(const auto& p:entry.at("grid"))r.grid.push_back(sample(p.at("lane"),p.at("along")));
        if(r.grid.size()!=World::MaxPlayers)throw std::runtime_error("A route needs five starting positions");
        float previous=-1;for(const auto& c:entry.at("checkpoints")){
            uint32_t lane=c.at("lane");float at=c.at("along");auto it=std::find(r.lanes.begin(),r.lanes.end(),lane);
            if(it==r.lanes.end())throw std::runtime_error("Checkpoint outside route");
            float distance=at;for(auto iter=r.lanes.begin();iter!=it;++iter)distance+=world.roadLength(*iter);
            if(distance<=previous)throw std::runtime_error("Unordered checkpoints");previous=distance;
            auto s=sample(lane,at);Checkpoint g{s.ground,s.forward,c.at("half_width"),c.at("height"),{}};
            if(g.halfWidth<1||g.halfWidth>12||g.height<1||g.height>5)throw std::runtime_error("Invalid checkpoint dimensions");
            for(float recover:c.at("recovery_along")){if(recover>=at)throw std::runtime_error("Recovery must precede its gate");g.recovery.push_back(sample(lane,recover));}
            if(g.recovery.empty())throw std::runtime_error("Checkpoint has no recovery position");r.checkpoints.push_back(g);
        }
        if(r.checkpoints.size()<3)throw std::runtime_error("Route needs at least three checkpoints");
        r.length=previous-entry.at("grid").at(0).at("along").get<float>();buildGuide(r,entry,world);routes.push_back(std::move(r));
    }
    if(routes.size()!=3)throw std::runtime_error("Expected three race routes");return routes;
}
