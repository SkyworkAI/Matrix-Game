#pragma once
#include "Minimap.h"
#include "RaceSession.h"
#include <nlohmann/json.hpp>

namespace reference {
struct PresentationFrameInfo {
    uint64_t frame=0,inputId=0; // Frame is zero-based; IDs are never renumbered here.
    double inputReceivedMonotonic=0,captureMonotonic=0; // Seconds, same monotonic clock.
    int width=864,height=480; // Actual reference RGB dimensions, not browser dimensions.
    bool muted=false;
    // Stream-cumulative gates crossed, not a one-frame cue. Caller accumulates
    // RaceProgress::next deltas across substeps/retries and resets with the stream.
    uint64_t checkpoint=0;
};

// Pass all accumulated substep events; physics entity IDs are car index + 1.
inline nlohmann::json presentationImpacts(const std::vector<CollisionEvent>& events,uint64_t playerEntity) {
    auto result=nlohmann::json::array();
    for(const auto& event:events)
        if(event.entityA==playerEntity||event.entityB==playerEntity)
            result.push_back({{"id",event.id},{"speed",event.speed}});
    return result;
}

namespace presentation_detail {
using Json=nlohmann::json;
inline constexpr size_t MaxRoads=512,MaxActive=128,MaxPreview=128;
inline bool finite(V3 p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}

inline std::optional<std::array<float,4>> mapSegment(V3 a,V3 b,V3 center,float range=RaceMapHalfRange) {
    if(!finite(a)||!finite(b)||!finite(center))return {};
    auto p=localMapPoint(a,center,range),q=localMapPoint(b,center,range);
    if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(q.x)||!std::isfinite(q.y)||!clipMapLine(p,q))return {};
    if(std::hypot(q.x-p.x,q.y-p.y)<1e-7f)return {};
    return std::array<float,4>{std::clamp(p.x,-1.f,1.f),std::clamp(p.y,-1.f,1.f),
        std::clamp(q.x,-1.f,1.f),std::clamp(q.y,-1.f,1.f)};
}

inline Json makeMap(const std::vector<MapLine>& roads,V3 center,const RouteGuide* guide=nullptr,int next=0,float along=0,float range=RaceMapHalfRange,size_t roadLimit=MaxRoads) {
    Json result={{"roads",Json::array()},{"active",Json::array()},{"preview",Json::array()},{"gate",nullptr}};
    for(const auto& road:roads){
        if(result["roads"].size()==roadLimit)break;
        if(auto line=mapSegment(road.a,road.b,center,range))
            result["roads"].push_back({(*line)[0],(*line)[1],(*line)[2],(*line)[3],road.freeway?1:0});
    }
    if(!guide||!std::isfinite(along))return result;
    const auto window=mapRouteWindow(*guide,next,along);
    if(!window.valid)return result;
    auto routeLine=[&](float begin,float end,Json& lines,size_t limit){
        for(size_t i=1;i<guide->points.size()&&lines.size()<limit;++i){
            const float a=guide->distances[i-1],b=guide->distances[i];
            if(b<=begin||a>=end||b-a<.001f)continue;
            const V3 delta=guide->points[i]-guide->points[i-1];
            if(auto line=mapSegment(guide->points[i-1]+delta*std::clamp((begin-a)/(b-a),0.f,1.f),
                guide->points[i-1]+delta*std::clamp((end-a)/(b-a),0.f,1.f),center))lines.push_back(*line);
        }
    };
    routeLine(window.begin,window.gate,result["active"],MaxActive);
    // Match RendererHud: ten metres on, ten off, starting at the active gate.
    for(double at=window.gate;at<window.previewEnd&&result["preview"].size()<MaxPreview;at+=20)
        routeLine(float(at),float(std::min(at+10,double(window.previewEnd))),result["preview"],MaxPreview);
    const auto gate=localMapPoint(mapRoutePoint(*guide,window.gate),center);
    if(std::isfinite(gate.x)&&std::isfinite(gate.y)&&std::abs(gate.x)<=1&&std::abs(gate.y)<=1)
        result["gate"]={gate.x,gate.y};
    return result;
}

class GateProjector {
    struct Point {double x,y,z;};
    Camera camera;
    V3 forward{},right{},up{};
    double tx=0,ty=0;
    bool valid=false;
    Point view(V3 point)const {
        const double x=double(point.x)-camera.eye.x,y=double(point.y)-camera.eye.y,z=double(point.z)-camera.eye.z;
        return {x*right.x+y*right.y+z*right.z,x*up.x+y*up.y+z*up.z,x*forward.x+y*forward.y+z*forward.z};
    }
public:
    GateProjector(const Camera& value,int width,int height):camera(value) {
        if(width<=0||height<=0||!finite(camera.eye)||!finite(camera.target)||
            !std::isfinite(camera.horizontalFov)||camera.horizontalFov<=0||camera.horizontalFov>=180)return;
        forward=(camera.target-camera.eye).normalized();right=cross({0,1,0},forward).normalized();up=cross(forward,right);
        if(!finite(forward)||!finite(right)||forward.length()<.5f||right.length()<.5f)return;
        // Same world-up LH basis, horizontal FOV, and .08..2000 m clip range as RendererGL.
        tx=std::tan(camera.horizontalFov*Pi/360);ty=tx*double(height)/width;
        valid=std::isfinite(tx)&&std::isfinite(ty)&&tx>0&&ty>0;
    }
    std::optional<std::array<float,4>> segment(V3 from,V3 to)const {
        if(!valid||!finite(from)||!finite(to))return {};
        const auto a=view(from),b=view(to);
        double begin=0,end=1;
        auto plane=[&](double start,double finish){
            if(!std::isfinite(start)||!std::isfinite(finish))return false;
            if(start<0&&finish<0)return false;
            if(start<0)begin=std::max(begin,start/(start-finish));
            else if(finish<0)end=std::min(end,start/(start-finish));
            return begin<=end;
        };
        // Clip in camera space BEFORE dividing by depth. Endpoint clamping or
        // projecting negative depth mirrors rear gates and creates border lines.
        if(!plane(a.z-.08,b.z-.08)||!plane(2000-a.z,2000-b.z)||
            !plane(a.x+a.z*tx,b.x+b.z*tx)||!plane(a.z*tx-a.x,b.z*tx-b.x)||
            !plane(a.y+a.z*ty,b.y+b.z*ty)||!plane(a.z*ty-a.y,b.z*ty-b.y))return {};
        auto screen=[&](double t){
            const double x=a.x+(b.x-a.x)*t,y=a.y+(b.y-a.y)*t,z=a.z+(b.z-a.z)*t;
            return MapPoint{float(std::clamp(.5+.5*x/(z*tx),0.,1.)),float(std::clamp(.5-.5*y/(z*ty),0.,1.))};
        };
        const auto p=screen(begin),q=screen(end);
        if(std::hypot(q.x-p.x,q.y-p.y)<1e-7f)return {};
        return std::array<float,4>{p.x,p.y,q.x,q.y};
    }
};

inline Json gateSegments(const Checkpoint& gate,const Camera& camera,int width,int height) {
    auto result=Json::array();
    if(!finite(gate.center)||!finite(gate.forward)||!std::isfinite(gate.halfWidth)||
        !std::isfinite(gate.height)||gate.halfWidth<=0||gate.height<=0)return result;
    const V3 right=cross({0,1,0},gate.forward).normalized();
    if(right.length()<.5f)return result;
    GateProjector projector(camera,width,height);
    V3 previous=gate.center-right*gate.halfWidth+V3{0,.06f,0};
    auto append=[&](V3 point){
        if(auto line=projector.segment(previous,point))result.push_back(*line);
        previous=point;
    };
    // Centreline of RendererGL's two posts and twenty-segment rounded arch.
    const float shoulder=std::max(1.f,gate.height-1.2f);
    for(int i=0;i<=20;++i){const float angle=Pi-float(i)*Pi/20;
        append(gate.center+right*(std::cos(angle)*gate.halfWidth)+V3{0,shoulder+std::sin(angle)*1.2f,0});}
    append(gate.center+right*gate.halfWidth+V3{0,.06f,0});
    return result;
}
}

// Read-only snapshot of player slot 0. Call under the caller's world/race lock
// using the pose and actual camera captured for this RGB frame. No rendering,
// collision consumption, clocks, race progression, or recovery occurs here.
// speed_kmh is already km/h in CarPose; heading is radians. Map is north-up,
// player-centred [-1,1] over 400 m; gate screen coordinates have top-left origin.
// race.next is the zero-based next gate index / crossed count, total is gate count.
// Gate distance uses route navigation, falling back to straight-line metres only
// when no guide exists. A rear/offscreen gate has an empty segments array.
inline nlohmann::json makeGameFrame(const World& world,const Camera& camera,const CarPose& pose,
    const PresentationFrameInfo& info,const RaceSession* race=nullptr,
    const nlohmann::json& audioImpacts=nlohmann::json::array()) {
    using Json=nlohmann::json;
    if(!audioImpacts.is_array())throw std::invalid_argument("Presentation impacts must be a JSON array");
    const V3 location=position(pose.body);
    const RaceSession::Player* player=race&&race->players()[0]?&*race->players()[0]:nullptr;
    const RaceRoute* route=player?&race->route():nullptr;
    RouteNavigation navigation;
    if(player)navigation=navigateRoute(*route,player->progress,location,pose.heading,pose.speed);
    const bool pursuit=race&&race->isPursuit();
    const bool hasGate=player&&!pursuit&&!player->progress.finished&&player->progress.next<route->checkpoints.size();
    Json result={{"frame",info.frame},{"simulation_s",world.simulationTime},{"speed_kmh",pose.speed},
        {"heading",pose.heading},{"position",{location.x,location.y,location.z}},{"input_id",info.inputId},
        {"input_received_monotonic",info.inputReceivedMonotonic},{"capture_monotonic",info.captureMonotonic},
        {"audio",{{"rpm",pose.engineRPM},{"throttle",pose.engineThrottle},{"front_slip",pose.frontLateralSlipSpeed},
            {"rear_slip",pose.rearLateralSlipSpeed},{"front_contacts",pose.groundedFrontWheels},
            {"rear_contacts",pose.groundedRearWheels},{"muted",info.muted||(race&&race->phase()!="racing")},
            {"impacts",audioImpacts},{"checkpoint",info.checkpoint}}},
        {"race",nullptr},{"map",presentation_detail::makeMap(world.mapLines(),location,hasGate?&route->guide:nullptr,
            hasGate?int(player->progress.next):0,navigation.along)},{"gate",nullptr}};
    if(!player)return result;
    if(pursuit){
        result["pursuit"]=race->pursuitState();
        const V3 center=world.mapCenter();const float range=std::max(100.f,world.mapHalfSize())*1.04f;
        // Static full-city roads live once in the take manifest, not in every frame.
        result["map"]=presentation_detail::makeMap({},center);
        const auto p=localMapPoint(location,center,range);const auto target=result["pursuit"]["position"];
        const auto q=localMapPoint({target[0],target[1],target[2]},center,range);
        result["map"].update({{"center",{center.x,center.y,center.z}},{"half_range_m",range},
            {"player",{p.x,p.y}},{"suspect",{q.x,q.y}},{"north_up",true}});
    }
    const auto& progress=player->progress;
    result["race"]={{"phase",race->phase()},{"countdown",race->countdownSeconds()},{"elapsed",progress.elapsed},
        {"penalty",progress.penalty},{"next",progress.next},{"total",route->checkpoints.size()},
        {"finished",progress.finished},{"route_id",route->id},{"route_name",route->name},{"notice",player->notice}};
    if(pursuit)result["race"]["mode"]="pursuit";
    if(!hasGate)return result;
    const auto& gate=route->checkpoints[progress.next];
    result["gate"]={{"segments",presentation_detail::gateSegments(gate,camera,info.width,info.height)},
        {"distance_m",navigation.valid?navigation.distanceToGate:(gate.center-location).length()}};
    return result;
}

inline nlohmann::json makeGameFrame(const World& world,const Camera& camera,const CarPose& pose,
    const PresentationFrameInfo& info,const RaceSession* race,const std::vector<CollisionEvent>& collisions) {
    return makeGameFrame(world,camera,pose,info,race,presentationImpacts(collisions,race?world.playerCar(0)+1:1));
}
}
