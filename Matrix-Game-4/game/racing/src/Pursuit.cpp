#include "Pursuit.h"
#include <limits>
#include <queue>

namespace {
float flat(V3 v){return std::hypot(v.x,v.z);}
float pointSegment(V3 p,V3 a,V3 b){
    V3 d=b-a;d.y=0;p.y=a.y;
    return flat(p-(a+d*std::clamp(dot(p-a,d)/std::max(dot(d,d),1e-8f),0.f,1.f)));
}
bool inside(V3 p,const std::array<V3,4>& box){
    bool positive=false,negative=false;
    for(size_t i=0;i<4;++i){V3 a=box[i],b=box[(i+1)%4];float c=(b.x-a.x)*(p.z-a.z)-(b.z-a.z)*(p.x-a.x);
        positive|=c>1e-5f;negative|=c< -1e-5f;}
    return !(positive&&negative);
}
bool intersects(V3 a,V3 b,V3 c,V3 d){
    auto side=[](V3 a,V3 b,V3 p){return (b.x-a.x)*(p.z-a.z)-(b.z-a.z)*(p.x-a.x);};
    return side(a,b,c)*side(a,b,d)<0&&side(c,d,a)*side(c,d,b)<0;
}
}
float pursuitBodyGap(const std::array<V3,4>& a,const std::array<V3,4>& b){
    if(inside(a[0],b)||inside(b[0],a))return 0;
    float best=std::numeric_limits<float>::max();
    for(size_t i=0;i<4;++i)for(size_t j=0;j<4;++j){
        if(intersects(a[i],a[(i+1)%4],b[j],b[(j+1)%4]))return 0;
        best=std::min({best,pointSegment(a[i],b[j],b[(j+1)%4]),pointSegment(b[j],a[i],a[(i+1)%4])});
    }
    return best;
}

void Pursuit::reset(const RaceRoute& route){
    if(route.lanes.empty())throw std::runtime_error("Pursuit needs a connected road route");
    // Close the existing route through actual road links, never a straight-line seam.
    auto lanes=route.lanes;const auto first=lanes.front(),last=lanes.back();
    std::vector<int> parent(world.roadCount(),-1);std::queue<uint32_t> queue;
    parent.at(last)=int(last);queue.push(last);
    while(!queue.empty()&&parent.at(first)<0){
        auto lane=queue.front();queue.pop();
        for(auto next:world.roadLinks(lane)){
            if(parent.at(next)>=0||(world.roadFlags(next)&(2|4)))continue;
            const auto a=world.roadSample(lane,world.roadLength(lane)).position,b=world.roadSample(next,0).position;
            if((a-b).length()>3)continue;
            bool supported=true;
            for(float at=0;at<=world.roadLength(next);at+=5){const auto p=world.roadSample(next,at).position;auto y=world.groundSupport(p);
                if(!y||std::abs(*y-p.y)>.5f){supported=false;break;}}
            if(!supported)continue;
            parent[next]=int(lane);queue.push(next);
        }
    }
    if(parent.at(first)<0)throw std::runtime_error("Cannot close pursuit route on supported roads");
    std::vector<uint32_t> closing;
    for(auto lane=first;lane!=last;lane=uint32_t(parent.at(lane)))closing.push_back(lane);
    std::reverse(closing.begin(),closing.end());
    for(auto lane:closing)if(lane!=first)lanes.push_back(lane);
    points.clear();
    for(auto lane:lanes){
        const float length=world.roadLength(lane);
        for(float at=0;;at=std::min(length,at+5)){
            auto p=world.roadSample(lane,at).position;
            if(points.empty()||(p-points.back()).length()>.1f)points.push_back(p);
            if(at==length)break;
        }
    }
    if(points.size()<10||(points.front()-points.back()).length()>3)throw std::runtime_error("Invalid pursuit loop");
    if((points.front()-points.back()).length()<.1f)points.pop_back();
    const auto start=route.grid.at(0);float best=std::numeric_limits<float>::max();size_t startPoint=0;
    for(size_t i=0;i<points.size();++i){
        const float distance=(points[i]-start.ground).length();
        if(distance<best){best=distance;startPoint=i;}
    }
    // Measure forward along the road loop so a nearby spawn does not require a long detour.
    constexpr float desired=300.f,margin=30.f;
    float along=0;best=std::numeric_limits<float>::max();nearest=0;
    for(size_t offset=1;offset<points.size();++offset){
        const size_t i=(startPoint+offset)%points.size(),previous=(i+points.size()-1)%points.size();
        along+=flat(points[i]-points[previous]);
        if(along>desired+margin)break;
        if(along<desired-margin)continue;
        V3 forward=points[(i+1)%points.size()]-points[i];
        if(flat(forward)<1)continue;
        const float angle=std::atan2(forward.x,forward.z);
        auto support=world.groundSupport(points[i]);
        if(!support||std::abs(*support-points[i].y)>.45f||!world.playerPlacementClear(1,points[i],angle))continue;
        const float score=std::abs(along-desired);
        if(score<best){best=score;nearest=i;initialRouteDistance=along;}
    }
    if(best==std::numeric_limits<float>::max())throw std::runtime_error("No nearby supported pursuit spawn ahead on the route");
    V3 forward=points[(nearest+1)%points.size()]-points[nearest];
    world.placePlayer(0,start.ground,std::atan2(start.forward.x,start.forward.z));
    world.placePlayer(1,points[nearest],std::atan2(forward.x,forward.z));
    world.clearStartingGrid({start.ground,points[nearest]});
    world.setAutonomousPlayer(1,true);
    initialDistance=flat(points[nearest]-start.ground);stuck=reverse=0;progress={};advance(0);
}

Input Pursuit::drive(double dt){
    const auto pose=world.playerPose(1);const V3 here=position(pose.body);
    float best=std::numeric_limits<float>::max();size_t found=nearest;
    for(int offset=-2;offset<=25;++offset){
        const size_t i=(nearest+points.size()+offset)%points.size();
        const float distance=(points[i]-here).length();
        if(distance<best){best=distance;found=i;}
    }
    nearest=found;
    float speed=pose.speed/3.6f,lookAhead=std::clamp(7.5f+std::abs(speed)*.4f,8.f,19.f);
    size_t target=nearest;
    for(size_t k=0;k<25&&(points[target]-here).length()<lookAhead;++k)target=(target+1)%points.size();
    const V3 delta=points[target]-here;
    const float angle=std::remainder(std::atan2(delta.x,delta.z)-pose.heading,2*Pi);
    float desired=std::abs(angle)>.8f?3.f:std::abs(angle)>.22f?6.f:13.f;
    const float obstacle=world.obstacleAhead(1,22.f,.25f,true);
    desired=std::min(desired,std::max(0.f,(obstacle-5.5f)*.85f));
    if(reverse>0){reverse=std::max(0.,reverse-dt);return {-.4f,-std::clamp(angle*1.7f,-1.f,1.f),0,0};}
    if(std::abs(speed)<.4f&&obstacle>8)stuck+=dt;else stuck=0;
    if(stuck>3){reverse=1.5;stuck=0;}
    return {speed<desired?.55f:0,std::clamp(angle*1.7f,-1.f,1.f),speed>desired+.7f?.7f:0,0};
}
void Pursuit::advance(double dt){
    const auto a=world.playerPose(0),b=world.playerPose(1);
    V3 p=position(a.body),q=position(b.body);
    centerDistance=(p-q).length();gap=pursuitBodyGap(world.playerFootprint(0),world.playerFootprint(1));
    eligible=gap<=5&&std::abs(p.y-q.y)<2&&a.groundedWheels>0&&b.groundedWheels>0&&world.staticLineClear(p,q);
    progress.advance(dt,gap,eligible);
}
nlohmann::json Pursuit::state()const{
    const auto suspect=world.playerPose(1);const auto p=position(suspect.body);
    return {{"suspect_id",world.playerCar(1)+1},{"position",{p.x,p.y,p.z}},{"heading",suspect.heading},{"speed_kmh",suspect.speed},
        {"distance_m",gap},{"center_distance_m",centerDistance},{"initial_distance_m",initialDistance},
        {"initial_route_distance_m",initialRouteDistance},
        {"distance_kind","body_footprint"},{"capture_radius_m",5},{"capture_required_s",5},{"capture_held_s",progress.held},
        {"capture_eligible",eligible},{"captured",progress.captured}};
}
