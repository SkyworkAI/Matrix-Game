#pragma once
#include "Navigation.h"

inline constexpr float RaceMapHalfRange=200.f;

inline float raceMapHalfSize(int width,int height){
    const float scale=std::min({1.f,width/1280.f,height/720.f});
    const float desired=height<480?160.f:224.f;
    // Double the panel, but reserve space for the left HUD and top navigation.
    const float leftHud=std::max(218.f,484*scale)+16,topHud=112*scale+12;
    return std::floor(std::min({desired,(width-24-leftHud)*.5f,(height-24-topHud)*.5f}));
}

struct MapRouteWindow {float begin=0,gate=0,previewEnd=0;bool valid=false;};
inline MapRouteWindow mapRouteWindow(const RouteGuide& guide,int next,float along){
    if(next<0||size_t(next)>=guide.checkpoints.size()||guide.points.size()<2||guide.distances.size()!=guide.points.size())return {};
    const float gate=guide.checkpoints[size_t(next)];
    const float previous=next>0?guide.checkpoints[size_t(next-1)]:0.f;
    const size_t previewGate=std::min(size_t(next)+3,guide.checkpoints.size()-1);
    // Progress, not nearest XY, disambiguates repeated streets and overpasses.
    return {std::max(previous,std::min(along-30,gate)),gate,guide.checkpoints[previewGate],true};
}
inline V3 mapRoutePoint(const RouteGuide& guide,float at){
    if(guide.points.empty()||guide.points.size()!=guide.distances.size())return {};
    auto it=std::lower_bound(guide.distances.begin(),guide.distances.end(),at);
    if(it==guide.distances.begin())return guide.points.front();
    if(it==guide.distances.end())return guide.points.back();
    size_t i=size_t(it-guide.distances.begin());float a=guide.distances[i-1],b=guide.distances[i];
    return guide.points[i-1]+(guide.points[i]-guide.points[i-1])*std::clamp((at-a)/std::max(b-a,.001f),0.f,1.f);
}

struct MapPoint {float x=0,y=0;};
inline MapPoint localMapPoint(V3 point,V3 center,float halfRange=RaceMapHalfRange) {
    return {(point.x-center.x)/halfRange,-(point.z-center.z)/halfRange};
}
// Liang-Barsky clipping keeps long road segments inside the map, rather than
// clamping their endpoints and creating spurious lines along its border.
inline bool clipMapLine(MapPoint& a,MapPoint& b) {
    float begin=0,end=1,dx=b.x-a.x,dy=b.y-a.y;
    auto edge=[&](float p,float q){
        if(std::abs(p)<1e-8f)return q>=0;
        float t=q/p;
        if(p<0)begin=std::max(begin,t);else end=std::min(end,t);
        return begin<=end;
    };
    if(!edge(-dx,a.x+1)||!edge(dx,1-a.x)||!edge(-dy,a.y+1)||!edge(dy,1-a.y))return false;
    b={a.x+end*dx,a.y+end*dy};a={a.x+begin*dx,a.y+begin*dy};return true;
}
