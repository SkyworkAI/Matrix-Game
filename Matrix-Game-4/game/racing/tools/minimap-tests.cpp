#include "../src/Minimap.h"
#include <iostream>

int main(){try{
    auto require=[](bool ok){if(!ok)throw std::runtime_error("Local minimap projection/clipping failed");};
    V3 center{140,10,-200};auto origin=localMapPoint(center,center);
    require(origin.x==0&&origin.y==0);
    auto north=localMapPoint(center+V3{0,20,100},center);
    auto east=localMapPoint(center+V3{100,0,0},center);
    require(north.x==0&&north.y==-.5f&&east.x==.5f&&east.y==0);
    require(localMapPoint(center+V3{200,0,0},center).x==1);
    require(localMapPoint(center+V3{0,0,200},center).y==-1);
    require(localMapPoint(center+V3{100,0,0},center,400).x==.25f);
    require(raceMapHalfSize(1280,720)==224&&raceMapHalfSize(1920,1080)==224);
    for(auto [width,height]:{std::pair{640,360},std::pair{640,480},std::pair{960,540},std::pair{1280,720}}){
        float half=raceMapHalfSize(width,height),scale=std::min({1.f,width/1280.f,height/720.f});
        require(half>(height<480?80.f:112.f));
        require(width-24-2*half>=std::max(218.f,484*scale)+16);
        require(height-24-2*half>=112*scale+12);
    }
    MapPoint a{-2,0},b{2,0};require(clipMapLine(a,b)&&a.x==-1&&b.x==1);
    a={0,-2};b={0,2};require(clipMapLine(a,b)&&a.y==-1&&b.y==1);
    a={2,-2};b={2,2};require(!clipMapLine(a,b));
    a={-2,-2};b={2,2};require(clipMapLine(a,b)&&a.x==-1&&a.y==-1&&b.x==1&&b.y==1);
    a={0,0};b={0,0};require(clipMapLine(a,b));
    a={2,0};b={2,0};require(!clipMapLine(a,b));
    a={-.5f,-.25f};b={.25f,.75f};require(clipMapLine(a,b)&&a.x==-.5f&&b.y==.75f);
    RouteGuide guide;
    guide.points={{0,0,0},{100,0,0},{0,0,0},{0,0,100},{0,20,0}};
    guide.distances={0,100,200,300,402};guide.checkpoints={100,200,300,402};
    auto outbound=mapRouteWindow(guide,0,50),inbound=mapRouteWindow(guide,1,150);
    require(outbound.valid&&outbound.begin==20&&outbound.gate==100&&outbound.previewEnd==402);
    require(inbound.valid&&inbound.begin==120&&inbound.gate==200&&inbound.previewEnd==402);
    // Same XY position, different progress: only the proper directed leg is gold.
    require(mapRoutePoint(guide,50).x==mapRoutePoint(guide,150).x);
    require(mapRoutePoint(guide,60).x>mapRoutePoint(guide,40).x);
    require(mapRoutePoint(guide,160).x<mapRoutePoint(guide,140).x);
    require(mapRouteWindow(guide,1,95).begin==100);
    require(mapRouteWindow(guide,3,350).previewEnd==402);
    require(!mapRouteWindow(guide,4,402).valid&&!mapRouteWindow(guide,-1,0).valid);
    require(std::abs(mapRoutePoint(guide,351).y-10)<1e-5);
    RouteGuide longGuide;
    for(int i=0;i<=10;++i){longGuide.points.push_back({float(i)*100,0,0});longGuide.distances.push_back(float(i)*100);if(i)longGuide.checkpoints.push_back(float(i)*100);}
    require(mapRouteWindow(longGuide,0,50).previewEnd==400);
    require(mapRouteWindow(longGuide,2,250).previewEnd==600);
    require(mapRouteWindow(longGuide,8,850).previewEnd==1000);
    require(mapRouteWindow(longGuide,9,950).previewEnd==1000);
    std::cout<<"Player-centered north-up map and bounded road segments passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
