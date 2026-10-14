#include "Renderer.h"
#include "Minimap.h"
#include <map>
#include <sstream>
#include <iomanip>
namespace {
const std::map<char,std::array<uint8_t,7>> Font={
 {'0',{14,17,19,21,25,17,14}},{'1',{4,12,4,4,4,4,14}},{'2',{14,17,1,2,4,8,31}},
 {'3',{30,1,1,14,1,1,30}},{'4',{2,6,10,18,31,2,2}},{'5',{31,16,16,30,1,1,30}},
 {'6',{6,8,16,30,17,17,14}},{'7',{31,1,2,4,8,8,8}},{'8',{14,17,17,14,17,17,14}},
 {'9',{14,17,17,15,1,2,12}},{'A',{14,17,17,31,17,17,17}},{'B',{30,17,17,30,17,17,30}},
 {'C',{14,17,16,16,16,17,14}},{'D',{30,17,17,17,17,17,30}},{'E',{31,16,16,30,16,16,31}},
 {'F',{31,16,16,30,16,16,16}},{'G',{14,17,16,23,17,17,15}},{'H',{17,17,17,31,17,17,17}},
 {'I',{14,4,4,4,4,4,14}},{'J',{7,2,2,2,2,18,12}},{'K',{17,18,20,24,20,18,17}},
 {'L',{16,16,16,16,16,16,31}},{'M',{17,27,21,21,17,17,17}},{'N',{17,25,21,19,17,17,17}},
 {'O',{14,17,17,17,17,17,14}},{'P',{30,17,17,30,16,16,16}},{'Q',{14,17,17,17,21,18,13}},
 {'R',{30,17,17,30,20,18,17}},{'S',{15,16,16,14,1,1,30}},{'T',{31,4,4,4,4,4,4}},
 {'U',{17,17,17,17,17,17,14}},{'V',{17,17,17,17,17,10,4}},{'W',{17,17,17,21,21,27,17}},
 {'X',{17,17,10,4,10,17,17}},{'Y',{17,17,10,4,4,4,4}},{'Z',{31,1,2,4,8,16,31}},
 {'.',{0,0,0,0,0,6,6}},{'-',{0,0,0,31,0,0,0}},{'/',{1,1,2,4,8,16,16}},
 {':',{0,6,6,0,6,6,0}},{'%',{17,2,4,4,8,16,17}},{'+',{0,4,4,31,4,4,0}}
};
}
void Renderer::rectangle(float x,float y,float w,float h,Color c) {
    ui.insert(ui.end(),{{x,y,c},{x+w,y,c},{x+w,y+h,c},{x,y,c},{x+w,y+h,c},{x,y+h,c}});
}
void Renderer::line(float x1,float y1,float x2,float y2,float thickness,Color c) {
    float dx=x2-x1,dy=y2-y1,l=std::sqrt(dx*dx+dy*dy);if(l<.01f)return;
    float ox=-dy/l*thickness*.5f,oy=dx/l*thickness*.5f;
    ui.insert(ui.end(),{{x1+ox,y1+oy,c},{x2+ox,y2+oy,c},{x2-ox,y2-oy,c},{x1+ox,y1+oy,c},{x2-ox,y2-oy,c},{x1-ox,y1-oy,c}});
}
void Renderer::text(float x,float y,const std::string& value,float scale,Color c) {
    for(char ch:value) { auto it=Font.find(ch);if(it!=Font.end())
        for(int row=0;row<7;++row)for(int col=0;col<5;++col)if(it->second[row]&(1<<(4-col)))rectangle(x+col*scale,y+row*scale,scale,scale,c);
        x+=6*scale;
    }
}
void Renderer::drawMinimap(const HudState& hud,int width,int height) {
    Color green=rgb(94,215,159),gold=rgb(246,190,48);
    const float half=hud.race?raceMapHalfSize(width,height):73.f;
    const float mx=width-half-24,my=height-half-24,pad=10,inner=half-pad;
    const V3 center=hud.race&&!hud.pursuit?hud.location:hud.mapCenter;
    const float range=hud.pursuit?std::max(100.f,hud.mapHalfSize)*1.04f:hud.race?RaceMapHalfRange:std::max(100.f,hud.mapHalfSize);
    rectangle(mx-half,my-half,2*half,2*half,{.025f,.04f,.04f,.92f});
    auto segment=[&](V3 a,V3 b,float thickness,Color color){
        auto p=localMapPoint(a,center,range),q=localMapPoint(b,center,range);
        if(clipMapLine(p,q))line(mx+p.x*inner,my+p.y*inner,mx+q.x*inner,my+q.y*inner,thickness,color);
    };
    if(hud.roads)for(const auto& road:*hud.roads)
        segment(road.a,road.b,road.freeway?1.8f:1.2f,road.freeway?rgb(126,122,106):rgb(91,104,109));
    if(hud.routeGuide&&!hud.finished){const auto& guide=*hud.routeGuide;
        const auto window=mapRouteWindow(guide,hud.nextCheckpoint,hud.navigation.along);
        if(window.valid){
            auto routeLine=[&](float begin,float end,float thickness,Color color){for(size_t i=1;i<guide.points.size();++i){
                float a=guide.distances[i-1],b=guide.distances[i];
                if(b<begin||a>end||b-a<.001f)continue;
                V3 delta=guide.points[i]-guide.points[i-1];
                segment(guide.points[i-1]+delta*std::clamp((begin-a)/(b-a),0.f,1.f),
                    guide.points[i-1]+delta*std::clamp((end-a)/(b-a),0.f,1.f),thickness,color);
            }};
            // The active leg always sits on top of the three-checkpoint preview.
            for(float at=window.gate;at<window.previewEnd;at+=20)
                routeLine(at,std::min(at+10,window.previewEnd),2,rgb(109,170,190));
            routeLine(window.begin,window.gate,7,{.02f,.03f,.03f,1});
            routeLine(window.begin,window.gate,3.5f,gold);
            // Map-only direction chevrons distinguish opposite traversals of a road.
            for(float at=window.begin+18;at<window.gate-12;at+=60){
                V3 point=mapRoutePoint(guide,at),direction=mapRoutePoint(guide,at+2)-mapRoutePoint(guide,at-2);
                direction.y=0;direction=direction.normalized();V3 side{direction.z,0,-direction.x};
                segment(point-direction*7+side*6,point,2,gold);
                segment(point-direction*7-side*6,point,2,gold);
            }
            auto p=localMapPoint(mapRoutePoint(guide,window.gate),center,range);
            if(std::abs(p.x)<=1&&std::abs(p.y)<=1){
                rectangle(mx+p.x*inner-5,my+p.y*inner-5,10,10,{.02f,.03f,.03f,1});
                rectangle(mx+p.x*inner-3,my+p.y*inner-3,6,6,gold);
                const float labelX=std::clamp(mx+p.x*inner+9,mx-inner,mx+inner-22);
                const float labelY=std::clamp(my+p.y*inner-14,my-inner,my+inner-9);
                rectangle(labelX-2,labelY-2,24,11,{.02f,.03f,.03f,1});
                text(labelX,labelY,std::to_string(hud.nextCheckpoint+1),1,gold);
            }
        }
    }
    auto p=localMapPoint(hud.location,center,range);
    float px=mx+std::clamp(p.x,-1.f,1.f)*inner,py=my+std::clamp(p.y,-1.f,1.f)*inner;
    float dx=std::sin(hud.heading),dy=-std::cos(hud.heading);
    line(px-dx*4,py-dy*4,px+dx*8,py+dy*8,3,green);
    line(px+dx*8,py+dy*8,px-dy*4,py+dx*4,2,green);
    line(px+dx*8,py+dy*8,px+dy*4,py-dx*4,2,green);
    if(hud.pursuit){
        auto target=localMapPoint(hud.suspectPosition,center,range);
        float sx=mx+std::clamp(target.x,-.97f,.97f)*inner,sy=my+std::clamp(target.y,-.97f,.97f)*inner;
        const Color red=rgb(255,102,105);float dx=std::sin(hud.suspectHeading),dy=-std::cos(hud.suspectHeading);
        rectangle(sx-6,sy-6,12,12,{.015f,.02f,.025f,1});
        line(sx,sy-5,sx+5,sy,2.5f,red);line(sx+5,sy,sx,sy+5,2.5f,red);
        line(sx,sy+5,sx-5,sy,2.5f,red);line(sx-5,sy,sx,sy-5,2.5f,red);
        line(sx,sy,sx+dx*12,sy+dy*12,2,red);
        const float labelScale=std::min(1.3f,half/110.f);
        rectangle(mx-half+5,my+half-25,2*half-10,21,{.025f,.04f,.04f,.95f});
        text(mx-half+10,my+half-19,"YOU",labelScale,green);
        text(mx-half+50,my+half-19,"SUSPECT",labelScale,red);
    }
    text(mx-half+8,my-half+6,"N",1,rgb(231,236,232));
    if(hud.race)text(mx+half-55,my+half-13,std::to_string(int(2*range))+" M",1,rgb(185,198,200));
}
void Renderer::buildHud(const HudState& hud) {
    buildHudLayout(hud,int(std::lround(width/hudScale)),int(std::lround(height/hudScale)));
    for(auto& vertex:ui){vertex.x*=hudScale;vertex.y*=hudScale;}
}
void Renderer::buildHudLayout(const HudState& hud,int width,int height) {
    ui.clear();
    if(!hud.visible)return;Color white=rgb(231,236,232),green=rgb(94,215,159);
    if(hud.visible) {
        float x=28,y=float(height)-86;
        rectangle(x-12,y-12,202,69,{.03f,.045f,.045f,.82f});
        std::ostringstream speed;speed<<std::setfill('0')<<std::setw(3)<<int(std::abs(hud.speed));
        text(x,y,speed.str(),5,white);text(x+105,y+20,"KM/H",2,white);
        text(x+169,y+7,hud.speed<-.5?"R":"D",3,green);
        drawMinimap(hud,width,height);
    }
    if(hud.pursuit){
        const float scale=std::min({1.f,width/1280.f,height/720.f}),x=24*scale,y=24*scale;
        const Color accent=hud.finished?green:rgb(255,102,105);
        rectangle(x,y,360*scale,130*scale,{.025f,.035f,.04f,.88f});rectangle(x,y,4*scale,130*scale,accent);
        text(x+16*scale,y+14*scale,hud.racePhase,2.5f*scale,accent);
        std::ostringstream distance;distance<<std::fixed<<std::setprecision(hud.suspectDistance<10?1:0)<<hud.suspectDistance<<" M";
        text(x+16*scale,y+42*scale,"SUSPECT  "+distance.str(),2*scale,white);
        std::ostringstream held;held<<std::fixed<<std::setprecision(1)<<hud.captureSeconds;
        text(x+16*scale,y+70*scale,"CAPTURE "+held.str()+" / 5.0 S",2*scale,hud.captureEligible||hud.finished?green:white);
        rectangle(x+16*scale,y+101*scale,328*scale,9*scale,{.15f,.2f,.2f,1});
        rectangle(x+16*scale,y+101*scale,328*scale*float(std::clamp(hud.captureSeconds/5,0.,1.)),9*scale,green);
        if(!hud.notice.empty())text(x+16*scale,y+144*scale,hud.notice,2*scale,white);
    }else if(hud.race){
        auto seconds=[](double value){std::ostringstream s;s<<std::fixed<<std::setprecision(2)<<value;return s.str();};
        float scale=std::min({1.f,width/1280.f,height/720.f});
        Color gold=rgb(246,190,48);float x=24*scale,y=24*scale;
        rectangle(x,y,382*scale,136*scale,{.025f,.035f,.04f,.85f});rectangle(x,y,4*scale,136*scale,gold);
        text(x+18*scale,y+16*scale,"P"+std::to_string(hud.playerNumber)+"  "+hud.racePhase,2*scale,gold);
        text(x+18*scale,y+44*scale,seconds(hud.elapsed+hud.penalty)+" S",4*scale,white);
        text(x+18*scale,y+84*scale,"TIME "+seconds(hud.elapsed)+"  +"+seconds(hud.penalty),2*scale,white);
        text(x+18*scale,y+110*scale,"GATE "+std::to_string(hud.nextCheckpoint)+" / "+std::to_string(hud.totalCheckpoints),2*scale,gold);
        const auto& nav=hud.navigation;
        if(!hud.finished&&nav.valid){
            float cx=width*.5f;rectangle(cx-174*scale,24*scale,348*scale,88*scale,{.025f,.035f,.04f,.86f});
            text(cx-154*scale,40*scale,"GATE "+std::to_string(hud.nextCheckpoint+1)+"  "+std::to_string(int(std::ceil(nav.distanceToGate)))+" M",2*scale,white);
            text(cx-154*scale,76*scale,"FINISH  "+std::to_string(int(std::ceil(nav.distanceToFinish)))+" M",2*scale,gold);
        }
        if(!hud.notice.empty()){rectangle(24*scale,174*scale,460*scale,34*scale,{.025f,.035f,.04f,.8f});text(38*scale,184*scale,hud.notice,2*scale,gold);}
    }
    if(hud.performance) {
        auto ms=[](double v) {std::ostringstream s;s<<std::fixed<<std::setprecision(2)<<v;return s.str();};
        rectangle(16,16,352,74,{.015f,.025f,.025f,.9f});
        text(28,27,"FRAME "+ms(hud.frameMs)+" MS",2,white);
        text(28,49,"CPU "+ms(hud.cpuMs)+"  GPU "+ms(hud.gpuMs),2,white);
        text(28,71,"OBJECTS "+std::to_string(visibleCount),1,green);
    }
    if(!hud.latencyStatus.empty()) {
        rectangle(16,16,420,18+float(hud.latencyStatus.size())*22,{.015f,.025f,.025f,.9f});
        for(size_t i=0;i<hud.latencyStatus.size();++i)
            text(28,27+float(i)*22,hud.latencyStatus[i],i==0?2.f:1.f,i==0?green:white);
    }
    if(!hud.menu.empty()) {
        rectangle(0,0,float(width),float(height),{0,0,0,.74f});
        RaceMenuLayout layout(width,height,hud.handling.has_value());
        float scale=layout.scale,px=layout.left,py=layout.top;
        text(px,py,hud.menuTitle,4*scale,white);
        if(!hud.menuDetail.empty())text(px,py+48*scale,hud.menuDetail,2*scale,white);
        for(size_t i=0;i<hud.menu.size();++i) {
            float y=layout.rowY(int(i));bool slider=hud.handling&&i<3;
            if(int(i)==hud.pauseSelection)rectangle(px-14*scale,y-14*scale,520*scale,(slider?72:42)*scale,{.15f,.25f,.21f,1});
            text(px,y,hud.menu[i],2*scale,int(i)==hud.pauseSelection?green:white);
            if(slider){
                const auto& settings=*hud.handling;std::ostringstream value;
                value<<std::fixed<<std::setprecision(2)<<settings.value(int(i));
                if(i==0)value<<" / "<<1-settings.frontDrive;
                text(px+(492-float(value.str().size())*12)*scale,y,value.str(),2*scale,white);
                float tx=layout.trackX(),ty=layout.trackY(int(i)),filled=layout.trackWidth()*settings.fraction(int(i));
                rectangle(tx,ty-2*scale,layout.trackWidth(),4*scale,rgb(100,116,121));
                rectangle(tx,ty-2*scale,filled,4*scale,green);
                rectangle(tx+filled-5*scale,ty-9*scale,10*scale,18*scale,white);
            }
        }
    } else if(hud.paused) {
        rectangle(0,0,float(width),float(height),{0,0,0,.66f});
        float px=width*.5f-150,py=height*.5f-112;
        text(px,py,"PAUSED",4,white);
        const char* labels[]={"RESUME","RESET VEHICLE","EXIT"};
        for(int i=0;i<3;++i) {
            float y=py+65+i*48;
            if(i==hud.pauseSelection)rectangle(px-10,y-12,305,38,{.15f,.25f,.21f,1});
            text(px,y,labels[i],2,i==hud.pauseSelection?green:white);
        }
    }
}
