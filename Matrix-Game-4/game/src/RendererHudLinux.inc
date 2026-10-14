void Renderer::rectangle(float x,float y,float w,float h,Color c) {
    ui.insert(ui.end(),{{x,y,c},{x+w,y,c},{x+w,y+h,c},{x,y,c},{x+w,y+h,c},{x,y+h,c}});
}
void Renderer::line(float x1,float y1,float x2,float y2,float thickness,Color c) {
    float dx=x2-x1,dy=y2-y1,l=std::sqrt(dx*dx+dy*dy);if(l<.01f)return;
    float ox=-dy/l*thickness*.5f,oy=dx/l*thickness*.5f;
    ui.insert(ui.end(),{{x1+ox,y1+oy,c},{x2+ox,y2+oy,c},{x2-ox,y2-oy,c},{x1+ox,y1+oy,c},{x2-ox,y2-oy,c},{x1-ox,y1-oy,c}});
}
void Renderer::text(float x,float y,const std::string& value,float scale,Color c) {
    for(char ch:value) { auto it=Font.find(char(std::toupper(static_cast<unsigned char>(ch))));if(it!=Font.end())
        for(int row=0;row<7;++row)for(int col=0;col<5;++col)if(it->second[row]&(1<<(4-col)))rectangle(x+col*scale,y+row*scale,scale,scale,c);
        x+=6*scale;
    }
}
void Renderer::drawHud(const HudState& hud) {
    if(!hud.visible&&!hud.showFps&&!hud.combat)return;
    ui.clear();Color white=rgb(231,236,232),green=rgb(94,215,159);
    if(hud.showFps) {
        std::ostringstream label;
        label<<"FPS ";
        if(hud.fps>0&&std::isfinite(hud.fps)) label<<std::fixed<<std::setprecision(1)<<hud.fps;
        else label<<"--";
        rectangle(12,12,156,56,{.025f,.025f,.025f,.85f});
        text(22,22,label.str(),2,white);
        std::ostringstream timing;
        if(hud.fps>0&&std::isfinite(hud.fps)) timing<<std::fixed<<std::setprecision(2)<<1000/hud.fps<<" MS";
        else timing<<"-- MS";
        text(22,44,timing.str(),2,hud.fps>=1000.0/3?green:white);
    }
    if(hud.combat) {
        float cx=width*.5f,cy=height*.5f;
        line(cx-12,cy,cx-5,cy,1.5f,white);line(cx+5,cy,cx+12,cy,1.5f,white);
        line(cx,cy-12,cx,cy-5,1.5f,white);line(cx,cy+5,cx,cy+12,1.5f,white);
        float right=float(width)-(width<600?168:240);
        text(right,22,"P"+std::to_string(hud.player)+(hud.automatic?"  AUTO":"  MANUAL"),2,white);
        if(hud.recording){rectangle(right,48,8,8,rgb(255,73,73));text(right+16,46,"REC "+std::to_string(hud.recordFrames),2,rgb(255,100,100));}
        float y=float(height)-80-CombatUi::footerHeight(width);bool compact=width<600;
        float leftWidth=compact?136.f:210.f,weaponWidth=compact?float(width)-172:246.f,weaponX=float(width)-weaponWidth-12;
        rectangle(12,y,leftWidth,62,{.025f,.025f,.025f,.8f});text(22,y+10,"HP "+std::to_string(int(hud.health)),compact?1.5f:2.f,white);
        rectangle(22,y+30,leftWidth-20,5,{.2f,.2f,.2f,1});rectangle(22,y+30,(leftWidth-20)*std::clamp(hud.health/100,0.f,1.f),5,green);
        text(22,y+44,"SHIELD "+std::to_string(int(hud.shield)),compact?1.2f:1.5f,hud.raised?rgb(105,177,255):white);
        rectangle(weaponX,y,weaponWidth,62,{.025f,.025f,.025f,.8f});const char* names[]={"PISTOL","RIFLE","SHOTGUN"};
        text(weaponX+12,y+10,names[hud.weapon],compact?1.5f:2.f,white);text(float(width)-76,y+10,std::to_string(hud.ammo),compact?2.f:3.f,white);
        text(weaponX+12,y+36,hud.combatAction,1.5f,green);
        text(weaponX+12,y+50,"TRAP "+(hud.trapCooldown>0?std::to_string(int(std::ceil(hud.trapCooldown)))+" S":"READY"),1.2f,white);
        rectangle(weaponX,y-30,weaponWidth,26,{.025f,.025f,.025f,.8f});
        text(weaponX+12,y-22,"H COVER "+hud.coverStatus,1.3f,hud.coverDurability>0?green:white);
        if(!hud.coverHint.empty())text(weaponX-24,y-45,hud.coverHint,1.2f,rgb(255,200,120));
        if(width>=850)for(int i=0;i<3;++i){float x=cx-60+i*42;rectangle(x,y+25,34,3,hud.owned[i]?i==hud.weapon?white:green:rgb(66,66,66));text(x+12,y+8,std::to_string(i+1),1.6f,hud.owned[i]?white:rgb(88,88,88));}
    }
    if(hud.visible) {
        float x=28,y=float(height)-86;
        rectangle(x-12,y-12,202,69,{.03f,.045f,.045f,.82f});
        std::ostringstream speed;speed<<std::setfill('0')<<std::setw(3)<<int(std::abs(hud.speed));
        text(x,y,speed.str(),5,white);text(x+105,y+20,"KM/H",2,white);
        text(x+169,y+7,hud.speed<-.5?"R":"D",3,green);
        float mx=float(width)-153,my=float(height)-153;
        rectangle(mx-73,my-73,146,146,{.025f,.04f,.04f,.86f});
        float mapScale=62/std::max(100.f,hud.mapHalfSize);
        if(hud.roads)for(const auto& road:*hud.roads){
            line(mx+(road.a.x-hud.mapCenter.x)*mapScale,my-(road.a.z-hud.mapCenter.z)*mapScale,mx+(road.b.x-hud.mapCenter.x)*mapScale,my-(road.b.z-hud.mapCenter.z)*mapScale,road.freeway?1.7f:.8f,road.freeway?rgb(164,149,99):rgb(99,107,105));}
        float px=mx+std::clamp((hud.location.x-hud.mapCenter.x)*mapScale,-62.f,62.f),py=my-std::clamp((hud.location.z-hud.mapCenter.z)*mapScale,-62.f,62.f);
        float dx=std::sin(hud.heading),dy=-std::cos(hud.heading);
        line(px-dx*3,py-dy*3,px+dx*6,py+dy*6,3,green);rectangle(px-2,py-2,4,4,green);
        text(mx-62,my-64,"N",1,white);
    }
    if(hud.performance) {
        auto ms=[](double v) {std::ostringstream s;s<<std::fixed<<std::setprecision(2)<<v;return s.str();};
        rectangle(16,16,352,74,{.015f,.025f,.025f,.9f});
        text(28,27,"FRAME "+ms(hud.frameMs)+" MS",2,white);
        text(28,49,"CPU "+ms(hud.cpuMs)+"  GPU "+ms(hud.gpuMs),2,white);
        text(28,71,"OBJECTS "+std::to_string(visibleCount),1,green);
    }
    if(hud.combat)drawCombatOverlay(hud);
    if(hud.paused&&!hud.combat) {
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
    if(ui.empty()) return;
    if(ui.size()>MaxUiVertices)throw std::runtime_error("HUD buffer exceeded");
    submitHud();
}
#include "RendererCombatHud.inc"
