#pragma once
#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace CombatUi {
constexpr float HelpScale=1.25f,HelpLineHeight=13;
inline std::vector<std::string> helpLines(int width) {
    constexpr std::array hints={"WASD MOVE","MOUSE LOOK","LMB FIRE","RMB AIM",
        "1/2/3 WEAPONS","R RELOAD","SPACE JUMP","SHIFT DASH","C CROUCH",
        "T MINE","F SHIELD","H COVER","F9 RECORD","ESC MENU"};
    std::vector<std::string> lines(1);
    for(const auto* hint:hints) {
        std::string next=lines.back().empty()?hint:lines.back()+"  "+hint;
        if(next.size()*6*HelpScale>width-32&&!lines.back().empty())lines.emplace_back(hint);
        else lines.back()=std::move(next);
    }
    return lines;
}
inline float footerHeight(int width) {return 8+helpLines(width).size()*HelpLineHeight;}
constexpr std::array MenuLabels={"RESUME","1 PISTOL","2 EXPLOSIVE RIFLE","3 SHOTGUN","EXIT"};
constexpr std::array DeathLabels={"RETRY","EXIT"};
struct Rect {
    float x,y,w,h;
    bool contains(float px,float py)const{return px>=x&&px<x+w&&py>=y&&py<y+h;}
};
inline Rect menuPanel(int width,int height) {
    float w=std::min(416.f,float(width)-24),h=216;
    return {(width-w)*.5f,(height-h)*.5f,w,h};
}
inline Rect menuItem(int width,int height,int index) {
    auto p=menuPanel(width,height);
    return {p.x+12,p.y+44+index*28,p.w-24,26};
}
inline int hitMenu(int width,int height,float x,float y,bool dead=false) {
    for(int i=0;i<int(dead?DeathLabels.size():MenuLabels.size());++i)if(menuItem(width,height,i).contains(x,y))return i;
    return -1;
}
inline int moveSelection(int current,int direction,bool dead=false) {
    const int count=int(dead?DeathLabels.size():MenuLabels.size());return (current+direction+count)%count;
}
}
