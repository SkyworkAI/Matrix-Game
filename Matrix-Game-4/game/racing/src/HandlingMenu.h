#pragma once
#include "Common.h"

struct PlayerHandling {
    float frontDrive=.7f,frontGrip=1.15f,rearGrip=.9f;
    float value(int row)const{return row==0?frontDrive:row==1?frontGrip:rearGrip;}
    static float minimum(int row){return row==0?0.f:.9f;}
    static float maximum(int row){return row==0?1.f:1.2f;}
    void set(int row,float value){
        if(row<0||row>2||!std::isfinite(value))return;
        value=std::clamp(std::round(value*100)/100,minimum(row),maximum(row));
        (row==0?frontDrive:row==1?frontGrip:rearGrip)=value;
    }
    void adjust(int row,int steps){set(row,value(row)+float(steps)*.01f);}
    float fraction(int row)const{return std::clamp((value(row)-minimum(row))/(maximum(row)-minimum(row)),0.f,1.f);}
    void slide(int row,float fraction){set(row,minimum(row)+std::clamp(fraction,0.f,1.f)*(maximum(row)-minimum(row)));}
};

// Drawing and hit-testing share drawable-pixel geometry, including HiDPI windows.
struct RaceMenuLayout {
    float scale,left,top;
    bool handling;
    RaceMenuLayout(int width,int height,bool customize=false):
        scale(std::min({1.f,width/1280.f,height/720.f})),
        left(width*.5f-260*scale),top(height*.5f-(customize?184:150)*scale),handling(customize){}
    float rowY(int row)const{return top+((handling?80:90)+float(row)*(handling?82:52))*scale;}
    float trackX()const{return left;}
    float trackWidth()const{return 492*scale;}
    float trackY(int row)const{return rowY(row)+32*scale;}
    int hit(float x,float y,int count)const{
        if(x<left-14*scale||x>=left+506*scale)return -1;
        for(int row=0;row<count;++row){float top=rowY(row)-14*scale;
            if(y>=top&&y<top+((handling&&row<3)?72:42)*scale)return row;}
        return -1;
    }
    bool hitTrack(float x,float y,int row)const{
        return handling&&row>=0&&row<3&&x>=trackX()-8*scale&&x<=trackX()+trackWidth()+8*scale&&std::abs(y-trackY(row))<=14*scale;
    }
    float fraction(float x)const{return std::clamp((x-trackX())/trackWidth(),0.f,1.f);}
};
