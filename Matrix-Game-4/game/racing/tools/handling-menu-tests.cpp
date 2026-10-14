#include "../src/HandlingMenu.h"
#include <iostream>
#include <limits>

int main(){try{
    auto require=[](bool ok){if(!ok)throw std::runtime_error("Handling menu regression");};
    auto near=[](float a,float b){return std::abs(a-b)<1e-5f;};
    PlayerHandling tuning;
    require(near(tuning.frontDrive,.7f)&&near(tuning.frontGrip,1.15f)&&near(tuning.rearGrip,.9f));
    for(int row=0;row<3;++row){
        tuning.set(row,-10);require(near(tuning.value(row),PlayerHandling::minimum(row)));
        tuning.adjust(row,-1);require(near(tuning.fraction(row),0));
        tuning.set(row,10);require(near(tuning.value(row),PlayerHandling::maximum(row)));
        tuning.adjust(row,1);require(near(tuning.fraction(row),1));
        tuning.slide(row,.5f);float value=tuning.value(row);
        tuning.adjust(row,1);require(near(tuning.value(row),value+.01f));
        tuning.set(row,std::numeric_limits<float>::quiet_NaN());require(near(tuning.value(row),value+.01f));
    }
    for(auto [width,height]:{std::pair{1280,720},std::pair{640,360},std::pair{2560,1440}}){
        for(bool settings:{false,true}){
            RaceMenuLayout menu(width,height,settings);int count=settings?4:5;
            require(menu.top>0&&menu.rowY(count-1)+28*menu.scale<height);
            for(int row=0;row<count;++row){
                require(menu.hit(menu.left,menu.rowY(row),count)==row);
                require(menu.hit(menu.left-20*menu.scale,menu.rowY(row),count)==-1);
                if(settings&&row<3){
                    require(menu.hitTrack(menu.trackX()+menu.trackWidth()*.5f,menu.trackY(row),row));
                    require(near(menu.fraction(menu.trackX()+menu.trackWidth()*.5f),.5f));
                    require(menu.hit(menu.trackX(),menu.trackY(row),count)==row);
                    require(near(menu.fraction(menu.trackX()-200),0));
                    require(near(menu.fraction(menu.trackX()+menu.trackWidth()+200),1));
                }
            }
            require(!menu.hitTrack(menu.trackX(),menu.trackY(3),3));
        }
    }
    std::cout<<"Handling ranges, keyboard increments, slider bounds and responsive hit targets passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
