#include "../src/CameraInput.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main(){
    constexpr float dt=1.f/120;
    constexpr float degreesPerPixel=CameraInput::RadiansPerPixel*180/std::numbers::pi_v<float>;
    for(int fps:{20,24,30,40,60,120}){
        const int ticks=120/fps;
        for(float sign:{-1.f,1.f}){
            const float step=CameraInput::pixelsPerTick(sign*1000,ticks,dt);
            assert(std::abs(step*ticks*fps*degreesPerPixel-sign*90)<.0001f);
        }
        const float small=CameraInput::pixelsPerTick(.5f,ticks,dt);
        assert(std::abs(small*ticks-.5f)<.00001f);
        assert(CameraInput::pixelsPerTick(0,ticks,dt)==0);
        for(int packetHz:{20,24,30,40,60,120})for(float sign:{-1.f,1.f}){
            CameraInput::NetworkAxis axis;float total=0,frameDegrees=0;
            for(int tick=0;tick<120;++tick){
                const double now=double(tick)/120;
                if(tick%(120/packetHz)==0)axis.add(sign*1000,now);
                const float applied=axis.consume(dt,now)*degreesPerPixel;
                total+=applied;frameDegrees+=std::abs(applied);
                if((tick+1)%ticks==0){assert(frameDegrees<=90.f/fps+.0001f);frameDegrees=0;}
            }
            assert(std::abs(total-sign*90)<.001f);
        }
        // Sub-limit displacement is conserved, including two packets coalesced
        // by the network; zero heartbeats must not discard pending movement.
        for(int packetHz:{20,60,120}){
            CameraInput::NetworkAxis axis;float total=0;
            for(int tick=0;tick<120;++tick){
                const double now=double(tick)/120;
                if(tick%(120/packetHz)==0){axis.add(50.f/packetHz,now);axis.add(50.f/packetHz,now);}
                axis.add(0,now);total+=axis.consume(dt,now);
            }
            assert(std::abs(total-100)<.001f);
        }
    }
    assert(CameraInput::pixelsPerTick(100,0,dt)==0);
    assert(CameraInput::pixelsPerTick(100,5,0)==0);
    assert(CameraInput::pixelsPerTick(100,5,dt)>0);
    assert(CameraInput::pixelsPerTick(-100,5,dt)<0);
    CameraInput::NetworkAxis burst;burst.add(10000,0);float total=0;
    for(int tick=0;tick<12;++tick)total+=burst.consume(dt,double(tick)/120);
    assert(std::abs(total*degreesPerPixel-4.5f)<.0001f);
    assert(burst.consume(dt,.2)==0);
    burst.add(1000,1);assert(burst.consume(dt,1.051)==0); // Paused game drops old motion.
    burst.add(1000,2);burst.add(-1,2.001);assert(burst.consume(dt,2.002)==-1); // No reversal lag.
    burst.add(1000,3);burst.clear();assert(burst.consume(dt,3)==0); // Focus loss.
    burst.add(1,4);assert(burst.consume(dt,4)==1);assert(burst.consume(dt,4.01)==0);
    std::cout<<"CAMERA_INPUT_TEST_PASSED\n";
}
