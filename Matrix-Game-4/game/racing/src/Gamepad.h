#pragma once
#include "Common.h"
#include <Xinput.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

class Gamepad {
public:
    Gamepad():worker([this]{poll();}){}
    ~Gamepad(){running=false;wake.notify_one();if(worker.joinable())worker.join();}
    Input read()const{return {throttle.load(),steer.load(),0,handbrake.load()};}
private:
    std::atomic<bool> running{true};
    std::atomic<float> throttle{0},steer{0},handbrake{0};
    std::mutex mutex;std::condition_variable wake;std::thread worker;
    void poll(){
        while(running){
            XINPUT_STATE state{};bool connected=XInputGetState(0,&state)==ERROR_SUCCESS;
            if(connected){const auto& g=state.Gamepad;float x=std::clamp(g.sThumbLX/32767.f,-1.f,1.f);
                steer=std::abs(x)>.12f?std::copysign((std::abs(x)-.12f)/.88f,x):0;
                float t=g.bRightTrigger/255.f-g.bLeftTrigger/255.f;throttle=std::abs(t)>.05f?t:0;
                handbrake=(g.wButtons&XINPUT_GAMEPAD_A)?1.f:0.f;
            }else{steer=0;throttle=0;handbrake=0;}
            // Disconnected XInput enumeration can block for hundreds of milliseconds.
            std::unique_lock lock(mutex);wake.wait_for(lock,std::chrono::milliseconds(connected?2:1000),[this]{return !running.load();});
        }
    }
};
