#include "VehicleAudio.h"
#include <SDL.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
double rms(const std::vector<float>& samples){double sum=0;for(float v:samples){require(std::isfinite(v)&&std::abs(v)<=1,"Non-finite or clipped output");sum+=v*v;}return std::sqrt(sum/samples.size());}
std::vector<float> render(VehicleAudioMixer& mixer,size_t count=12000){std::vector<float> samples(count*2);mixer.render(samples.data(),count);return samples;}
double difference(const std::vector<float>& a,const std::vector<float>& b){double sum=0;for(size_t i=0;i<a.size();++i)sum+=std::abs(a[i]-b[i]);return sum/a.size();}
double tireRoughness(const std::vector<float>& samples,const std::vector<float>& baseline){
    double energy=0,change=0,previous=0;
    for(size_t i=0;i<samples.size();i+=2){
        // Undo the soft limiter to isolate tire audio from the identical engine signal.
        const double tire=samples[i]/(1-std::abs(samples[i]))-baseline[i]/(1-std::abs(baseline[i]));
        energy+=tire*tire;if(i)change+=(tire-previous)*(tire-previous);previous=tire;
    }
    return change/std::max(energy,1e-12);
}

void queueTests(){
    VehicleAudioQueue queue;VehicleAudioFrame frame,out;
    for(int pass=0;pass<4;++pass){
        for(uint64_t i=0;i<64;++i){frame.frame=i;require(queue.push(frame),"Queue prematurely full");}
        require(!queue.push(frame),"Unbounded queue");
        for(uint64_t i=0;i<64;++i){require(queue.pop(out)&&out.frame==i,"Queue ordering/wraparound");}
        require(!queue.pop(out),"Queue should be empty");
    }
    std::thread producer([&](){for(uint64_t i=0;i<100000;++i){VehicleAudioFrame next;next.frame=i;next.rpm=float(i);next.addImpact(i+1,3);while(!queue.push(next))std::this_thread::yield();}});
    bool valid=true;
    for(uint64_t i=0;i<100000;++i){while(!queue.pop(out))std::this_thread::yield();valid&=out.frame==i&&out.rpm==float(i)&&out.impacts[0].id==i+1;}
    producer.join();require(valid,"Concurrent snapshot corruption");
    frame={};for(uint64_t i=1;i<=20;++i)frame.addImpact(i,float(i));
    require(frame.impactCount==8,"Impact collector not bounded");
    for(const auto& impact:frame.impacts)require(impact.speed>=13,"Impact collector did not retain strongest events");
    frame.addImpact(20,30);require(frame.impactCount==8,"Duplicate impact in frame");
}
}

int main(int argc,char** argv){
    try {
        if(argc!=2)throw std::runtime_error("Usage: VehicleAudioTests ASSET_DIRECTORY");
        const std::filesystem::path assets=argv[1];queueTests();
        VehicleAudioFrame frame;frame.frame=1;frame.groundedFrontWheels=frame.groundedRearWheels=2;
        VehicleAudioMixer idle(assets),rev(assets),skid(assets),airborne(assets),muted(assets,0);
        require(rms(render(idle))==0,"Sound before first presented frame");
        idle.submit(frame);muted.submit(frame);auto idleSamples=render(idle);
        require(rms(idleSamples)>.005,"Silent engine sample");require(rms(render(muted))==0,"Volume zero is not silent");
        frame.rpm=5500;frame.throttle=1;rev.submit(frame);auto revSamples=render(rev);
        require(difference(idleSamples,revSamples)>.02,"RPM/load does not change engine sound");
        frame.rpm=1000;frame.throttle=0;frame.rearLateralSlipSpeed=12;skid.submit(frame);auto skidSamples=render(skid);
        // Use a fresh idle mixer to align sample phases for channel-isolation checks.
        VehicleAudioMixer baseline(assets);VehicleAudioFrame quiet;quiet.frame=1;quiet.groundedFrontWheels=quiet.groundedRearWheels=2;baseline.submit(quiet);auto baselineSamples=render(baseline);
        require(difference(baselineSamples,skidSamples)>.005,"No tire sound while skidding");
        auto axleSamples=[&](bool front,float slip,int contacts=2){VehicleAudioMixer mixer(assets);auto probe=quiet;
            (front?probe.frontLateralSlipSpeed:probe.rearLateralSlipSpeed)=slip;
            (front?probe.groundedFrontWheels:probe.groundedRearWheels)=contacts;
            mixer.submit(probe);auto samples=render(mixer);rms(samples);return samples;};
        for(bool front:{false,true}){
            auto tireDifference=[&](float slip,int contacts=2){return difference(baselineSamples,axleSamples(front,slip,contacts));};
            const float threshold=front?3.f:1.f;
            for(float slip:{0.f,.5f,threshold-.001f,threshold})require(tireDifference(slip)<1e-7,"Tire sound at or below axle threshold");
            require(tireDifference(threshold+.01f)>1e-7,"No tire sound just above axle threshold");
            const double gentle=tireDifference(threshold+.1f),doubled=tireDifference(threshold+.2f),medium=tireDifference((threshold+8.f)*.5f),strong=tireDifference(8.f),capped=tireDifference(12.f);
            // Allow for the output soft limiter and small playback-rate changes.
            require(doubled>gentle*1.8&&doubled<gentle*2.2,"Tire gain should rise linearly above axle threshold");
            require(gentle<medium&&medium>strong*.35&&medium<strong*.7,"Midpoint tire gain is not near half strength");
            require(strong>.005&&capped>strong*.8&&capped<strong*1.2,"Tire sound missing or not capped above 8 m/s");
            require(tireDifference(12,0)<1e-7,"Other axle contact incorrectly enabled airborne axle audio");
            require(tireDifference(12,1)>.005,"One contacting wheel on the axle should be sufficient");
        }
        const auto frontSamples=axleSamples(true,8),rearSamples=axleSamples(false,8);
        require(difference(frontSamples,rearSamples)>.005,"Front and rear tire timbres are indistinguishable");
        require(tireRoughness(frontSamples,baselineSamples)<tireRoughness(rearSamples,baselineSamples)*.75,"Front rubbing sound is not lower-frequency than rear squeal");
        VehicleAudioMixer both(assets),fadeReference(assets);auto bothFrame=quiet;
        bothFrame.frontLateralSlipSpeed=bothFrame.rearLateralSlipSpeed=12;
        both.submit(bothFrame);fadeReference.submit(quiet);rms(render(both));render(fadeReference);
        bothFrame.frame=2;bothFrame.groundedFrontWheels=bothFrame.groundedRearWheels=0;both.submit(bothFrame);
        render(both,24000);render(fadeReference,24000);
        require(difference(render(both),render(fadeReference))<1e-7,"Axle sounds did not fade out after losing contact");
        bothFrame.frame=3;bothFrame.epoch=1;bothFrame.frontLateralSlipSpeed=bothFrame.rearLateralSlipSpeed=0;both.submit(bothFrame);
        fadeReference.submit(bothFrame);
        require(difference(render(both),render(fadeReference))<1e-7,"Reset retained an old tire sound");
        frame.groundedRearWheels=0;airborne.submit(frame);
        require(difference(baselineSamples,render(airborne))<1e-7,"Tire squeal while airborne");
        frame.frame=2;frame.muted=true;idle.submit(frame);render(idle);require(rms(render(idle))<1e-7,"Pause failed to fade to silence");

        VehicleAudioMixer impacts(assets);frame={};frame.frame=1;
        for(uint64_t i=1;i<=8;++i)frame.addImpact(i,10);
        impacts.submit(frame);require(impacts.activeImpacts()==4,"Impact voice limit");
        rms(render(impacts,96000));require(impacts.activeImpacts()==0,"Impact did not finish");
        frame.frame=2;impacts.submit(frame);require(impacts.activeImpacts()==0,"Duplicate event replayed on a different frame");
        frame.addImpact(50,50);frame.frame=1;impacts.submit(frame);require(impacts.activeImpacts()==0,"Stale frame played audio");
        frame.frame=3;impacts.submit(frame);require(impacts.activeImpacts()==1,"New collision missing");
        frame.epoch=1;frame.frame=0;frame.impactCount=0;impacts.submit(frame);require(impacts.activeImpacts()==0,"Reset retained old voices");
        frame.epoch=0;frame.frame=999;frame.addImpact(99,10);impacts.submit(frame);require(impacts.activeImpacts()==0,"Stale epoch accepted");
        frame.epoch=1;frame.frame=1;frame.muted=true;impacts.submit(frame);require(impacts.activeImpacts()==0,"Paused collision played");
        frame.frame=2;frame.impactCount=0;frame.muted=false;impacts.submit(frame);require(impacts.activeImpacts()==0,"Resume replayed paused collision");

        VehicleAudioMixer checkpoint(assets),silentCheckpoint(assets,0);VehicleAudioFrame cue;
        cue.frame=1;cue.checkpoint=1;cue.muted=true;
        checkpoint.submit(cue);silentCheckpoint.submit(cue);
        require(rms(render(checkpoint))>.01,"Missing checkpoint cue with vehicle audio muted at finish");
        require(rms(render(silentCheckpoint))==0,"Checkpoint cue ignored master volume zero");
        render(checkpoint);cue.frame=2;checkpoint.submit(cue);
        require(rms(render(checkpoint))==0,"Duplicate checkpoint cue replayed");
        cue.frame=3;cue.checkpoint=2;checkpoint.submit(cue);
        require(rms(render(checkpoint))>.01,"Next gate did not play a new cue");
        cue.frame=4;cue.checkpoint=0;checkpoint.submit(cue);render(checkpoint);
        require(rms(render(checkpoint))==0,"Cue continued beyond one shot");
        cue.frame=3;cue.checkpoint=3;checkpoint.submit(cue);
        require(rms(render(checkpoint))==0,"Stale display frame replayed a checkpoint");
        cue.frame=5;cue.checkpoint=3;checkpoint.submit(cue);render(checkpoint,512);
        cue.epoch=1;cue.frame=0;cue.checkpoint=0;checkpoint.submit(cue);
        require(rms(render(checkpoint))==0,"Reset retained the old checkpoint cue");
        cue.frame=1;cue.checkpoint=1;checkpoint.submit(cue);
        require(rms(render(checkpoint))>.01,"Retry failed to play first checkpoint again");
        render(checkpoint);cue.epoch=0;cue.frame=999;cue.checkpoint=4;checkpoint.submit(cue);
        require(rms(render(checkpoint))==0,"Old race epoch replayed a checkpoint cue");

        std::array<float,1024> block{};const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<5000;++i){frame.frame=uint64_t(i+10);frame.impactCount=0;frame.frontLateralSlipSpeed=frame.rearLateralSlipSpeed=15;frame.groundedFrontWheels=frame.groundedRearWheels=2;
            for(int j=0;j<4;++j)frame.addImpact(uint64_t(1000+i*4+j),20);
            impacts.submit(frame);impacts.render(block.data(),512);}
        const double milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/5000;
        std::cout<<"Offline mixer: "<<milliseconds<<" ms per 512 stereo frames (10.67 ms of audio, 4 impact voices).\n";

        if(SDL_Init(SDL_INIT_AUDIO)!=0)throw std::runtime_error(SDL_GetError());
        {VehicleAudio audio(assets,true);require(audio.enabled(),"Could not open test audio device");
            for(int i=0;i<20;++i){frame.frame=uint64_t(i+10000);audio.present(frame);SDL_Delay(2);}
            require(audio.droppedFrames()==0,"Unexpected audio queue overflow");}
        SDL_Quit();std::cout<<"Vehicle audio tests passed.\n";return 0;
    }catch(const std::exception& error){SDL_Quit();std::cerr<<error.what()<<'\n';return 1;}
}
