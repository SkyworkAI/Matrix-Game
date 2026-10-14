#include "VehicleAudio.h"
#include <SDL.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
float finiteClamp(float value,float low,float high){return std::isfinite(value)?std::clamp(value,low,high):low;}
float tireStrength(float slip,int contacts,float threshold){return contacts>0?finiteClamp((slip-threshold)/(8.f-threshold),0,1):0;}
float tirePlaybackRate(float slip,int contacts){return .95f+.1f*(contacts>0?finiteClamp((slip-1.5f)/9,0,1):0);}

std::vector<float> loadSample(const std::filesystem::path& path,bool loop){
    SDL_AudioSpec spec{};Uint8* bytes=nullptr;Uint32 length=0;
    if(!SDL_LoadWAV(path.string().c_str(),&spec,&bytes,&length))throw std::runtime_error(path.string()+": "+SDL_GetError());
    std::unique_ptr<Uint8,decltype(&SDL_FreeWAV)> storage(bytes,SDL_FreeWAV);
    SDL_AudioCVT convert{};
    if(SDL_BuildAudioCVT(&convert,spec.format,spec.channels,spec.freq,AUDIO_F32SYS,1,VehicleAudioMixer::SampleRate)<0)
        throw std::runtime_error(SDL_GetError());
    std::vector<Uint8> converted(size_t(length)*std::max(1,convert.len_mult));
    std::memcpy(converted.data(),bytes,length);convert.buf=converted.data();convert.len=int(length);
    if(SDL_ConvertAudio(&convert)<0)throw std::runtime_error(SDL_GetError());
    const size_t count=size_t(convert.len_cvt)/sizeof(float);
    if(count<32)throw std::runtime_error("Audio sample is empty or too short: "+path.string());
    std::vector<float> samples(count);std::memcpy(samples.data(),converted.data(),count*sizeof(float));
    for(float& value:samples)value=finiteClamp(value,-1,1);
    if(loop){
        // Overlap the tail with the head once at load time to remove loop seam clicks.
        const size_t overlap=std::min(size_t(1024),count/8),end=count-overlap;
        for(size_t i=0;i<overlap;++i){float mix=float(i)/float(overlap);samples[i]=samples[end+i]*(1-mix)+samples[i]*mix;}
        samples.resize(end);
    }
    return samples;
}

float readLoop(const std::vector<float>& samples,double& position,float rate){
    const size_t a=size_t(position),b=(a+1)%samples.size();const float mix=float(position-double(a));
    const float value=samples[a]+(samples[b]-samples[a])*mix;
    position+=rate;if(position>=samples.size())position-=double(samples.size());return value;
}
}

void VehicleAudioFrame::addImpact(uint64_t id,float speed){
    if(!id||!std::isfinite(speed)||speed<.7f)return;
    for(size_t i=0;i<impactCount;++i)if(impacts[i].id==id){impacts[i].speed=std::max(impacts[i].speed,speed);return;}
    if(impactCount<impacts.size())impacts[impactCount++]={id,speed};
    else {auto weakest=std::min_element(impacts.begin(),impacts.end(),[](auto a,auto b){return a.speed<b.speed;});
        if(speed>weakest->speed)*weakest={id,speed};}
}

bool VehicleAudioQueue::push(const VehicleAudioFrame& frame){
    static_assert(std::atomic<uint32_t>::is_always_lock_free);
    const uint32_t w=write.load(std::memory_order_relaxed);
    if(w-read.load(std::memory_order_acquire)>=Capacity)return false;
    frames[w%Capacity]=frame;write.store(w+1,std::memory_order_release);return true;
}
bool VehicleAudioQueue::pop(VehicleAudioFrame& frame){
    const uint32_t r=read.load(std::memory_order_relaxed);
    if(r==write.load(std::memory_order_acquire))return false;
    frame=frames[r%Capacity];read.store(r+1,std::memory_order_release);return true;
}

VehicleAudioMixer::VehicleAudioMixer(const std::filesystem::path& directory,float v):volume(finiteClamp(v,0,1)){
    engine=loadSample(directory/"engine-loop.wav",true);tire=loadSample(directory/"tire-loop.wav",true);
    frontTire=loadSample(directory/"tire-front-loop.wav",true);
    impactSamples[0]=loadSample(directory/"impact-medium.wav",false);impactSamples[1]=loadSample(directory/"impact-heavy.wav",false);
    // Precompute a short ascending chime; no synthesis or allocation in callback.
    checkpointSample.resize(size_t(SampleRate*.26));
    for(size_t i=0;i<checkpointSample.size();++i){
        const double t=double(i)/SampleRate;
        for(int note=0;note<2;++note){double age=t-note*.08;if(age<0)continue;
            double envelope=std::min(age/.005,1.)*std::exp(-age*25);
            checkpointSample[i]+=float(.32*envelope*std::sin(6.283185307179586*(note?1174.66:880)*age));}
    }
}
void VehicleAudioMixer::submit(const VehicleAudioFrame& frame){
    if(received&&(frame.epoch<state.epoch||(frame.epoch==state.epoch&&frame.frame<=state.frame)))return;
    if(!received||frame.epoch!=state.epoch){voices={};recentImpacts={};recentIndex=0;engineGain=tireGain=frontTireGain=master=0;enginePosition=tirePosition=frontTirePosition=0;
        lastCheckpoint=0;checkpointPosition=0;checkpointActive=false;}
    state=frame;received=true;
    if(frame.checkpoint>lastCheckpoint){lastCheckpoint=frame.checkpoint;checkpointPosition=0;checkpointActive=true;}
    if(state.muted){voices={};return;}
    for(size_t i=0;i<std::min(frame.impactCount,frame.impacts.size());++i){
        const auto impact=frame.impacts[i];if(!impact.id||!std::isfinite(impact.speed)||impact.speed<.7f)continue;
        if(std::find(recentImpacts.begin(),recentImpacts.end(),impact.id)!=recentImpacts.end())continue;
        recentImpacts[recentIndex++%recentImpacts.size()]=impact.id;
        auto voice=std::find_if(voices.begin(),voices.end(),[](const auto& v){return !v.active;});
        if(voice==voices.end())voice=std::min_element(voices.begin(),voices.end(),[](const auto& a,const auto& b){return a.gain<b.gain;});
        *voice={0,finiteClamp(.1f+impact.speed*.035f,.1f,.8f),.96f+float(impact.id%5)*.02f,impact.speed>=5?1:0,true};
    }
}
size_t VehicleAudioMixer::activeImpacts()const{return size_t(std::count_if(voices.begin(),voices.end(),[](auto v){return v.active;}));}
void VehicleAudioMixer::render(float* stereo,size_t frames){
    const bool audible=received&&!state.muted;
    const float rpm=finiteClamp(state.rpm,600,7000),throttle=finiteClamp(std::abs(state.throttle),0,1);
    const float targetRate=std::clamp(rpm/2200.f,.35f,3.f);
    const float targetEngine=.22f+.22f*throttle+.10f*rpm/6500;
    const float tireRate=tirePlaybackRate(state.rearLateralSlipSpeed,state.groundedRearWheels);
    const float frontRate=tirePlaybackRate(state.frontLateralSlipSpeed,state.groundedFrontWheels);
    const float slip=tireStrength(state.rearLateralSlipSpeed,state.groundedRearWheels,1.f);
    const float frontSlip=tireStrength(state.frontLateralSlipSpeed,state.groundedFrontWheels,3.f);
    for(size_t i=0;i<frames;++i){
        master+=((audible?volume:0)-master)*.003f;rate+=(targetRate-rate)*.0005f;
        engineGain+=(targetEngine-engineGain)*.001f;tireGain+=(slip*.42f-tireGain)*.0007f;
        frontTireGain+=(frontSlip*.42f*.6f-frontTireGain)*.0007f;
        float mixed=readLoop(engine,enginePosition,rate)*engineGain+readLoop(tire,tirePosition,tireRate)*tireGain
            +readLoop(frontTire,frontTirePosition,frontRate)*frontTireGain;
        for(auto& voice:voices)if(voice.active){
            const auto& samples=impactSamples[voice.sample];const size_t a=size_t(voice.position),b=std::min(a+1,samples.size()-1);
            mixed+=(samples[a]+(samples[b]-samples[a])*float(voice.position-double(a)))*voice.gain;
            voice.position+=voice.rate;if(voice.position>=samples.size())voice.active=false;
        }
        mixed*=master;
        if(checkpointActive){mixed+=volume*checkpointSample[checkpointPosition++];checkpointActive=checkpointPosition<checkpointSample.size();}
        mixed/=1+std::abs(mixed);stereo[2*i]=stereo[2*i+1]=mixed;
    }
}

struct VehicleAudio::Impl {
    VehicleAudioMixer mixer;
    VehicleAudioQueue queue;
    SDL_AudioDeviceID device=0;
    uint64_t dropped=0;
    explicit Impl(const std::filesystem::path& directory,float volume):mixer(directory,volume){}
    ~Impl(){if(device)SDL_CloseAudioDevice(device);}
    static void callback(void* user,Uint8* bytes,int length){
        auto& self=*static_cast<Impl*>(user);VehicleAudioFrame frame;
        for(int i=0;i<64&&self.queue.pop(frame);++i)self.mixer.submit(frame);
        self.mixer.render(reinterpret_cast<float*>(bytes),size_t(length)/(2*sizeof(float)));
    }
};
VehicleAudio::VehicleAudio(const std::filesystem::path& directory,bool enabled,float volume){
    if(!enabled||volume<=0)return;
    if(SDL_InitSubSystem(SDL_INIT_AUDIO)!=0){std::cerr<<"Audio disabled: "<<SDL_GetError()<<'\n';return;}
    try {
        auto candidate=std::make_unique<Impl>(directory,volume);
        SDL_AudioSpec spec{};spec.freq=VehicleAudioMixer::SampleRate;spec.format=AUDIO_F32SYS;spec.channels=2;spec.samples=512;
        spec.callback=Impl::callback;spec.userdata=candidate.get();
        candidate->device=SDL_OpenAudioDevice(nullptr,0,&spec,nullptr,0);
        if(!candidate->device)throw std::runtime_error(SDL_GetError());
        impl=std::move(candidate);SDL_PauseAudioDevice(impl->device,0);
        std::cout<<"Vehicle audio: engine, tire slip, impacts, checkpoint chime; 48 kHz stereo, volume "<<volume<<".\n";
    }catch(const std::exception& error){std::cerr<<"Audio disabled: "<<error.what()<<'\n';}
}
VehicleAudio::~VehicleAudio(){if(impl&&impl->dropped)std::cerr<<"Audio queue dropped "<<impl->dropped<<" presentation snapshots.\n";}
void VehicleAudio::present(const VehicleAudioFrame& frame){if(impl&&!impl->queue.push(frame))++impl->dropped;}
bool VehicleAudio::enabled()const{return bool(impl);}
uint64_t VehicleAudio::droppedFrames()const{return impl?impl->dropped:0;}
