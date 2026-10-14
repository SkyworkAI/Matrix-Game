#include "CombatAudio.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

CombatAudio::CombatAudio(float gain):volume(gain){
    if(!std::isfinite(gain)||gain<0||gain>1)throw std::invalid_argument("Audio volume must be in [0,1]");
    constexpr float durations[]={.16f,.30f,.32f,.85f,.10f,.08f,.20f};
    constexpr float pitches[]={180,85,310,48,100,1300,1800};
    uint32_t rng=0xC4571E;
    for(size_t kind=0;kind<sounds.size();++kind){auto& wave=sounds[kind];wave.resize(size_t(durations[kind]*SampleRate));float low=0,phase=0;
        for(size_t i=0;i<wave.size();++i){
            rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
            const float noise=float(rng)/float(UINT32_MAX)*2-1,t=float(i)/SampleRate;
            low+=(noise-low)*(kind==3?.025f:.16f);
            phase+=2*std::numbers::pi_v<float>*pitches[kind]*(1-.65f*std::min(1.f,t/durations[kind]))/SampleRate;
            float signal=.65f*noise+.35f*std::sin(phase);
            if(kind==1)signal=.6f*noise+.8f*low+.4f*std::sin(phase);
            if(kind==2)signal=.25f*noise+.6f*std::sin(phase)+.2f*std::sin(phase*2.3f);
            if(kind==3)signal=2*low+.55f*std::sin(phase);
            if(kind==4)signal=low*2+.2f*std::sin(phase);
            if(kind==6)signal=.12f*noise+.5f*std::sin(phase)+.3f*std::sin(phase*1.47f);
            const float attack=std::min(1.f,t/.001f),envelope=std::exp(-6*t/durations[kind]);
            wave[i]=signal*attack*envelope;
        }
    }
}
void CombatAudio::reset(){voices={};}
void CombatAudio::play(const SoundCue& cue){
    const auto type=size_t(cue.sound);if(type>=sounds.size())return;
    auto it=std::find_if(voices.begin(),voices.end(),[](const auto& v){return !v.data;});
    if(it==voices.end())it=std::min_element(voices.begin(),voices.end(),[](const auto& a,const auto& b){return a.data->size()-a.cursor<b.data->size()-b.cursor;});
    const float pan=std::clamp(cue.pan,-1.f,1.f),gain=volume*std::clamp(cue.gain,0.f,1.f);
    *it={&sounds[type],0,gain*std::sqrt((1-pan)*.5f),gain*std::sqrt((1+pan)*.5f)};
}
std::vector<float> CombatAudio::render(int samples,const std::vector<SoundCue>& cues){
    if(samples<0)throw std::invalid_argument("Negative audio sample count");
    auto ordered=cues;std::stable_sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.sample<b.sample;});
    std::vector<float> result(size_t(samples)*2);size_t next=0;
    for(int i=0;i<samples;++i){while(next<ordered.size()&&ordered[next].sample<=i)play(ordered[next++]);
        float left=0,right=0;for(auto& voice:voices)if(voice.data){const float value=(*voice.data)[voice.cursor++];left+=value*voice.left;right+=value*voice.right;if(voice.cursor==voice.data->size())voice.data=nullptr;}
        result[i*2]=std::tanh(left);result[i*2+1]=std::tanh(right);
    }return result;
}
