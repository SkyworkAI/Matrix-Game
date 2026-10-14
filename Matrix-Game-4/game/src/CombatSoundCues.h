#pragma once
#include "CombatAudio.h"
#include "Encounter.h"

// Shared event mapping for local PCM and frame-synchronized browser playback.
inline std::vector<SoundCue> combatSoundCues(const std::vector<CombatEvent>& events,
        const Camera& camera,uint32_t player,double start,int samples,double speed=1){
    if(samples<=0||!std::isfinite(speed)||speed<=0)throw std::invalid_argument("Invalid sound interval");
    std::vector<SoundCue> result;
    const V3 direction=(camera.target-camera.eye).normalized(),right=cross({0,1,0},direction).normalized();
    int impacts=0;
    for(const auto& e:events){CombatSound sound;
        if(e.type=="WeaponFire")sound=e.weapon==2?CombatSound::Shotgun:e.weapon==1?CombatSound::Launcher:CombatSound::Pistol;
        else if(e.type=="Explosion")sound=CombatSound::Explosion;
        else if(e.type=="HitByBullet")sound=CombatSound::Body;
        else if(e.type=="BulletHitScene")sound=CombatSound::Stone;
        else if(e.type=="ShieldBlock")sound=CombatSound::Shield;
        else continue;
        if(sound==CombatSound::Body||sound==CombatSound::Stone||sound==CombatSound::Shield){if(impacts++>=8)continue;}
        const V3 delta=e.position-camera.eye;const float distance=delta.length();
        const bool own=e.actor==player&&e.type=="WeaponFire";
        float gain=own?1.f:1.f/(1+distance*.065f),pan=own?0.f:dot(delta.normalized(),right)*.8f;
        if(sound==CombatSound::Stone)gain*=.35f;
        const int sample=std::clamp(int(std::lround((e.time-start)/speed*CombatAudio::SampleRate)),0,samples-1);
        result.push_back({sound,sample,gain,pan});
    }
    return result;
}
