#include "CombatSoundCues.h"
#include <cassert>
int main(){
    const Camera camera{{0,0,0},{0,0,1},75};
    CombatEvent fire;fire.type="WeaponFire";fire.actor=1;fire.time=1.01;
    for(int weapon=0;weapon<3;++weapon){
        fire.weapon=weapon;const auto cues=combatSoundCues({fire},camera,1,1,1200);
        assert(cues.size()==1);const auto& c=cues[0];
        assert(c.sound==(weapon==2?CombatSound::Shotgun:weapon==1?CombatSound::Launcher:CombatSound::Pistol));
        assert(c.sample==480&&c.gain==1&&c.pan==0);
        assert(combatSoundCues({fire},camera,1,1,1000,2)[0].sample==240);
    }
    fire.actor=2;fire.position={10,0,0};
    const auto npc=combatSoundCues({fire},camera,1,1,1200)[0];assert(npc.gain<1&&npc.pan>.7);
    CombatEvent impact=fire;impact.type="BulletHitScene";
    assert(combatSoundCues(std::vector<CombatEvent>(20,impact),camera,1,1,1200).size()==8);
    const char* types[]={"Explosion","HitByBullet","BulletHitScene","ShieldBlock"};
    const CombatSound sounds[]={CombatSound::Explosion,CombatSound::Body,CombatSound::Stone,CombatSound::Shield};
    for(int i=0;i<4;++i){impact.type=types[i];assert(combatSoundCues({impact},camera,1,1,1200)[0].sound==sounds[i]);}
    impact.type="Unrelated";assert(combatSoundCues({impact},camera,1,1,1200).empty());
    CombatAudio audio,other;
    for(int i=0;i<int(CombatSound::Count);++i){
        const auto& wave=audio.waveform(CombatSound(i));assert(wave==other.waveform(CombatSound(i)));
        assert(wave.size()>1000);float energy=0;for(float value:wave){assert(std::isfinite(value)&&std::abs(value)<4);energy+=value*value;}assert(energy>1);
    }
}
