#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

enum class CombatSound {Pistol,Shotgun,Launcher,Explosion,Body,Stone,Shield,Count};
struct SoundCue {CombatSound sound;int sample=0;float gain=1,pan=0;};

// The same PCM block feeds the local audio device and the lossless recording.
class CombatAudio {
public:
    static constexpr int SampleRate=48000;
    explicit CombatAudio(float volume=.6f);
    std::vector<float> render(int samples,const std::vector<SoundCue>& cues);
    void reset();
    const std::vector<float>& waveform(CombatSound sound)const{return sounds.at(size_t(sound));}
private:
    struct Voice {const std::vector<float>* data=nullptr;size_t cursor=0;float left=0,right=0;};
    std::array<std::vector<float>,size_t(CombatSound::Count)> sounds;
    std::array<Voice,32> voices{};
    float volume;
    void play(const SoundCue& cue);
};
