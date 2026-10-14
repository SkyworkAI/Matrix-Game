#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

struct VehicleAudioImpact {uint64_t id=0;float speed=0;};

// A value snapshot, suitable for attaching to a displayed frame in a delayed pipeline.
struct VehicleAudioFrame {
    uint64_t frame=0,epoch=0;
    float rpm=1000,throttle=0,frontLateralSlipSpeed=0,rearLateralSlipSpeed=0;
    int groundedFrontWheels=0,groundedRearWheels=0;
    bool muted=false;
    // Highest gate crossed in this displayed frame, one-based; zero means no cue.
    // Race UI cues may play over results while vehicle audio is muted.
    uint64_t checkpoint=0;
    std::array<VehicleAudioImpact,8> impacts{};
    size_t impactCount=0;
    void addImpact(uint64_t id,float speed);
};

// One producer (presentation), one consumer (SDL callback). Never blocks either.
class VehicleAudioQueue {
public:
    bool push(const VehicleAudioFrame& frame);
    bool pop(VehicleAudioFrame& frame);
private:
    static constexpr uint32_t Capacity=64;
    std::array<VehicleAudioFrame,Capacity> frames{};
    std::atomic<uint32_t> write{0},read{0};
};

class VehicleAudioMixer {
public:
    static constexpr int SampleRate=48000;
    explicit VehicleAudioMixer(const std::filesystem::path& assetDirectory,float volume=.6f);
    void submit(const VehicleAudioFrame& frame);
    void render(float* stereo,size_t frames);
    size_t activeImpacts()const;
private:
    struct Voice {double position=0;float gain=0,rate=1;int sample=0;bool active=false;};
    std::vector<float> engine,tire,frontTire;
    std::vector<float> checkpointSample;
    size_t checkpointPosition=0;
    uint64_t lastCheckpoint=0;
    bool checkpointActive=false;
    std::array<std::vector<float>,2> impactSamples;
    std::array<Voice,4> voices{};
    std::array<uint64_t,64> recentImpacts{};
    size_t recentIndex=0;
    VehicleAudioFrame state{};
    bool received=false;
    float volume=0,engineGain=0,tireGain=0,frontTireGain=0,master=0,rate=1;
    double enginePosition=0,tirePosition=0,frontTirePosition=0;
};

class VehicleAudio {
public:
    VehicleAudio(const std::filesystem::path& assetDirectory,bool enabled,float volume=.6f);
    ~VehicleAudio();
    VehicleAudio(const VehicleAudio&)=delete;
    VehicleAudio& operator=(const VehicleAudio&)=delete;
    void present(const VehicleAudioFrame& frame);
    bool enabled()const;
    uint64_t droppedFrames()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
