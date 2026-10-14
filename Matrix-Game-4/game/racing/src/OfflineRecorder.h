#pragma once
#include "VehicleAudio.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <memory>
#include <vector>

// A bounded, lossless recording queue. Saturation blocks the fixed-step producer.
class OfflineRecorder {
public:
    OfflineRecorder(const std::filesystem::path& directory,const std::filesystem::path& audioAssets,
        int width,int height,int fps,float volume,nlohmann::json metadata);
    ~OfflineRecorder();
    OfflineRecorder(const OfflineRecorder&)=delete;
    OfflineRecorder& operator=(const OfflineRecorder&)=delete;
    void input(nlohmann::json event);
    void submit(std::vector<uint8_t> scene,std::vector<uint8_t> ui,nlohmann::json state,VehicleAudioFrame audio);
    void finish();
    uint64_t frames()const;
    const std::filesystem::path& directory()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
