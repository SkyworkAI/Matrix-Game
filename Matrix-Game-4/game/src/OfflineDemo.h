#pragma once
#include <filesystem>
#include <cstdint>
#include <string>
struct DemoOptions {
    int width=1280,height=720;
    uint32_t seed=1729;
    std::string scene="castle";
    float volume=.6f;
    bool automaticRecord=false,noAudio=false,test=false;
    std::filesystem::path output;
};
int runOfflineDemo(const std::filesystem::path& root,const DemoOptions& options);
