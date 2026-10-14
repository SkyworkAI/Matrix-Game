#pragma once
#include <json.hpp>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Adapted from the racing game's bounded lossless offline recorder.
class DemoRecorder {
public:
    DemoRecorder(const std::filesystem::path& path,int width,int height,int fps,nlohmann::json metadata);
    ~DemoRecorder();
    void input(nlohmann::json event);
    void submit(std::vector<uint8_t> scene,std::vector<uint8_t> ui,nlohmann::json state,std::vector<float> audio);
    void finish(const std::string& reason="stopped");
    uint64_t frames()const;
    const std::filesystem::path& directory()const;
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
