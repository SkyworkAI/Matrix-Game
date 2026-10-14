#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

// Encodes one view's frames on a worker thread and appends the Annex B H.264
// stream to a file. Screen capture cannot be used for the multi-window race:
// the X screen is capped at 2560 px wide, so the windows always overlap and a
// grab only ever sees the topmost one. Reading each window's own framebuffer
// and encoding it independently is what makes N simultaneous 1080p60
// recordings possible.
class ViewRecorder {
public:
    ViewRecorder(const std::filesystem::path& path,int width,int height,int fps,int bitrateKbps,int threads=2);
    ~ViewRecorder();
    void submit(std::vector<uint8_t>&& rgba,int64_t timestampMicroseconds);
    std::vector<uint8_t> takeReusable();
    void finish();
    uint64_t encoded()const;
    uint64_t dropped()const;
    double spanSeconds()const;
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
