#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
struct EncodedFrame {std::vector<uint8_t> bytes;bool key=false;};
class VideoEncoder {
public:
    VideoEncoder(int width,int height,int fps,const std::string& codec,int threads=2,const std::string& preset="veryfast");
    ~VideoEncoder();
    void setBitrate(int kbps);
    EncodedFrame encode(const std::vector<uint8_t>& rgba,int64_t timestamp,bool forceKey);
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
