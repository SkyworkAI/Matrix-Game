#pragma once
#include "CaptionTypes.h"
#include <nlohmann/json.hpp>
#include <map>
#include <span>
#include <optional>

namespace captions {
class Stream {
    struct Name {std::string shortName,full,kind;};
    struct Track {Visible last;int absent=0,present=0;bool announced=false;};
    std::map<uint32_t,Name> names;
    std::map<uint64_t,Track> tracks;
    std::string previousText;
    uint64_t lastFrame=0;
    double lastTime=-1;
    int tokenBudget;
    std::string direction(const Visible& v,const Camera& camera,bool distance)const;
public:
    nlohmann::json provenance;
    explicit Stream(const std::filesystem::path& catalog,int estimatedTokenBudget=300);
    void reset();
    nlohmann::json chunk(uint64_t epoch,uint64_t chunkId,std::span<const Frame> frames);
    static int estimatedTokens(const std::string& text);
    static nlohmann::json frameJson(const Frame& frame);
};
}
