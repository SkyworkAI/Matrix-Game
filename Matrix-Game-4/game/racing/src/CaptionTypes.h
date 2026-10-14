#pragma once
#include "Common.h"

namespace captions {
// Stable entity identity is independent of both catalog class and GPU token.
struct Entity {
    uint64_t id=0;
    uint32_t captionId=0;
    V3 position;
    bool dynamic=false,ego=false;
};
struct Visible {
    Entity entity;
    double fraction=0;
    float screenX=.5f,screenY=.5f;
};
struct Frame {
    uint64_t id=0;
    double time=0;
    Camera camera;
    float speedKmh=0;
    std::vector<Visible> visible;
};
struct PixelStats {uint32_t token=0,pixels=0;float x=.5f,y=.5f;};
}
