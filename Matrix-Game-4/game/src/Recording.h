#pragma once
#include "Encounter.h"

class Recording {
public:
    Recording(const std::filesystem::path& directory,const std::filesystem::path& assets,uint32_t seed,const std::filesystem::path& sceneAssets={});
    ~Recording();
    void capture(Renderer& renderer,Encounter& encounter);
    void captureFinalEvents(Renderer& renderer,Encounter& encounter);
    void finish();
    int frames()const;
    const std::filesystem::path& directory()const;
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
