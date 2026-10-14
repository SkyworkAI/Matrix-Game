#pragma once
#include "GameRuntime.h"
#include "ReferenceStream.h"

struct ReferenceDemoOptions {
    reference::Settings stream;
    double duration=0;
    bool captions=false,record=false;
    int captionWidth=432;
    std::filesystem::path inputScript,screenshot;
};
int runReferenceDemo(SDL_Window* window,Renderer& renderer,World& world,
    const Options& options,const std::filesystem::path& root,const ReferenceDemoOptions& demo);
