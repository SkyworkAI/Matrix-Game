#pragma once
#include "GameRuntime.h"
#include "LatencySimulation.h"

struct LatencyDemoOptions {
    latency::Mode mode=latency::Mode::Pulse;
    int chunkFrames=12;
    double referenceFps=40,inferenceMs=300,fixedMs=600,duration=0;
    std::filesystem::path inputScript,screenshot;
};
int runLatencyDemo(SDL_Window* window,Renderer& renderer,World& world,
    const Options& options,const std::filesystem::path& root,const LatencyDemoOptions& demo);
