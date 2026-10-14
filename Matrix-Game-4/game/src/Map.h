#pragma once
#include "Renderer.h"
#include "SceneConfig.h"
#include "RoadSurface.h"
#include <map>

struct Bounds {
    V3 min, max;
    V3 center() const { return (min + max) * .5f; }
};

struct MapObject {
    MeshData mesh;
    Bounds bounds;
    uint32_t catalogIndex=0, primitiveCount=0;
    bool tree=false,terrain=false,road=false;
};

struct MapData {
    SceneConfig scene;
    Bounds bounds, castleBounds;
    Bounds navigationBounds{};
    std::vector<V3> starts;
    bool hasNavigationBounds=false, catalogColors=false;
    std::vector<MapObject> objects;
    std::map<size_t,RoadSurfaceSplit> roadSurfaceSplits;
    uint32_t primitiveCount=0, treeCount=0, terrainCount=0, roadCount=0;
    uint64_t triangleCount=0;
    explicit MapData(const SceneConfig& scene);
    bool usesBatches() const { return objects.size()>2048; }
    void configureRenderer(Renderer& renderer) const;
    std::vector<DrawObject> upload(Renderer& renderer) const;
private:
    void filterCourtyardTrees();
    void loadRoadSurfaceOverrides();
};
