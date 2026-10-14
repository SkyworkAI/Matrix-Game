#pragma once
#include "Common.h"

// Render-only partition; the original mesh remains authoritative for collision.
struct RoadSurfaceSplit {
    MeshData floor,structure;
};
RoadSurfaceSplit splitRoadSurface(const MeshData& mesh,float floorMaxY,float floorMinNormalY,
                                 float creaseDegrees,uint32_t structureColor);
