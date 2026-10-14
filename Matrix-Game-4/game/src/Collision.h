#pragma once
#include "Map.h"
#include <random>

struct RayHit { bool hit=false; float fraction=1; V3 point,normal; uint32_t token=0,part=0; };
class Collision {
public:
    explicit Collision(const MapData& map);
    ~Collision();
    RayHit ray(V3 from,V3 to,uint32_t ignore=0,bool staticOnly=false) const;
    RayHit sweepBullet(V3 from,V3 to,float radius,uint32_t ignore,bool staticOnly=false) const;
    bool overlapBox(const Matrix& volume,uint32_t ignore=0) const;
    void setObstacle(uint32_t token,const Matrix* collisionVolume,const Matrix* navigationVolume=nullptr);
    void addDebris(const Matrix& meshTransform,V3 localCenter,V3 localSize,V3 velocity,V3 angularVelocity);
    void stepDebris(float dt);
    void clearDebris();
    std::vector<Matrix> debris() const;
    size_t addCharacter(V3 feet);
    void teleport(size_t i,V3 feet);
    V3 move(size_t i,V3 velocity,float dt,bool& grounded);
    V3 characterPosition(size_t i) const;
    void setHitBoxes(size_t actor,uint32_t token,const std::vector<Matrix>& boxes,uint32_t shieldToken=0,const Matrix* shield=nullptr);
    bool project(V3 position,V3& nearest,float horizontal=4,float vertical=6) const;
    std::vector<V3> path(V3 from,V3 to) const;
    bool randomPoint(V3 center,float minRadius,float maxRadius,std::mt19937& rng,V3& point) const;
    V3 camera(V3 pivot,V3 desired) const;
    int navPolygons() const;
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
