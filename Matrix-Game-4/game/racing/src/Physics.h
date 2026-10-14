#pragma once
#include "Common.h"
#include "CollisionEvents.h"
#include <optional>
struct CarModel {
    uint32_t bodyMesh=0;
    std::array<uint32_t,4> wheelMeshes{};
    std::array<V3,4> wheelPositions{};
    std::array<std::array<float,2>,4> wheelSizes{};
    V3 halfExtent,bodyCenter;
};
struct CarPose {
    Matrix body;std::array<Matrix,4> wheels;V3 velocity;
    float speed=0,heading=0;
    // Diagnostic angles are radians; lateral speed is m/s, yaw rate is rad/s.
    float lateralSpeed=0,yawRate=0,steerAngle=0,frontSlipAngle=0,rearSlipAngle=0;
    std::array<float,4> wheelAngularSpeed{};
    int groundedWheels=0,groundedFrontWheels=0,groundedRearWheels=0;
    // Audio slip is the maximum lateral speed (m/s) of contacting wheels on each axle.
    float engineRPM=1000,engineThrottle=0,frontLateralSlipSpeed=0,rearLateralSlipSpeed=0;
};
class Physics {
public:
    Physics();~Physics();
    void addBox(V3 center,V3 halfExtent,float yaw=0,bool ground=false);
    void addStaticMesh(const MeshData& mesh,const Matrix& matrix,bool ground=false,bool bridge=false);
    size_t addCar(const CarModel& model,const MeshData& body,V3 pos,float heading,float longitudinalImpulseScale=10);
    size_t addPedestrian(V3 pos);
    // A positive forward limit tapers throttle over the last 5 km/h; zero keeps legacy behavior.
    void drive(size_t index,Input input,bool steeringSpeedLimit=true,float forwardLimitKmh=0);
    void setCarLateralGrip(size_t index,float front,float rear);
    void setCarFrontDrive(size_t index,float frontShare);
    void setCarDetail(size_t index,float distance);
    void step(float dt);
    void resetCar(size_t index,V3 position,float heading);
    void setCarEnabled(size_t index,bool enabled);
    std::vector<CollisionEvent> takeCollisions();
    CarPose car(size_t index) const;
    V3 carPosition(size_t index) const;
    Matrix pedestrian(size_t index) const;
    bool pedestrianFallen(size_t index) const;
    void walk(size_t index,V3 velocity);
    void resetPedestrian(size_t index,V3 position);
    float groundHeight(V3 reference,bool streetOnly=false)const;
    std::optional<float> groundSupport(V3 reference)const;
    bool clearCarPlacement(size_t index,V3 position,float heading)const;
    V3 cameraCollision(V3 target,V3 desired) const;
    void optimize();
    uint32_t collisions() const;
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
