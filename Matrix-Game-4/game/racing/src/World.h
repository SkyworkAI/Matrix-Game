#pragma once
#include "Common.h"
#include "Physics.h"
#include "CaptionTypes.h"
#include <optional>
class Renderer;
class World {
public:
    static constexpr float PlayerMaxSpeedKmh=80.f;
    explicit World(const std::filesystem::path& assetPath,int trafficCount=64,int pedestrianCount=160,bool testTrack=false);
    void upload(Renderer& renderer);
    void uploadAll(const std::vector<Renderer*>& renderers);
    void update(float dt,Input input,bool autodrive=false);
    void setPlayerSteeringLimit(bool enabled){playerSteeringLimit=enabled;}
    void setPlayerLateralGrip(float front,float rear);
    void setPlayerFrontDrive(float share);
    void reset();
    Camera camera(int mode,bool lookBack=false)const;
    const std::vector<DrawObject>& drawObjects();
#ifndef _WIN32
    void enableCaptions(const std::filesystem::path& root);
#endif
    const std::vector<captions::Entity>& captionEntities()const{return frameEntities;}
    std::vector<DrawObject> outlineFixture()const;
    std::vector<DrawObject> nearClipFixture()const;
    bool collisionFixture();
    std::array<float,4> steeringFixture();
    void placeAtFreeway();
    void placeAtUnderpass();
    bool underpassFixture(const std::filesystem::path& root);
    void placeAtRampSide();
    bool rampGroundFixture(const std::filesystem::path& root);
    std::vector<std::pair<std::string,Camera>> rampSurveyCameras(const std::filesystem::path& root)const;
    bool allRampsFixture(const std::filesystem::path& root)const;
    Camera overviewCamera()const;
    CarPose player()const{return physics.car(0);}
    Input playerControls()const{return controls;}
    static constexpr size_t MaxPlayers=5;
    struct RoadSample {V3 position,forward;float width;};
    void enableMultiplayer();
    void setPlayerActive(size_t slot,bool active);
    CarPose playerPose(size_t slot)const;
    size_t playerCar(size_t slot)const{return traffic.at(playerTraffic.at(slot)).car;}
    Camera playerCamera(size_t slot,int mode=0,bool lookBack=false)const;
    void updatePlayers(float dt,const std::array<Input,MaxPlayers>& inputs);
    float obstacleAhead(size_t slot,float range,float halfAngle,bool includeParked=false)const;
    void setAutonomousPlayer(size_t slot,bool value){autonomousPlayers.at(slot)=value;}
    std::array<V3,4> playerFootprint(size_t slot)const;
    bool playerPlacementClear(size_t slot,V3 ground,float heading)const;
    bool staticLineClear(V3 a,V3 b)const{return (physics.cameraCollision(a,b)-b).length()<.01f;}
    void placePlayer(size_t slot,V3 ground,float heading);
    bool clearRecovery(size_t slot,V3 ground,float heading)const;
    void clearStartingGrid(const std::vector<V3>& ground);
    RoadSample roadSample(uint32_t lane,float along)const;
    const std::vector<uint32_t>& roadLinks(uint32_t lane)const{return lanes.at(lane).next;}
    float roadLength(uint32_t lane)const{return lanes.at(lane).length;}
    uint32_t roadFlags(uint32_t lane)const{return lanes.at(lane).flags;}
    size_t roadCount()const{return lanes.size();}
    std::optional<float> groundSupport(V3 p)const{return physics.groundSupport(p);}
    std::vector<CollisionEvent> takeCollisions(){return physics.takeCollisions();}
    uint32_t collisions()const{return physics.collisions();}
    size_t staticObjectCount()const{return sourceObjectCount;}
    size_t buildingCount()const{return originalBuildings;}
    size_t bridgeDeckCount()const{return originalDecks;}
    std::array<int,2> nearbyPopulation()const;
    std::array<int,2> populationMotionFixture();
    std::array<int,2> groundCoverageFixture(std::vector<std::array<float,4>>* misses=nullptr)const;
    const std::vector<MapLine>& mapLines()const{return mapRoads;}
    V3 mapCenter()const{return (cityMin+cityMax)*.5f;}
    float mapHalfSize()const{return std::max(cityMax.x-cityMin.x,cityMax.z-cityMin.z)*.5f;}
    double simulationTime=0;
    struct WorkTiming {
        uint64_t calls=0;double totalMs=0,maxMs=0;
        void add(double seconds){++calls;double ms=seconds*1000;totalMs+=ms;maxMs=std::max(maxMs,ms);}
    };
    // Cumulative simulation work, sampled by the server under its world lock.
    std::array<WorkTiming,3> workTimings{}; // NPC steering, physics, recycling
    float distance=0;
private:
    struct Building {uint32_t mesh;V3 size;};
    struct Lane {std::vector<V3> points;std::vector<float> cumulative;std::vector<uint32_t> next;float width=4,length=0;uint32_t flags=0;};
    struct Traffic {size_t car=0,model=0;uint32_t lane=0,nextLane=0,random=1;float along=0,stalled=0;bool parked=false,enabled=true;int playerSlot=-1;uint64_t captionEntity=0;};
    struct Walker {size_t body=0;uint32_t lane=0,random=1;float along=0,phase=0,speed=1.3f;uint64_t captionEntity=0;uint32_t captionId=0;};
    struct Parking {V3 p;float yaw;uint32_t model;};
    struct Spawn {uint32_t lane;float along;V3 p;float rankDistance=0;};
    Physics physics;
    std::vector<MeshData> meshes;
    std::vector<uint32_t> cityRenderMeshes,vehicleCaptionIds,personCaptionIds;
    std::vector<captions::Entity> staticEntities,frameEntities;
    uint64_t nextCaptionEntity=1;
    bool captionsEnabled=false;
    std::vector<CarModel> models;
    std::vector<Building> buildings;
    std::vector<DrawObject> statics,frameObjects;
    std::vector<Traffic> traffic;
    std::vector<Walker> people;
    std::vector<Lane> lanes;
    std::vector<uint32_t> carLanes,footLanes;
    std::vector<Parking> parking;
    std::vector<Spawn> carSpawns,footSpawns;
    std::vector<uint32_t> parkingSpawns;
    std::vector<MapLine> mapRoads;
    std::vector<V3> groundGapProbes;
    V3 cityMin,cityMax;
    V3 spawnCenter;
    bool testing=false;
    bool playerSteeringLimit=true;
    float playerFrontGrip=1,playerRearGrip=1;
    float playerFrontDrive=.5f;
    Input controls;
    std::array<size_t,MaxPlayers> playerTraffic{};
    std::array<bool,MaxPlayers> autonomousPlayers{};
    bool multiplayer=false;
    std::vector<V3> playerPositions()const;
    float playerDistance(V3 p)const;
    void advance(float dt);
    size_t sourceObjectCount=0,originalBuildings=0,originalDecks=0;
    uint32_t cubeMesh=0,leafMesh=0;
    void load(const std::filesystem::path& path);
    void build(int trafficCount,int pedestrianCount);
    void loadCity(const std::filesystem::path& path);
    void buildTestTrack();
    V3 lanePoint(uint32_t lane,float along)const;
    V3 laneForward(uint32_t lane,float along)const;
    float closestAlong(uint32_t lane,V3 p)const;
    uint32_t chooseNext(uint32_t lane,uint32_t& random)const;
    void refreshSpawns();
    std::optional<Spawn> findSpawn(bool pedestrian,uint32_t& random,bool recycling=false)const;
    void recyclePopulation();
    void box(V3 center,V3 size,Color color,bool collision=false,float yaw=0,float surface=0,float range=240);
    void object(uint32_t mesh,const Matrix& transform,Color color,V3 center,float radius,float range=300,float surface=0);
    Input driver(Traffic& car,float dt);
};
