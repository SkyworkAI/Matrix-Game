#pragma once
#include "Collision.h"
#include "CameraInput.h"
#include <json.hpp>
using Json=nlohmann::json;

struct CombatInput {
    float forward=0,right=0,lookX=0,lookY=0;
    bool fire=false,ads=false,shield=false,crouch=false,jump=false,dash=false,reload=false,trap=false,cover=false,grenade=false;
    int weapon=-1;
};
struct CombatEvent {
    uint64_t id=0,step=0;
    double time=0;
    std::string type;
    V3 position;
    uint32_t actor=0;
    int weapon=-1;
    std::vector<uint32_t> indices,tokens,visibilityTokens;
    bool pointEvent=false,allowPositionOnly=false;
    V3 surfaceNormal;
};
class Encounter {
public:
    Encounter(const std::filesystem::path& root,const MapData& map,Renderer& renderer,uint32_t seed);
    ~Encounter();
    void advance(float dt,const CombatInput& input={});
    void setAutopilot(bool enabled);
    bool autopilot()const;
    void setRecording(bool enabled);
    void setCoverProbability(float probability);
    void prepareClip();
    Camera camera(int player)const;
    V3 playerPosition(int player)const;
    uint32_t playerToken(int player)const;
    const std::vector<DrawObject>& draws(float projectileAlpha=1);
    const std::vector<DrawObject>& mapDraws() const;
    const std::vector<CatalogObject>& catalog()const;
    std::vector<CombatEvent> takeEvents();
    bool hasEvents()const;
    Json eventState()const;
    bool eventVisible(const CombatEvent& event,const Camera& camera,const CapturedView& view,uint32_t observer)const;
    std::string action(int player)const;
    void hud(HudState& state,int player)const;
    bool recoveryRequested()const;
    bool recover();
    bool playerAlive()const;
    bool retry();
    uint64_t step()const;
    double seconds()const;
    Json diagnostics()const;
    Json sceneTest();
    Json mapSurfaceTest();
    Json selfTest();
    Json poseTest();
    Json uePoseTest();
    Json coverTest();
    Json trailTest();
    Json firingTest();
    Json enemyFiringTest();
    Json eventTest();
    Json loadoutTest();
    Json demoRulesTest();
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
