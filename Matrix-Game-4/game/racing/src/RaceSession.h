#pragma once
#include "World.h"
#include "RaceRules.h"
#include "Pursuit.h"
#include <nlohmann/json.hpp>
#include <set>

class RaceSession {
public:
    using Json=nlohmann::json;
    struct Player {
        int id=0;size_t slot=0;std::string name;
        bool connected=true,visible=true,ready=false;
        Input input;uint64_t inputSequence=0,appliedInputSequence=0;
        double lastInput=0;
        int camera=0;bool lookBack=false;
        RaceProgress progress;
        std::string recovery,notice;
        bool recoveryAutomatic=false;
        std::set<std::string> recoveryRequests;
    };
    RaceSession(World& world,std::vector<RaceRoute> routes,bool chase=false);
    Json command(const Json& command);
    Json state()const;
    void tick(double dt);
    // Default on. Disable fall teleports when a consumer cannot reset its history.
    // Cancels queued automatic recovery only; manual requests remain available.
    void setAutomaticRecovery(bool enabled);
    const RaceRoute& route()const{return routes.at(selected);}
    const std::array<std::optional<Player>,World::MaxPlayers>& players()const{return roster;}
    const std::string& phase()const{return stage;}
    bool isPursuit()const{return pursuit.has_value();}
    Json pursuitState()const{return pursuit?pursuit->state():Json(nullptr);}
    uint64_t simulationTick()const{return tickId;}
    uint64_t raceId()const{return race;}
    uint64_t epoch()const{return pauseEpoch;}
    double countdownSeconds()const{return countdown;}
    const std::vector<CollisionEvent>& collisions()const{return stepCollisions;}
private:
    World& world;
    std::optional<Pursuit> pursuit;
    std::vector<RaceRoute> routes;
    std::array<std::optional<Player>,World::MaxPlayers> roster;
    std::vector<CollisionEvent> stepCollisions;
    size_t selected=0;
    int nextId=1,host=0;
    uint64_t race=0,pauseEpoch=1,tickId=0;
    double countdown=3;
    bool automaticRecovery=true;
    std::string stage="lobby",resumeStage="racing",pauseReason;
    Player& player(int id);
    void pause(const std::string& reason);
    void clearReady();
    void prepareGrid();
    bool allReady()const;
    void recover(Player& p);
};
