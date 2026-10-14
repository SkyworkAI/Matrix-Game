#pragma once
#include "RaceSession.h"

// The local window owns the roster; player 1 is the manual driver.
class LocalRaceControls {
public:
    LocalRaceControls(RaceSession& session,std::vector<int> players):race(session),ids(std::move(players)) {
        if(ids.empty())throw std::runtime_error("Local race needs a player");
        for(const auto& p:race.players())if(p&&p->id==ids.front())sequence=p->inputSequence;
    }
    void lobby() {
        if(race.phase()!="lobby"&&race.phase()!="results")send(ids.front(),"end_race");
        if(race.phase()=="results")send(ids.front(),"next_race");
    }
    void start(const std::string& route) {
        lobby();send(ids.front(),"select_route",{{"route_id",route}});ready();
    }
    void pause() {
        if(race.phase()=="racing"||race.phase()=="countdown")send(ids.front(),"pause");
    }
    void resume() {
        if(race.phase()=="paused")ready();
    }
    bool recover() {
        if(race.isPursuit())return false;
        if((race.phase()!="racing"&&race.phase()!="paused")||race.countdownSeconds()>0)return false;
        for(const auto& p:race.players())if(p&&p->id==ids.front()&&(p->progress.finished||!p->recovery.empty()))return false;
        send(ids.front(),"recover",{{"request_id","local-"+std::to_string(++recovery)}});return true;
    }
    void drive(const Input& input,int camera,bool lookBack) {
        send(ids.front(),"input",{{"seq",++sequence},{"throttle",input.throttle},{"steer",input.steer},
            {"brake",input.brake},{"handbrake",input.handbrake},{"camera",camera},{"look_back",lookBack}});
    }
private:
    RaceSession& race;
    std::vector<int> ids;
    uint64_t sequence=0,recovery=0;
    void send(int id,const char* type,RaceSession::Json extra=RaceSession::Json::object()) {
        extra["type"]=type;extra["player_id"]=id;extra["race_id"]=race.raceId();extra["epoch"]=race.epoch();
        race.command(extra);
    }
    void ready() {
        for(int id:ids)send(id,"focus",{{"value",true}});
        for(int id:ids)send(id,"ready",{{"value",true}});
    }
};
