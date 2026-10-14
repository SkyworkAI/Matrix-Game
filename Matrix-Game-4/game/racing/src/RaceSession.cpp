#include "RaceSession.h"

namespace {
float control(const nlohmann::json& j,const char* name,float low,float high){
    float value=j.value(name,0.f);if(!std::isfinite(value)||value<low||value>high)throw std::runtime_error("Invalid control value");return value;
}
}
RaceSession::RaceSession(World& w,std::vector<RaceRoute> r,bool chase):world(w),routes(std::move(r)){
    if(routes.empty())throw std::runtime_error("No race routes");world.enableMultiplayer();
    if(chase){pursuit.emplace(world);automaticRecovery=false;}
}
void RaceSession::setAutomaticRecovery(bool enabled){
    automaticRecovery=enabled;
    if(!enabled)for(auto& p:roster)if(p&&p->recoveryAutomatic){
        p->recovery.clear();p->recoveryAutomatic=false;
        if(p->notice=="RECOVERY - WAITING FOR SPACE")p->notice.clear();
    }
}
RaceSession::Player& RaceSession::player(int id){for(auto& p:roster)if(p&&p->id==id)return *p;throw std::runtime_error("Unknown player");}
void RaceSession::clearReady(){++pauseEpoch;for(auto& p:roster)if(p){p->ready=false;p->input={};}}
void RaceSession::pause(const std::string& reason){
    if(stage=="racing"||stage=="countdown"){resumeStage=stage;stage="paused";}
    if(stage=="paused"){pauseReason=reason;clearReady();}
}
bool RaceSession::allReady()const{
    bool any=false;for(const auto& p:roster)if(p){any=true;if(!p->connected||!p->visible||!p->ready)return false;}return any;
}
void RaceSession::prepareGrid(){
    if(pursuit&&roster[0]&&roster[1]){pursuit->reset(route());return;}
    std::vector<V3> positions;
    for(auto& p:roster)if(p){auto s=route().grid.at(p->slot);world.placePlayer(p->slot,s.ground,std::atan2(s.forward.x,s.forward.z));positions.push_back(s.ground);}
    world.clearStartingGrid(positions);
}
RaceSession::Json RaceSession::command(const Json& c){
    auto type=c.at("type").get<std::string>();
    if(type=="join"){
        if(pursuit&&roster[0]&&roster[1])throw std::runtime_error("Pursuit has one driver and one suspect");
        if(stage!="lobby"&&stage!="results")throw std::runtime_error("比赛进行中，请等待房主结束本局再加入");
        std::string name=c.at("name");if(name.empty()||name.size()>72)throw std::runtime_error("昵称长度不正确");
        for(size_t slot=0;slot<roster.size();++slot)if(!roster[slot]){
            Player p;p.id=nextId++;p.name=name;p.slot=slot;p.visible=c.value("visible",true);roster[slot]=p;
            world.setPlayerActive(slot,true);auto s=route().grid.at(slot);world.placePlayer(slot,s.ground,std::atan2(s.forward.x,s.forward.z));
            if(!host)host=p.id;clearReady();return {{"player_id",p.id},{"slot",slot}};
        }
        throw std::runtime_error("房间已满，最多 5 人");
    }
    auto& p=player(c.at("player_id"));
    if(type=="connected"){
        bool connected=c.at("value");bool changed=p.connected!=connected;p.connected=connected;p.ready=false;p.input={};p.visible=false;
        if(changed||connected)pause(connected?"有玩家重新连接，请全员确认准备":"有玩家掉线，等待重连");return {{"ok",true}};
    }
    if(!p.connected)throw std::runtime_error("Player is disconnected");
    if(type=="focus"){
        bool visible=c.at("value");bool changed=visible!=p.visible;p.visible=visible;
        if(!visible){p.ready=false;p.input={};if(changed)pause("有玩家切出页面，请全员恢复准备");}return {{"ok",true}};
    }
    if(type=="pause"){pause("玩家暂停，请全员确认后继续");return {{"ok",true}};}
    if(type=="leave"||type=="remove"){
        if(stage!="lobby"&&stage!="results")throw std::runtime_error("比赛中保留玩家，请先结束本局");
        int id=p.id;if(type=="remove"){if(p.id!=host)throw std::runtime_error("仅房主可移除玩家");id=c.at("target");}
        auto& target=player(id);size_t slot=target.slot;world.setPlayerActive(slot,false);roster[slot].reset();
        if(host==id){host=0;for(const auto& other:roster)if(other){host=other->id;break;}}
        if(!host){stage="lobby";pauseReason.clear();}
        clearReady();return {{"ok",true}};
    }
    if(type=="select_route"||type=="next_race"||type=="end_race"){
        if(p.id!=host)throw std::runtime_error("仅房主可操作");
        if(type=="end_race"){
            if(stage=="lobby"||stage=="results")throw std::runtime_error("尚无正在进行的比赛");stage="results";clearReady();
            for(auto& member:roster)if(member&&!member->progress.finished)member->notice=pursuit?"PURSUIT ENDED":"DNF - RACE ENDED";
        }else{
            if(stage!="lobby"&&stage!="results")throw std::runtime_error("请先结束本局");
            if(type=="select_route"){
                if(stage!="lobby")throw std::runtime_error("返回大厅后选择路线");
                auto id=c.at("route_id").get<std::string>();auto it=std::find_if(routes.begin(),routes.end(),[&](const auto& r){return r.id==id;});
                if(it==routes.end())throw std::runtime_error("Unknown route");selected=size_t(it-routes.begin());
            }
            stage="lobby";clearReady();for(auto& member:roster)if(member){member->progress={};member->recovery.clear();member->recoveryAutomatic=false;member->recoveryRequests.clear();member->notice.clear();}
            prepareGrid();
        }
        return {{"ok",true}};
    }
    if(c.value("race_id",uint64_t(-1))!=race||c.value("epoch",uint64_t(-1))!=pauseEpoch)throw std::runtime_error("状态已更新，请重新操作");
    if(type=="ready"){
        if(stage!="lobby"&&stage!="paused")throw std::runtime_error("当前无需准备");
        if(!p.visible)throw std::runtime_error("请先返回游戏页面");p.ready=c.value("value",true);
        if(allReady()){
            if(stage=="lobby"){
                if(pursuit&&(!roster[0]||!roster[1]))throw std::runtime_error("Pursuit needs a suspect");
                prepareGrid();++race;tickId=0;countdown=3;stage="countdown";
                for(auto& member:roster)if(member){member->progress={};member->input={};member->recovery.clear();member->recoveryAutomatic=false;member->recoveryRequests.clear();member->notice.clear();}
            }else{stage=resumeStage;pauseReason.clear();for(auto& member:roster)if(member)member->input={};}
        }
    }else if(type=="input"){
        if(!c.at("seq").is_number_unsigned())throw std::runtime_error("Invalid input sequence");
        uint64_t seq=c.at("seq");if(seq<=p.inputSequence)return {{"ok",true}};
        Input input{control(c,"throttle",-1,1),control(c,"steer",-1,1),control(c,"brake",0,1),control(c,"handbrake",0,1)};
        int camera=c.value("camera",0);if(camera<0||camera>2)throw std::runtime_error("Invalid camera");
        p.inputSequence=seq;p.camera=camera;p.lookBack=c.value("look_back",false);p.lastInput=now();
        p.input=stage=="racing"&&p.visible?input:Input{};
    }else if(type=="recover"){
        if(pursuit)throw std::runtime_error("Restart pursuit instead of teleporting vehicles");
        if(stage!="racing"&&stage!="paused")throw std::runtime_error("比赛开始后可使用救援");
        if(p.progress.finished)throw std::runtime_error("本局已完赛");
        auto id=c.at("request_id").get<std::string>();if(id.empty()||id.size()>80)throw std::runtime_error("Invalid recovery request");
        if(p.recoveryRequests.contains(id))return {{"ok",true},{"duplicate",true}};
        if(!p.recovery.empty())return {{"ok",true},{"pending",true}};
        if(p.recoveryRequests.size()>=1024)throw std::runtime_error("Too many recovery requests");
        p.recoveryRequests.insert(id);p.recovery=id;p.recoveryAutomatic=false;p.notice="RECOVERY PENDING";
    }else throw std::runtime_error("Unknown command");
    return {{"ok",true}};
}
void RaceSession::recover(Player& p){
    const auto& spots=p.progress.next?route().checkpoints.at(p.progress.next-1).recovery:route().grid;
    for(size_t i=0;i<spots.size();++i){auto s=spots[(p.slot+i)%spots.size()];if(!world.clearRecovery(p.slot,s.ground,std::atan2(s.forward.x,s.forward.z)))continue;
        world.placePlayer(p.slot,s.ground,std::atan2(s.forward.x,s.forward.z));p.progress.penalty+=5;p.input={};p.recovery.clear();p.recoveryAutomatic=false;p.notice="RECOVERED  +5 S";return;
    }
    p.notice="RECOVERY - WAITING FOR SPACE";
}
void RaceSession::tick(double dt){
    stepCollisions.clear();
    if(stage=="countdown"){countdown=std::max(0.,countdown-dt);if(countdown==0){stage="racing";for(auto& p:roster)if(p)p->ready=false;}return;}
    if(stage!="racing")return;
    std::array<Input,World::MaxPlayers> inputs{};std::array<V3,World::MaxPlayers> before{};
    for(auto& p:roster)if(p){
        if(automaticRecovery&&!p->progress.finished&&position(world.playerPose(p->slot).body).y< -20&&p->recovery.empty()){
            p->recovery="automatic";p->recoveryAutomatic=true;
        }
        if(!p->recovery.empty())recover(*p);
        before[p->slot]=position(world.playerPose(p->slot).body);
        inputs[p->slot]=p->progress.finished||!p->recovery.empty()||now()-p->lastInput>.35?Input{0,0,1,1}:p->input;
    }
    if(pursuit)inputs[1]=pursuit->drive(dt);
    world.updatePlayers(float(dt),inputs);++tickId;
    for(auto& p:roster)if(p)p->appliedInputSequence=p->inputSequence;
    // Contacts belong to this step; apply before freezing a finish score.
    stepCollisions=world.takeCollisions();
    if(pursuit){
        pursuit->advance(dt);
        for(auto& p:roster)if(p)p->progress.elapsed+=dt;
        if(pursuit->captured()){
            roster[0]->progress.finished=true;roster[0]->notice="SUSPECT CAPTURED";stage="results";clearReady();
        }
        return;
    }
    for(const auto& event:stepCollisions)for(auto& p:roster)if(p&&!p->progress.finished){
        uint64_t entity=world.playerCar(p->slot)+1;
        if(event.entityA!=entity&&event.entityB!=entity)continue;
        uint64_t other=event.entityA==entity?event.entityB:event.entityA;
        double penalty=other>=1000?5:other>0?3:2;p->progress.penalty+=penalty;
        p->notice="CONTACT "+std::to_string(event.id)+"  +"+std::to_string(int(penalty))+" S";
    }
    bool allFinished=true;
    for(auto& p:roster)if(p){p->progress.advance(route(),before[p->slot],position(world.playerPose(p->slot).body),dt);if(!p->progress.finished)allFinished=false;}
    if(allFinished){stage="results";clearReady();}
}
RaceSession::Json RaceSession::state()const{
    Json members=Json::array(),options=Json::array(),ranking=Json::array();std::vector<const Player*> finished;
    for(const auto& r:routes)options.push_back({{"id",r.id},{"name",r.name},{"description",r.description},{"length",r.length},{"checkpoints",r.checkpoints.size()}});
    for(const auto& p:roster)if(p){auto pose=world.playerPose(p->slot);auto pos=position(pose.body);
        members.push_back({{"id",p->id},{"slot",p->slot},{"name",p->name},{"connected",p->connected},{"visible",p->visible},{"ready",p->ready},{"next",p->progress.next},{"elapsed",p->progress.elapsed},{"penalty",p->progress.penalty},{"total",p->progress.total()},{"finished",p->progress.finished},{"recovery_pending",!p->recovery.empty()},{"notice",p->notice},{"position",{pos.x,pos.y,pos.z}},{"heading",pose.heading},{"speed",pose.speed},{"input_seq",p->inputSequence}});
        if(p->progress.finished)finished.push_back(&*p);
    }
    std::stable_sort(finished.begin(),finished.end(),[](auto a,auto b){return a->progress.total()<b->progress.total();});for(auto p:finished)ranking.push_back(p->id);
    return {{"type","state"},{"phase",stage},{"host_id",host},{"race_id",race},{"epoch",pauseEpoch},{"simulation_tick",tickId},{"countdown",countdown},{"pause_reason",pauseReason},{"route_id",route().id},{"routes",options},{"players",members},{"ranking",ranking},{"pursuit",pursuitState()}};
}
