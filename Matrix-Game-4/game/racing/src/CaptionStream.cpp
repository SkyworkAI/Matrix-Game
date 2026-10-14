#include "CaptionStream.h"
#include <set>
#include <zlib.h>

namespace captions {
using Json=nlohmann::json;
namespace {
constexpr double TrainingPixels=1280*720;
bool important(const Visible& v){return v.fraction*TrainingPixels>=(v.entity.dynamic?500:800);}
std::string countName(size_t n){const char* words[]={"zero","one","two","three","four","five","six","seven","eight","nine"};return n<10?words[n]:std::to_string(n);}
std::string plural(std::string s){return s+(s.ends_with("s")||s.ends_with("x")||s.ends_with("ch")||s.ends_with("sh")?"es":"s");}
}
Stream::Stream(const std::filesystem::path& catalog,int budget):tokenBudget(budget){
    std::ifstream file(catalog);if(!file)throw std::runtime_error("Missing caption catalog: "+catalog.string());
    Json data;file>>data;if(data.at("schema_version")!=1||budget<80||budget>500)throw std::runtime_error("Invalid caption catalog/budget");
    std::ifstream labels(catalog.parent_path()/"caption-labels.bin",std::ios::binary);if(!labels)throw std::runtime_error("Missing caption label sidecar");
    std::array<char,65536> buffer{};uLong crc=crc32(0,nullptr,0);
    while(labels){labels.read(buffer.data(),buffer.size());crc=crc32(crc,reinterpret_cast<const Bytef*>(buffer.data()),uInt(labels.gcount()));}
    if(!labels.eof()||uint32_t(crc)!=data.at("labels_crc32").get<uint32_t>())throw std::runtime_error("Caption catalog/label sidecar mismatch");
    for(const auto& [id,value]:data.at("objects").items())names.emplace(uint32_t(std::stoul(id)),Name{value.at("short"),value.at("full"),value.at("kind")});
    provenance=data;provenance.erase("objects");
}
void Stream::reset(){tracks.clear();previousText.clear();lastFrame=0;lastTime=-1;}
int Stream::estimatedTokens(const std::string& text){std::istringstream in(text);int n=0;std::string word;while(in>>word)++n;return int(n*1.33);}
std::string Stream::direction(const Visible& v,const Camera& camera,bool distance)const{
    // Match training's world-angle 9-way bins (not normalized screen thirds).
    V3 f=camera.target-camera.eye;f.y=0;f=f.normalized();V3 right=cross({0,1,0},f);
    V3 d=v.entity.position-camera.eye;float flat=std::hypot(d.x,d.z);
    float h=std::atan2(dot(d,right),dot(d,f))*180/Pi,a=std::atan2(d.y,flat)*180/Pi;
    std::string x=h<-12?"left":h>12?"right":"",y=a>2?"top":a<-6?"bottom":"";
    std::string p=y.empty()?x:x.empty()?y:y+"-"+x;if(p.empty())p="center";
    return (distance&&flat>=60?"far ":"")+p;
}
Json Stream::frameJson(const Frame& f){
    Json objects=Json::array();for(const auto& v:f.visible)objects.push_back({{"entity_id",std::to_string(v.entity.id)},
        {"caption_id",v.entity.captionId},{"role",v.entity.ego?"ego":v.entity.dynamic?"dynamic":"static"},
        {"position_m",{v.entity.position.x,v.entity.position.y,v.entity.position.z}},{"visible_fraction",v.fraction},{"screen_center",{v.screenX,v.screenY}}});
    return {{"frame_id",f.id},{"simulation_s",f.time},{"speed_kmh",f.speedKmh},
        {"camera",{{"eye",{f.camera.eye.x,f.camera.eye.y,f.camera.eye.z}},{"target",{f.camera.target.x,f.camera.target.y,f.camera.target.z}},{"horizontal_fov",f.camera.horizontalFov}}},
        {"objects",objects}};
}
Json Stream::chunk(uint64_t epoch,uint64_t chunkId,std::span<const Frame> frames){
    if(frames.empty())throw std::runtime_error("Empty caption chunk");
    struct Event {Visible v;std::string text;uint64_t frame;bool enter;};std::vector<Event> events;
    double speed=0;
    for(const auto& f:frames){
        if(!f.id||(lastFrame&&f.id!=lastFrame+1)||!std::isfinite(f.time)||f.time<=lastTime)throw std::runtime_error("Noncontiguous caption frames: reset on stream discontinuity");
        const bool initial=lastFrame==0;lastFrame=f.id;lastTime=f.time;speed+=std::abs(f.speedKmh);
        std::set<uint64_t> seen,allIds;
        for(const auto& v:f.visible){
            if(!v.entity.id||!allIds.insert(v.entity.id).second)throw std::runtime_error("Invalid/duplicate caption entity in one frame");
            seen.insert(v.entity.id);
            if(!names.contains(v.entity.captionId))throw std::runtime_error("Visible entity has no catalog description");
            const auto existing=tracks.find(v.entity.id);
            const bool retained=existing!=tracks.end()&&existing->second.announced&&v.fraction*TrainingPixels>=(v.entity.dynamic?250:400);
            if(!important(v)&&!retained){seen.erase(v.entity.id);continue;}
            auto& t=tracks[v.entity.id];t.last=v;t.absent=0;++t.present;
            if(initial)t.announced=true;
            if(!t.announced&&t.present>=2){t.announced=true;if(!v.entity.ego)events.push_back({v,names.at(v.entity.captionId).shortName+" enters at "+direction(v,f.camera,false)+".",f.id,true});}
        }
        for(auto it=tracks.begin();it!=tracks.end();){auto& [id,t]=*it;
            if(!seen.contains(id)){t.present=0;if(++t.absent>=2){
                if(t.announced&&!t.last.entity.ego)events.push_back({t.last,names.at(t.last.entity.captionId).shortName+" exits at "+direction(t.last,f.camera,false)+".",f.id,false});
                it=tracks.erase(it);continue;}}
            ++it;
        }
    }
    const auto& first=frames.front();std::vector<Visible> selected;std::optional<Visible> hero;
    for(const auto& v:first.visible)if(names.contains(v.entity.captionId)&&important(v)){
        if(v.entity.ego)hero=v;else if(v.fraction*TrainingPixels>=1200||v.entity.dynamic)selected.push_back(v);}
    const auto rank=[](const Visible& a,const Visible& b){return a.fraction!=b.fraction?a.fraction>b.fraction:a.entity.id<b.entity.id;};
    std::sort(selected.begin(),selected.end(),rank);
    // Fill empty directions from smaller but genuinely visible objects.
    std::set<std::string> covered;std::set<uint64_t> included;for(const auto& v:selected){covered.insert(direction(v,first.camera,false));included.insert(v.entity.id);}
    auto all=first.visible;std::sort(all.begin(),all.end(),rank);
    for(const auto& v:all)if(!v.entity.ego&&names.contains(v.entity.captionId)&&important(v)&&!included.contains(v.entity.id)&&covered.insert(direction(v,first.camera,false)).second)selected.push_back(v);
    std::sort(selected.begin(),selected.end(),rank);
    std::string prefix="Daytime city, cloudy sky. ";
    if(hero){prefix+="Center: player-controlled "+names.at(hero->entity.captionId).full;
        if(hero->fraction>=.06)prefix+=", dominating foreground";else if(hero->fraction>=.02)prefix+=", large in view";
        speed/=frames.size();prefix+=speed<2?", stationary. ":", ~"+std::to_string(int(std::round(speed/5)*5))+" km/h. ";}
    const auto compose=[&](){std::string text=prefix;std::vector<uint32_t> order;std::map<uint32_t,std::vector<Visible>> groups;
        for(const auto& v:selected){if(!groups.contains(v.entity.captionId))order.push_back(v.entity.captionId);groups[v.entity.captionId].push_back(v);}
        std::map<std::string,int> shortCounts;for(auto id:order)++shortCounts[names.at(id).shortName];
        if(!order.empty())text+="Around: ";
        for(size_t i=0;i<order.size();++i){if(i)text+=", ";const auto& group=groups.at(order[i]);const auto& name=names.at(order[i]);
            if(group.size()==1){const auto& v=group.front();text+=(direction(v,first.camera,false)=="center"||shortCounts[name.shortName]>1?name.full:name.shortName)+" at "+direction(v,first.camera,true);}
            else{ text+=countName(group.size())+" "+plural(name.shortName)+", ";std::map<std::string,size_t> bins;std::vector<std::string> directions;
                for(const auto& v:group){auto p=direction(v,first.camera,true);if(!bins.contains(p))directions.push_back(p);++bins[p];}
                for(size_t j=0;j<directions.size();++j){if(j)text+=j+1==directions.size()?" and ":", ";const auto& p=directions[j];text+=countName(bins[p])+" at "+p;}}}
        if(!order.empty())text+=".";
        for(const auto& event:events)text+="\n"+event.text;
        while(!text.empty()&&text.back()==' ')text.pop_back();return text;};
    std::string text=compose();
    // Keep a bounded event tail, then trim least-visible overview objects.
    while(estimatedTokens(text)>tokenBudget&&events.size()>2){events.pop_back();text=compose();}
    while(estimatedTokens(text)>tokenBudget&&!selected.empty()){selected.pop_back();text=compose();}
    while(estimatedTokens(text)>tokenBudget&&!events.empty()){events.pop_back();text=compose();}
    if(estimatedTokens(text)>tokenBudget)throw std::runtime_error("Ego description exceeds caption budget");
    Json entities=Json::array(),eventJson=Json::array();if(hero)entities.push_back(std::to_string(hero->entity.id));
    for(const auto& v:selected)entities.push_back(std::to_string(v.entity.id));
    for(const auto& e:events)eventJson.push_back({{"entity_id",std::to_string(e.v.entity.id)},{"caption_id",e.v.entity.captionId},{"frame_id",e.frame},{"type",e.enter?"enters":"exits"},{"text",e.text}});
    Json out={{"schema_version",1},{"epoch",epoch},{"chunk_id",chunkId},{"frame_start",first.id},{"frame_end",frames.back().id},
        {"frame_count",frames.size()},{"simulation_start_s",first.time},{"simulation_end_s",frames.back().time},
        {"prompt",text},{"text_changed",text!=previousText},{"estimated_tokens",estimatedTokens(text)},{"token_count_is_estimate",true},
        {"overview_entity_ids",entities},{"events",eventJson},{"overview_frame_id",first.id}};
    previousText=text;return out;
}
}
