#include "CaptionStream.h"
#include "World.h"
#include <iostream>
#include <set>

void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
captions::Visible visible(uint64_t id,uint32_t cls,float x,bool ego=false){return {{id,cls,{x,0,15},true,ego},ego?.1:.025,.5f,.5f};}
captions::Frame frame(uint64_t id,std::vector<captions::Visible> objects){return {id,id/40.,{{0,2,0},{0,2,1},90},40,std::move(objects)};}
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Pass repository root");const std::filesystem::path root=argv[1];
    captions::Stream stream(root/"assets/caption-catalog.json");
    const auto hero=visible(100,1247,0,true),left=visible(101,1243,-10),right=visible(102,1243,10),newCar=visible(103,1243,8);
    std::vector<captions::Frame> frames={frame(1,{hero,left,right}),frame(2,{hero,left,right})};
    auto a=stream.chunk(1,1,frames);std::string text=a.at("prompt");
    require(text.find("two compact station wagons")!=std::string::npos,"same-model instances retain count");
    require(text.find("one at bottom-left and one at bottom-right")!=std::string::npos,"training-style directional grouping");
    require(a.at("overview_entity_ids").size()==3&&a.at("events").empty(),"explicit ego and first-frame overview");
    require(text.find("100")==std::string::npos&&text.find("1247")==std::string::npos,"IDs never leak into text");
    frames={frame(3,{hero,left}),frame(4,{hero,left,newCar}),frame(5,{hero,left,newCar})};
    auto b=stream.chunk(1,2,frames);require(b.at("events").size()==2,"independent exit and enter for same model");
    require(b["events"][0]["entity_id"]=="102"&&b["events"][0]["type"]=="exits","correct instance exits");
    require(b["events"][1]["entity_id"]=="103"&&b["events"][1]["type"]=="enters","recycled identity enters");
    require(b["overview_entity_ids"]==nlohmann::json::array({"100","101"}),"late entrants not described as present at chunk start");
    frames={frame(6,{hero,left,newCar}),frame(7,{hero,left,newCar})};auto c=stream.chunk(1,3,frames);
    frames={frame(8,{hero,left,newCar}),frame(9,{hero,left,newCar})};auto d=stream.chunk(1,4,frames);
    require(d["text_changed"]==false&&c["prompt"]==d["prompt"],"identical text can reuse full encoding");
    frames={frame(10,{hero,left}),frame(11,{hero,left,newCar})};require(stream.chunk(1,5,frames)["events"].empty(),"one-frame occlusion is debounced");
    bool bad=false;try{frames={frame(13,{hero})};stream.chunk(1,6,frames);}catch(const std::exception&){bad=true;}
    require(bad,"discontinuous frame IDs rejected");stream.reset();frames={frame(1,{}),frame(2,{})};
    require(stream.chunk(2,1,frames)["prompt"]=="Daytime city, cloudy sky.","no invisible or guessed ego");
    stream.reset();std::vector<captions::Visible> many{hero};nlohmann::json catalog;std::ifstream catalogFile(root/"assets/caption-catalog.json");catalogFile>>catalog;
    for(const auto& [id,name]:catalog.at("objects").items()){many.push_back(visible(1000+many.size(),uint32_t(std::stoul(id)),0));if(many.size()>=70)break;}
    frames={frame(1,many)};auto large=stream.chunk(3,1,frames);require(large["estimated_tokens"].get<int>()<=300&&large["overview_entity_ids"].size()<many.size(),"long distinct descriptions are actually trimmed to budget");
    World world(root/"assets/bbox-assets.bin",16,20);world.enableCaptions(root);const auto& objects=world.drawObjects();const auto& entities=world.captionEntities();
    std::map<uint64_t,int> parts;std::set<uint64_t> ids;int egos=0;
    for(const auto& e:entities)if(e.id){require(ids.insert(e.id).second,"world identities unique");egos+=e.ego;}
    for(const auto& o:objects)if(o.instance.pad[0]>0){auto token=size_t(o.instance.pad[0]);require(token<entities.size(),"token indexes this frame only");++parts[entities[token].id];}
    require(egos==1&&!parts.empty(),"exactly one explicit ego");for(auto [id,n]:parts)require(n==5||n==6,"wheels/limbs share body identity");
    auto previous=parts;world.drawObjects();parts.clear();for(const auto& o:world.drawObjects())if(o.instance.pad[0]>0)++parts[world.captionEntities().at(size_t(o.instance.pad[0])).id];
    require(parts==previous,"entity identity stable across captures");
    std::cout<<"PASS: same-model tracking, grouping, enter/exit, debounce, budget, reset, world identities and part aggregation\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
