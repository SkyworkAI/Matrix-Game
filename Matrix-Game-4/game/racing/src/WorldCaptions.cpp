#include "World.h"
#include <zlib.h>

namespace {
uint32_t checksum(const std::filesystem::path& path){
    std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("Missing caption source asset: "+path.string());
    std::array<char,65536> buffer{};uLong crc=crc32(0,nullptr,0);
    while(file){file.read(buffer.data(),buffer.size());crc=crc32(crc,reinterpret_cast<const Bytef*>(buffer.data()),uInt(file.gcount()));}
    if(!file.eof())throw std::runtime_error("Cannot read caption source asset");return uint32_t(crc);
}
}
void World::enableCaptions(const std::filesystem::path& root){
    if(captionsEnabled)throw std::runtime_error("Captions already enabled");
    std::ifstream file(root/"assets/caption-labels.bin",std::ios::binary);
    const auto read=[&]<class T>(){T v{};file.read(reinterpret_cast<char*>(&v),sizeof(v));if(!file)throw std::runtime_error("Missing/truncated caption labels");return v;};
    const auto u32=[&](){return read.operator()<uint32_t>();};
    if(u32()!=0x31504143)throw std::runtime_error("Unsupported caption label version");
    const auto mapCrc=u32(),vehicleCrc=u32(),meshCount=u32(),objectCount=u32();
    if(mapCrc!=checksum(root/"assets/small-city.bin")||vehicleCrc!=checksum(root/"assets/bbox-assets.bin"))
        throw std::runtime_error("Caption labels do not match map/vehicle assets; rebuild the sidecar");
    if(meshCount!=cityRenderMeshes.size()||objectCount>1000000)throw std::runtime_error("Caption mesh/object count mismatch");
    for(auto index:cityRenderMeshes){auto& mesh=meshes.at(index);auto count=u32();
        if(count!=mesh.indices.size()/3)throw std::runtime_error("Caption triangle count mismatch");
        mesh.captionTokens.resize(count);for(auto& token:mesh.captionTokens){token=u32();if(token>objectCount)throw std::runtime_error("Invalid caption token");}}
    staticEntities.push_back({});
    for(uint32_t i=1;i<=objectCount;++i){auto captionId=u32();u32();auto center=read.operator()<V3>();staticEntities.push_back({i,captionId,center,false,false});}
    const auto list=[&](std::vector<uint32_t>& ids){auto count=u32();if(!count||count>256)throw std::runtime_error("Invalid caption model list");
        for(uint32_t i=0;i<count;++i){auto id=u32();if(!id)throw std::runtime_error("Missing model caption");ids.push_back(id);}};
    list(vehicleCaptionIds);list(personCaptionIds);
    if(vehicleCaptionIds.size()!=models.size()||file.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("Caption sidecar layout mismatch");
    nextCaptionEntity=staticEntities.size();
    for(auto& w:people)w.captionId=personCaptionIds.at(w.random%personCaptionIds.size());
    captionsEnabled=true;
}
