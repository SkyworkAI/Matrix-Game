#include "Map.h"
#include <limits>
#include <map>
#include <tuple>
#include <cctype>
#include <json.hpp>

namespace {
class Reader {
    std::ifstream file;
    uint64_t remaining;
public:
    explicit Reader(const std::filesystem::path& path):file(path,std::ios::binary) {
        if(!file) throw std::runtime_error("Cannot open map binary: "+path.string());
        remaining=std::filesystem::file_size(path);
    }
    template<class T> void read(T* data,size_t count=1) {
        if(count>remaining/sizeof(T)) throw std::runtime_error("Truncated map data");
        const auto bytes=count*sizeof(T);
        if(!file.read(reinterpret_cast<char*>(data),std::streamsize(bytes))) throw std::runtime_error("Map read failed");
        remaining-=bytes;
    }
    uint32_t integer() { uint32_t value; read(&value); return value; }
    Bounds bounds() {
        Bounds b; read(&b);
        if(!std::isfinite(b.min.x)||!std::isfinite(b.min.y)||!std::isfinite(b.min.z)||
           !std::isfinite(b.max.x)||!std::isfinite(b.max.y)||!std::isfinite(b.max.z)||
           b.min.x>b.max.x||b.min.y>b.max.y||b.min.z>b.max.z) throw std::runtime_error("Invalid map bounds");
        return b;
    }
    void finish() { if(remaining) throw std::runtime_error("Unexpected bytes after map data"); }
};
bool contains(const Bounds& b,V3 p) {
    constexpr float tolerance=.002f;
    return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&
        p.x>=b.min.x-tolerance&&p.y>=b.min.y-tolerance&&p.z>=b.min.z-tolerance&&
        p.x<=b.max.x+tolerance&&p.y<=b.max.y+tolerance&&p.z<=b.max.z+tolerance;
}
}

MapData::MapData(const SceneConfig& config):scene(config) {
    const auto path=scene.assets/"castle-map.bin";
    std::ifstream profileFile(scene.assets/"map.json");
    if(profileFile){
        nlohmann::json profile;profileFile>>profile;
        if(profile.at("id")!=scene.id)throw std::runtime_error("Map profile does not match selected scene");
        const auto encoding=profile.value("bboxColorEncoding",std::string("display-srgb"));
        if(encoding!="display-srgb"&&encoding!="catalog-srgb")throw std::runtime_error("Unsupported map color encoding");
        catalogColors=encoding=="catalog-srgb";
        const auto vector=[](const nlohmann::json& v){
            if(!v.is_array()||v.size()!=3)throw std::runtime_error("Invalid map profile vector");
            V3 p{v[0].get<float>(),v[1].get<float>(),v[2].get<float>()};
            if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))throw std::runtime_error("Nonfinite map profile vector");
            return p;
        };
        for(const auto& p:profile.at("startsMeters"))starts.push_back(vector(p));
        const auto& b=profile.at("navigationBoundsMeters");
        navigationBounds={vector(b.at("min")),vector(b.at("max"))};
        if(navigationBounds.min.x>=navigationBounds.max.x||navigationBounds.min.y>=navigationBounds.max.y||navigationBounds.min.z>=navigationBounds.max.z||starts.empty())
            throw std::runtime_error("Invalid map navigation bounds or starts");
        hasNavigationBounds=true;
    }
    static_assert(sizeof(V3)==12 && sizeof(Bounds)==24);
    Reader reader(path);
    if(reader.integer()!=0x314d4242) throw std::runtime_error("Unsupported map binary format");
    const auto version=reader.integer();
    if(version<1||version>3) throw std::runtime_error("Unsupported map binary version");
    const auto count=reader.integer();
    primitiveCount=reader.integer();
    if(count==0||count>100000) throw std::runtime_error("Invalid map object count");
    bounds=reader.bounds(); castleBounds=reader.bounds();
    objects.reserve(count);
    uint64_t actualPrimitives=0;
    for(uint32_t i=0;i<count;++i) {
        MapObject o;
        o.catalogIndex=reader.integer();
        const auto flags=reader.integer();
        if(flags>(version==1?1u:version==2?2u:4u)||(flags&&(flags&(flags-1)))) throw std::runtime_error("Invalid object flags");
        o.tree=(flags&1)!=0; treeCount+=o.tree;
        o.terrain=(flags&2)!=0; terrainCount+=o.terrain;
        o.road=(flags&4)!=0; roadCount+=o.road;
        o.primitiveCount=reader.integer(); actualPrimitives+=o.primitiveCount;
        const auto vertices=reader.integer(),indices=reader.integer(),edges=reader.integer();
        if(vertices<3||vertices>1000000||indices==0||indices%3||indices>6000000||edges>3000000||!o.primitiveCount)
            throw std::runtime_error("Invalid mesh dimensions");
        o.bounds=reader.bounds();
        if(!contains(bounds,o.bounds.min)||!contains(bounds,o.bounds.max)) throw std::runtime_error("Object outside scene bounds");
        o.mesh.vertices.resize(vertices); o.mesh.colors.resize(vertices);
        o.mesh.indices.resize(indices); o.mesh.edges.resize(edges);
        reader.read(o.mesh.vertices.data(),vertices); reader.read(o.mesh.colors.data(),vertices);
        reader.read(o.mesh.indices.data(),indices); reader.read(o.mesh.edges.data(),edges);
        for(const auto& v:o.mesh.vertices) if(!contains(o.bounds,v)) throw std::runtime_error("Mesh outside object bounds");
        for(auto index:o.mesh.indices) if(index>=vertices) throw std::runtime_error("Invalid triangle index");
        for(auto edge:o.mesh.edges) if(edge[0]>=vertices||edge[1]>=vertices) throw std::runtime_error("Invalid edge index");
        triangleCount+=indices/3;
        objects.push_back(std::move(o));
    }
    if(actualPrimitives!=primitiveCount) throw std::runtime_error("Primitive count mismatch");
    reader.finish();
    if(scene.keepCourtyardTrees)filterCourtyardTrees();
    loadRoadSurfaceOverrides();
}

void MapData::loadRoadSurfaceOverrides(){
    std::ifstream file(scene.assets/"road-surface-overrides.json");if(!file)return;
    nlohmann::json rules,manifest,catalog;file>>rules;
    std::ifstream(scene.assets/"manifest.json")>>manifest;
    std::ifstream(scene.catalogDirectory()/"CatalogManifest.json")>>catalog;
    if(rules.at("schemaVersion")!=1||rules.at("scene")!=scene.id||manifest.at("objects").size()!=objects.size())
        throw std::runtime_error("Road surface override metadata does not match map");
    const auto lower=[](std::string s){for(auto& c:s)c=char(std::tolower(static_cast<unsigned char>(c)));return s;};
    for(const auto& rule:rules.at("meshes")){
        const auto identity=lower(rule.at("sourceMesh").get<std::string>());
        std::string hex;
        for(const auto& c:catalog.at("bboxColorsByIndex"))if(lower(c.value("identity",std::string()))==identity)hex=c.at("srgbHex");
        if(hex.size()!=8||!std::all_of(hex.begin(),hex.end(),[](unsigned char c){return std::isxdigit(c);}))
            throw std::runtime_error("Road structure is missing its catalog color");
        const uint32_t color=0xff000000u|uint32_t(std::stoul(hex.substr(0,2),nullptr,16))|
            (uint32_t(std::stoul(hex.substr(2,2),nullptr,16))<<8)|(uint32_t(std::stoul(hex.substr(4,2),nullptr,16))<<16);
        const float height=rule.at("floorMaxHeightMeters");
        if(!std::isfinite(height)||height<=0)throw std::runtime_error("Invalid mixed-road floor height");
        size_t matches=0;
        for(size_t i=0;i<objects.size();++i){
            const auto& metadata=manifest.at("objects").at(i);
            if(lower(metadata.value("sourceMeshPath",std::string()))!=identity)continue;
            const auto& o=objects[i];
            if(!o.road||o.mesh.vertices.size()!=metadata.at("vertexCount")||o.mesh.indices.size()/3!=metadata.at("triangleCount"))
                throw std::runtime_error("Road surface override object does not match binary");
            const auto& b=metadata.at("boundsMeters");
            for(int axis=0;axis<3;++axis){
                const float lo[]={o.bounds.min.x,o.bounds.min.y,o.bounds.min.z},hi[]={o.bounds.max.x,o.bounds.max.y,o.bounds.max.z};
                if(std::abs(lo[axis]-b.at("min").at(axis).get<float>())>.002f||std::abs(hi[axis]-b.at("max").at(axis).get<float>())>.002f)
                    throw std::runtime_error("Road surface override bounds do not match binary");
            }
            auto split=splitRoadSurface(o.mesh,o.bounds.min.y+height,rule.at("floorMinNormalY"),rule.at("creaseDegrees"),color);
            if(!roadSurfaceSplits.emplace(i,std::move(split)).second)throw std::runtime_error("Duplicate road surface override");
            ++matches;
        }
        if(matches!=rule.at("expectedInstances").get<size_t>())throw std::runtime_error("Road surface override instance count changed");
    }
}

void MapData::filterCourtyardTrees(){
    // Keep one tree near each courtyard perimeter anchor; all consumers share this map.
    // Select by position, since all tree instances share the same catalog index.
    const V3 treeAnchors[]={{65,0,-255},{180,0,-255},{65,0,-160},{180,0,-160}};
    std::vector<size_t> keptTrees;
    for(const auto& anchor:treeAnchors){
        size_t nearest=objects.size();float distance=std::numeric_limits<float>::max();
        for(size_t i=0;i<objects.size();++i){
            if(!objects[i].tree||std::find(keptTrees.begin(),keptTrees.end(),i)!=keptTrees.end())continue;
            const V3 p=objects[i].bounds.center();
            const float d=(p.x-anchor.x)*(p.x-anchor.x)+(p.z-anchor.z)*(p.z-anchor.z);
            if(d<distance){distance=d;nearest=i;}
        }
        if(nearest!=objects.size())keptTrees.push_back(nearest);
    }
    size_t objectIndex=0;
    std::erase_if(objects,[&](const MapObject& o){
        const bool keep=std::find(keptTrees.begin(),keptTrees.end(),objectIndex++)!=keptTrees.end();
        if(!o.tree||keep)return false;
        --treeCount;
        primitiveCount-=o.primitiveCount;
        triangleCount-=o.mesh.indices.size()/3;
        return true;
    });
}

std::vector<DrawObject> MapData::upload(Renderer& renderer) const {
    configureRenderer(renderer);
    const bool castleChecker=scene.groundPattern==SceneConfig::GroundPattern::CastleNested;
    const auto surface=[&](const MapObject& o){return o.terrain?(castleChecker?2.f:4.f):o.road?(castleChecker?3.f:5.f):0.f;};
    const auto color=[&](uint32_t packed,float material){
        if(!catalogColors||material!=0)return packed;
        const auto c=bboxFill(rgb(packed&255,(packed>>8)&255,(packed>>16)&255));
        return (packed&0xff000000u)|uint32_t(std::lround(c.r*255))|(uint32_t(std::lround(c.g*255))<<8)|(uint32_t(std::lround(c.b*255))<<16);
    };
    const auto visitParts=[&](auto&& emit){
        for(size_t i=0;i<objects.size();++i){
            const auto& o=objects[i];const auto split=roadSurfaceSplits.find(i);
            if(split==roadSurfaceSplits.end())emit(i,o.mesh,surface(o));
            else{emit(i,split->second.floor,surface(o));emit(i,split->second.structure,0.f);}
        }
    };
    if(usesBatches()){
        using Key=std::tuple<int,int,int>;
        struct Batch{MeshData mesh;Bounds bounds{{1e9f,1e9f,1e9f},{-1e9f,-1e9f,-1e9f}};};
        std::map<Key,Batch> batches;
        visitParts([&](size_t oi,const MeshData& part,float material){
            const auto& o=objects[oi];const V3 center=o.bounds.center();
            auto& b=batches[{int(std::floor(center.x/32)),int(std::floor(center.z/32)),int(material)}];
            auto& m=b.mesh;const auto offset=uint32_t(m.vertices.size());
            m.vertices.insert(m.vertices.end(),part.vertices.begin(),part.vertices.end());
            for(auto c:part.colors)m.colors.push_back(color(c,material));
            // Logical object identity survives batching, for ID capture and collision parity.
            m.tokens.insert(m.tokens.end(),part.vertices.size(),uint32_t(oi+1));
            for(auto i:part.indices)m.indices.push_back(offset+i);
            for(auto e:part.edges)m.edges.push_back({offset+e[0],offset+e[1]});
            b.bounds.min={std::min(b.bounds.min.x,o.bounds.min.x),std::min(b.bounds.min.y,o.bounds.min.y),std::min(b.bounds.min.z,o.bounds.min.z)};
            b.bounds.max={std::max(b.bounds.max.x,o.bounds.max.x),std::max(b.bounds.max.y,o.bounds.max.y),std::max(b.bounds.max.z,o.bounds.max.z)};
        });
        std::vector<DrawObject> result;
        for(const auto& [key,b]:batches){
            DrawObject draw;draw.mesh=renderer.addMesh(b.mesh);draw.instance.world=transform({});
            draw.instance.surface=float(std::get<2>(key));draw.instance.objectToken=1;
            draw.center=b.bounds.center();draw.radius=(b.bounds.max-b.bounds.min).length()*.5f;draw.range=100000;
            result.push_back(draw);
        }
        return result;
    }
    std::vector<DrawObject> result;
    result.reserve(objects.size());
    visitParts([&](size_t oi,const MeshData& part,float material){
        const auto& o=objects[oi];
        DrawObject draw;
        if(catalogColors&&material==0){auto mesh=part;for(auto& c:mesh.colors)c=color(c,material);draw.mesh=renderer.addMesh(mesh);}
        else draw.mesh=renderer.addMesh(part);
        draw.instance.world=transform({});
        draw.instance.surface=material;
        draw.instance.objectToken=uint32_t(oi+1);
        draw.center=o.bounds.center();
        draw.radius=(o.bounds.max-o.bounds.min).length()*.5f;
        draw.range=2000;
        result.push_back(draw);
    });
    return result;
}

void MapData::configureRenderer(Renderer& renderer) const{
    renderer.outlineWidth=scene.outlineWidth;
    renderer.fadeSmallEdges=scene.fadeSmallEdges;
}
