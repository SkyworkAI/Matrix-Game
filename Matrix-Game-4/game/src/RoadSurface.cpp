#include "RoadSurface.h"
#include <map>
#include <tuple>

namespace {
std::vector<std::array<uint32_t,2>> creaseEdges(const MeshData& mesh,float degrees){
    // LOD meshes duplicate vertices at UV/normal seams. Weld only for adjacency,
    // retaining the original vertex positions and indices in the rendered mesh.
    std::map<std::tuple<long long,long long,long long>,uint32_t> positions;
    std::vector<uint32_t> welded;
    for(const auto& v:mesh.vertices){
        auto key=std::make_tuple(std::llround(v.x*10000),std::llround(v.y*10000),std::llround(v.z*10000));
        welded.push_back(positions.emplace(key,uint32_t(positions.size())).first->second);
    }
    struct Edge{std::vector<V3> normals;std::vector<std::array<uint32_t,2>> copies;};
    std::map<std::pair<uint32_t,uint32_t>,Edge> edges;
    for(size_t i=0;i<mesh.indices.size();i+=3){
        const uint32_t tri[]={mesh.indices[i],mesh.indices[i+1],mesh.indices[i+2]};
        V3 n=cross(mesh.vertices[tri[1]]-mesh.vertices[tri[0]],mesh.vertices[tri[2]]-mesh.vertices[tri[0]]);
        if(n.length()<1e-8f)continue;
        n=n.normalized();
        for(int k=0;k<3;++k){
            uint32_t a=tri[k],b=tri[(k+1)%3],wa=welded[a],wb=welded[b];if(wa==wb)continue;
            auto& e=edges[{std::min(wa,wb),std::max(wa,wb)}];
            e.normals.push_back(n);e.copies.push_back({std::min(a,b),std::max(a,b)});
        }
    }
    std::vector<std::array<uint32_t,2>> result;
    const float cosine=std::cos(degrees*Pi/180);
    for(const auto& [key,e]:edges){
        bool sharp=e.normals.size()==1;
        for(size_t i=0;i<e.normals.size();++i)for(size_t j=i+1;j<e.normals.size();++j)
            sharp|=std::abs(dot(e.normals[i],e.normals[j]))<cosine;
        if(sharp)result.insert(result.end(),e.copies.begin(),e.copies.end());
    }
    std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());
    return result;
}
}

RoadSurfaceSplit splitRoadSurface(const MeshData& mesh,float floorMaxY,float floorMinNormalY,
                                 float creaseDegrees,uint32_t structureColor){
    if(!std::isfinite(floorMaxY)||!std::isfinite(floorMinNormalY)||floorMinNormalY<=0||floorMinNormalY>1||
       !std::isfinite(creaseDegrees)||creaseDegrees<=0||creaseDegrees>=90||mesh.indices.size()%3)
        throw std::runtime_error("Invalid mixed-road surface parameters");
    RoadSurfaceSplit result;
    for(auto* part:{&result.floor,&result.structure}){
        part->vertices=mesh.vertices;part->colors=mesh.colors;part->tokens=mesh.tokens;
    }
    result.structure.colors.assign(mesh.vertices.size(),structureColor);
    for(size_t i=0;i<mesh.indices.size();i+=3){
        const V3 a=mesh.vertices.at(mesh.indices[i]),b=mesh.vertices.at(mesh.indices[i+1]),c=mesh.vertices.at(mesh.indices[i+2]);
        const V3 n=cross(b-a,c-a);
        const bool floor=std::max({a.y,b.y,c.y})<=floorMaxY&&std::abs(n.y)>=floorMinNormalY*n.length();
        auto& indices=floor?result.floor.indices:result.structure.indices;
        indices.insert(indices.end(),mesh.indices.begin()+i,mesh.indices.begin()+i+3);
    }
    if(result.floor.indices.empty()||result.structure.indices.empty())throw std::runtime_error("Mixed road must contain floor and structure triangles");
    result.structure.edges=creaseEdges(result.structure,creaseDegrees);
    return result;
}
