#include "RoadSurface.h"
#include <cassert>
#include <iostream>
#include <set>

int main(){
    MeshData mesh;
    // A floor and a wall with duplicated vertices along its triangulation seam.
    mesh.vertices={{0,0,0},{2,0,0},{2,0,2},{0,0,2},
                   {0,0,2},{2,0,2},{2,2,2},{0,0,2},{2,2,2},{0,2,2},
                   {0,2,2},{2,2,2},{2,2,3},{0,2,3}};
    mesh.indices={0,1,2,0,2,3,4,5,6,7,8,9,10,11,12,10,12,13};
    mesh.colors.assign(mesh.vertices.size(),0xffffffff);mesh.tokens.assign(mesh.vertices.size(),42);
    const uint32_t color=0xffffc83f;
    const auto split=splitRoadSurface(mesh,.6f,.98f,20,color);
    assert(split.floor.indices==std::vector<uint32_t>({0,1,2,0,2,3}));
    assert(split.structure.indices==std::vector<uint32_t>(mesh.indices.begin()+6,mesh.indices.end()));
    assert(split.floor.edges.empty()&&!split.structure.edges.empty());
    assert(split.structure.colors==std::vector<uint32_t>(mesh.vertices.size(),color));
    assert(split.floor.colors==mesh.colors&&split.structure.tokens==mesh.tokens);
    assert(split.floor.vertices.size()==mesh.vertices.size()&&split.structure.vertices.size()==mesh.vertices.size());
    for(size_t i=0;i<mesh.vertices.size();++i){
        assert((split.floor.vertices[i]-mesh.vertices[i]).length()==0);
        assert((split.structure.vertices[i]-mesh.vertices[i]).length()==0);
    }
    const std::set<std::array<uint32_t,2>> edges(split.structure.edges.begin(),split.structure.edges.end());
    assert(!edges.contains({4,6})&&!edges.contains({7,8})&&!edges.contains({10,12}));
    assert(edges.contains({8,9})&&edges.contains({10,11}));
    for(float normal:{0.f,1.1f}){
        bool rejected=false;try{splitRoadSurface(mesh,.6f,normal,20,color);}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
    std::cout<<"PASS road floor partition, elevated wall caps, welded crease edges, unchanged geometry/colors/tokens\n";
}
