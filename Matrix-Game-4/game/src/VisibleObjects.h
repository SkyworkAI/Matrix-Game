#pragma once
#include "Common.h"
#include <numeric>
#include <unordered_map>

struct CatalogObject {
    enum Flags : uint32_t { Exempt=1,Rotation=2,MergeNearby=4 };
    uint32_t index=0,flags=0;
    int32_t barrierRun=-1;
    V3 min,max,rotation;
};
static_assert(sizeof(CatalogObject)==48);
struct VisibleObject {
    uint32_t index=0,count=1,pixels=0,flags=0;
    std::array<double,3> position{},rotation{};
};
static_assert(sizeof(VisibleObject)==64);

inline std::vector<VisibleObject> reduceVisibleObjects(const std::vector<CatalogObject>& objects,
    const std::vector<uint32_t>& pixels,double pixelScale) {
    if(pixels.size()!=objects.size())throw std::runtime_error("Catalog pixel count mismatch");
    std::vector<size_t> visible;
    for(size_t i=1;i<objects.size();++i)if(objects[i].index&&pixels[i]>=4)visible.push_back(i);
    std::vector<size_t> parents(visible.size());std::iota(parents.begin(),parents.end(),0);
    const auto root=[&](size_t i){while(parents[i]!=i){parents[i]=parents[parents[i]];i=parents[i];}return i;};
    const auto canMerge=[](const CatalogObject& o){return o.index&&(o.flags&CatalogObject::MergeNearby)&&!(o.flags&(CatalogObject::Exempt|CatalogObject::Rotation));};
    const auto gap=[](float amin,float amax,float bmin,float bmax){return std::max(0.f,std::max(amin-bmax,bmin-amax));};
    // Each token already combines split Parts. Count each logical object once.
    for(size_t i=0;i<visible.size();++i){const auto& a=objects[visible[i]];if(!canMerge(a))continue;
        for(size_t j=0;j<i;++j){const auto& b=objects[visible[j]];if(a.index!=b.index||!canMerge(b))continue;
            const float x=gap(a.min.x,a.max.x,b.min.x,b.max.x),y=gap(a.min.y,a.max.y,b.min.y,b.max.y),z=gap(a.min.z,a.max.z,b.min.z,b.max.z);
            const bool barrier=a.barrierRun>=0&&a.barrierRun==b.barrierRun;
            const bool nearby=a.barrierRun<0&&b.barrierRun<0&&x*x+y*y+z*z<=4;
            if(barrier||nearby)parents[root(i)]=root(j);
        }
    }
    struct Group {VisibleObject object;V3 min,max;bool exempt=false;};
    std::vector<Group> groups;std::unordered_map<size_t,size_t> groupIds;
    for(size_t i=0;i<visible.size();++i){const auto& o=objects[visible[i]];const auto r=root(i);auto it=groupIds.find(r);
        if(it==groupIds.end()){
            groupIds[r]=groups.size();Group g;g.object.index=o.index;g.object.count=0;g.object.flags=o.flags;
            g.object.rotation={o.rotation.x,o.rotation.y,o.rotation.z};g.min=o.min;g.max=o.max;
            g.exempt=(o.flags&(CatalogObject::Exempt|CatalogObject::Rotation))!=0;groups.push_back(g);it=groupIds.find(r);
        }
        auto& g=groups[it->second];++g.object.count;g.object.pixels+=pixels[visible[i]];
        g.min={std::min(g.min.x,o.min.x),std::min(g.min.y,o.min.y),std::min(g.min.z,o.min.z)};
        g.max={std::max(g.max.x,o.max.x),std::max(g.max.y,o.max.y),std::max(g.max.z,o.max.z)};
    }
    for(auto& g:groups){const V3 center=(g.min+g.max)*.5f;g.object.position={double(center.z)*100,double(center.x)*100,double(center.y)*100};}
    std::sort(groups.begin(),groups.end(),[](const Group& a,const Group& b){
        if(a.object.pixels!=b.object.pixels)return a.object.pixels>b.object.pixels;
        if(a.object.index!=b.object.index)return a.object.index<b.object.index;
        if(a.object.position!=b.object.position)return a.object.position<b.object.position;
        return a.object.count>b.object.count;
    });
    std::vector<VisibleObject> result;size_t staticCount=0;
    for(auto& g:groups){if(!g.exempt&&staticCount++>=20)continue;g.object.pixels=uint32_t(std::llround(g.object.pixels*pixelScale));result.push_back(g.object);}
    return result;
}
