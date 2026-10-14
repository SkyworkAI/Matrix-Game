#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

struct CollisionEvent {
    uint64_t id=0,entityA=0,entityB=0;
    float speed=0;
};

// Merge mesh triangles into body contacts. Brief solver separations do not turn
// one scrape into repeated penalties. All time here is simulation time.
class CollisionEvents {
public:
    using Contact=std::array<uint32_t,4>; // body A, body B, subshape A, subshape B
    void touch(Contact key,uint64_t a,uint64_t b,float speed) {
        if(key[0]>key[1]) {std::swap(key[0],key[1]);std::swap(key[2],key[3]);std::swap(a,b);}
        auto& p=pairs[{key[0],key[1]}];
        p.contacts.insert({key[2],key[3]});p.absent=0;
        if(!p.reported&&speed>=.7f){p.reported=true;events.push_back({++serial,a,b,speed});}
    }
    void remove(Contact key) {
        if(key[0]>key[1]) {std::swap(key[0],key[1]);std::swap(key[2],key[3]);}
        auto it=pairs.find({key[0],key[1]});if(it!=pairs.end())it->second.contacts.erase({key[2],key[3]});
    }
    void step(float dt) {
        for(auto it=pairs.begin();it!=pairs.end();) {
            if(it->second.contacts.empty())it->second.absent+=dt;
            if(it->second.absent>=.2f)it=pairs.erase(it);else ++it;
        }
    }
    std::vector<CollisionEvent> take(){std::vector<CollisionEvent> out;out.swap(events);return out;}
private:
    struct Pair {std::set<std::array<uint32_t,2>> contacts;float absent=0;bool reported=false;};
    std::map<std::array<uint32_t,2>,Pair> pairs;
    std::vector<CollisionEvent> events;
    uint64_t serial=0;
};
