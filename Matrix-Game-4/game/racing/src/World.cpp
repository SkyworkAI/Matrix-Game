#include "World.h"
#include "Renderer.h"
#include <limits>

namespace {
const Color CarColor=rgb(79,116,140),WheelColor=rgb(129,130,128);
const Color AsphaltColor=rgb(132,134,132);
constexpr uint32_t AsphaltPacked=0xff848684u,SidewalkPacked=0xffa9adacu;
uint32_t randomStep(uint32_t& state){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
float flatDistance(V3 a,V3 b){V3 d=a-b;d.y=0;return d.length();}
MeshData cube(){
    MeshData m;m.vertices={{-.5f,-.5f,-.5f},{.5f,-.5f,-.5f},{.5f,-.5f,.5f},{-.5f,-.5f,.5f},{-.5f,.5f,-.5f},{.5f,.5f,-.5f},{.5f,.5f,.5f},{-.5f,.5f,.5f}};
    const uint32_t faces[][4]={{0,3,2,1},{4,5,6,7},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7}};
    for(auto& f:faces)m.indices.insert(m.indices.end(),{f[0],f[1],f[2],f[0],f[2],f[3]});
    m.edges={{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};return m;
}
struct Reader {
    std::ifstream file;
    explicit Reader(const std::filesystem::path& p):file(p,std::ios::binary){if(!file)throw std::runtime_error("Missing game asset: "+p.filename().string());}
    template<class T>T read(){T value{};file.read(reinterpret_cast<char*>(&value),sizeof(T));if(!file)throw std::runtime_error("Truncated asset file");return value;}
    template<class T>void list(std::vector<T>& list,uint32_t count){if(count>4000000)throw std::runtime_error("Invalid asset count");list.resize(count);file.read(reinterpret_cast<char*>(list.data()),std::streamsize(count)*sizeof(T));if(!file)throw std::runtime_error("Truncated geometry");}
};
MeshData readMesh(Reader& r,bool colored=false){
    MeshData m;auto vc=r.read<uint32_t>(),ic=r.read<uint32_t>(),ec=r.read<uint32_t>();
    if(vc<3||ic<3||ic%3)throw std::runtime_error("Invalid geometry counts");r.list(m.vertices,vc);if(colored)r.list(m.colors,vc);r.list(m.indices,ic);r.list(m.edges,ec);
    for(auto p:m.vertices)if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))throw std::runtime_error("Non-finite vertex");
    for(auto i:m.indices)if(i>=vc)throw std::runtime_error("Invalid triangle index");
    for(auto e:m.edges)if(e[0]>=vc||e[1]>=vc)throw std::runtime_error("Invalid edge index");return m;
}
uint32_t assetCrc32(const std::filesystem::path& path){
    static const auto table=[](){std::array<uint32_t,256> values{};
        for(uint32_t i=0;i<256;++i){auto c=i;for(int bit=0;bit<8;++bit)c=(c>>1)^((c&1)?0xedb88320u:0);values[i]=c;}return values;}();
    std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("Missing stripe base map");
    std::array<char,65536> bytes{};uint32_t crc=0xffffffffu;
    while(file){file.read(bytes.data(),bytes.size());for(std::streamsize i=0;i<file.gcount();++i)crc=table[(crc^uint8_t(bytes[i]))&255]^(crc>>8);}
    if(!file.eof())throw std::runtime_error("Cannot checksum stripe base map");return crc^0xffffffffu;
}
struct RampSite {uint32_t id;V3 origin,forward;float width,length;};
std::vector<RampSite> readRampSites(const std::filesystem::path& root){
    Reader r(root/L"assets/ramp-ground-fixtures.bin");
    if(r.read<uint32_t>()!=0x31504d52)throw std::runtime_error("Unsupported ramp survey format");
    auto count=r.read<uint32_t>();if(!count||count>64)throw std::runtime_error("Invalid ramp survey count");
    std::vector<RampSite> sites;
    for(uint32_t i=0;i<count;++i)sites.push_back({r.read<uint32_t>(),r.read<V3>(),r.read<V3>(),r.read<float>(),r.read<float>()});
    return sites;
}
}
World::World(const std::filesystem::path& path,int trafficCount,int pedestrianCount,bool testTrack):testing(testTrack){
    load(path);if(testing)buildTestTrack();else{loadCity(path.parent_path()/L"small-city.bin");build(trafficCount,pedestrianCount);}
}
void World::load(const std::filesystem::path& path){
    Reader r(path);if(r.read<uint32_t>()!=0x31474242)throw std::runtime_error("Unsupported BBox asset format");
    auto mc=r.read<uint32_t>(),vc=r.read<uint32_t>(),bc=r.read<uint32_t>();
    if(mc>512||vc<1||vc>32||bc>128)throw std::runtime_error("Invalid asset header");
    for(uint32_t i=0;i<mc;++i)meshes.push_back(readMesh(r));
    for(uint32_t i=0;i<vc;++i){CarModel model;model.bodyMesh=r.read<uint32_t>();for(auto& m:model.wheelMeshes)m=r.read<uint32_t>();
        for(auto& p:model.wheelPositions)p=r.read<V3>();for(auto& s:model.wheelSizes)s=r.read<std::array<float,2>>();model.halfExtent=r.read<V3>();model.bodyCenter=r.read<V3>();
        if(model.bodyMesh>=meshes.size())throw std::runtime_error("Invalid vehicle mesh");for(auto m:model.wheelMeshes)if(m>=meshes.size())throw std::runtime_error("Invalid wheel mesh");models.push_back(model);}
    for(uint32_t i=0;i<bc;++i){Building b{r.read<uint32_t>(),r.read<V3>()};if(b.mesh>=meshes.size())throw std::runtime_error("Invalid building mesh");buildings.push_back(b);}
    cubeMesh=uint32_t(meshes.size());meshes.push_back(cube());
}
void World::loadCity(const std::filesystem::path& path){
    Reader r(path);if(r.read<uint32_t>()!=0x32435459)throw std::runtime_error("Unsupported Small City format");
    auto chunks=r.read<uint32_t>(),routes=r.read<uint32_t>(),parked=r.read<uint32_t>(),roads=r.read<uint32_t>();
    sourceObjectCount=r.read<uint32_t>();originalBuildings=r.read<uint32_t>();originalDecks=r.read<uint32_t>();
    if(chunks>20000||routes<10||routes>60000||parked>30000||roads>30000)throw std::runtime_error("Invalid Small City counts");
    cityMin={1e9f,1e9f,1e9f};cityMax={-1e9f,-1e9f,-1e9f};
    for(uint32_t i=0;i<chunks;++i){
        V3 center=r.read<V3>();float radius=r.read<float>(),range=r.read<float>(),surface=r.read<float>();auto flags=r.read<uint32_t>();auto m=readMesh(r,true);
        for(auto p:m.vertices){cityMin.x=std::min(cityMin.x,p.x);cityMin.y=std::min(cityMin.y,p.y);cityMin.z=std::min(cityMin.z,p.z);cityMax.x=std::max(cityMax.x,p.x);cityMax.y=std::max(cityMax.y,p.y);cityMax.z=std::max(cityMax.z,p.z);}
        if(flags&2)physics.addStaticMesh(m,transform({0,0,0}),bool(flags&4),bool(flags&16));
        if(flags&8)for(size_t j=0;j<m.indices.size();j+=3){
            V3 a=m.vertices[m.indices[j]],b=m.vertices[m.indices[j+1]],c=m.vertices[m.indices[j+2]];
            if(std::abs(cross(b-a,c-a).y)>.0001f)groundGapProbes.push_back((a+b+c)*(1.f/3));
        }
        // Appearance only: keep the serialized meshes, physics and caption
        // triangle ordering intact. Flags distinguish sidewalks from pillars.
        if(surface>.5f&&(flags==7||flags==23))std::fill(m.colors.begin(),m.colors.end(),AsphaltPacked);
        else if(surface== -1.f&&flags==7)std::fill(m.colors.begin(),m.colors.end(),SidewalkPacked);
        if(flags&1){cityRenderMeshes.push_back(uint32_t(meshes.size()));object(uint32_t(meshes.size()),transform({0,0,0}),{},center,radius,range,surface);meshes.push_back(std::move(m));}
    }
    for(uint32_t i=0;i<routes;++i){Lane lane;lane.flags=r.read<uint32_t>();lane.width=r.read<float>();auto points=r.read<uint32_t>(),links=r.read<uint32_t>();
        if(points<2||points>10000||links>100)throw std::runtime_error("Invalid lane");r.list(lane.points,points);r.list(lane.next,links);
        for(auto next:lane.next)if(next>=routes)throw std::runtime_error("Invalid route link");lanes.push_back(std::move(lane));}
    for(uint32_t i=0;i<parked;++i){Parking p{r.read<V3>(),r.read<float>(),r.read<uint32_t>()};p.model%=uint32_t(models.size());parking.push_back(p);}
    for(uint32_t i=0;i<roads;++i){MapLine line{r.read<V3>(),r.read<V3>(),r.read<uint32_t>()!=0};mapRoads.push_back(line);}
    physics.optimize();
    for(uint32_t i=0;i<lanes.size();++i){auto& l=lanes[i];
        for(auto& p:l.points)p.y=physics.groundHeight(p);
        l.cumulative.push_back(0);for(size_t j=1;j<l.points.size();++j){l.length+=(l.points[j]-l.points[j-1]).length();l.cumulative.push_back(l.length);}
        if(!(l.flags&1)&&l.length>5){if(l.flags&4)footLanes.push_back(i);else carLanes.push_back(i);}
    }
    if(carLanes.empty()||footLanes.empty())throw std::runtime_error("Small City has no driving or walking routes");
    // Paint follows the exported lane boundaries and the actual collision surface height.
    for(uint32_t id:carLanes){const auto& lane=lanes[id];
        for(float s=4;s<lane.length-4;s+=9){V3 p=lanePoint(id,s),f=laneForward(id,s),right=cross({0,1,0},f).normalized();p=p+right*(lane.width*.5f);p.y=physics.groundHeight(p)+.018f;
            box(p,{.12f,.012f,3.0f},rgb(202,202,196),false,std::atan2(f.x,f.z),-1,220);}
    }
    const auto stripePath=path.parent_path()/"reference-building-stripes.bin";
    if(std::filesystem::exists(stripePath)){
        Reader stripes(stripePath);
        if(stripes.read<uint32_t>()!=0x31525453||stripes.read<uint32_t>()!=assetCrc32(path))
            throw std::runtime_error("Building stripes do not match the base map");
        const auto count=stripes.read<uint32_t>();if(count!=originalBuildings)throw std::runtime_error("Building stripe count mismatch");
        for(uint32_t i=0;i<count;++i){
            const auto center=stripes.read<V3>();const float radius=stripes.read<float>(),range=stripes.read<float>();
            if(!std::isfinite(center.x)||!std::isfinite(center.y)||!std::isfinite(center.z)||!std::isfinite(radius)||!std::isfinite(range)||radius<=0||range<=0)
                throw std::runtime_error("Invalid building stripe bounds");
            auto mesh=readMesh(stripes,true);if(!mesh.edges.empty())throw std::runtime_error("Unexpected stripe outlines");
            // Visual overlays only: do not add collision, caption IDs or map bounds.
            object(uint32_t(meshes.size()),transform({0,0,0}),{},center,radius,range,-2);
            meshes.push_back(std::move(mesh));
        }
        if(stripes.file.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("Trailing stripe asset data");
    }
    const auto junctionPath=path.parent_path()/"reference-junction-surfaces.bin";
    if(std::filesystem::exists(junctionPath)){
        Reader junctions(junctionPath);
        if(junctions.read<uint32_t>()!=0x314e554a||junctions.read<uint32_t>()!=assetCrc32(path))
            throw std::runtime_error("Junction surfaces do not match the base map");
        const auto count=junctions.read<uint32_t>();if(!count||count>20000)throw std::runtime_error("Invalid junction surface count");
        for(uint32_t i=0;i<count;++i){
            const auto center=junctions.read<V3>();const float radius=junctions.read<float>(),range=junctions.read<float>();
            if(!std::isfinite(center.x)||!std::isfinite(center.y)||!std::isfinite(center.z)||!std::isfinite(radius)||!std::isfinite(range)||radius<=0||range<=0)
                throw std::runtime_error("Invalid junction surface bounds");
            auto mesh=readMesh(junctions);if(!mesh.edges.empty())throw std::runtime_error("Unexpected junction outlines");
            // No collision or caption entity: this only paints existing road gaps.
            object(uint32_t(meshes.size()),transform({0,0,0}),AsphaltColor,center,radius,range,2);
            meshes.push_back(std::move(mesh));
        }
        if(junctions.file.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("Trailing junction surface data");
    }
}
void World::upload(Renderer& renderer){uploadAll({&renderer});}
void World::uploadAll(const std::vector<Renderer*>& renderers){for(auto* renderer:renderers){renderer->makeCurrent();for(uint32_t i=0;i<meshes.size();++i)if(renderer->addMesh(meshes[i])!=i)throw std::runtime_error("Mesh ordering mismatch");}}
void World::object(uint32_t mesh,const Matrix& m,Color color,V3 center,float radius,float range,float surface){statics.push_back({mesh,{m,color,surface},center,radius,range});}
void World::box(V3 p,V3 size,Color color,bool collision,float yaw,float surface,float range){object(cubeMesh,transform(p,size,yaw),color,p,size.length()*.5f,range,surface);if(collision)physics.addBox(p,size*.5f,yaw,surface>.5f);}
V3 World::lanePoint(uint32_t id,float along)const{
    const auto& l=lanes.at(id);along=std::clamp(along,0.f,l.length);auto it=std::upper_bound(l.cumulative.begin(),l.cumulative.end(),along);
    size_t i=std::clamp(size_t(it-l.cumulative.begin()),size_t(1),l.points.size()-1);float span=l.cumulative[i]-l.cumulative[i-1];
    return l.points[i-1]+(l.points[i]-l.points[i-1])*(span>.001f?(along-l.cumulative[i-1])/span:0);
}
V3 World::laneForward(uint32_t id,float along)const{return (lanePoint(id,along+1)-lanePoint(id,along-1)).normalized();}
float World::closestAlong(uint32_t id,V3 p)const{
    const auto& l=lanes[id];float best=std::numeric_limits<float>::max(),along=0;
    for(size_t i=1;i<l.points.size();++i){V3 a=l.points[i-1],d=l.points[i]-a;float t=std::clamp(dot(p-a,d)/std::max(.00001f,dot(d,d)),0.f,1.f),distance=(a+d*t-p).length();
        if(distance<best){best=distance;along=l.cumulative[i-1]+(l.cumulative[i]-l.cumulative[i-1])*t;}}
    return along;
}
uint32_t World::chooseNext(uint32_t id,uint32_t& random)const{const auto& next=lanes[id].next;return next.empty()?id:next[randomStep(random)%next.size()];}
void World::refreshSpawns(){
    V3 p=position(player().body);if(!multiplayer&&!carSpawns.empty()&&flatDistance(p,spawnCenter)<35)return;
    spawnCenter=p;carSpawns.clear();footSpawns.clear();parkingSpawns.clear();
    const auto positions=playerPositions();
    // Player bodies cannot move during this simulation update. Read their
    // positions once, instead of taking Jolt body locks inside every sort
    // comparison (thousands of candidates, up to five players each time).
    auto distance=[&](V3 point){float best=1e9f;for(auto center:positions)best=std::min(best,flatDistance(point,center));return best;};
    auto collect=[&](const auto& ids,auto& out,float spacing,float radius){for(auto id:ids){const auto& l=lanes[id];
        bool near=false;for(auto center:positions)if(flatDistance(lanePoint(id,closestAlong(id,center)),center)<radius){near=true;break;}if(!near)continue;
        for(float at=5;at<l.length-4;at+=spacing){V3 point=lanePoint(id,at);float d=distance(point);if(d<radius)out.push_back({id,at,point,d});}}
        std::sort(out.begin(),out.end(),[](const Spawn& a,const Spawn& b){return a.rankDistance<b.rankDistance;});};
    collect(carLanes,carSpawns,15.f,multiplayer?460.f:340.f);collect(footLanes,footSpawns,3.8f,multiplayer?280.f:220.f);
    for(uint32_t i=0;i<parking.size();++i)if(distance(parking[i].p)<(multiplayer?460:300))parkingSpawns.push_back(i);
}
std::optional<World::Spawn> World::findSpawn(bool pedestrian,uint32_t& random,bool recycling)const{
    const auto& candidates=pedestrian?footSpawns:carSpawns;
    if(candidates.empty())return {};
    V3 playerPos=position(player().body),forward{std::sin(player().heading),0,std::cos(player().heading)};
    size_t pool=recycling?candidates.size():std::min(candidates.size(),pedestrian?size_t(700):size_t(320)),start=randomStep(random)%pool;
    for(size_t attempt=0;attempt<pool;++attempt){
        const auto& s=candidates[(start+attempt)%pool];float distance=playerDistance(s.p);
        if(distance<(pedestrian?8.f:17.f))continue;
        bool ahead=dot(s.p-playerPos,forward)>0;
        if(recycling&&(multiplayer?(distance<(pedestrian?235.f:400.f)):(distance<(ahead?(pedestrian?120.f:180.f):(pedestrian?65.f:80.f))||distance>(ahead?(pedestrian?180.f:250.f):(pedestrian?90.f:130.f)))))continue;
        bool free=true;
        if(pedestrian){for(const auto& person:people)if((position(physics.pedestrian(person.body))-s.p).length()<2.3f){free=false;break;}}
        else {for(const auto& car:traffic)if(car.enabled&&(physics.carPosition(car.car)-s.p).length()<10){free=false;break;}}
        if(free)return s;
    }
    return {};
}
void World::build(int trafficCount,int pedestrianCount){
    uint32_t seed=20260908,id=carLanes.front();float best=1e9f;
    for(auto candidate:carLanes){const auto& l=lanes[candidate];if(l.flags&2||l.length<75)continue;V3 p=lanePoint(candidate,l.length*.35f);float score=flatDistance(p,{-120,0,-50});if(score<best){best=score;id=candidate;}}
    float along=std::min(25.f,lanes[id].length*.25f);V3 p=lanePoint(id,along),f=laneForward(id,along);
    const auto& model=models[0];physics.addCar(model,meshes[model.bodyMesh],p+V3{0,.9f,0},std::atan2(f.x,f.z));
    Traffic playerCar;playerCar.playerSlot=0;playerCar.lane=id;playerCar.along=along;playerCar.random=12993;playerCar.nextLane=chooseNext(id,playerCar.random);traffic.push_back(playerCar);
    refreshSpawns();
    for(int i=0;i<trafficCount;++i){auto chosen=findSpawn(false,seed);if(!chosen)throw std::runtime_error("Insufficient clear traffic spawn locations");auto s=*chosen;V3 direction=laneForward(s.lane,s.along);size_t mi=(i+1)%models.size();const auto& m=models[mi];
        Traffic t;t.car=physics.addCar(m,meshes[m.bodyMesh],s.p+V3{0,.9f,0},std::atan2(direction.x,direction.z));t.model=mi;t.lane=s.lane;t.along=s.along;t.random=randomStep(seed);t.nextLane=chooseNext(t.lane,t.random);traffic.push_back(t);}
    std::sort(parking.begin(),parking.end(),[&](const Parking& a,const Parking& b){return flatDistance(a.p,p)<flatDistance(b.p,p);});
    for(size_t i=0;i<std::min(parking.size(),size_t(40));++i){auto place=parking[i];place.p.y=physics.groundHeight(place.p)+.72f;const auto& m=models[place.model];
        Traffic t;t.car=physics.addCar(m,meshes[m.bodyMesh],place.p,place.yaw);t.model=place.model;t.random=randomStep(seed);t.parked=true;traffic.push_back(t);physics.drive(t.car,{0,0,1,1});}
    carSpawns.clear();refreshSpawns();
    for(int i=0;i<pedestrianCount;++i){auto chosen=findSpawn(true,seed);if(!chosen)throw std::runtime_error("Insufficient clear pedestrian spawn locations");auto s=*chosen;Walker w;w.body=physics.addPedestrian(s.p+V3{0,.78f,0});w.lane=s.lane;w.along=s.along;w.random=randomStep(seed);w.phase=float(i%10)*.6f;w.speed=1.1f+float(i%7)*.08f;people.push_back(w);}
    physics.optimize();frameObjects.reserve(statics.size()+traffic.size()*5+people.size()*6);
}
void World::buildTestTrack(){
    box({0,-.2f,0},{1240,.4f,1240},rgb(153,154,152),true,0,1,2000);
    box({0,.6f,616},{1240,1.2f,.65f},rgb(139,101,108),true);
    Lane l;l.points={{4.2f,0,-600},{4.2f,0,600}};l.cumulative={0,1200};l.length=1200;l.next={0};lanes.push_back(l);carLanes.push_back(0);
    const auto& m=models[0];physics.addCar(m,meshes[m.bodyMesh],{4.2f,.9f,-38},0);Traffic p;p.playerSlot=0;p.along=562;traffic.push_back(p);
    Traffic parked;parked.car=physics.addCar(m,meshes[m.bodyMesh],{15,.9f,42},0);parked.parked=true;traffic.push_back(parked);physics.drive(1,{0,0,1,1});
    physics.optimize();sourceObjectCount=statics.size();
}
Input World::driver(Traffic& t,float dt){
    if(t.parked)return {0,0,1,1};
    auto pose=physics.car(t.car);V3 p=position(pose.body);t.along=closestAlong(t.lane,p);
    if(lanes[t.lane].length-t.along<2.5f){t.lane=t.nextLane;t.along=closestAlong(t.lane,p);t.nextLane=chooseNext(t.lane,t.random);}
    float speed=pose.speed/3.6f,lookAhead=std::clamp(7.5f+std::abs(speed)*.4f,8.f,19.f),remaining=lanes[t.lane].length-t.along;
    V3 target=remaining<lookAhead?lanePoint(t.nextLane,lookAhead-remaining):lanePoint(t.lane,t.along+lookAhead);V3 delta=target-p;delta.y=0;
    float angle=std::remainder(std::atan2(delta.x,delta.z)-pose.heading,2*Pi),desiredSpeed=(lanes[t.lane].flags&2)?22.f:12.5f;
    if(std::abs(angle)>.22f||(lanes[t.lane].flags&1))desiredSpeed=7.f;
    if(std::abs(angle)>.8f)desiredSpeed=4.f;
    if(lanes[t.lane].next.empty())desiredSpeed=std::min(desiredSpeed,std::max(0.f,(remaining-5)*.7f));
    for(const auto& other:traffic)if(other.enabled&&other.car!=t.car){
        V3 d=physics.carPosition(other.car)-p;if(std::abs(d.y)>2.5f)continue;d.y=0;V3 f{std::sin(pose.heading),0,std::cos(pose.heading)};
        float ahead=dot(d,f);if(ahead>0&&ahead<21&&std::abs(dot(d,cross({0,1,0},f)))<2.15f)desiredSpeed=std::min(desiredSpeed,std::max(0.f,(ahead-7)*.85f));
    }
    if(std::abs(speed)<.4f&&desiredSpeed>1)t.stalled+=dt;else t.stalled=0;
    if(t.stalled>3&&t.stalled<4.8f)return {-.4f,-std::clamp(angle,-1.f,1.f),0,0};if(t.stalled>=4.8f)t.stalled=0;
    return {speed<desiredSpeed?.65f:0,std::clamp(angle*1.7f,-1.f,1.f),speed>desiredSpeed+.7f?.7f:0,0};
}
void World::recyclePopulation(){
    if(testing)return;refreshSpawns();V3 p=position(player().body),forward{std::sin(player().heading),0,std::cos(player().heading)};
    auto distant=[&](V3 q,bool foot){float d=multiplayer?playerDistance(q):flatDistance(q,p);return d>(multiplayer?(foot?240.f:410.f):(foot?210.f:300.f))||(!multiplayer&&dot(q-p,forward)<0&&d>(foot?105.f:155.f))||q.y< -15;};
    for(size_t i=1;i<traffic.size();++i){auto& t=traffic[i];if(t.playerSlot>=0||!t.enabled||!distant(physics.carPosition(t.car),false))continue;
        if(t.parked){
            if(parkingSpawns.empty())continue;size_t start=randomStep(t.random)%parkingSpawns.size();
            for(size_t attempt=0;attempt<parkingSpawns.size();++attempt){auto place=parking[parkingSpawns[(start+attempt)%parkingSpawns.size()]];float d=flatDistance(place.p,p);
                bool ahead=dot(place.p-p,forward)>0;if(multiplayer?playerDistance(place.p)<400:(d<(ahead?150:70)||d>(ahead?250:130)))continue;
                bool free=true;for(const auto& car:traffic)if(car.enabled&&car.car!=t.car&&flatDistance(physics.carPosition(car.car),place.p)<7){free=false;break;}if(!free)continue;
                place.p.y=physics.groundHeight(place.p)+.9f;physics.resetCar(t.car,place.p,place.yaw);physics.drive(t.car,{0,0,1,1});t.captionEntity=0;break;
            }continue;
        }
        auto chosen=findSpawn(false,t.random,true);if(!chosen)continue;auto s=*chosen;auto f=laneForward(s.lane,s.along);physics.resetCar(t.car,s.p+V3{0,.9f,0},std::atan2(f.x,f.z));t.lane=s.lane;t.along=s.along;t.nextLane=chooseNext(t.lane,t.random);t.stalled=0;t.captionEntity=0;}
    for(auto& w:people)if(distant(position(physics.pedestrian(w.body)),true)){auto chosen=findSpawn(true,w.random,true);if(!chosen)continue;auto s=*chosen;physics.resetPedestrian(w.body,s.p+V3{0,.78f,0});w.lane=s.lane;w.along=s.along;w.captionEntity=0;}
}
void World::update(float dt,Input input,bool autodrive){
    physics.setCarLateralGrip(0,autodrive?1:playerFrontGrip,autodrive?1:playerRearGrip);
    physics.setCarFrontDrive(0,autodrive?.5f:playerFrontDrive);
    controls=autodrive?driver(traffic[0],dt):input;
    physics.drive(0,controls,autodrive||playerSteeringLimit,autodrive?0:PlayerMaxSpeedKmh);advance(dt);
    if(position(player().body).y< -20)reset();
}
void World::advance(float dt){
    simulationTime+=dt;auto before=player();
    if(int(simulationTime*25)!=int((simulationTime-dt)*25)){
        const double start=now();
        for(size_t i=1;i<traffic.size();++i)if(traffic[i].enabled&&traffic[i].playerSlot<0){physics.setCarDetail(traffic[i].car,playerDistance(physics.carPosition(traffic[i].car)));physics.drive(traffic[i].car,driver(traffic[i],.04f));}
        for(auto& w:people){if(physics.pedestrianFallen(w.body))continue;V3 p=position(physics.pedestrian(w.body));w.along=closestAlong(w.lane,p);
            if(lanes[w.lane].length-w.along<.8f){w.lane=chooseNext(w.lane,w.random);w.along=closestAlong(w.lane,p);}
            V3 d=lanePoint(w.lane,w.along+1.5f)-p;d.y=0;physics.walk(w.body,d.normalized()*w.speed);}
        workTimings[0].add(now()-start);
    }
    const double physicsStart=now();physics.step(dt);workTimings[1].add(now()-physicsStart);
    auto after=player();distance+=(position(after.body)-position(before.body)).length();
    if(!testing&&int(simulationTime)!=int(simulationTime-dt)){const double start=now();recyclePopulation();workTimings[2].add(now()-start);}
}
std::vector<V3> World::playerPositions()const{
    std::vector<V3> out;
    for(size_t slot=0;slot<(multiplayer?MaxPlayers:1);++slot){const auto& t=traffic.at(playerTraffic[slot]);if(t.enabled)out.push_back(physics.carPosition(t.car));}
    if(out.empty())out.push_back(physics.carPosition(0));return out;
}
float World::playerDistance(V3 p)const{
    float best=1e9f;
    for(size_t slot=0;slot<(multiplayer?MaxPlayers:1);++slot){const auto& t=traffic.at(playerTraffic[slot]);if(t.enabled)best=std::min(best,flatDistance(p,physics.carPosition(t.car)));}
    return best==1e9f?flatDistance(p,physics.carPosition(0)):best;
}
void World::setPlayerLateralGrip(float front,float rear){
    physics.setCarLateralGrip(0,front,rear);playerFrontGrip=front;playerRearGrip=rear;
    if(multiplayer)for(size_t slot=1;slot<MaxPlayers;++slot)physics.setCarLateralGrip(playerCar(slot),front,rear);
}
void World::setPlayerFrontDrive(float share){
    physics.setCarFrontDrive(0,share);playerFrontDrive=share;
    if(multiplayer)for(size_t slot=1;slot<MaxPlayers;++slot)physics.setCarFrontDrive(playerCar(slot),share);
}
void World::enableMultiplayer(){
    if(multiplayer)return;
    for(size_t slot=1;slot<MaxPlayers;++slot){
        const auto& m=models[0];Traffic t;t.playerSlot=int(slot);t.lane=traffic[0].lane;t.nextLane=traffic[0].nextLane;
        t.car=physics.addCar(m,meshes[m.bodyMesh],physics.carPosition(0)+V3{float(slot)*6,0,0},0);
        physics.setCarLateralGrip(t.car,playerFrontGrip,playerRearGrip);
        physics.setCarFrontDrive(t.car,playerFrontDrive);
        playerTraffic[slot]=traffic.size();traffic.push_back(t);
    }
    multiplayer=true;
    for(size_t slot=0;slot<MaxPlayers;++slot)setPlayerActive(slot,false);
}
void World::setPlayerActive(size_t slot,bool active){auto& t=traffic.at(playerTraffic.at(slot));t.enabled=active;physics.setCarEnabled(t.car,active);}
CarPose World::playerPose(size_t slot)const{return physics.car(playerCar(slot));}
void World::updatePlayers(float dt,const std::array<Input,MaxPlayers>& inputs){
    for(size_t slot=0;slot<MaxPlayers;++slot){auto& t=traffic.at(playerTraffic[slot]);if(t.enabled){
        const bool autonomous=autonomousPlayers[slot];physics.setCarDetail(t.car,0);
        physics.setCarLateralGrip(t.car,autonomous?1:playerFrontGrip,autonomous?1:playerRearGrip);
        physics.setCarFrontDrive(t.car,autonomous?.5f:playerFrontDrive);
        physics.drive(t.car,inputs[slot],autonomous||playerSteeringLimit,autonomous?0:PlayerMaxSpeedKmh);}}
    controls=inputs[0];advance(dt);
}
std::array<V3,4> World::playerFootprint(size_t slot)const{
    const auto& model=models.at(traffic.at(playerTraffic.at(slot)).model);const auto pose=playerPose(slot);
    const auto c=model.bodyCenter,h=model.halfExtent;
    return {transformPoint(c+V3{-h.x,0,-h.z},pose.body),transformPoint(c+V3{h.x,0,-h.z},pose.body),
        transformPoint(c+V3{h.x,0,h.z},pose.body),transformPoint(c+V3{-h.x,0,h.z},pose.body)};
}
bool World::playerPlacementClear(size_t slot,V3 ground,float heading)const{
    return physics.clearCarPlacement(playerCar(slot),ground+V3{0,.9f,0},heading);
}
World::RoadSample World::roadSample(uint32_t lane,float along)const{return {lanePoint(lane,along),laneForward(lane,along),lanes.at(lane).width};}
float World::obstacleAhead(size_t slot,float range,float halfAngle,bool includeParked)const{
    auto pose=playerPose(slot);V3 p=position(pose.body),forward{std::sin(pose.heading),0,std::cos(pose.heading)};float best=range;
    auto consider=[&](V3 q){V3 d=q-p;if(includeParked&&std::abs(d.y)>2.5f)return;d.y=0;float along=dot(d,forward);if(along<=.5f||along>=best)return;
        if(std::abs(d.x*forward.z-d.z*forward.x)>std::tan(halfAngle)*along+1.1f)return;best=along;};
    for(const auto& t:traffic)if(t.enabled&&(includeParked||!t.parked)&&t.playerSlot!=int(slot))consider(physics.carPosition(t.car));
    for(const auto& w:people)consider(position(physics.pedestrian(w.body)));
    return best;
}
void World::placePlayer(size_t slot,V3 ground,float heading){physics.resetCar(playerCar(slot),ground+V3{0,.9f,0},heading);}
bool World::clearRecovery(size_t slot,V3 p,float heading)const{
    auto support=physics.groundSupport(p);if(!support||std::abs(*support-p.y)>.45f)return false;
    const V3 center=p+V3{0,.9f,0};
    for(const auto& t:traffic)if(t.enabled&&t.car!=playerCar(slot)&&(physics.carPosition(t.car)-center).length()<6.5f)return false;
    for(const auto& w:people)if((position(physics.pedestrian(w.body))-center).length()<3.5f)return false;
    return physics.clearCarPlacement(playerCar(slot),center,heading);
}
void World::clearStartingGrid(const std::vector<V3>& grid){
    if(testing)return;carSpawns.clear();refreshSpawns();
    auto near=[&](V3 p){for(auto g:grid)if((p-g).length()<8)return true;return false;};
    for(auto& t:traffic)if(t.playerSlot<0&&near(physics.carPosition(t.car))){
        auto spawn=findSpawn(false,t.random,true);if(!spawn)throw std::runtime_error("No safe NPC relocation for starting grid");
        auto s=*spawn;auto f=laneForward(s.lane,s.along);physics.resetCar(t.car,s.p+V3{0,.9f,0},std::atan2(f.x,f.z));
        t.lane=s.lane;t.along=s.along;t.nextLane=chooseNext(s.lane,t.random);t.stalled=0;t.captionEntity=0;
    }
    for(auto& w:people)if(near(position(physics.pedestrian(w.body)))){
        auto spawn=findSpawn(true,w.random,true);if(!spawn)throw std::runtime_error("No safe pedestrian relocation for starting grid");
        auto s=*spawn;physics.resetPedestrian(w.body,s.p+V3{0,.78f,0});w.lane=s.lane;w.along=s.along;w.captionEntity=0;
    }
}
void World::reset(){
    if(testing){physics.resetCar(0,{4.2f,.9f,-38},0);traffic[0].along=562;return;}
    V3 p=position(player().body);float best=1e9f;uint32_t id=carLanes.front();float at=0;
    for(auto candidate:carLanes){float s=std::clamp(closestAlong(candidate,p),4.f,std::max(4.f,lanes[candidate].length-6));float d=(lanePoint(candidate,s)-p).length();if(d<best){best=d;id=candidate;at=s;}}
    for(int attempt=0;attempt<20;++attempt){bool free=true;V3 q=lanePoint(id,at);for(size_t i=1;i<traffic.size();++i)if((physics.carPosition(traffic[i].car)-q).length()<8){free=false;break;}if(free)break;at=std::max(4.f,at-9);}
    V3 f=laneForward(id,at);physics.resetCar(0,lanePoint(id,at)+V3{0,.9f,0},std::atan2(f.x,f.z));auto& t=traffic[0];t.lane=id;t.along=at;t.nextLane=chooseNext(id,t.random);t.stalled=0;
}
void World::placeAtFreeway(){
    for(auto id:carLanes)if((lanes[id].flags&2)&&lanes[id].length>140){float s=lanes[id].length*.4f;V3 p=lanePoint(id,s);if(p.y<6)continue;V3 f=laneForward(id,s);
        physics.resetCar(0,p+V3{0,.9f,0},std::atan2(f.x,f.z));auto& t=traffic[0];t.lane=id;t.along=s;t.nextLane=chooseNext(id,t.random);return;}
    throw std::runtime_error("No elevated freeway route found");
}
Camera World::overviewCamera()const{V3 center=(cityMin+cityMax)*.5f;center.y=0;return {center+V3{0,1600,-420},center,90};}
void World::placeAtUnderpass(){
    physics.resetCar(0,{172.4f,1.55f,327.5f},Pi-.15f);
    for(int i=0;i<500;++i){physics.drive(0,{0,0,1,1});physics.step(.002f);}
}
bool World::underpassFixture(const std::filesystem::path& root){
    std::filesystem::create_directories(root/L"benchmarks");
    std::ofstream samples(root/L"benchmarks/underpass-heights.csv");samples<<"x,z,ground_height_m,step_m\n";
    float maxStep=0,minHeight=1e9f,maxHeight=-1e9f;int count=0;
    for(float x:{171.5f,172.25f,173.f}){
        float previous=0;
        for(int i=0;i<=248;++i){
            float z=284+i*.25f,height=physics.groundHeight({x,-2.2f,z});
            float step=i?std::abs(height-previous):0;previous=height;
            minHeight=std::min(minHeight,height);maxHeight=std::max(maxHeight,height);maxStep=std::max(maxStep,step);++count;
            samples<<x<<','<<z<<','<<height<<','<<step<<'\n';
        }
    }
    std::ofstream trace(root/L"benchmarks/underpass-driving.csv");trace<<"direction,time_s,x,y,z,speed_kmh,grounded_wheels\n";
    int supported=0,drivingSteps=0;float minDistance=1e9f,maxVerticalSpeed=0;
    for(int direction:{-1,1}){
        float start=direction<0?333.f:294.5f;
        physics.resetCar(0,{173.5f,1.65f,start},direction<0?Pi:0);
        for(int i=0;i<600;++i){physics.drive(0,{0,0,1,1});physics.step(.002f);}
        V3 before=physics.carPosition(0);
        for(int i=0;i<6500;++i){
            auto pose=physics.car(0);float speed=std::abs(pose.speed);
            V3 p=position(pose.body);float desired=std::atan2(173.5f-p.x,float(direction)*8);
            float steer=std::clamp(std::remainder(desired-pose.heading,2*Pi)*1.7f,-.5f,.5f);
            physics.drive(0,{speed<18?.3f:0,steer,speed>20?.3f:0,0});physics.step(.002f);
            pose=physics.car(0);supported+=pose.groundedWheels>=2;++drivingSteps;
            maxVerticalSpeed=std::max(maxVerticalSpeed,std::abs(pose.velocity.y));
            if(i%50==0){auto p=position(pose.body);trace<<direction<<','<<i*.002f<<','<<p.x<<','<<p.y<<','<<p.z<<','<<pose.speed<<','<<pose.groundedWheels<<'\n';}
            if((position(pose.body).z-before.z)*direction>=38.5f)break;
        }
        minDistance=std::min(minDistance,std::abs(physics.carPosition(0).z-before.z));
    }
    bool passed=minHeight>.5f&&maxHeight<.8f&&maxStep<.04f&&minDistance>=38.5f&&supported>drivingSteps*.98f&&maxVerticalSpeed<1.5f;
    std::ofstream result(root/L"benchmarks/underpass-tests.json");
    result<<"{\"passed\":"<<(passed?"true":"false")<<",\"samples\":"<<count<<",\"min_height_m\":"<<minHeight<<",\"max_height_m\":"<<maxHeight<<",\"max_step_m\":"<<maxStep
        <<",\"min_drive_distance_m\":"<<minDistance<<",\"supported_percent\":"<<100.f*supported/drivingSteps<<",\"max_vertical_speed_m_s\":"<<maxVerticalSpeed<<"}\n";
    return passed;
}
void World::placeAtRampSide(){
    physics.resetCar(0,{-.3f,1.65f,-114.5f},-.05f);
    for(int i=0;i<500;++i){physics.drive(0,{0,0,1,1});physics.step(.002f);}
}
bool World::rampGroundFixture(const std::filesystem::path& root){
    std::filesystem::create_directories(root/L"benchmarks");
    std::ofstream samples(root/L"benchmarks/ramp-ground-heights.csv");samples<<"x,z,ground_height_m,longitudinal_step_m,lateral_step_m\n";
    float minHeight=1e9f,maxHeight=-1e9f,maxStep=0;int count=0;
    std::array<float,373> previousRow{};
    for(int col=0;col<=8;++col){
        float x=-1.5f+col*.25f,previous=0;
        for(int row=0;row<=372;++row){
            float z=-119+row*.25f,height=physics.groundHeight({x,-2.2f,z});
            float along=row?std::abs(height-previous):0,across=col?std::abs(height-previousRow[row]):0;
            previous=height;previousRow[row]=height;
            minHeight=std::min(minHeight,height);maxHeight=std::max(maxHeight,height);maxStep=std::max({maxStep,along,across});++count;
            samples<<x<<','<<z<<','<<height<<','<<along<<','<<across<<'\n';
        }
    }
    float maxApproachStep=0,rampMin=1e9f,rampMax=-1e9f,rampMaxStep=0;
    for(float x:{-1.5f,-.5f,.5f})for(float start:{-126.f,-24.f}){
        float previous=physics.groundHeight({x,-2.2f,start});
        for(int i=1;i<=16;++i){
            float height=physics.groundHeight({x,-2.2f,start+i*.25f});
            maxApproachStep=std::max(maxApproachStep,std::abs(height-previous));previous=height;
        }
    }
    float previousRamp=0;
    for(int i=0;i<=384;++i){
        float height=physics.groundHeight({-9,6,-120+i*.25f});
        rampMin=std::min(rampMin,height);rampMax=std::max(rampMax,height);
        if(i)rampMaxStep=std::max(rampMaxStep,std::abs(height-previousRamp));previousRamp=height;
    }
    std::ofstream trace(root/L"benchmarks/ramp-ground-driving.csv");trace<<"direction,time_s,x,y,z,speed_kmh,grounded_wheels\n";
    int supported=0,steps=0;float minDistance=1e9f,maxVerticalSpeed=0;
    for(int direction:{-1,1}){
        // Cross the former drop, stopping before the original access bollards.
        float start=direction>0?-119.5f:-109.5f;
        physics.resetCar(0,{-.3f,1.65f,start},direction>0?0:Pi);
        for(int i=0;i<600;++i){physics.drive(0,{0,0,1,1});physics.step(.002f);}
        V3 before=physics.carPosition(0);
        for(int i=0;i<4000;++i){
            auto pose=physics.car(0);auto p=position(pose.body);float speed=std::abs(pose.speed);
            float desired=std::atan2(-.3f-p.x,float(direction)*6),steer=std::clamp(std::remainder(desired-pose.heading,2*Pi)*1.7f,-.5f,.5f);
            physics.drive(0,{speed<12?.25f:0,steer,speed>14?.3f:0,0});physics.step(.002f);
            pose=physics.car(0);supported+=pose.groundedWheels>=2;++steps;maxVerticalSpeed=std::max(maxVerticalSpeed,std::abs(pose.velocity.y));
            if(i%50==0){auto q=position(pose.body);trace<<direction<<','<<i*.002f<<','<<q.x<<','<<q.y<<','<<q.z<<','<<pose.speed<<','<<pose.groundedWheels<<'\n';}
            if((position(pose.body).z-before.z)*direction>=10.f)break;
        }
        minDistance=std::min(minDistance,std::abs(physics.carPosition(0).z-before.z));
    }
    bool passed=minHeight>.5f&&maxHeight<.8f&&maxStep<.04f&&maxApproachStep<.04f&&minDistance>=10.f&&supported>steps*.98f&&maxVerticalSpeed<1.5f&&rampMax>4&&rampMin>.5f&&rampMaxStep<.1f;
    std::ofstream result(root/L"benchmarks/ramp-ground-tests.json");
    result<<"{\"passed\":"<<(passed?"true":"false")<<",\"samples\":"<<count<<",\"min_height_m\":"<<minHeight<<",\"max_height_m\":"<<maxHeight<<",\"max_step_m\":"<<maxStep
        <<",\"max_approach_step_m\":"<<maxApproachStep<<",\"min_drive_distance_m\":"<<minDistance<<",\"supported_percent\":"<<100.f*supported/steps<<",\"max_vertical_speed_m_s\":"<<maxVerticalSpeed
        <<",\"elevated_ramp_min_height_m\":"<<rampMin<<",\"elevated_ramp_max_height_m\":"<<rampMax<<",\"elevated_ramp_max_step_m\":"<<rampMaxStep<<"}\n";
    return passed;
}
std::vector<std::pair<std::string,Camera>> World::rampSurveyCameras(const std::filesystem::path& root)const{
    std::vector<std::pair<std::string,Camera>> result;
    for(const auto& site:readRampSites(root))for(int side:{-1,1})for(int end:{0,1}){
        V3 right=cross({0,1,0},site.forward),offset=right*(side*(site.width*.5f-1));
        float along=end?site.length-4:4,direction=end?-1.f:1.f;
        V3 eye=site.origin+site.forward*along+offset,target=eye+site.forward*(25*direction);eye.y=3.4f;target.y=.74f;
        result.push_back({"ramp-audit-"+std::to_string(site.id)+(side<0?"-left":"-right")+(end?"-end":"-start"),{eye,target,90}});
    }
    return result;
}
bool World::allRampsFixture(const std::filesystem::path& root)const{
    std::filesystem::create_directories(root/L"benchmarks");
    std::ofstream csv(root/L"benchmarks/all-ramps-heights.csv");csv<<"ramp,side,along_m,across_m,x,z,height_m,longitudinal_step_m,lateral_step_m\n";
    std::ofstream result(root/L"benchmarks/all-ramps-tests.json");
    const auto sites=readRampSites(root);bool passed=true;int total=0,failed=0;
    result<<"{\"regions\":[";bool first=true;
    for(const auto& site:sites)for(int side:{-1,1}){
        V3 right=cross({0,1,0},site.forward);int rows=int((site.length+4)*4)+1,count=0;std::vector<float> previousRow(rows);
        float low=1e9f,high=-1e9f,maxStep=0,approachStep=0;
        for(int col=0;col<=16;++col){
            float across=site.width*.5f-2.25f+col*.25f,previous=0;
            for(int row=0;row<rows;++row){
                float along=-4+row*.25f;V3 p=site.origin+site.forward*along+right*(side*across);p.y=-2.2f;
                float height=physics.groundHeight(p,true),longitudinal=row?std::abs(height-previous):0,lateral=col?std::abs(height-previousRow[row]):0;
                previous=height;previousRow[row]=height;low=std::min(low,height);high=std::max(high,height);maxStep=std::max({maxStep,longitudinal,lateral});++count;
                csv<<site.id<<','<<side<<','<<along<<','<<across<<','<<p.x<<','<<p.z<<','<<height<<','<<longitudinal<<','<<lateral<<'\n';
            }
        }
        for(float start:{-13.25f,site.length-.25f}){
            float previous=0;
            for(int i=0;i<=34;++i){V3 p=site.origin+site.forward*(start+i*.25f)+right*(side*(site.width*.5f-1));p.y=-2.2f;
                float height=physics.groundHeight(p,true),step=i?std::abs(height-previous):0;approachStep=std::max(approachStep,step);previous=height;
                csv<<site.id<<','<<side<<','<<start+i*.25f<<','<<site.width*.5f-1<<','<<p.x<<','<<p.z<<','<<height<<','<<step<<",0\n";}
        }
        bool ok=low>.5f&&high<.8f&&high-low<.02f&&maxStep<.02f&&approachStep<.02f;
        passed&=ok;failed+=!ok;total+=count;
        result<<(first?"":",")<<"{\"ramp\":"<<site.id<<",\"side\":"<<side<<",\"passed\":"<<(ok?"true":"false")<<",\"samples\":"<<count<<",\"min_height_m\":"<<low<<",\"max_height_m\":"<<high<<",\"max_step_m\":"<<maxStep<<",\"max_approach_step_m\":"<<approachStep<<'}';first=false;
    }
    passed&=sites.size()==8;
    result<<"],\"passed\":"<<(passed?"true":"false")<<",\"entrances\":"<<sites.size()<<",\"side_corridors\":"<<sites.size()*2<<",\"samples\":"<<total<<",\"failed_regions\":"<<failed<<"}\n";
    return passed;
}
Camera World::camera(int mode,bool lookBack)const{
    return playerCamera(0,mode,lookBack);
}
Camera World::playerCamera(size_t slot,int mode,bool lookBack)const{
    auto pose=playerPose(slot);V3 p=position(pose.body),forward{std::sin(pose.heading),0,std::cos(pose.heading)};if(lookBack)forward=forward*-1;
    Camera c;c.horizontalFov=90;
    if(mode==0){c.eye=p-forward*6.9f+V3{0,2.75f,0};c.target=p+forward*5.3f+V3{0,.65f,0};}
    else if(mode==1){
        // UE camera offsets use the actor origin; convert-assets.mjs raises our
        // vehicle origin by 0.65 m. Raise the dataset-matched view by 0.30 m
        // to expose more road surface without changing its pitch or FOV.
        c.eye=p-forward*6.f+V3{0,2.45f-.65f+.30f,0};
        c.target=c.eye+forward*12.f+V3{0,-12.f*std::tan(13.15f*Pi/180),0};
    }
    else{c.eye=p+forward*1.75f+V3{0,.92f,0};c.target=c.eye+forward*12+V3{0,-.15f,0};}
    c.eye=physics.cameraCollision(p+V3{0,1,0},c.eye);return c;
}
std::array<int,2> World::nearbyPopulation()const{
    std::array<int,2> count{};V3 p=position(player().body);for(size_t i=1;i<traffic.size();++i)if(!traffic[i].parked&&flatDistance(physics.carPosition(traffic[i].car),p)<180)++count[0];
    for(const auto& w:people)if(flatDistance(position(physics.pedestrian(w.body)),p)<180)++count[1];return count;
}
std::array<int,2> World::populationMotionFixture(){
    std::vector<V3> carsBefore,peopleBefore;for(const auto& t:traffic)carsBefore.push_back(physics.carPosition(t.car));for(const auto& w:people)peopleBefore.push_back(position(physics.pedestrian(w.body)));
    for(int i=0;i<3000;++i)update(.002f,{0,0,1,1});
    std::array<int,2> result{};for(size_t i=1;i<traffic.size();++i){float d=flatDistance(physics.carPosition(traffic[i].car),carsBefore[i]);if(!traffic[i].parked&&d>3&&d<100)++result[0];}
    for(size_t i=0;i<people.size();++i){float d=flatDistance(position(physics.pedestrian(people[i].body)),peopleBefore[i]);if(d>1&&d<15)++result[1];}return result;
}
std::array<int,2> World::groundCoverageFixture(std::vector<std::array<float,4>>* misses)const{
    int failures=0;for(auto p:groundGapProbes){
        // Start just above the repair so an overhead bridge cannot hide a missing floor.
        float actual=physics.groundHeight(p+V3{0,-2.8f,0});
        // Overlapping original curbs may sit above the repair and also provide valid support.
        if(actual<p.y-.04f){++failures;if(misses)misses->push_back({p.x,p.y,p.z,actual});}
    }
    return {int(groundGapProbes.size()),failures};
}
const std::vector<DrawObject>& World::drawObjects(){
    frameObjects=statics;
    if(captionsEnabled)frameEntities=staticEntities;
    float token=0;
    const auto entity=[&](uint64_t& id,uint32_t captionId,V3 p,bool ego){
        if(!captionsEnabled)return;
        if(!id)id=nextCaptionEntity++;
        token=float(frameEntities.size());frameEntities.push_back({id,captionId,p,true,ego});
    };
    auto push=[&](uint32_t mesh,Matrix m,Color c,float radius,float range=240,float surface=0){frameObjects.push_back({mesh,{m,c,surface,{token,0,0}},position(m),radius,range});};
    for(auto& t:traffic){if(!t.enabled)continue;auto pose=physics.car(t.car);const auto& model=models[t.model];
        entity(t.captionEntity,captionsEnabled?vehicleCaptionIds.at(t.model):0,position(pose.body),t.car==playerCar(0));
        push(model.bodyMesh,pose.body,CarColor,4,380);
        for(int i=0;i<4;++i)push(model.wheelMeshes[i],pose.wheels[i],WheelColor,1,300);}
    for(auto& w:people){Matrix root=physics.pedestrian(w.body);entity(w.captionEntity,w.captionId,position(root),false);bool fallen=physics.pedestrianFallen(w.body);float swing=fallen?0:std::sin(float(simulationTime)*6+w.phase)*.35f;
        auto part=[&](V3 p,V3 size,Color c,float roll=0){push(cubeMesh,multiply(transform(p,size,0,roll),root),c,1,220);};
        part({0,.78f,0},{.26f,.29f,.25f},rgb(202,180,74));part({0,.34f,0},{.43f,.56f,.26f},rgb(76,127,158));
        part({-.32f,.31f,swing*.3f},{.16f,.6f,.17f},rgb(67,157,167),swing);part({.32f,.31f,-swing*.3f},{.16f,.6f,.17f},rgb(172,115,76),-swing);
        part({-.14f,-.40f,-swing*.4f},{.18f,.74f,.21f},rgb(77,153,93),-swing*.5f);part({.14f,-.40f,swing*.4f},{.18f,.74f,.21f},rgb(140,102,156),swing*.5f);
    }
    return frameObjects;
}
std::vector<DrawObject> World::outlineFixture()const{
    std::vector<DrawObject> objects;const float distances[]={12,40,140,400,900};
    for(int row=0;row<2;++row)for(int i=0;i<5;++i){float z=distances[i];V3 p{(-.8f+i*.4f)*z,(row==0?.22f:-.22f)*z,z};Color c=rgb(79+i*16,116+row*35,140-i*8);
        objects.push_back({row==0?models[0].bodyMesh:cubeMesh,{transform(p,row==0?V3{1,1,1}:V3{2,5,2},i*.27f),c,0},p,4,3000});}return objects;
}
std::vector<DrawObject> World::nearClipFixture()const{
    return {{cubeMesh,{transform({-3,-2,3},{3,8,20}),rgb(149,96,94),0},{-3,-2,3},12,100},
        {cubeMesh,{transform({3,-2,3},{3,8,20}),rgb(119,108,146),0},{3,-2,3},12,100},
        {models[0].bodyMesh,{transform({0,-.9f,.4f},{1,1,1},Pi*.5f),rgb(79,116,140),0},{0,-.9f,.4f},4,100}};
}
bool World::collisionFixture(){
    physics.resetCar(0,{4.2f,.9f,588},0);uint32_t before=physics.collisions();
    for(int i=0;i<3000;++i){physics.drive(0,{1,0,0,0});physics.step(.002f);}V3 p=physics.carPosition(0);
    bool wall=physics.collisions()>before&&p.z<616&&std::abs(physics.car(0).speed)<2;
    physics.resetCar(0,{4.2f,.9f,558},0);size_t walker=physics.addPedestrian({4.2f,.77f,574});
    for(int i=0;i<1700;++i){physics.drive(0,{.8f,0,0,0});physics.step(.002f);}bool person=physics.pedestrianFallen(walker);
    auto parked=physics.carPosition(1);physics.resetCar(0,parked+V3{0,.15f,-18},0);before=physics.collisions();
    for(int i=0;i<1800;++i){physics.drive(0,{1,0,0,0});physics.step(.002f);}bool car=physics.collisions()>before&&(physics.carPosition(1)-parked).length()>.5f;
    return wall&&person&&car;
}
std::array<float,4> World::steeringFixture(){
    physics.addBox({0,-.2f,1400},{90,.2f,90});physics.optimize();std::array<float,4> result{};
    for(int side=0;side<2;++side){physics.resetCar(0,{0,.9f,1400},0);
        for(int i=0;i<700;++i){physics.drive(0,{.7f,0,0,0});physics.step(.002f);}
        for(int i=0;i<450;++i){physics.drive(0,{.7f,side==0?.6f:-.6f,0,0});physics.step(.002f);}
        result[side*2]=physics.carPosition(0).x;result[side*2+1]=physics.car(0).heading;}
    return result;
}
