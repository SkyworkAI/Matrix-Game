#include "Encounter.h"
#include "Recording.h"
#include <map>
#include <set>
#include <numeric>
#include <limits>

namespace {
V3 native(const Json& p){return {p[1].get<float>()*.01f,p[2].get<float>()*.01f,p[0].get<float>()*.01f};}
V3 animPoint(const Json& p){return {-p[0].get<float>()*.01f,p[2].get<float>()*.01f,p[1].get<float>()*.01f};}
using Quat=DirectX::XMFLOAT4;
Quat animRotation(const Json& q){Quat out{-q[0].get<float>(),q[2].get<float>(),q[1].get<float>(),q[3].get<float>()};DirectX::XMStoreFloat4(&out,DirectX::XMQuaternionNormalize(DirectX::XMLoadFloat4(&out)));return out;}
Quat multiply(Quat a,Quat b){Quat q;DirectX::XMStoreFloat4(&q,DirectX::XMQuaternionMultiply(DirectX::XMLoadFloat4(&b),DirectX::XMLoadFloat4(&a)));return q;}
Quat inverse(Quat q){return {-q.x,-q.y,-q.z,q.w};}
Quat slerp(Quat a,Quat b,float t){Quat q;DirectX::XMStoreFloat4(&q,DirectX::XMQuaternionSlerp(DirectX::XMLoadFloat4(&a),DirectX::XMLoadFloat4(&b),t));return q;}
V3 rotate(V3 p,Quat q){DirectX::XMFLOAT3 v;DirectX::XMStoreFloat3(&v,DirectX::XMVector3Rotate(DirectX::XMVectorSet(p.x,p.y,p.z,0),DirectX::XMLoadFloat4(&q)));return {v.x,v.y,v.z};}
Quat axisRotation(V3 axis,float angle){Quat q;DirectX::XMStoreFloat4(&q,DirectX::XMQuaternionRotationAxis(DirectX::XMVectorSet(axis.x,axis.y,axis.z,0),angle));return q;}
Quat between(V3 from,V3 to){from=from.normalized();to=to.normalized();float d=std::clamp(dot(from,to),-1.f,1.f);
    if(d<-.999999f){V3 axis=std::abs(from.x)<.8f?V3{1,0,0}:V3{0,1,0};return axisRotation(cross(from,axis).normalized(),Pi);}
    V3 axis=cross(from,to);Quat q{axis.x,axis.y,axis.z,1+d};DirectX::XMStoreFloat4(&q,DirectX::XMQuaternionNormalize(DirectX::XMLoadFloat4(&q)));return q;}
Matrix boneTransform(V3 p,V3 scale,Quat q){Matrix m;DirectX::XMStoreFloat4x4(&m,DirectX::XMMatrixScaling(scale.x,scale.y,scale.z)*DirectX::XMMatrixRotationQuaternion(DirectX::XMLoadFloat4(&q))*DirectX::XMMatrixTranslation(p.x,p.y,p.z));return m;}
V3 bboxRotationDegreesUE(Matrix m){
    for(int row=0;row<3;++row){V3 axis{m.m[row][0],m.m[row][1],m.m[row][2]};axis=axis.normalized();m.m[row][0]=axis.x;m.m[row][1]=axis.y;m.m[row][2]=axis.z;}
    Quat q;DirectX::XMStoreFloat4(&q,DirectX::XMQuaternionRotationMatrix(DirectX::XMLoadFloat4x4(&m)));
    float x=-q.x,y=q.z,z=q.y,w=q.w,s=z*x-w*y,yaw=std::atan2(2*(w*z+x*y),1-2*(y*y+z*z))*180/Pi;
    float pitch,roll;if(std::abs(s)>.4999995f){pitch=s<0?-90.f:90.f;roll=(s<0?-yaw:yaw)-2*std::atan2(x,w)*180/Pi;roll=std::remainder(roll,360.f);}
    else{pitch=std::asin(std::clamp(2*s,-1.f,1.f))*180/Pi;roll=std::atan2(-2*(w*x+y*z),1-2*(x*x+y*y))*180/Pi;}
    return {pitch,yaw,roll};
}
V3 forward(float yaw,float pitch=0){return {std::sin(yaw)*std::cos(pitch),std::sin(pitch),std::cos(yaw)*std::cos(pitch)};}
V3 rotate(V3 p,float yaw){return {p.x*std::cos(yaw)+p.z*std::sin(yaw),p.y,-p.x*std::sin(yaw)+p.z*std::cos(yaw)};}
float flatLength(V3 p){return std::sqrt(p.x*p.x+p.z*p.z);}
float approach(float from,float to,float maxDelta){return from+std::clamp(to-from,-maxDelta,maxDelta);}
constexpr std::array<int,21> boneParents{-1,0,1,2,3,4,3,6,7,3,9,10,0,12,13,14,0,16,17,18,11};
std::vector<V3> blendPose(const std::vector<V3>& from,const std::vector<V3>& to,float alpha){
    if(from.size()!=boneParents.size()||to.size()!=boneParents.size())throw std::runtime_error("Unexpected animation skeleton");
    std::vector<V3> out(from.size());
    for(size_t b=0;b<out.size();++b){int p=boneParents[b];if(p<0){out[b]=from[b]*(1-alpha)+to[b]*alpha;continue;}
        V3 old=from[b]-from[p],target=to[b]-to[p];float oldLength=old.length(),targetLength=target.length();
        V3 start=old.normalized(),end=target.normalized();if(oldLength<.00001f)start=end;if(targetLength<.00001f)end=start;
        float cosine=std::clamp(dot(start,end),-1.f,1.f);V3 direction;
        if(cosine>.9995f)direction=(start*(1-alpha)+end*alpha).normalized();
        else{V3 tangent=(end-start*cosine).normalized();
            if(tangent.length()<.1f)tangent=cross(start,std::abs(start.x)<.8f?V3{1,0,0}:V3{0,1,0}).normalized();
            float angle=std::acos(cosine)*alpha;direction=start*std::cos(angle)+tangent*std::sin(angle);}
        // Blend segment directions and lengths separately so fast rotations cannot shorten limbs.
        out[b]=out[p]+direction*(oldLength*(1-alpha)+targetLength*alpha);
    }
    return out;
}
void solveLimb(V3 root,V3& joint,V3& tip,V3 target,V3 bendHint){
    float upper=(joint-root).length(),lower=(tip-joint).length();V3 delta=target-root;
    float distance=std::clamp(delta.length(),std::abs(upper-lower)+.001f,upper+lower-.001f);
    V3 direction=delta.normalized();if(direction.length()<.1f)direction={0,-1,0};
    V3 bend=(bendHint-direction*dot(bendHint,direction)).normalized();
    if(bend.length()<.1f)bend=cross(direction,{1,0,0}).normalized();
    float along=(upper*upper-lower*lower+distance*distance)/(2*distance);
    joint=root+direction*along+bend*std::sqrt(std::max(0.f,upper*upper-along*along));tip=root+direction*distance;
}
Color hexColor(const std::string& value){return rgb(std::stoi(value.substr(0,2),nullptr,16),std::stoi(value.substr(2,2),nullptr,16),std::stoi(value.substr(4,2),nullptr,16));}
Matrix aimTransform(V3 p,V3 scale,float yaw,float pitch){Matrix m;DirectX::XMStoreFloat4x4(&m,DirectX::XMMatrixScaling(scale.x,scale.y,scale.z)*DirectX::XMMatrixRotationRollPitchYaw(-pitch,yaw,0)*DirectX::XMMatrixTranslation(p.x,p.y,p.z));return m;}
V3 point(const Matrix& m,V3 p){DirectX::XMFLOAT3 out;DirectX::XMStoreFloat3(&out,DirectX::XMVector3TransformCoord(DirectX::XMVectorSet(p.x,p.y,p.z,1),DirectX::XMLoadFloat4x4(&m)));return {out.x,out.y,out.z};}
MeshData boxMesh(){MeshData m;for(int i=0;i<8;++i)m.vertices.push_back({i&1?.5f:-.5f,i&2?.5f:-.5f,i&4?.5f:-.5f});m.indices={0,2,3,0,3,1,4,5,7,4,7,6,0,1,5,0,5,4,2,6,7,2,7,3,0,4,6,0,6,2,1,3,7,1,7,5};for(uint32_t i=0;i<8;++i)for(uint32_t b:{1u,2u,4u})if(!(i&b))m.edges.push_back({i,i|b});return m;}
MeshData cylinderMesh(){MeshData m;constexpr uint32_t n=20;for(int y=0;y<2;++y)for(uint32_t i=0;i<n;++i){float a=2*Pi*i/n;m.vertices.push_back({std::sin(a)*.5f,float(y)-.5f,std::cos(a)*.5f});}m.vertices.push_back({0,-.5f,0});m.vertices.push_back({0,.5f,0});for(uint32_t i=0;i<n;++i){uint32_t j=(i+1)%n;m.indices.insert(m.indices.end(),{i,j,n+j,i,n+j,n+i,2*n,j,i,2*n+1,n+i,n+j});m.edges.insert(m.edges.end(),{{i,j},{n+i,n+j},{i,n+i}});}return m;}
MeshData sphereMesh(){MeshData m;constexpr uint32_t n=16,r=10;for(uint32_t j=0;j<=r;++j)for(uint32_t i=0;i<=n;++i){float a=2*Pi*i/n,b=Pi*j/r;m.vertices.push_back({std::sin(a)*std::sin(b)*.5f,std::cos(b)*.5f,std::cos(a)*std::sin(b)*.5f});}for(uint32_t j=0;j<r;++j)for(uint32_t i=0;i<n;++i){uint32_t a=j*(n+1)+i,b=a+n+1;m.indices.insert(m.indices.end(),{a,b,a+1,a+1,b,b+1});if(i%4==0)m.edges.push_back({a,b});if(j==r/2)m.edges.push_back({a,a+1});}return m;}
V3 launch(V3 from,V3 target,float speed,float gravity,bool compensate=true){
    V3 delta=target-from,direct=delta.normalized()*speed;float distance=flatLength(delta);
    if(!compensate||gravity<=.000001f||flatLength(direct)<=.0001f||distance<.000001f)return direct;
    double v2=double(speed)*speed,g=gravity,x=distance,y=delta.y,disc=v2*v2-g*(g*x*x+2*y*v2);
    if(disc<0){V3 direction=(delta.normalized()+V3{0,1,0}).normalized();double horizontal=flatLength(direction);
        double denominator=2*(x*direction.y/horizontal-y)*horizontal*horizontal;
        return denominator>0?direction*float(std::sqrt(g*x*x/denominator)):direct;}
    float angle=float(std::atan((v2-std::sqrt(std::max(0.0,disc)))/(g*x)));V3 h={delta.x/distance,0,delta.z/distance};return h*(speed*std::cos(angle))+V3{0,speed*std::sin(angle),0};
}
}
struct Encounter::Impl {
    static constexpr int ActorCount=3;
    static constexpr int PlayerIndex=0;
    static constexpr int EnemyFirst=1;
    static constexpr int EnemyCount=ActorCount-EnemyFirst;
    static constexpr int EnemyWeapon=2;
    static constexpr int EnemyShotgunPellets=2;
    struct Weapon {std::string name;uint32_t index=0,projectileIndex=0,mesh=0;Color color,bulletColor;float speed=0,gravity=0,damage=0,head=2,range=250,life=12,radius=.005f,blastRadius=4.5f,spread=0,spreadExponent=1,visualDiameter=.48f,cadence=.2f,reload=1.5f;int pellets=1,magazine=12;bool explosive=false,compensate=true;V3 muzzle,attachOffset,scale{1,1,1};Quat attachRotation{0,0,0,1};};
    struct Animation {float duration=1,strideSpeed=0;std::vector<std::vector<V3>> frames;std::vector<std::vector<Quat>> rotations;};
    struct Part {V3 min,max;V3 center()const{return (min+max)*.5f;}V3 size()const{return max-min;}};
    struct BboxPart {Part bounds;Quat referenceFrame{0,0,0,1};float length=0;int start=0,end=-1;};
    struct Appearance {uint32_t index=0;std::string name,animationSet;float scale=1;bool gripIK=true;V3 rightGrip,leftGrip,weaponOffset;std::vector<V3> reference;std::vector<Quat> referenceRotations;std::array<BboxPart,14> bboxParts;std::array<Quat,2> wristCorrection;};
    struct Actor {
        double previousShot=-100;int shotWeapon=-1;
        float weaponReadyAlpha=0;
        uint32_t token=0,index=0,weaponToken=0,shieldToken=0,heldTrapToken=0;size_t body=0,appearance=0;V3 pos,velocity,anchor,goal,device;float yaw=0,pitch=0,health=100,shield=0,shieldAlpha=0,shieldHit=0,scale=1,walkPhase=0;
        int weapon=0,target=-1,requestedWeapon=-1;std::array<int,3> ammo{12,30,8};std::array<bool,3> owned{true,false,false};
        bool grounded=true,raised=false,ads=false,crouch=false,trapCommitted=false;
        float crouchAlpha=0,airAlpha=0;double airborneTime=0,posedAt=-1,lastShot=-100;int gait=0;V3 dashDirection;std::vector<V3> localPose,sourcePose,posedBones;std::vector<Quat> localRotations,sourceRotations,posedRotations;float gripError=0;std::string animation;
        double fireAt=0,reloadEnd=0,reloadStart=0,thinkAt=0,switchAt=12,actionEnd=0,actionStart=0,respawnAt=0,deathAt=0,holdUntil=0,shieldUntil=0,shieldCheck=0,shieldRecovery=0,nextDash=10,nextJump=12,trapReady=0,nextTrap=4,lastProgress=0,lastGoal=0,shielded=0,unshielded=0;
        std::string action;std::vector<V3> path;size_t waypoint=0;std::vector<Matrix> boxes;Matrix gun{},shieldMatrix{};V3 muzzle;float flash=0;
    };
    struct Bullet {uint32_t token=0,owner=0,index=0;int weapon=0;V3 pos,previous,velocity;double age=0;float distance=0;};
    struct Pickup {uint32_t token=0,padToken=0;V3 pos;int kind=0,owner=-1;float durability=200;double available=0,expires=1e9;bool active=true;};
    struct Supply {int kind=1,owner=0;uint32_t token=0;double next=0;};
    struct Trap {uint32_t token=0,owner=0;V3 pos;double planted=0;bool active=true;};
    struct Effect {V3 pos;Color color;double born=0;float radius=1;int kind=0;V3 velocity;};
    struct Cover {uint32_t token=0,owner=0;V3 base;float yaw=0,height=0,durability=300;double born=0;bool placedEvent=false;};
    struct CoverSettings {V3 size;float distance=3,cooldown=30,lifetime=20,durability=300,rise=.6f,probability=.5f,delay=3,retry=1;bool automatic=true;uint32_t index=0;Color color;std::array<uint32_t,4> fragments;std::array<Part,4> fragmentBounds;};
    std::filesystem::path root;const MapData& map;Renderer& renderer;Collision collision;std::mt19937 rng;uint32_t seed;
    std::array<Weapon,3> weapons;std::array<Actor,ActorCount> actors;std::vector<Supply> supplies;std::vector<Pickup> pickups;std::vector<Bullet> bullets;std::vector<Trap> traps;std::vector<Effect> effects;
    std::vector<DrawObject> staticDraws,renderDraws;std::vector<CatalogObject> objects;std::vector<CombatEvent> events;
    std::vector<V3> starts;std::map<std::string,Animation> animations;std::map<std::string,std::map<std::string,Animation>> targetAnimations;std::vector<Appearance> appearances;size_t playerAppearance=0;Json ballisticProbes;
    float automaticFireMin=.5f,automaticFireMax=2,postSwitchMin=1,postSwitchMax=1.5f;
    CoverSettings coverSettings;Cover wall;std::mt19937 coverRng;double coverReady=0,coverAttempt=0;uint64_t coverGeneration=0;bool coverSelected=false,coverUsed=false;std::string coverMessage;
    float maximumEnemyShotAngle=0;uint64_t enemyShotgunShots=0;
    std::map<std::string,uint64_t> counts;std::array<Color,6> partColors;Color shieldColor=rgb(97,134,236);
    uint32_t box=0,cylinder=0,sphere=0;uint64_t tick=0,eventId=0;double time=0,nextSupply=0,nextShield=0;bool automatic=true,recording=false,recovery=false;int lastShieldHolder=EnemyFirst+EnemyCount-1;V3 start;
    Impl(const std::filesystem::path& r,const MapData& m,Renderer& render,uint32_t s):root(r),map(m),renderer(render),collision(map),rng(s),seed(s),coverRng(s^0x434f5645){
        Json data,manifest,armData,bboxData;std::ifstream f(root/L"assets/shooter/source.json"),mf(map.scene.catalogDirectory()/"CatalogManifest.json"),af(root/L"assets/shooter/arm-reference.json"),bf(root/L"assets/shooter/ue-bbox-parts.json");
        if(!f||!mf||!af||!bf)throw std::runtime_error("Missing exported ShooterCore assets/reference BBox parts");f>>data;mf>>manifest;af>>armData;bf>>bboxData;
        if(bboxData.at("sourceSha256")!=armData.at("sourceSha256"))throw std::runtime_error("UE BBox and wrist references came from different source exports");
        if(data.value("schema",0)!=2)throw std::runtime_error("Export UE retargeted animations with tools/export-shooter.ps1");ballisticProbes=data.at("ballisticProbes");
        if(armData.value("schema",0)!=1||armData.at("characters").size()!=data.at("characters").size())throw std::runtime_error("Arm reference export does not match ShooterCore appearances");
        const auto& timing=data.at("combatTiming");automaticFireMin=timing.at("automaticFireMin");automaticFireMax=timing.at("automaticFireMax");postSwitchMin=timing.at("postSwitchMin");postSwitchMax=timing.at("postSwitchMax");
        box=renderer.addMesh(boxMesh());cylinder=renderer.addMesh(cylinderMesh());sphere=renderer.addMesh(sphereMesh());
        objects.push_back({});staticDraws=map.upload(renderer);
        for(const auto& o:map.objects){CatalogObject c;c.index=o.catalogIndex;c.min=o.bounds.min;c.max=o.bounds.max;c.flags=o.tree?CatalogObject::Exempt:0;objects.push_back(c);}
        for(size_t i=0;i<staticDraws.size();++i){if(!map.usesBatches())staticDraws[i].instance.objectToken=uint32_t(i+1);staticDraws[i].range=200;}
        auto identity=[&](const std::string& needle){for(const auto& [key,value]:manifest["semanticIndices"].items())if(key.find(needle)!=std::string::npos)return value.get<uint32_t>();throw std::runtime_error("Missing catalog identity "+needle);};
        if(map.catalogColors)shieldColor=hexColor(manifest.at("bboxColorsByIndex").at(std::to_string(identity("mesh:/shootercore/equipment/shield/sm_soloshield"))).at("srgbHex"));
        Json coverData;std::ifstream cf(root/L"assets/shooter/cover.json");if(!cf)throw std::runtime_error("Missing read-only UE cover export");cf>>coverData;
        if(coverData.at("schema")!=1||coverData.at("inputKeys")!=Json::array({"H"})||coverData.at("meshes").size()!=5)throw std::runtime_error("Incomplete UE cover export");
        const auto& defaults=coverData.at("defaults");coverSettings.size=native(defaults.at("dimensionsCm"));
        coverSettings.distance=defaults.at("placement_distance").get<float>()*.01f;coverSettings.cooldown=defaults.at("cooldown");coverSettings.lifetime=defaults.at("lifetime");coverSettings.durability=defaults.at("initial_durability");coverSettings.rise=defaults.at("rise_duration");coverSettings.automatic=defaults.at("place_during_recording");
        coverSettings.probability=coverData.at("automaticProbability");coverSettings.delay=coverData.at("automaticDelay");coverSettings.retry=coverData.at("automaticRetry");
        coverSettings.index=identity("actorclass:/script/shootercoreruntime.solocoveractor");coverSettings.color=hexColor(manifest.at("bboxColorsByIndex").at(std::to_string(coverSettings.index)).at("srgbHex"));
        V3 fullMin=native(coverData.at("meshes")[0].at("minCm")),fullMax=native(coverData.at("meshes")[0].at("maxCm")),fullCenter=(fullMin+fullMax)*.5f,fullSize=fullMax-fullMin;
        const auto normalized=[&](const Json& p){V3 v=native(p)-fullCenter;return V3{v.x/fullSize.x,v.y/fullSize.y,v.z/fullSize.z};};
        for(int i=0;i<4;++i){const auto& piece=coverData.at("meshes")[i+1];MeshData mesh;for(const auto& p:piece.at("verticesCm"))mesh.vertices.push_back(normalized(p));for(auto j:piece.at("indices"))mesh.indices.push_back(j.get<uint32_t>());
            coverSettings.fragments[i]=renderer.addMesh(mesh);coverSettings.fragmentBounds[i]={normalized(piece.at("minCm")),normalized(piece.at("maxCm"))};}
        for(int i=0;i<3;++i){const auto& w=data["weapons"][i];auto& o=weapons[i];o.name=w["name"];std::string name=o.name;std::transform(name.begin(),name.end(),name.begin(),[](char c){return char(std::tolower(c));});
            o.index=identity("mesh:/shootercore/weapons/visuals/adapted/skm_solo_"+name);o.projectileIndex=identity("projectile:/script/lyraruntime.lyrabulletprojectile|/shootercore/weapons/"+name);
            o.color=hexColor(manifest["bboxColorsByIndex"][std::to_string(o.index)]["srgbHex"]);
            // Projectile hue identifies the weapon in both gameplay and model input.
            o.bulletColor=hexColor(manifest["bboxColorsByIndex"][std::to_string(o.projectileIndex)]["srgbHex"]);
            o.speed=w["projectile_speed"].get<float>()*.01f;o.gravity=w["projectile_gravity_scale"].get<float>()*9.8f;o.range=w["max_damage_range"].get<float>()*.01f;o.life=w["projectile_life_time"];
            o.pellets=w["bullets_per_cartridge"];o.magazine=i==0?12:i==1?30:8;o.damage=i==0?35.f:i==1?100.f:24.f;o.head=i==2?1.75f:2;o.spread=i==2?8.f:i==1?1.f:2.5f;o.cadence=i==0?.25f:i==1?.1f:1.f;
            o.magazine=w.at("inventoryStats").at("Lyra.ShooterGame.Weapon.MagazineSize").get<int>();o.cadence=w.at("fireDelay");o.spread=w.at("heat_to_spread_curve")[0][1];
            o.explosive=w.at("projectile_explodes_on_impact");
            // Local gameplay tuning; retain the original exported weapon assets.
            if(i==1){o.cadence=2.f;o.blastRadius=8.f;o.damage=150.f;}
            o.compensate=w.at("compensate_bullet_drop");o.spreadExponent=w.at("spread_exponent");
            // Preserve the existing projectile head size without drawing a trail.
            o.visualDiameter=w.at("projectile_trail_width").get<float>()*.04f;
            if(w.at("attachBone")!="weapon_r")throw std::runtime_error("Unexpected equipment attachment bone");
            const auto& attach=w.at("attachTransform");const auto& socket=w.at("socketTransform");
            o.attachRotation=multiply(animRotation(socket.at("q")),animRotation(attach.at("q")));
            o.attachOffset=animPoint(socket.at("p"))+rotate(animPoint(attach.at("p")),animRotation(socket.at("q")));
            o.scale={attach.at("s")[0],attach.at("s")[2],attach.at("s")[1]};
            o.radius=w.at("projectile_radius").get<float>()*.01f;o.muzzle=animPoint(w["muzzle"]["p"]);
            MeshData mesh;for(const auto& part:w["parts"]){uint32_t offset=uint32_t(mesh.vertices.size());for(const auto& p:part["vertices"])mesh.vertices.push_back(animPoint(p));for(auto n:part["indices"])mesh.indices.push_back(offset+n.get<uint32_t>());for(const auto& e:part["edges"])mesh.edges.push_back({offset+e[0].get<uint32_t>(),offset+e[1].get<uint32_t>()});}
            o.mesh=renderer.addMesh(mesh);
        }
        const char* parts[]={"head","body","leftArm","rightArm","leftLeg","rightLeg"};for(int i=0;i<6;++i)partColors[i]=hexColor(manifest["bboxColorsByIndex"]["2"]["parts"][parts[i]]["srgbHex"]);
        const char* targetBones[]={"pelvis","spine_01","spine_02","spine_03","neck_01","head","upperarm_l","lowerarm_l","hand_l","upperarm_r","lowerarm_r","hand_r","thigh_l","calf_l","foot_l","ball_l","thigh_r","calf_r","foot_r","ball_r","hand_r"};
        for(const auto& entry:data["characters"]){std::string label=entry["label"];Appearance out;out.name=label;std::transform(label.begin(),label.end(),label.begin(),[](char c){return char(std::tolower(c));});
            out.index=identity("|"+label);out.scale=entry.at("scale");
            out.animationSet=entry.at("animationSet");out.gripIK=entry.at("useWeaponGripIK");out.rightGrip=animPoint(entry.at("rightGripOffset"));out.leftGrip=animPoint(entry.at("leftGripOffset"));out.weaponOffset=animPoint(entry.at("weaponPoseOffset"));
            const auto& armReference=armData.at("characters").at(appearances.size());
            if(armReference.at("label")!=out.name||armReference.at("mesh")!=entry.at("mesh"))throw std::runtime_error("Arm reference does not match appearance export");
            out.wristCorrection={animRotation(armReference.at("wristCorrection").at("hand_l")),animRotation(armReference.at("wristCorrection").at("hand_r"))};
            const auto& bboxReference=bboxData.at("characters").at(appearances.size());if(bboxReference.at("label")!=out.name||bboxReference.at("mesh")!=entry.at("mesh"))throw std::runtime_error("BBox reference does not match appearance");
            const char* bboxNames[]={"head","pelvis","upperarm_l","lowerarm_l","upperarm_r","lowerarm_r","thigh_l","calf_l","thigh_r","calf_r","hand_l","hand_r","foot_l","foot_r"};
            const int partStarts[]={5,0,6,7,9,10,12,13,16,17,8,11,14,18},ends[]={-1,3,7,8,10,11,13,14,17,18,-1,-1,-1,-1};
            for(int p=0;p<14;++p){const auto& part=bboxReference.at("parts").at(bboxNames[p]);auto& target=out.bboxParts[p];V3 lo=animPoint(part.at("min")),hi=animPoint(part.at("max"));
                target.bounds={{hi.x,lo.y,lo.z},{lo.x,hi.y,hi.z}};target.referenceFrame=animRotation(part.at("referenceRotation"));target.length=part.at("referenceLength").get<float>()*.01f;target.start=partStarts[p];target.end=ends[p];}
            for(const char* name:targetBones){out.reference.push_back(animPoint(entry.at("bones").at(name).at("p")));out.referenceRotations.push_back(animRotation(entry.at("bones").at(name).at("q")));}
            appearances.push_back(std::move(out));
        }
        if(appearances.empty())throw std::runtime_error("No enabled catalog character appearances");
        // Keep the player on a stable human-shaped cartoon body. Some exported
        // enemy skins, such as Chicken_SK, intentionally have oversized depth.
        auto humanAppearance=[&](const char* suffix){for(size_t i=0;i<appearances.size();++i)if(appearances[i].name.ends_with(suffix))return i;return appearances.size();};
        playerAppearance=humanAppearance("/Soldier_SK");
        if(playerAppearance==appearances.size())playerAppearance=humanAppearance("/Ninja_SK");
        if(playerAppearance==appearances.size())playerAppearance=0;
        const auto loadAnimations=[&](const Json& entries,std::map<std::string,Animation>& bank){for(const auto& [name,a]:entries.items()){
            if(!a.value("rootMotionRemoved",false)||a.value("space",std::string{})!="root-local")throw std::runtime_error("Animation export contains root motion. Run tools/export-shooter.ps1.");
            Animation out;out.duration=a["duration"];for(const auto& frame:a["frames"]){std::vector<V3> bones;std::vector<Quat> rotations;for(const auto& b:frame){bones.push_back(animPoint(b["p"]));rotations.push_back(animRotation(b["q"]));}out.frames.push_back(std::move(bones));out.rotations.push_back(std::move(rotations));}
            if(name.find("_Jog_")!=std::string::npos||name.find("_Walk_")!=std::string::npos){
                V3 direction=name.ends_with("Left")?V3{-1,0,0}:name.ends_with("Right")?V3{1,0,0}:name.ends_with("Bwd")?V3{0,0,-1}:V3{0,0,1};std::vector<float> speeds;
                for(int foot:{14,18}){float floor=1e9f;for(const auto& f:out.frames)floor=std::min(floor,f[foot].y);
                    for(size_t i=1;i<out.frames.size();++i){float speed=-dot(out.frames[i][foot]-out.frames[i-1][foot],direction)*float(out.frames.size()-1)/out.duration;
                        if(out.frames[i][foot].y<floor+.05f&&speed>.05f)speeds.push_back(speed);}}
                if(!speeds.empty()){std::sort(speeds.begin(),speeds.end());out.strideSpeed=speeds[speeds.size()/2];}
            }
            bank.emplace(name,std::move(out));}};
        loadAnimations(data.at("animations"),animations);
        for(const auto& [key,bank]:data.at("retargetedAnimationSets").items())loadAnimations(bank,targetAnimations[key]);
        for(int i=0;i<3;++i){auto it=animations.find("MM_"+weapons[i].name+"_Reload");if(it!=animations.end())weapons[i].reload=it->second.duration;}
        if(!map.starts.empty()){for(auto p:map.starts){V3 projected;if(collision.project(p,projected,5,10))starts.push_back(projected);}}
        else for(const auto& t:data["starts"]){V3 projected;if(collision.project(native(t["p"]),projected,5,10))starts.push_back(projected);}
        if(starts.empty()){V3 p=map.castleBounds.center();if(collision.project(p,p,50,100))starts.push_back(p);}
        if(starts.empty())throw std::runtime_error("No reachable PlayerStart in the BBox map");
        for(auto& a:actors)a.body=collision.addCharacter({0,-1000,0});
        if(!restart())throw std::runtime_error("Could not place player and enemies on connected navigation");
        for(int w=1;w<3;++w)supplies.push_back({w,PlayerIndex,0,0});
        maintenance();pose();
    }
    float random(float lo,float hi){return std::uniform_real_distribution<float>(lo,hi)(rng);}
    uint32_t token(uint32_t index,uint32_t flags=CatalogObject::Exempt){CatalogObject o;o.index=index;o.flags=flags;objects.push_back(o);return uint32_t(objects.size()-1);}
    uint32_t index(uint32_t t)const{return t<objects.size()?objects[t].index:0;}
    void bounds(uint32_t t,const Matrix& m){if(!t)return;auto& c=objects.at(t);V3 p=position(m),e{(std::abs(m._11)+std::abs(m._21)+std::abs(m._31))*.5f,(std::abs(m._12)+std::abs(m._22)+std::abs(m._32))*.5f,(std::abs(m._13)+std::abs(m._23)+std::abs(m._33))*.5f};c.min=p-e;c.max=p+e;}
    void emit(std::string type,int actor,V3 p,std::initializer_list<uint32_t> tokens,bool pointEvent=false,bool allowPositionOnly=false){
        CombatEvent e;e.id=++eventId;e.step=tick;e.time=time;e.type=type;e.actor=actor>=0?actors[actor].token:0;e.position=p;e.tokens=tokens;e.pointEvent=pointEvent;e.allowPositionOnly=allowPositionOnly;
        std::string base=type.starts_with("Enemy")?type.substr(5):type;
        static const std::map<std::string,size_t> roles={{"WeaponFire",3},{"WeaponSwitch",3},{"WeaponPickup",2},{"ReloadStart",2},{"ReloadEnd",2},{"HitByBullet",2},{"BulletHitScene",2},{"ShieldPickup",2},{"ShieldEquipped",2},{"ShieldRaise",2},{"ShieldLower",2},{"ShieldDrop",2},{"ShieldBlock",2},{"ShieldBroken",2},{"TrapPlaced",2},{"TrapTriggered",2},{"CoverPlaced",2},{"CoverBroken",2},{"Jump",1},{"Land",1},{"Dash",1},{"Spawned",1},{"Died",1},{"Explosion",1},{"ExplosionDamage",2}};
        if(!roles.contains(base)||roles.at(base)!=e.tokens.size()||!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))throw std::runtime_error("Invalid gameplay event: "+type);
        for(auto t:tokens){if(!t||t>=objects.size()||!index(t))throw std::runtime_error("Gameplay event has an unresolved object: "+type);e.indices.push_back(index(t));}
        if(base=="WeaponFire"&&actor>=0)e.weapon=actors[actor].weapon;
        else if(!e.indices.empty())for(int w=0;w<3;++w)if(e.indices.front()==weapons[w].projectileIndex)e.weapon=w;
        if(base=="WeaponPickup"||base=="ShieldPickup")e.visibilityTokens=e.tokens;
        else if(base=="WeaponFire"||base=="WeaponSwitch")e.visibilityTokens.assign(e.tokens.begin()+1,e.tokens.end());
        else if(base.starts_with("Shield")||base.starts_with("Reload")||base=="TrapPlaced"||base.starts_with("Cover"))e.visibilityTokens={e.tokens[1]};
        else e.visibilityTokens=e.tokens;
        if(base=="TrapPlaced"||base=="TrapTriggered"){e.surfaceNormal={0,1,0};}
        events.push_back(e);++counts[type];
    }
    void spawn(int slot,V3 p,bool initial=false){size_t body=actors[slot].body;Actor a;a.body=body;a.pos=p;a.anchor=p;a.lastProgress=time;a.appearance=slot==PlayerIndex?playerAppearance:size_t(random(0,float(appearances.size())-.001f));const auto& appearance=appearances[a.appearance];a.index=appearance.index;a.scale=appearance.scale;a.token=token(a.index);if(slot!=PlayerIndex){a.weapon=EnemyWeapon;a.owned.fill(false);a.owned[EnemyWeapon]=true;}a.weaponToken=token(weapons[a.weapon].index);a.yaw=random(-Pi,Pi);a.fireAt=time+random(automaticFireMin,automaticFireMax);a.switchAt=time+random(12,20);a.nextDash=time+random(10,16);a.nextJump=time+random(12,20);a.nextTrap=time+random(4,12);for(int i=0;i<3;++i)a.ammo[i]=weapons[i].magazine;actors[slot]=a;collision.teleport(body,p);if(slot>=EnemyFirst&&!initial)emit("EnemySpawned",slot,p,{a.token});}
    bool restart(){
        V3 p=starts[size_t(random(0,float(starts.size())-.001f))];std::array<V3,ActorCount> spots;spots[PlayerIndex]=p;
        for(int i=EnemyFirst;i<ActorCount;++i){bool valid=false;for(int k=0;k<12&&!valid;++k){if(!collision.randomPoint(p,8.f,35.f,rng,spots[i]))continue;valid=true;for(int j=0;j<i;++j)if((spots[j]-spots[i]).length()<1.5f)valid=false;}if(!valid)return false;}
        start=p;bullets.clear();traps.clear();pickups.clear();effects.clear();resetCover();actors[PlayerIndex].weaponReadyAlpha=0;actors[PlayerIndex].previousShot=actors[PlayerIndex].lastShot=-100;actors[PlayerIndex].shotWeapon=-1;
        for(int i=0;i<ActorCount;++i){V3 spot=spots[i];
            if(i==PlayerIndex&&actors[i].token){auto& a=actors[i];a.pos=spot;a.velocity={};a.path.clear();a.target=-1;a.shield=0;a.shieldToken=0;a.shieldAlpha=0;a.shieldHit=0;a.raised=false;a.grounded=true;a.airborneTime=0;a.airAlpha=0;a.crouchAlpha=0;a.localPose.clear();a.posedAt=-1;a.action.clear();a.actionEnd=0;a.reloadEnd=0;a.holdUntil=0;a.trapReady=0;a.heldTrapToken=0;a.trapCommitted=false;a.anchor=spot;a.lastProgress=time;a.thinkAt=0;collision.teleport(a.body,spot);}
            else spawn(i,spot,true);
        }actors[PlayerIndex].owned.fill(true);actors[PlayerIndex].requestedWeapon=-1;
        for(auto& s:supplies){s.token=0;s.next=time;}recovery=false;nextSupply=nextShield=0;return true;
    }
    bool isAuto(int i)const{return i!=0||automatic;}
    bool hostile(int i,int j)const{return (i==PlayerIndex)!=(j==PlayerIndex);}
    Matrix coverMesh()const{return transform(wall.base+V3{0,wall.height-coverSettings.size.y*.5f,0},coverSettings.size,wall.yaw);}
    Matrix coverVolume()const{return transform(wall.base+V3{0,wall.height*.5f,0},{coverSettings.size.x,wall.height,coverSettings.size.z},wall.yaw);}
    void invalidatePaths(){for(auto& a:actors){a.path.clear();a.thinkAt=0;a.lastGoal=-100;}}
    void removeCover(){if(wall.token){collision.setObstacle(wall.token,nullptr);wall={};invalidatePaths();}}
    void resetCover(){removeCover();collision.clearDebris();coverReady=coverAttempt=0;coverMessage.clear();}
    bool findCoverPlacement(V3& base,std::string& reason)const{
        const auto& a=actors[PlayerIndex];const auto size=coverSettings.size;V3 desired=a.pos+forward(a.yaw)*coverSettings.distance;
        float low=1e9f,high=-1e9f;int steps=std::clamp(int(std::ceil(size.x/.5f)),2,64);
        for(int z=-1;z<=1;++z)for(int x=0;x<=steps;++x){V3 p=desired+rotate({size.x*(float(x)/steps-.5f),0,z*size.z*.5f},a.yaw);
            auto ground=collision.ray(p+V3{0,.8f,0},p-V3{0,1.2f,0},a.token);
            if(!ground.hit||ground.normal.y<.95f||ground.token>map.objects.size()){reason="UNSUPPORTED GROUND";return false;}low=std::min(low,ground.point.y);high=std::max(high,ground.point.y);}
        if(high-low>.08f){reason="UNEVEN GROUND";return false;}base={desired.x,high+.005f,desired.z};
        Matrix volume=transform(base+V3{0,size.y*.5f,0},{size.x+.2f,size.y-.005f,size.z+.2f},a.yaw);
        if(collision.overlapBox(volume,a.token)){reason="SPACE OCCUPIED";return false;}
        // UE independently guards pawn capsule AABBs, including nonblocking pawns.
        auto temp=volume;V3 center=position(temp),extent{(std::abs(temp._11)+std::abs(temp._31))*.5f,size.y*.5f,(std::abs(temp._13)+std::abs(temp._33))*.5f};
        for(int i=EnemyFirst;i<ActorCount;++i){const auto& other=actors[i];if(other.health<=0)continue;V3 c=other.pos+V3{0,.9f,0};if(std::abs(c.x-center.x)<=extent.x+.42f&&std::abs(c.z-center.z)<=extent.z+.42f&&std::abs(c.y-center.y)<=extent.y+.9f){reason="SPACE OCCUPIED";return false;}}
        if(collision.sweepBullet(a.pos+V3{0,.9f,0},base+V3{0,size.y*.5f,0},.08f,a.token).hit){reason="DEPLOYMENT BLOCKED";return false;}return true;
    }
    bool canPlaceCover(std::string& reason)const{const auto& a=actors[PlayerIndex];
        if(a.health<=0||!a.grounded||a.crouch||a.actionEnd>time||a.reloadEnd>time){reason="STAND ON GROUND";return false;}
        if(recording&&coverUsed){reason="USED THIS CLIP";return false;}if(wall.token){reason="COVER ACTIVE";return false;}if(time<coverReady){reason="COOLDOWN";return false;}
        V3 p;return findCoverPlacement(p,reason);
    }
    bool placeCover(int actor=PlayerIndex){if(actor!=PlayerIndex)return false;++counts["CoverRequests"];std::string reason;V3 base;
        if(!canPlaceCover(reason)||!findCoverPlacement(base,reason)){coverMessage=reason;return false;}
        auto& a=actors[PlayerIndex];wall={token(coverSettings.index,CatalogObject::Exempt|CatalogObject::Rotation),a.token,base,a.yaw,0,coverSettings.durability,time,false};
        if(recording)coverUsed=true;coverReady=time+std::max(1.f,coverSettings.cooldown);coverMessage.clear();invalidatePaths();return true;
    }
    void updateCover(){if(!wall.token)return;if(time-wall.born>=std::max(1.f,coverSettings.lifetime)){removeCover();return;}
        float progress=coverSettings.rise>0?std::clamp(float((time-wall.born)/coverSettings.rise),0.f,1.f):1;
        float remainder=1-progress,height=coverSettings.size.y*(1-remainder*remainder*remainder);if(height<=0)return;
        if(height!=wall.height){wall.height=height;auto volume=coverVolume();auto nav=transform(wall.base+V3{0,coverSettings.size.y*.5f,0},coverSettings.size,wall.yaw);collision.setObstacle(wall.token,&volume,&nav);bounds(wall.token,coverMesh());objects[wall.token].rotation={0,wall.yaw*180/Pi,0};}
        if(!wall.placedEvent){wall.placedEvent=true;emit("CoverPlaced",PlayerIndex,position(coverVolume()),{wall.owner,wall.token});}
    }
    bool automaticCover(int target){if(!recording||!automatic||!coverSettings.automatic||!coverSelected||coverUsed||time<coverAttempt||target<EnemyFirst||actors[target].health<=0)return false;
        const auto& a=actors[PlayerIndex];V3 offset=actors[target].pos-a.pos;float distance=dot(offset,forward(a.yaw)),side=std::abs(dot(offset,rotate({1,0,0},a.yaw)));
        if(distance<coverSettings.distance+1.5f||flatLength(offset)>30||std::abs(offset.y)>3.5f||side*coverSettings.distance/distance>coverSettings.size.x*.5f-.2f)return false;
        coverAttempt=time+coverSettings.retry;return placeCover();
    }
    void breakCover(){if(!wall.token)return;emit("CoverBroken",PlayerIndex,position(coverVolume()),{wall.owner,wall.token},true,true);
        Matrix mesh=coverMesh();for(int i=0;i<4;++i){const auto& part=coverSettings.fragmentBounds[i];V3 velocity=forward(wall.yaw)+rotate({i%2?1.8f:-1.8f,2,0},wall.yaw);
            // UE angular velocity is expressed in world degrees; convert to native radians.
            V3 spin{(i%2?1:-1)*120*Pi/180,80*Pi/180,90*Pi/180};collision.addDebris(mesh,part.center(),part.size(),velocity,spin);}removeCover();
    }
    bool hitCover(const Bullet& bullet,const RayHit& hit,int shooter){if(!wall.token||hit.token!=wall.token)return false;
        const uint32_t target=wall.token;const auto& w=weapons[bullet.weapon];
        float falloff=bullet.weapon==2?(bullet.distance<=6?1.f:bullet.distance<=20?.7f:.5f):1.f;wall.durability=std::max(0.f,wall.durability-w.damage*falloff);if(wall.durability<=0)breakCover();
        // UE applies the blocker's damage first, so the fatal break fact precedes
        // the projectile's scene-hit fact while both retain the wall identity.
        emit("BulletHitScene",shooter,hit.point,{bullet.token,target},true,true);return true;
    }
    bool finishingShot(const Actor& a)const{return a.shotWeapon==a.weapon&&time-a.lastShot>=0&&time-a.lastShot<animations.at("MM_"+weapons[a.weapon].name+"_Fire").duration;}
    void equip(int i,int w){auto& a=actors[i];if((i!=PlayerIndex&&w!=EnemyWeapon)||w==a.weapon||!a.owned[w]||a.reloadEnd>time||a.raised||a.actionEnd>time||(isAuto(i)&&finishingShot(a)))return;auto old=a.weaponToken;a.weapon=w;a.weaponReadyAlpha=0;a.weaponToken=token(weapons[w].index);a.fireAt=std::max(a.fireAt,time+(isAuto(i)?random(postSwitchMin,postSwitchMax):.15));a.switchAt=time+random(12,20);emit("WeaponSwitch",i,a.pos,{a.token,old,a.weaponToken});}
    void reload(int i){auto& a=actors[i];if(a.ammo[a.weapon]>=weapons[a.weapon].magazine||a.reloadEnd>time||a.raised||a.actionEnd>time||(isAuto(i)&&a.ammo[a.weapon]==0&&finishingShot(a)))return;a.reloadStart=time;a.reloadEnd=time+weapons[a.weapon].reload;emit("ReloadStart",i,a.pos,{a.token,a.weaponToken});}
    void shield(int i,bool raised){auto& a=actors[i];raised=raised&&a.shield>0&&a.health>0&&a.actionEnd<=time;if(a.raised==raised)return;a.raised=raised;if(raised){a.reloadEnd=0;a.ads=false;}else a.shieldRecovery=time+.45;emit(raised?"ShieldRaise":"ShieldLower",i,a.pos,{a.token,a.shieldToken});}
    void drop(int i){auto& a=actors[i];if(a.shield<=0)return;shield(i,false);emit("ShieldDrop",i,a.pos,{a.token,a.shieldToken});pickups.push_back({a.shieldToken,token(12),a.pos,3,-1,a.shield,time+1,time+120});a.shield=0;a.shieldToken=0;}
    void damage(int victim,float amount,int instigator,uint32_t source,V3 p,bool explosion=false){
        auto& a=actors[victim];if(a.health<=0||a.token==source)return;
        if(!explosion)emit("HitByBullet",victim,p,{source,a.token},true);
        if(victim==PlayerIndex)amount=1.f;
        float applied=std::min(a.health,amount);a.health-=applied;
        if(explosion&&applied>0)emit("ExplosionDamage",victim,p,{source,a.token},true,true);
        if(a.health<=0){emit(victim==PlayerIndex?"Died":"EnemyDied",victim,a.pos,{a.token});a.deathAt=time;a.respawnAt=victim==PlayerIndex?std::numeric_limits<double>::infinity():time+random(0,10);a.reloadEnd=0;a.actionEnd=0;a.path.clear();drop(victim);collision.setHitBoxes(a.body,a.token,{},0,nullptr);
            for(int j=0;j<ActorCount;++j)if(actors[j].target==victim){actors[j].holdUntil=time+2;actors[j].path.clear();actors[j].target=-1;}
            (void)instigator;
        }
    }
    void explode(V3 p,int actor,uint32_t source,uint32_t directToken=0,float blastDamage=100,float blastRadius=4.5f){emit("Explosion",actor,p,{source},true,true);
        for(int i=0;i<ActorCount;++i){auto& a=actors[i];if(a.health<=0||i==actor)continue;
            // A confirmed body collision takes full damage instead of splash, never both.
            if(directToken&&a.token==directToken){damage(i,blastDamage,actor,source,p,true);continue;}
            V3 target=a.pos+V3{0,.9f,0};float d=(target-p).length();if(d>blastRadius)continue;auto h=collision.ray(p+(target-p).normalized()*.06f,target,0,true);if(h.hit&&h.fraction<.98f)continue;damage(i,blastDamage*std::max(.1f,1-d/blastRadius),actor,source,target,true);}
    }
    bool visible(int i,int j)const{V3 a=actors[i].pos+V3{0,1.5f,0},b=actors[j].pos+V3{0,1.1f,0};return !collision.ray(a,b,0,true).hit;}
    void goal(int i,V3 p){auto& a=actors[i];auto path=collision.path(a.pos,p);if(path.size()>1){a.goal=p;a.path=std::move(path);a.waypoint=1;a.lastGoal=time;}}
    bool frontGoal(int i){
        auto& a=actors[i];const auto& player=actors[PlayerIndex];
        const float side=i==EnemyFirst?-1.f:1.f;V3 bestPoint;float best=1e9f;
        for(float angle:{side*.45f,side*.75f,0.f,-side*.45f})for(float distance:{12.f,17.f,9.f}){
            V3 p;const V3 desired=player.pos+forward(player.yaw+angle)*distance;
            if(!collision.project(desired,p,3,6)||flatLength(p-player.pos)<6)continue;
            if(dot((p-player.pos).normalized(),forward(player.yaw))<.35f)continue;
            if(collision.path(a.pos,p).size()<2||collision.ray(p+V3{0,1.4f,0},player.pos+V3{0,1,0},0,true).hit)continue;
            float score=flatLength(p-a.pos)+std::abs(angle-side*.45f)*8+std::abs(distance-12)*.3f;
            for(int j=EnemyFirst;j<ActorCount;++j)if(j!=i&&actors[j].health>0&&flatLength(p-actors[j].goal)<4)score+=15;
            if(score<best){best=score;bestPoint=p;}
        }
        if(best==1e9f)return false;goal(i,bestPoint);return true;
    }
    void think(int i){auto& a=actors[i];if(time<a.thinkAt||time<a.holdUntil||a.health<=0)return;a.thinkAt=time+.25;
        float best=1e9f;int chosen=-1;for(int j=0;j<ActorCount;++j)if(i!=j&&hostile(i,j)&&actors[j].health>0){float d=(a.pos-actors[j].pos).length();if(d<best&&visible(i,j)){best=d;chosen=j;}}
        if(chosen>=0)a.target=chosen;else if(a.target<0||!hostile(i,a.target)||actors[a.target].health<=0)a.target=-1;
        if(a.actionEnd>time)return;
        Pickup* pickup=nullptr;float pd=35;
        bool wantsShield=i==PlayerIndex?a.shielded+30<a.unshielded:(std::count_if(actors.begin()+EnemyFirst,actors.end(),[](const auto& b){return b.health>0&&b.shield>0;})<1);
        for(auto& p:pickups)if(p.active&&p.available<=time&&((i==PlayerIndex&&p.kind<3&&!a.owned[p.kind])||(p.kind==3&&!a.shield&&wantsShield))){float d=(p.pos-a.pos).length();if(d<pd){pickup=&p;pd=d;}}
        if(pickup){if(time-a.lastGoal>2||flatLength(a.goal-pickup->pos)>1)goal(i,pickup->pos);return;}
        if(i!=PlayerIndex&&actors[PlayerIndex].health>0){
            a.target=PlayerIndex;
            const auto& player=actors[PlayerIndex];
            const bool behind=dot((a.goal-player.pos).normalized(),forward(player.yaw))<.35f;
            if((time-a.lastGoal>1.25&&(a.path.empty()||behind))||time-a.lastGoal>3){
                if(frontGoal(i))return;
            }else if(!a.path.empty())return;
        }
        if(a.target>=0){auto& target=actors[a.target];float desired=i==PlayerIndex?13.f:(i==ActorCount-1?27.f:11.f);float d=flatLength(a.pos-target.pos);
            if(a.path.empty()||time-a.lastGoal>3||d>desired+7||d<desired-4){V3 p;float lo=std::max(4.f,desired-3),hi=desired+3;
                for(int k=0;k<4;++k)if(collision.randomPoint(target.pos,lo,hi,rng,p)&&!collision.path(a.pos,p).empty()&&!collision.ray(p+V3{0,1.4f,0},target.pos+V3{0,1,0},0,true).hit){goal(i,p);break;}}
        }else if(a.path.empty()||time-a.lastGoal>8){V3 p;if(collision.randomPoint(a.pos,6,25,rng,p))goal(i,p);}
        if(i==PlayerIndex&&time>=a.switchAt){std::vector<int> slots;for(int w=0;w<3;++w)if(a.owned[w]&&w!=a.weapon&&a.ammo[w]>0)slots.push_back(w);if(!slots.empty())equip(i,slots[size_t(random(0,float(slots.size())-.001f))]);else a.switchAt=time+3;}
    }
    bool actionStart(int i,const std::string& action,V3 direction={}){auto& a=actors[i];if(a.health<=0||!a.grounded||a.actionEnd>time||a.reloadEnd>time||a.raised)return false;
        if(action=="Jump"){a.velocity.y=5.0f;a.grounded=false;a.actionEnd=time+.8;a.nextJump=time+random(12,20);a.nextDash=std::max(a.nextDash,time+random(4,6));emit("Jump",i,a.pos,{a.token});}
        else if(action=="Dash"){if(direction.length()<.1f)direction=forward(a.yaw);V3 end=a.pos+direction*(12.95f+flatLength(a.velocity)*.7f)*.892857143f;V3 projected;
            if(!collision.project(end,projected,1,2)||collision.path(a.pos,projected).empty()||collision.ray(a.pos+V3{0,.8f,0},end+V3{0,.8f,0},0,true).hit)return false;
            a.dashDirection=rotate(direction,-a.yaw);a.velocity=direction*(12.95f+flatLength(a.velocity)*.7f);a.actionEnd=time+1.321428571;a.nextDash=time+random(10,16);a.nextJump=std::max(a.nextJump,time+random(4,6));emit("Dash",i,a.pos,{a.token});}
        else if(action=="Trap"){if(i!=PlayerIndex||time<a.trapReady)return false;int active=0;for(const auto& t:traps)active+=t.active&&t.owner==a.token;if(active>=3)return false;V3 p=a.pos+forward(a.yaw)*.65f;auto floor=collision.ray(p+V3{0,.5f,0},p-V3{0,.6f,0},0,true);if(!floor.hit)return false;a.goal=floor.point;a.actionEnd=time+2.35;a.trapCommitted=false;a.velocity={};}
        else return false;
        if(action=="Trap")a.heldTrapToken=token(1);
        a.action=action;a.actionStart=time;a.fireAt=std::max(a.fireAt,a.actionEnd);return true;
    }
    void fire(int i,V3 target,bool grenade=false){auto& a=actors[i];if(a.health<=0||time<a.fireAt||a.reloadEnd>time||a.actionEnd>time||a.raised||a.shieldAlpha>.001f||a.shieldRecovery>time)return;
        // Actor/controller aim alone is insufficient: the jogging clip can leave
        // the visible barrel lowered or sideways. Automatic shots wait for the
        // actual posed gun to face the target after the upper-body ready blend.
        if(isAuto(i)){V3 barrel=(point(a.gun,{0,0,1})-position(a.gun)).normalized(),aim=(target-a.muzzle).normalized();
            if(dot(barrel,aim)<std::cos(10.f*Pi/180))return;}
        auto& w=weapons[a.weapon];if(!a.ammo[a.weapon]){reload(i);return;}if(grenade)return;
        --a.ammo[a.weapon];if(i>=EnemyFirst){V3 barrel=(point(a.gun,{0,0,1})-position(a.gun)).normalized();maximumEnemyShotAngle=std::max(maximumEnemyShotAngle,std::acos(std::clamp(dot(barrel,(target-a.muzzle).normalized()),-1.f,1.f))*180/Pi);if(a.weapon==2)++enemyShotgunShots;}
        a.flash=.06f;a.previousShot=a.shotWeapon==a.weapon?a.lastShot:-100;a.shotWeapon=a.weapon;a.lastShot=time;a.fireAt=time+(isAuto(i)&&a.weapon!=1?random(automaticFireMin,automaticFireMax):w.cadence);
        V3 muzzle=a.muzzle,eye=a.pos+V3{0,1.65f-.5f*a.crouchAlpha,0};auto cover=collision.sweepBullet(eye,muzzle,w.radius,a.token);
        if(cover.hit)muzzle=eye+(muzzle-eye)*std::max(0.f,cover.fraction)+cover.normal*.001f;
        V3 aim=target-muzzle;if(aim.length()>w.range)target=muzzle+aim.normalized()*w.range;
        V3 velocity=launch(muzzle,target,w.speed,w.gravity,w.compensate);if(velocity.length()<.0001f)velocity=forward(a.yaw,a.pitch)*w.speed;uint32_t first=0;
        const int pellets=i!=PlayerIndex&&a.weapon==2?EnemyShotgunPellets:w.pellets;
        for(int n=0;n<pellets;++n){Bullet b;b.owner=a.token;b.weapon=a.weapon;b.index=w.projectileIndex;b.token=token(b.index);if(!first)first=b.token;b.pos=b.previous=muzzle;
            V3 direction=velocity.normalized(),right=cross({0,1,0},direction).normalized();if(right.length()<.1f)right={1,0,0};V3 up=cross(direction,right);
            float angle=random(0,2*Pi),cone=std::pow(random(0,1),w.spreadExponent)*w.spread*.5f*Pi/180;
            b.velocity=(direction*std::cos(cone)+(right*std::cos(angle)+up*std::sin(angle))*std::sin(cone))*velocity.length();bullets.push_back(b);}
        emit("WeaponFire",i,muzzle,{a.token,a.weaponToken,first});
    }
    void maintenance(){
        if(time>=nextSupply){nextSupply=time+3;for(auto& s:supplies){bool active=false;for(const auto& p:pickups)if(p.token==s.token&&p.active)active=true;if(active||time<s.next)continue;V3 p;if(collision.randomPoint(actors[s.owner].pos,6,30,rng,p)){Pickup pickup;pickup.kind=s.kind;pickup.owner=s.owner;pickup.pos=p;pickup.token=token(weapons[s.kind].index);pickup.padToken=token(12);s.token=pickup.token;pickups.push_back(pickup);}}
            for(int i=PlayerIndex;i<=PlayerIndex;++i){auto& a=actors[i];if(a.shield||a.shielded+30>=a.unshielded)continue;bool supplied=false;for(const auto& p:pickups)if(p.active&&p.kind==3&&p.owner==i)supplied=true;
                if(!supplied){V3 p;if(collision.randomPoint(a.pos,6,25,rng,p))pickups.push_back({token(13),token(12),p,3,i});}}
        }
        if(time>=nextShield){nextShield=time+.25;int held=0;for(int i=EnemyFirst;i<ActorCount;++i)held+=actors[i].health>0&&actors[i].shield>0;
            if(!held)for(int j=1;j<=EnemyCount;++j){int i=EnemyFirst+(lastShieldHolder-EnemyFirst+j)%EnemyCount;auto& a=actors[i];if(a.health<=0)continue;Pickup* reusable=nullptr;for(auto& p:pickups)if(p.active&&p.kind==3&&p.owner<0){reusable=&p;break;}if(reusable&&time<reusable->available)break;
                if(reusable){a.shield=reusable->durability;a.shieldToken=reusable->token;reusable->active=false;}else{a.shield=200;a.shieldToken=token(13);}lastShieldHolder=i;emit("EnemyShieldEquipped",i,a.pos,{a.token,a.shieldToken});break;}}
        for(auto& p:pickups){if(!p.active)continue;if(time>=p.expires){p.active=false;continue;}if(time<p.available)continue;
            for(int i=0;i<ActorCount;++i){auto& a=actors[i];if(a.health<=0||flatLength(p.pos-a.pos)>1.15f||std::abs(p.pos.y-a.pos.y)>1.8f)continue;
                if(p.kind<3){if(i!=PlayerIndex||a.owned[p.kind])continue;a.owned[p.kind]=true;a.ammo[p.kind]=weapons[p.kind].magazine;emit("WeaponPickup",i,p.pos,{a.token,p.token});equip(i,p.kind);for(auto& s:supplies)if(s.token==p.token)s.next=time+30;}
                else{if(a.shield>0||(i==PlayerIndex&&isAuto(i)&&a.shielded+30>=a.unshielded))continue;if(i>=EnemyFirst&&std::any_of(actors.begin()+EnemyFirst,actors.end(),[](const auto& x){return x.health>0&&x.shield>0;}))continue;a.shield=p.durability;a.shieldToken=p.token;emit("ShieldPickup",i,p.pos,{a.token,p.token});}p.active=false;break;}}
    }
    void simulate(float dt,const CombatInput& input){if(actors[PlayerIndex].health<=0)return;time+=dt;++tick;updateCover();collision.stepDebris(dt);std::array<bool,ActorCount> requestedFire{};
        for(int i=0;i<ActorCount;++i){auto& a=actors[i];if(a.health<=0){if(time>=a.respawnAt&&time>a.deathAt+2){V3 p;if(collision.randomPoint(actors[PlayerIndex].pos,8,35,rng,p))spawn(i,p);else a.respawnAt=time+3;}continue;}
            bool autoPlay=isAuto(i);if(a.shield>0)a.shielded+=dt;else a.unshielded+=dt;if(i==PlayerIndex&&autoPlay&&a.shield>0&&a.shielded>=a.unshielded+30)drop(i);
            if(a.reloadEnd>0&&time>=a.reloadEnd){a.ammo[a.weapon]=weapons[a.weapon].magazine;a.reloadEnd=0;emit("ReloadEnd",i,a.pos,{a.token,a.weaponToken});}
            if(a.action=="Trap"&&!a.trapCommitted&&time>=a.actionStart+1){uint32_t t=a.heldTrapToken;traps.push_back({t,a.token,a.goal,time});a.trapCommitted=true;a.trapReady=time+60;emit("TrapPlaced",i,a.goal,{a.token,t},true,true);}
            if(time>=a.actionEnd)a.action.clear();
            if(autoPlay)think(i);
            V3 desired{};if(autoPlay){if(time>=a.holdUntil&&a.waypoint<a.path.size()){V3 d=a.path[a.waypoint]-a.pos;d.y=0;if(d.length()<.4f){++a.waypoint;if(a.waypoint>=a.path.size())a.path.clear();}else desired=d.normalized()*2.5f;}}
            else{a.yaw=std::remainder(a.yaw+input.lookX*CameraInput::RadiansPerPixel,2*Pi);a.pitch=std::clamp(a.pitch-input.lookY*CameraInput::RadiansPerPixel,-1.15f,1.15f);a.crouch=input.crouch;a.ads=input.ads&&!a.raised;
                desired=(forward(a.yaw)*input.forward+rotate({1,0,0},a.yaw)*input.right).normalized()*(a.crouch?1.5f:a.ads?1.8f:2.5f);
                if(input.weapon>=0&&input.weapon<3&&a.owned[input.weapon])a.requestedWeapon=input.weapon;
                if(a.requestedWeapon>=0){equip(i,a.requestedWeapon);if(a.weapon==a.requestedWeapon)a.requestedWeapon=-1;}
                if(input.reload)reload(i);if(input.jump)actionStart(i,"Jump");if(input.dash)actionStart(i,"Dash",desired.normalized());if(input.trap)actionStart(i,"Trap");shield(i,input.shield);}
            if(autoPlay){V3 aim=desired;bool see=a.target>=0&&actors[a.target].health>0&&visible(i,a.target);if(see)aim=actors[a.target].pos+V3{0,1.15f,0}-(a.pos+V3{0,1.45f,0});
                if(time>=a.holdUntil&&aim.length()>.01f){float yaw=std::atan2(aim.x,aim.z),pitch=see?std::atan2(aim.y,flatLength(aim)):0;float limit=std::min(1.5f*Pi/180,(recording?36.f:90.f)*Pi/180*dt);a.yaw+=std::clamp(std::remainder(yaw-a.yaw,2*Pi),-limit,limit);a.pitch=approach(a.pitch,pitch,limit);}
                if(a.shield>0&&time>=a.shieldCheck){a.shieldCheck=time+.1;bool threat=false;
                    for(const auto& b:bullets)if(b.owner!=a.token){V3 offset=a.pos+V3{0,1,0}-b.pos;float arrival=dot(offset,b.velocity)/std::max(1.f,dot(b.velocity,b.velocity));if(arrival>0&&arrival<1.5f&&(offset-b.velocity*arrival).length()<1.2f&&dot((b.pos-a.pos).normalized(),forward(a.yaw))>.25f){threat=true;break;}}
                    if(threat&&a.actionEnd<=time){if(a.raised||time>=a.shieldRecovery){shield(i,true);a.shieldUntil=time+1.8;}}
                    else if(a.raised&&time>=a.shieldUntil)shield(i,false);}
                if(see&&time>=a.holdUntil){float d=flatLength(aim);if(i==PlayerIndex)automaticCover(a.target);if(i==PlayerIndex&&time>=a.nextTrap&&d>=3&&d<=22){if(actionStart(i,"Trap"))a.nextTrap=time+60;else a.nextTrap=time+2;}
                    if(flatLength(desired)>1&&d>=5&&d<=28){if(time>=a.nextDash&&!actionStart(i,"Dash",desired.normalized()))a.nextDash=time+2;else if(time>=a.nextJump)actionStart(i,"Jump");}
                    if(dot(aim.normalized(),forward(a.yaw,a.pitch))>.993f)requestedFire[i]=true;}
                if(!a.ammo[a.weapon])reload(i);
            }else requestedFire[i]=input.fire;
            bool readyWanted=(autoPlay?a.target>=0&&actors[a.target].health>0&&visible(i,a.target):input.fire||input.ads)||finishingShot(a);
            readyWanted=readyWanted&&a.health>0&&a.actionEnd<=time&&a.reloadEnd<=time&&!a.raised&&a.shieldAlpha<.01f;
            a.weaponReadyAlpha=approach(a.weaponReadyAlpha,readyWanted?1.f:0.f,dt*(readyWanted?8.f:3.f));
            if(i==PlayerIndex&&input.cover)placeCover();
            bool dashing=a.action=="Dash"&&time<a.actionStart+.892857143;
            if(a.action=="Trap"||time<a.holdUntil)desired={};
            if(!dashing){V3 horizontal{a.velocity.x,0,a.velocity.z},diff=desired-horizontal;float max=8.333333f*dt;if(diff.length()>max)diff=diff.normalized()*max;horizontal=horizontal+diff;a.velocity.x=horizontal.x;a.velocity.z=horizontal.z;}
            if(a.grounded&&a.velocity.y<=0)a.velocity.y=-.1f;else a.velocity.y-=9.8f*dt;
            bool wasGrounded=a.grounded;a.pos=collision.move(a.body,a.velocity,dt,a.grounded);if(!wasGrounded&&a.grounded){a.velocity.y=0;emit("Land",i,a.pos,{a.token});}
            a.airborneTime=a.grounded?0:a.airborneTime+dt;
            bool airPose=!a.grounded&&(a.action=="Jump"||a.airborneTime>.08)&&a.action!="Dash"&&a.health>0;
            a.airAlpha=approach(a.airAlpha,airPose?1.f:0.f,dt*10);a.crouchAlpha=approach(a.crouchAlpha,a.crouch?1.f:0.f,dt*7);
            advanceGait(a,dt);a.shieldAlpha=approach(a.shieldAlpha,a.raised?1.f:0.f,dt*(a.raised?4.f:1/.35f));a.shieldHit=std::max(0.f,a.shieldHit-dt*6);a.flash=std::max(0.f,a.flash-dt);
            if(flatLength(a.pos-a.anchor)>=.5f){a.anchor=a.pos;a.lastProgress=time;}else if(autoPlay&&time-a.lastProgress>=10){if(i==PlayerIndex)recovery=true;else{a.health=0;a.deathAt=time-3;a.respawnAt=time;drop(i);++counts["EnemyRecovery"];}}
        }
        pose();
        for(int i=0;i<ActorCount;++i)if(requestedFire[i]){auto& a=actors[i];if(isAuto(i)){if(a.target>=0)fire(i,actors[a.target].pos+V3{0,1.15f,0});}
            else{Camera c=camera(i);V3 target=c.eye+(c.target-c.eye).normalized()*weapons[a.weapon].range;auto hit=collision.ray(c.eye,target,a.token);fire(i,hit.hit?hit.point:target);}}
        for(auto& b:bullets){auto& w=weapons[b.weapon];b.previous=b.pos;V3 delta=b.velocity*dt+V3{0,-.5f*w.gravity*dt*dt,0};float remaining=std::max(0.f,w.range-b.distance);
            if(delta.length()>remaining)delta=delta.normalized()*remaining;V3 next=b.pos+delta;auto h=collision.sweepBullet(b.pos,next,w.radius,b.owner);b.velocity.y-=w.gravity*dt;b.age+=dt;b.distance+=delta.length()*(h.hit?std::clamp(h.fraction,0.f,1.f):1.f);
            if(h.hit){b.pos=h.point;int shooter=-1;for(int i=0;i<ActorCount;++i)if(actors[i].token==b.owner)shooter=i;bool blocked=false;for(int i=0;i<ActorCount;++i){auto& a=actors[i];if(h.token==a.shieldToken&&a.shield>0){a.shield-=w.damage;a.shieldHit=1;emit("ShieldBlock",i,h.point,{b.token,a.shieldToken},true);blocked=true;
                    if(a.shield<=0){uint32_t old=a.shieldToken;shield(i,false);emit("ShieldBroken",i,h.point,{a.token,old},true,true);a.shieldToken=0;a.shield=0;collision.setHitBoxes(a.body,a.token,a.boxes,0,nullptr);for(int f=0;f<4;++f)effects.push_back({h.point,shieldColor,time,1,1,{random(-2,2),random(1,3),random(-2,2)}});}break;}}
                if(!blocked)blocked=hitCover(b,h,shooter);
                if(!blocked){if(w.explosive)explode(h.point+h.normal*.01f,shooter,b.token,h.token,w.damage,w.blastRadius);else{bool victim=false;for(int i=0;i<ActorCount;++i)if(actors[i].token==h.token&&actors[i].health>0){float falloff=b.weapon==2?(b.distance<=6?1.f:b.distance<=20?.7f:.5f):1.f;damage(i,w.damage*falloff*(h.part==0?w.head:1),shooter,b.token,h.point);victim=true;break;}if(!victim&&index(h.token))emit("BulletHitScene",shooter,h.point,{b.token,h.token},true,true);}}
                b.age=1e6;
            }else b.pos=next;
        }
        std::erase_if(bullets,[&](const auto& b){return b.age>=weapons[b.weapon].life||b.distance>=weapons[b.weapon].range;});
        for(auto& t:traps)if(t.active){if(time-t.planted>=60){t.active=false;continue;}if(time-t.planted<.65)continue;for(int i=0;i<ActorCount;++i){const auto& a=actors[i];if(a.health<=0||a.token==t.owner||!a.grounded||flatLength(a.pos-t.pos)>.6f||std::abs(a.pos.y-t.pos.y)>.5f)continue;int owner=-1;for(int j=0;j<ActorCount;++j)if(actors[j].token==t.owner)owner=j;emit("TrapTriggered",owner,t.pos,{t.token,a.token},true,true);explode(t.pos+V3{0,.15f,0},owner,t.token);t.active=false;break;}}
        std::erase_if(effects,[&](const auto& e){return time-e.born>(e.kind==1?4:.65);});maintenance();
    }
    std::vector<V3> sample(const std::string& name,double t,bool loop=true,std::vector<Quat>* rotations=nullptr,const std::map<std::string,Animation>* bank=nullptr)const{
        const auto& a=(bank?*bank:animations).at(name);
        double phase=loop?std::fmod(t,a.duration):std::clamp(t,0.0,double(a.duration));if(phase<0)phase+=a.duration;
        float frame=float(phase)/a.duration*float(a.frames.size()-1);size_t i=size_t(frame),j=std::min(i+1,a.frames.size()-1);float alpha=frame-float(i);
        if(rotations){rotations->resize(boneParents.size());for(size_t b=0;b<rotations->size();++b)(*rotations)[b]=slerp(a.rotations[i][b],a.rotations[j][b],alpha);}return blendPose(a.frames[i],a.frames[j],alpha);
    }
    std::string locomotion(Actor& a,bool crouch)const{
        std::string gun=a.weapon==0?"Pistol":"Rifle";if(flatLength(a.velocity)<=.2f)return "MM_"+gun+(crouch?"_Crouch_Idle":"_Idle_Hipfire");
        V3 local=rotate(a.velocity,-a.yaw);if(std::abs(local.x)>std::abs(local.z)*1.25f)a.gait=local.x<0?-1:1;else if(std::abs(local.x)<std::abs(local.z)*.8f)a.gait=0;
        return "MM_"+gun+(crouch?"_Crouch_Walk_":"_Jog_")+(a.gait<0?"Left":a.gait>0?"Right":local.z<0?"Bwd":"Fwd");
    }
    void advanceGait(Actor& a,float dt){
        if(!a.grounded||!a.action.empty()||flatLength(a.velocity)<=.2f)return;
        const auto& bank=targetAnimations.at(appearances[a.appearance].animationSet);const auto& anim=bank.at(locomotion(a,a.crouchAlpha>.5f));
        float rate=std::clamp(flatLength(a.velocity)/std::max(.1f,anim.strideSpeed*a.scale),.25f,3.f);
        a.walkPhase+=dt*rate/anim.duration;
    }
    void recoil(std::vector<V3>& bones,std::vector<Quat>& rotations,const std::map<std::string,Animation>& bank,const Actor& actor)const{
        std::string name="MM_"+weapons[actor.weapon].name+"_Fire";const auto& anim=bank.at(name);double t=time-actor.lastShot;
        if(t<0||t>=anim.duration||actor.shotWeapon!=actor.weapon||actor.health<=0||actor.reloadEnd>time||!actor.action.empty()||actor.raised)return;
        std::vector<Quat> shotQ;auto shot=sample(name,t,false,&shotQ,&bank);const auto& rest=anim.rotations.back();const auto& restPose=anim.frames.back();auto old=bones;auto oldQ=rotations;
        float alpha=std::min(1.f,float(t)*30);
        // Fast manual fire restarts the authored clip before its previous shot
        // ends. Blend the two clip phases briefly, instead of snapping to rest.
        double previousTime=time-actor.previousShot;
        if(alpha<1&&previousTime>=0&&previousTime<anim.duration){std::vector<Quat> previousQ;auto previous=sample(name,previousTime,false,&previousQ,&bank);
            shot=blendPose(previous,shot,alpha);for(size_t b=0;b<shotQ.size();++b)shotQ[b]=slerp(previousQ[b],shotQ[b],alpha);alpha=1;}
        // The authored additive clip includes pelvis/clavicle/socket translation
        // as well as rotation. Keep the movement capsule independent of this
        // root-local pose; the standing feet are restored below with fixed lengths.
        rotations[0]=multiply(slerp({0,0,0,1},multiply(shotQ[0],inverse(rest[0])),alpha),oldQ[0]);
        bones[0]=old[0]+(shot[0]-restPose[0])*alpha;
        for(int b=1;b<21;++b){if(b>=12&&b<20)continue;int p=boneParents[b];
            Quat delta=multiply(multiply(inverse(shotQ[p]),shotQ[b]),inverse(multiply(inverse(rest[p]),rest[b])));
            rotations[b]=multiply(rotations[p],multiply(slerp({0,0,0,1},delta,alpha),multiply(inverse(oldQ[p]),oldQ[b])));
            V3 base=rotate(old[b]-old[p],inverse(oldQ[p]));
            V3 translation=rotate(shot[b]-shot[p],inverse(shotQ[p]))-rotate(restPose[b]-restPose[p],inverse(rest[p]));
            V3 offset=base+translation*alpha;
            if(b==7||b==8||b==10||b==11)offset=offset.normalized()*base.length();
            bones[b]=bones[p]+rotate(offset,rotations[p]);
        }
        if(actor.grounded){Quat pelvisDelta=multiply(rotations[0],inverse(oldQ[0]));
            for(int thigh:{12,16}){bones[thigh]=bones[0]+rotate(old[thigh]-old[0],pelvisDelta);bones[thigh+1]=old[thigh+1]+bones[thigh]-old[thigh];bones[thigh+2]=old[thigh+2]+bones[thigh]-old[thigh];
                solveLimb(bones[thigh],bones[thigh+1],bones[thigh+2],old[thigh+2],old[thigh+1]-old[thigh]+V3{0,0,.1f});bones[thigh+3]=old[thigh+3]+bones[thigh+2]-old[thigh+2];}}
    }
    std::array<Matrix,6> characterBoxes(const std::vector<V3>& bones,const std::vector<Quat>& rotations,const Appearance& appearance,float scale,float yaw)const{
        struct Box {V3 center,size;Quat rotation;};std::array<Box,14> resolved;Quat facing=axisRotation({0,1,0},yaw);
        const auto swing=[](V3 from,V3 to,V3 crossAxis){return dot(from,to)<-.9999f?axisRotation(crossAxis,Pi):between(from,to);};
        for(int i=0;i<14;++i){const auto& reference=appearance.bboxParts[i];Part bounds=reference.bounds;Quat rotation=rotations[reference.start];
            if(reference.end>=0){V3 delta=bones[reference.end]-bones[reference.start];float length=delta.length();Quat frame=multiply(facing,reference.referenceFrame);
                // UE's segment Z becomes native Y under [-X,Z,Y]. Exclude the
                // tracked bone's twist; swing the cached component frame only.
                V3 direction=delta.normalized(),crossAxis=rotate({1,0,0},frame);
                Quat alignment=swing(rotate({0,1,0},frame),direction,crossAxis);
                V3 x=rotate(crossAxis,alignment);x=(x-direction*dot(x,direction)).normalized();V3 z=cross(x,direction).normalized();x=cross(direction,z).normalized();
                // UE reconstructs MakeFromZX after the swing. This also aligns
                // near-antiparallel segments exactly instead of retaining -ref Z.
                Matrix basis(x.x,x.y,x.z,0,direction.x,direction.y,direction.z,0,z.x,z.y,z.z,0,0,0,0,1);
                if(length>1e-6f)DirectX::XMStoreFloat4(&rotation,DirectX::XMQuaternionNormalize(DirectX::XMQuaternionRotationMatrix(DirectX::XMLoadFloat4x4(&basis))));
                const auto posed=[&](float value){return value<0?value*scale:value>reference.length?length+(value-reference.length)*scale:value*length/reference.length;};
                bounds.min={bounds.min.x*scale,posed(bounds.min.y),bounds.min.z*scale};bounds.max={bounds.max.x*scale,posed(bounds.max.y),bounds.max.z*scale};
            }else{bounds.min=bounds.min*scale;bounds.max=bounds.max*scale;}
            resolved[i]={bones[reference.start]+rotate(bounds.center(),rotation),bounds.size(),rotation};}
        std::array<Matrix,6> boxes;for(int i=0;i<2;++i)boxes[i]=boneTransform(resolved[i].center,resolved[i].size,resolved[i].rotation);
        const int groups[][3]={{2,3,10},{4,5,11},{6,7,12},{8,9,13}};
        for(int group=0;group<4;++group){const auto& first=resolved[groups[group][0]];V3 direction=(resolved[groups[group][2]].center-first.center).normalized();Quat rotation=first.rotation;
            if(direction.length()>.1f)rotation=multiply(swing(rotate({0,1,0},rotation),direction,rotate({1,0,0},rotation)),rotation);
            const auto fit=[&](Quat frame){Part bounds{{1e9f,1e9f,1e9f},{-1e9f,-1e9f,-1e9f}};Quat inv=inverse(frame);
                for(int index:groups[group]){const auto& part=resolved[index];V3 center=rotate(part.center-first.center,inv);Quat relative=multiply(inv,part.rotation);
                    V3 x=rotate({part.size.x*.5f,0,0},relative),y=rotate({0,part.size.y*.5f,0},relative),z=rotate({0,0,part.size.z*.5f},relative);
                    V3 extent{std::abs(x.x)+std::abs(y.x)+std::abs(z.x),std::abs(x.y)+std::abs(y.y)+std::abs(z.y),std::abs(x.z)+std::abs(y.z)+std::abs(z.z)};
                    V3 lo=center-extent,hi=center+extent;bounds.min={std::min(bounds.min.x,lo.x),std::min(bounds.min.y,lo.y),std::min(bounds.min.z,lo.z)};bounds.max={std::max(bounds.max.x,hi.x),std::max(bounds.max.y,hi.y),std::max(bounds.max.z,hi.z)};}return bounds;};
            const auto volume=[](const Part& b){auto s=b.size();return s.x*s.y*s.z;};Part bounds=fit(rotation);
            for(int index:groups[group]){auto candidate=fit(resolved[index].rotation);if(volume(candidate)<volume(bounds)*.98f){bounds=candidate;rotation=resolved[index].rotation;}}
            V3 local=rotate(direction,inverse(rotation)),size=bounds.size();int longitudinal=0;if(std::abs(local.z)>std::abs(local.x))longitudinal=2;if(std::abs(local.y)>(longitudinal==0?std::abs(local.x):std::abs(local.z)))longitudinal=1;
            float width=group<2?.60f:.75f;for(int axis=0;axis<3;++axis)if(axis!=longitudinal)(&size.x)[axis]*=width;
            boxes[group+2]=boneTransform(first.center+rotate(bounds.center(),rotation),size,rotation);}
        return boxes;
    }
    void pose(){for(int i=0;i<ActorCount;++i){auto& a=actors[i];const auto& appearance=appearances[a.appearance];const auto& bank=targetAnimations.at(appearance.animationSet);
            bool walking=flatLength(a.velocity)>.2f;std::string name=locomotion(a,false);double t=walking?a.walkPhase*bank.at(name).duration:time;
            bool loop=true;if(a.reloadEnd>time){name="MM_"+weapons[a.weapon].name+"_Reload";t=time-a.reloadStart;loop=false;}if(a.health<=0){name="MM_Death_Front_01";t=time-a.deathAt;loop=false;}
            if(a.action=="Trap"&&animations.contains("A_TrapPlace")){name="A_TrapPlace";t=(time-a.actionStart)*animations.at(name).duration/2;loop=false;}
            if(a.action=="Dash"){const auto& direction=a.dashDirection;name=std::abs(direction.x)>std::abs(direction.z)?(direction.x<0?"MM_Dash_Left":"MM_Dash_Right"):(direction.z<0?"MM_Dash_Backward":"MM_Dash_Forward");t=(time-a.actionStart)*.7;loop=false;}
            a.animation=name;std::vector<Quat> rotations,sourceQ;auto bones=sample(name,t,loop,&rotations,&bank),source=sample(name,t,loop,&sourceQ);
            if(a.crouchAlpha>0&&a.health>0&&a.action.empty()&&a.reloadEnd<=time){std::string crouch=locomotion(a,true);double ct=walking?a.walkPhase*bank.at(crouch).duration:time;
                std::vector<Quat> cq,sq;auto cb=sample(crouch,ct,true,&cq,&bank),sb=sample(crouch,ct,true,&sq);bones=blendPose(bones,cb,a.crouchAlpha);source=blendPose(source,sb,a.crouchAlpha);
                for(size_t b=0;b<rotations.size();++b){rotations[b]=slerp(rotations[b],cq[b],a.crouchAlpha);sourceQ[b]=slerp(sourceQ[b],sq[b],a.crouchAlpha);}if(a.crouchAlpha>.5f)a.animation=crouch;}
            if(a.airAlpha>0&&a.health>0){std::string jumpName=a.velocity.y>0?"MM_Unarmed_Jump_Start":"MM_Unarmed_Jump_Fall_Loop";std::vector<Quat> jumpQ;auto jump=sample(jumpName,time-a.actionStart,a.velocity.y<=0,&jumpQ,&bank);V3 anchor=bones[0]-jump[0];auto target=bones;
                for(int b=12;b<20;++b){target[b]=jump[b]+anchor;rotations[b]=slerp(rotations[b],jumpQ[b],a.airAlpha);}bones=blendPose(bones,target,a.airAlpha);}
            if(a.weaponReadyAlpha>0&&a.health>0&&a.action.empty()&&a.reloadEnd<=time){
                const std::string ready="MM_"+std::string(a.weapon==0?"Pistol":"Rifle")+"_Idle_Hipfire";
                const auto upper=[&](std::vector<V3>& pose,std::vector<Quat>& q,const std::map<std::string,Animation>& clips){std::vector<Quat> rq;auto raised=sample(ready,time,true,&rq,&clips),target=pose;
                    for(int b=1;b<21;++b){if(b>=12&&b<20)continue;target[b]=pose[0]+raised[b]-raised[0];q[b]=slerp(q[b],rq[b],a.weaponReadyAlpha);}pose=blendPose(pose,target,a.weaponReadyAlpha);};
                upper(bones,rotations,bank);upper(source,sourceQ,animations);
            }
            // Filter transitions in one root-local skeleton, keeping connected segment lengths.
            float blend=1-std::exp(-24*float(std::max(0.0,time-a.posedAt)));
            if(a.localPose.empty()){a.localPose=bones;a.localRotations=rotations;a.sourcePose=source;a.sourceRotations=sourceQ;}
            else{a.localPose=blendPose(a.localPose,bones,blend);a.sourcePose=blendPose(a.sourcePose,source,blend);
                for(size_t b=0;b<rotations.size();++b){a.localRotations[b]=slerp(a.localRotations[b],rotations[b],blend);a.sourceRotations[b]=slerp(a.sourceRotations[b],sourceQ[b],blend);}}
            a.posedAt=time;bones=a.localPose;rotations=a.localRotations;source=a.sourcePose;sourceQ=a.sourceRotations;
            Quat readyCorrection{0,0,0,1};
            if(a.weaponReadyAlpha>.85f&&a.shieldAlpha<.01f&&a.health>0&&a.action.empty()&&a.reloadEnd<=time){
                // The ready clip itself has small breathing/weapon alignment
                // offsets. Aim-offset calibration belongs to the upper body,
                // and uses the neutral pose so it never erases authored recoil.
                V3 neutralBarrel=rotate({0,0,1},multiply(sourceQ[20],weapons[a.weapon].attachRotation));
                float weight=std::clamp((a.weaponReadyAlpha-.85f)/.15f,0.f,1.f);weight=weight*weight*(3-2*weight);
                readyCorrection=slerp({0,0,0,1},between(neutralBarrel,{0,0,1}),weight);
            }
            // Filter locomotion transitions before layering the short fire clip,
            // so the shot does not lose its attack to a second smoothing pass.
            recoil(bones,rotations,bank,a);recoil(source,sourceQ,animations,a);
            for(int n=1;n<21;++n){if(n>=12&&n<20)continue;bones[n]=bones[0]+rotate(bones[n]-bones[0],readyCorrection);rotations[n]=multiply(readyCorrection,rotations[n]);source[n]=source[0]+rotate(source[n]-source[0],readyCorrection);sourceQ[n]=multiply(readyCorrection,sourceQ[n]);}
            for(auto& p:bones)p=p*a.scale;
            bool aiming=a.health>0&&a.action!="Trap"&&a.reloadEnd<=time&&a.shieldAlpha<.01f;
            if(aiming){Quat bodyPitch=axisRotation({1,0,0},-a.pitch*.4f),gunPitch=axisRotation({1,0,0},-a.pitch);
                for(int n=1;n<21;++n){if(n>=12&&n<20)continue;bones[n]=bones[0]+rotate(bones[n]-bones[0],bodyPitch);rotations[n]=multiply(bodyPitch,rotations[n]);source[n]=source[0]+rotate(source[n]-source[0],gunPitch);sourceQ[n]=multiply(gunPitch,sourceQ[n]);}}
            a.gripError=0;
            if(appearance.gripIK&&a.action!="Trap"&&a.health>0){V3 separation=(source[8]-source[11])*weapons[a.weapon].scale.x;
                // UE's ApplyGrip keeps the current retargeted target wrist and
                // applies the source/target retarget-pose delta afterward.
                rotations[8]=multiply(rotations[8],appearance.wristCorrection[0]);
                rotations[11]=multiply(rotations[11],appearance.wristCorrection[1]);
                V3 rightOffset=rotate(appearance.rightGrip*a.scale,rotations[11]),leftOffset=rotate(appearance.leftGrip*a.scale,rotations[8]);V3 wrist=rightOffset+separation-leftOffset,goal=bones[11]+appearance.weaponOffset*a.scale;
                const auto reach=[&](int shoulder,V3 target){float length=((bones[shoulder+1]-bones[shoulder]).length()+(bones[shoulder+2]-bones[shoulder+1]).length())*.995f;V3 d=target-bones[shoulder];return bones[shoulder]+d.normalized()*std::min(d.length(),length);};
                for(int step=0;step<8;++step){goal=reach(9,goal);goal=goal+reach(6,goal+wrist)-(goal+wrist);}
                V3 rightPole=bones[10]-bones[9]+(bones[10]-(bones[9]+bones[11])*.5f).normalized()*(.3f*a.scale);
                V3 leftPole=bones[7]-bones[6]+(bones[7]-(bones[6]+bones[8])*.5f).normalized()*(.3f*a.scale);
                solveLimb(bones[9],bones[10],bones[11],goal,rightPole);solveLimb(bones[6],bones[7],bones[8],bones[11]+wrist,leftPole);
                a.gripError=(bones[8]+leftOffset-bones[11]-rightOffset-separation).length();}
            V3 gunPosition=bones[11]+rotate(appearance.rightGrip*a.scale,rotations[11])+source[20]-source[11];Quat gunRotation=sourceQ[20];
            if(a.shield>0&&a.health>0&&a.action!="Trap"){float u=a.shieldAlpha;V3 oldHand=bones[11];
                V3 oldForearm=bones[11]-bones[10];
                float left=(bones[7]-bones[6]).length()+(bones[8]-bones[7]).length(),right=(bones[10]-bones[9]).length()+(bones[11]-bones[10]).length();
                V3 carry=bones[6]+V3{-.15f,-.85f,.15f}*left,guard=bones[6]+V3{.22f,.12f,.80f-a.shieldHit*.06f}*left;float alpha=u*u*(3-2*u);
                solveLimb(bones[6],bones[7],bones[8],carry*(1-alpha)+guard*alpha,{-.18f,-.08f,0});
                // UE ApplyShield does not solve the right arm at all when
                // RaiseAlpha is zero. A carried shield only owns the left arm.
                if(u>0){auto guardPose=bones;solveLimb(guardPose[9],guardPose[10],guardPose[11],bones[9]+V3{.08f,-.72f,.28f}*right,{.2f,-1,-.25f});
                    auto blended=blendPose(bones,guardPose,alpha);bones[10]=blended[10];bones[11]=blended[11];
                    Quat delta=between(oldForearm,bones[11]-bones[10]);rotations[11]=multiply(delta,rotations[11]);gunPosition=bones[11]+rotate(gunPosition-oldHand,delta);gunRotation=multiply(delta,gunRotation);}}
            if(a.action=="Trap"&&a.health>0){
                const auto smooth=[](float lo,float hi,float v){float t=std::clamp((v-lo)/(hi-lo),0.f,1.f);return t*t*(3-2*t);};
                float phase=float(time-a.actionStart)/2,alpha=smooth(.12f,.42f,phase)*(1-smooth(.55f,.88f,phase));
                V3 target=rotate(a.goal+V3{0,.09f,0}-a.pos,-a.yaw),delta=bones[9]-target;
                float reach=((bones[10]-bones[9]).length()+(bones[11]-bones[10]).length())*.94f;
                float lowering=std::clamp(delta.y-std::sqrt(std::max(0.f,reach*reach-delta.x*delta.x-delta.z*delta.z)),0.f,reach*.65f)*alpha;
                V3 leftFoot=bones[14],rightFoot=bones[18],oldHand=bones[11];for(auto& p:bones)p.y-=lowering;
                solveLimb(bones[12],bones[13],bones[14],leftFoot,{0,0,1});solveLimb(bones[16],bones[17],bones[18],rightFoot,{0,0,1});
                bones[15]=bones[15]+bones[14]-(leftFoot-V3{0,lowering,0});bones[19]=bones[19]+bones[18]-(rightFoot-V3{0,lowering,0});
                V3 forearm=bones[11]-bones[10];solveLimb(bones[9],bones[10],bones[11],bones[11]*(1-alpha)+target*alpha,{1,0,.3f});
                rotations[11]=multiply(between(forearm,bones[11]-bones[10]),rotations[11]);gunPosition=gunPosition+bones[11]-oldHand;
            }
            gunPosition=gunPosition+rotate(weapons[a.weapon].attachOffset,gunRotation);gunRotation=multiply(gunRotation,weapons[a.weapon].attachRotation);bones[20]=gunPosition;
            for(auto& p:bones)p=a.pos+rotate(p,a.yaw);
            Quat facing=axisRotation({0,1,0},a.yaw);for(auto& q:rotations)q=multiply(facing,q);gunRotation=multiply(facing,gunRotation);
            a.posedBones=bones;a.posedRotations=rotations;
            a.device=bones[11]+V3{0,-.07f,0};
            auto fitted=characterBoxes(bones,rotations,appearance,a.scale,a.yaw);a.boxes.assign(fitted.begin(),fitted.end());
            a.gun=boneTransform(bones[20],weapons[a.weapon].scale,gunRotation);a.muzzle=point(a.gun,weapons[a.weapon].muzzle);
            float shieldYaw=a.yaw-(1-a.shieldAlpha)*55.f*Pi/180;V3 shieldForward=forward(shieldYaw),shieldCenter=bones[8]+rotate({.14f,.08f,.12f},shieldYaw);
            const auto& head=a.boxes[0];float headFront=dot(position(head)-shieldCenter,shieldForward)+.5f*(std::abs(dot({head._11,head._12,head._13},shieldForward))+std::abs(dot({head._21,head._22,head._23},shieldForward))+std::abs(dot({head._31,head._32,head._33},shieldForward)));
            // Keep the enlarged fitted head behind the shield's visible and collidable face.
            float guardAlpha=a.shieldAlpha*a.shieldAlpha*(3-2*a.shieldAlpha);shieldCenter=shieldCenter+shieldForward*(std::max(0.f,headFront+.08f)*guardAlpha);
            a.shieldMatrix=aimTransform(shieldCenter,{1.1f,1.3f,.10f},shieldYaw,-a.shieldHit*.06f);
            collision.setHitBoxes(a.body,a.token,a.health>0?a.boxes:std::vector<Matrix>{},a.shieldToken,a.shield>0&&a.action!="Trap"?&a.shieldMatrix:nullptr);
            V3 min{1e9f,1e9f,1e9f},max{-1e9f,-1e9f,-1e9f};for(const auto& m:a.boxes){bounds(a.token,m);const auto& object=objects[a.token];
                min={std::min(min.x,object.min.x),std::min(min.y,object.min.y),std::min(min.z,object.min.z)};max={std::max(max.x,object.max.x),std::max(max.y,object.max.y),std::max(max.z,object.max.z)};}
            objects[a.token].min=min;objects[a.token].max=max;bounds(a.weaponToken,a.gun);if(a.shieldToken)bounds(a.shieldToken,a.shieldMatrix);
        }}
    void draw(uint32_t mesh,const Matrix& matrix,Color color,uint32_t tokenId,float radius=2,float surface=0,float maxScreenDiameter=0){DrawObject d;d.mesh=mesh;d.instance.world=matrix;d.instance.color=color;d.instance.objectToken=tokenId;d.instance.surface=surface;d.center=position(matrix);d.radius=radius;d.range=200;d.maxScreenDiameter=maxScreenDiameter;renderDraws.push_back(d);}
    const std::vector<DrawObject>& draw(float projectileAlpha=1){renderDraws=staticDraws;for(auto& a:actors){for(int j=0;j<6;++j)draw(box,a.boxes[j],bboxFill(partColors[j]),a.token);if(a.action!="Trap"){draw(weapons[a.weapon].mesh,a.gun,bboxFill(weapons[a.weapon].color),a.weaponToken);if(a.shield>0)draw(box,a.shieldMatrix,bboxFill(shieldColor),a.shieldToken);}if(a.flash>0)draw(sphere,transform(a.muzzle,{.3f,.3f,.3f}),rgb(255,244,163),0,.3f,-1,.08f);}
        if(wall.token&&wall.height>0)draw(box,coverMesh(),bboxFill(coverSettings.color),wall.token,2.4f);
        auto pieces=collision.debris();for(size_t i=0;i<pieces.size();++i)draw(coverSettings.fragments[i%4],pieces[i],bboxFill(coverSettings.color),0,2.4f);
        for(const auto& a:actors)if(a.action=="Trap"&&!a.trapCommitted){Matrix m=transform(a.device,{.5f,.14f,.5f});draw(cylinder,m,bboxFill(rgb(56,149,144)),a.heldTrapToken,.4f);bounds(a.heldTrapToken,m);}
        for(const auto& b:bullets){const auto& w=weapons[b.weapon];V3 pos=b.previous*(1-projectileAlpha)+b.pos*projectileAlpha;float diameter=w.visualDiameter;
            Matrix m=transform(pos,{diameter,diameter,diameter});draw(sphere,m,bboxFill(w.bulletColor),b.token,diameter*.6f,0,.08f);bounds(b.token,m);}
        for(const auto& p:pickups)if(p.active){draw(cylinder,transform(p.pos+V3{0,.08f,0},{1.5f,.16f,1.5f}),bboxFill(p.kind==3?rgb(77,212,147):rgb(255,191,206)),p.padToken);bounds(p.padToken,transform(p.pos,{1.5f,.3f,1.5f}));
            V3 center=p.pos+V3{0,1.15f+.06f*std::sin(float(time)*2),0};Matrix m=transform(center,p.kind==3?V3{1.1f,1.3f,.1f}:V3{1.5f,1.5f,1.5f},float(time)*.698f);draw(p.kind==3?box:weapons[p.kind].mesh,m,bboxFill(p.kind==3?shieldColor:weapons[p.kind].color),p.token);bounds(p.token,m);}
        for(const auto& t:traps)if(t.active){Matrix m=transform(t.pos+V3{0,.07f,0},{.50f,.14f,.50f});draw(cylinder,m,bboxFill(rgb(56,149,144)),t.token,.4f);bounds(t.token,m);float phase=float(std::fmod(time,.85));if(time-t.planted>.65&&(phase<.12f||(phase>=.22f&&phase<.34f)))draw(sphere,transform(t.pos+V3{0,.15f,0},{.08f,.05f,.08f}),rgb(255,34,22),0,.1f,-1);}
        for(const auto& e:effects){float age=float(time-e.born);V3 p=e.pos+e.velocity*age+V3{0,-4.9f*age*age,0};auto h=collision.ray(e.pos,p,0,true);if(h.hit)p=h.point+V3{0,.08f,0};draw(box,transform(p,{.5f,.6f,.05f},age*2,age),bboxFill(e.color),0,.5f);}
        return renderDraws;
    }
    Camera camera(int i)const{
        const auto& a=actors.at(i);
        V3 f=forward(a.yaw,a.pitch),right=rotate({1,0,0},a.yaw),up=cross(f,right);
        V3 pivot=a.pos+V3{0,1.65f-.5f*a.crouchAlpha,0};
        V3 desired=pivot-f*(a.ads?2.2f:map.scene.cameraDistance)+right*(a.ads?1.1f:1.9f)+up*map.scene.cameraUp;
        V3 eye=collision.camera(pivot,desired);
        return {eye,eye+f*100,a.ads?70.f:80.f};
    }
};
Encounter::Encounter(const std::filesystem::path& root,const MapData& map,Renderer& renderer,uint32_t seed):impl(std::make_unique<Impl>(root,map,renderer,seed)){}
Encounter::~Encounter()=default;
void Encounter::advance(float dt,const CombatInput& input){int steps=std::max(1,int(std::ceil(dt*120-.0001f)));CombatInput in=input;for(int i=0;i<steps;++i){in.lookX=input.lookX/steps;in.lookY=input.lookY/steps;impl->simulate(dt/steps,in);in.jump=in.dash=in.trap=in.cover=in.reload=false;in.weapon=-1;}}
void Encounter::setAutopilot(bool enabled){impl->automatic=enabled;impl->actors[0].path.clear();impl->actors[0].target=-1;impl->actors[0].requestedWeapon=-1;if(!enabled)impl->shield(0,false);}
bool Encounter::autopilot()const{return impl->automatic;}
void Encounter::setRecording(bool enabled){if(enabled&&!impl->recording){++impl->coverGeneration;impl->coverUsed=false;impl->coverSelected=std::uniform_real_distribution<float>(0,1)(impl->coverRng)<impl->coverSettings.probability;impl->coverAttempt=impl->time+impl->coverSettings.delay;}impl->recording=enabled;}
void Encounter::setCoverProbability(float probability){impl->coverSettings.probability=std::clamp(probability,0.f,1.f);}
void Encounter::prepareClip(){auto& a=impl->actors[Impl::PlayerIndex];a.appearance=impl->playerAppearance;const auto& appearance=impl->appearances[a.appearance];a.index=appearance.index;a.scale=appearance.scale;a.localPose.clear();a.posedAt=-1;impl->objects[a.token].index=a.index;impl->pose();}
Camera Encounter::camera(int player)const{return impl->camera(player);}
V3 Encounter::playerPosition(int p)const{return impl->actors.at(p).pos+V3{0,.9f,0};}
uint32_t Encounter::playerToken(int p)const{return impl->actors.at(p).token;}
const std::vector<DrawObject>& Encounter::draws(float projectileAlpha){return impl->draw(std::clamp(projectileAlpha,0.f,1.f));}
const std::vector<DrawObject>& Encounter::mapDraws()const{return impl->staticDraws;}
const std::vector<CatalogObject>& Encounter::catalog()const{return impl->objects;}
std::vector<CombatEvent> Encounter::takeEvents(){auto result=std::move(impl->events);impl->events.clear();return result;}
bool Encounter::hasEvents()const{return !impl->events.empty();}
Json Encounter::eventState()const{Json actors=Json::array(),traps=Json::array();for(const auto& a:impl->actors)actors.push_back({{"Token",a.token},{"Index",a.index},{"Health",a.health},{"WeaponToken",a.weaponToken},{"Weapon",a.weapon},{"Ammo",a.ammo},{"Grounded",a.grounded},{"ShieldToken",a.shieldToken},{"ShieldRaised",a.raised},{"ShieldAlpha",a.shieldAlpha},{"Action",a.action},{"ActionStart",a.actionStart},{"HeldTrapToken",a.heldTrapToken},{"TrapCommitted",a.trapCommitted}});
    for(const auto& t:impl->traps)traps.push_back({{"Token",t.token},{"OwnerToken",t.owner},{"Active",t.active},{"PlantedTime",t.planted}});return {{"Actors",actors},{"Traps",traps}};}
bool Encounter::eventVisible(const CombatEvent& e,const Camera& c,const CapturedView& view,uint32_t observer)const{
    const auto& visibility=e.visibilityTokens.empty()?e.tokens:e.visibilityTokens;
    bool participant=false;for(auto t:visibility)if(t<view.pixels.size()&&view.pixels[t]>0)participant=true;
    // The observer sees its own successful placement gesture even when the
    // small device lands below the viewport or behind the observer's legs.
    if(e.type=="TrapPlaced"&&e.actor==observer&&observer<view.pixels.size()&&view.pixels[observer]>0)return true;
    if(!e.pointEvent)return participant;
    V3 probe=e.position+e.surfaceNormal*.02f,d=probe-c.eye,f=(c.target-c.eye).normalized(),r=cross({0,1,0},f).normalized(),u=cross(f,r);float z=dot(d,f),t=std::tan(c.horizontalFov*Pi/360);
    bool pointOnscreen=z>.08f&&std::abs(dot(d,r))<z*t&&std::abs(dot(d,u))<z*t/(1280.f/720.f);
    if(!pointOnscreen||view.ids.size()!=1280*720)return false;
    const auto matches=[&](uint32_t token){return std::find(visibility.begin(),visibility.end(),token)!=visibility.end();};
    auto hit=impl->collision.ray(c.eye,probe,observer);
    if(hit.hit&&(hit.point-probe).length()>.05f&&!matches(hit.token))return false;
    int x=std::clamp(int((.5f+.5f*dot(d,r)/(z*t))*1280),0,1279),y=std::clamp(int((.5f-.5f*dot(d,u)/(z*t)*1280/720)*720),0,719);
    for(int py=std::max(0,y-1);py<=std::min(719,y+1);++py)for(int px=std::max(0,x-1);px<=std::min(1279,x+1);++px)if(matches(view.ids[size_t(py)*1280+px]))return true;
    if(!e.allowPositionOnly)return false;
    uint32_t foreground=view.ids[size_t(y)*1280+x];if(!foreground||foreground>=impl->objects.size())return true;
    // Match UE's position-only fallback: a foreground object's world bounds
    // still reject a vanished event object hidden behind that object.
    const auto& object=impl->objects[foreground];float enter=0,leave=1;
    for(int axis=0;axis<3;++axis){float start=(&c.eye.x)[axis],delta=(&d.x)[axis],lo=(&object.min.x)[axis],hi=(&object.max.x)[axis];
        if(std::abs(delta)<1e-7f){if(start<lo||start>hi)return true;continue;}float a=(lo-start)/delta,b=(hi-start)/delta;if(a>b)std::swap(a,b);enter=std::max(enter,a);leave=std::min(leave,b);if(enter>leave)return true;}
    return false;
}
std::string Encounter::action(int p)const{const auto& a=impl->actors.at(p);if(!a.action.empty())return a.action;if(a.reloadEnd>impl->time)return "Reload";if(a.raised)return "Shield";
    if(a.health>0&&a.shotWeapon==a.weapon&&a.shieldAlpha<.001f&&impl->time>=a.lastShot&&impl->time-a.lastShot<impl->animations.at("MM_"+impl->weapons[a.weapon].name+"_Fire").duration)return "Fire";
    if(flatLength(a.velocity)>.15f)return "Walk";return "Idle";}
void Encounter::hud(HudState& h,int p)const{const auto& a=impl->actors.at(p);h.combat=true;h.player=p+1;h.automatic=p!=0||impl->automatic;h.ammo=a.ammo[a.weapon];h.weapon=a.weapon;h.owned=a.owned;h.health=a.health;h.shield=a.shield;h.raised=a.raised;h.trapCooldown=float(std::max(0.0,a.trapReady-impl->time));h.combatAction=action(p);h.kills=int(impl->counts.contains("EnemyDied")?impl->counts.at("EnemyDied"):0);
    h.coverDurability=impl->wall.token?impl->wall.durability:0;h.coverStatus=impl->wall.token?std::to_string(int(impl->wall.durability))+" HP":impl->recording&&impl->coverUsed?"USED IN CLIP":impl->time<impl->coverReady?std::to_string(int(std::ceil(impl->coverReady-impl->time)))+" S":"READY";h.coverHint=impl->coverMessage;}
bool Encounter::recoveryRequested()const{return impl->recovery;}
bool Encounter::recover(){if(!impl->restart())return false;impl->maintenance();impl->pose();++impl->counts["PairedRecovery"];return true;}
bool Encounter::playerAlive()const{return impl->actors[Impl::PlayerIndex].health>0;}
bool Encounter::retry(){
    if(!impl->restart())return false;
    impl->spawn(Impl::PlayerIndex,impl->actors[Impl::PlayerIndex].pos,true);
    impl->actors[Impl::PlayerIndex].owned.fill(true);impl->automatic=false;
    impl->counts.clear();impl->events.clear();impl->maintenance();impl->pose();return true;
}
uint64_t Encounter::step()const{return impl->tick;}
double Encounter::seconds()const{return impl->time;}
Json Encounter::diagnostics()const{Json actors=Json::array();for(const auto& a:impl->actors)actors.push_back({{"position",{a.pos.x,a.pos.y,a.pos.z}},{"health",a.health},{"index",a.index},{"shield",a.shield},{"weapon",a.weapon},{"owned",a.owned},{"ammo",a.ammo},{"action",a.action},{"yaw",a.yaw},{"pitch",a.pitch},{"grounded",a.grounded}});return {{"seconds",impl->time},{"step",impl->tick},{"seed",impl->seed},{"automatic",impl->automatic},{"navPolygons",impl->collision.navPolygons()},{"events",impl->counts},{"characters",actors},{"bullets",impl->bullets.size()},{"pickups",impl->pickups.size()},{"recoveryRequested",impl->recovery},{"cover",{{"active",impl->wall.token!=0},{"durability",impl->wall.durability},{"height",impl->wall.height},{"index",impl->coverSettings.index},{"generation",impl->coverGeneration},{"selected",impl->coverSelected},{"used",impl->coverUsed},{"cooldown",std::max(0.0,impl->coverReady-impl->time)}}}};}
Json Encounter::sceneTest(){
    auto& g=*impl;
    const auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    require(g.collision.navPolygons()>0,"Scene has no navigation");
    require(Impl::ActorCount==3&&g.actors[1].weapon==2&&g.actors[2].weapon==2,"Scene changed demo loadout policy");
    require(g.actors[0].owned==std::array<bool,3>{true,true,true},"Scene lost player weapons");
    for(int i=1;i<Impl::ActorCount;++i)require(!g.collision.path(g.actors[0].pos,g.actors[i].pos).empty(),"Disconnected scene spawn");
    const auto folder=g.map.scene.outputDirectory(g.root,"captures")/"scene-test";
    std::filesystem::create_directories(folder);
    Json samples=Json::array();std::set<uint32_t> allIdentities;
    for(int i=0;i<Impl::ActorCount;++i){
        const auto cam=g.camera(i);const auto capture=g.renderer.captureScene(g.draw(),cam,g.objects,false);
        const V3 f=(cam.target-cam.eye).normalized(),r=cross({0,1,0},f).normalized(),u=cross(f,r);
        const float tx=std::tan(cam.horizontalFov*Pi/360),ty=tx*720/1280;
        int checked=0,rayMatched=0,sweepMatched=0;std::set<uint32_t> identities;
        for(int y=20;y<700;y+=19)for(int x=20;x<1260;x+=19){
            const auto token=capture.ids[size_t(y)*1280+x];if(!token||token>g.map.objects.size())continue;
            const V3 dir=(f+r*((2*(x+.5f)/1280-1)*tx)+u*((1-2*(y+.5f)/720)*ty)).normalized();
            const auto hit=g.collision.ray(cam.eye,cam.eye+dir*1000,0,true);
            const auto sweep=g.collision.sweepBullet(cam.eye,cam.eye+dir*1000,.001f,0,true);
            ++checked;if(hit.hit&&hit.token==token){++rayMatched;identities.insert(token);}
            if(sweep.hit&&sweep.token==token)++sweepMatched;
        }
        samples.push_back({{"view",i},{"samples",checked},{"ray_matches",rayMatched},{"sweep_matches",sweepMatched},{"distinct_instances",identities.size()}});
        allIdentities.insert(identities.begin(),identities.end());
        std::ofstream(folder/"id-parity.json")<<samples.dump(2);
        require(checked>=20&&rayMatched>=checked*.95f&&sweepMatched>=checked*.95f,"Render/collision instance IDs disagree");
#ifndef _WIN32
        require(g.renderer.captureDemoScene(g.draw(),cam)==capture.color,"Offline and reference RGB differ");
#endif
        g.renderer.render(g.draw(),cam,{});
        require(g.renderer.screenshot(folder/("view-"+std::to_string(i)+".png")),"Scene screenshot failed");
    }
    require(allIdentities.size()>=4,"Scene test did not sample enough distinct instances");
    return {{"passed",true},{"scene",g.map.scene.id},{"objects",g.map.objects.size()},{"draw_batches",g.staticDraws.size()},{"trees",g.map.treeCount},{"nav_polygons",g.collision.navPolygons()},{"id_parity",samples},{"state",diagnostics()}};
}

Json Encounter::mapSurfaceTest(){
    auto& g=*impl;
    const auto folder=g.map.scene.outputDirectory(g.root,"captures")/"map-surface-test";
    std::filesystem::create_directories(folder);Json results=Json::array();
    for(const auto& [index,parts]:g.map.roadSurfaceSplits){
        const auto& object=g.map.objects[index];const uint32_t token=uint32_t(index+1);
        const auto makeDraw=[&](MeshData mesh,float surface){
            if(surface==0)for(auto& packed:mesh.colors){
                const auto c=bboxFill(rgb(packed&255,(packed>>8)&255,(packed>>16)&255));
                packed=0xff000000u|uint32_t(std::lround(c.r*255))|(uint32_t(std::lround(c.g*255))<<8)|(uint32_t(std::lround(c.b*255))<<16);
            }
            DrawObject d;d.mesh=g.renderer.addMesh(mesh);d.instance.world=transform({});d.instance.objectToken=token;
            d.instance.surface=surface;d.center=object.bounds.center();d.radius=(object.bounds.max-object.bounds.min).length()*.5f;d.range=2000;return d;
        };
        const float material=g.map.scene.groundPattern==SceneConfig::GroundPattern::EgyptNested?5.f:3.f;
        const std::vector<DrawObject> before{makeDraw(object.mesh,material)};
        const std::vector<DrawObject> after{makeDraw(parts.floor,material),makeDraw(parts.structure,0)};
        const auto center=object.bounds.center();
        const V3 pivot{center.x,object.bounds.min.y,center.z};
        const Camera camera{pivot+V3{8,9,31},pivot+V3{0,1.5f,0},75};
        const auto original=g.renderer.captureScene(before,camera,g.objects);
        const auto corrected=g.renderer.captureScene(after,camera,g.objects);
        if(original.ids!=corrected.ids||original.depth!=corrected.depth)throw std::runtime_error("Surface override changed geometry or instance IDs");
        size_t black=0,fill=0,unchangedFloor=0;
        const auto packed=parts.structure.colors.front();
        const auto c=bboxFill(rgb(packed&255,(packed>>8)&255,(packed>>16)&255));
        const int expected[]={int(std::lround(c.r*255)),int(std::lround(c.g*255)),int(std::lround(c.b*255))};
        for(size_t p=0;p<corrected.ids.size();++p)if(corrected.ids[p]==token){
            const auto* a=original.color.data()+4*p;const auto* b=corrected.color.data()+4*p;
            if(b[0]<=8&&b[1]<=8&&b[2]<=8)++black;
            if(b[0]==expected[0]&&b[1]==expected[1]&&b[2]==expected[2])++fill;
            if(b[0]==a[0]&&b[1]==a[1]&&b[2]==a[2]&&b[0]>150)++unchangedFloor;
        }
        if(black<50||fill<500||unchangedFloor<500)throw std::runtime_error("Courtyard floor, colored structures or black outlines missing");
        for(bool fixed:{false,true}){
            g.renderer.render(fixed?after:before,camera,{});
            if(!g.renderer.screenshot(folder/(std::to_string(index)+(fixed?"-after.png":"-before.png"))))throw std::runtime_error("Surface screenshot failed");
        }
        g.renderer.render(g.staticDraws,camera,{});g.renderer.screenshot(folder/(std::to_string(index)+"-scene.png"));
        results.push_back({{"object_index",index},{"floor_triangles",parts.floor.indices.size()/3},{"structure_triangles",parts.structure.indices.size()/3},
            {"black_pixels",black},{"fill_pixels",fill},{"unchanged_floor_pixels",unchangedFloor},{"geometry_and_ids_unchanged",true}});
    }
    if(g.map.scene.id=="ancient-egypt"&&results.size()!=3)throw std::runtime_error("Expected all three mixed courtyard meshes");
    return {{"passed",true},{"scene",g.map.scene.id},{"surfaces",results}};
}

Json Encounter::enemyFiringTest(){
    auto& g=*impl;auto& a=g.actors[1];auto folder=g.map.scene.outputDirectory(g.root,"captures")/"enemy-ready-regression";std::filesystem::create_directories(folder);Json checks=Json::array();
    const V3 directions[]={{0,0,2.5f},{0,0,-2.5f},{-2.5f,0,0},{2.5f,0,0}};
    const auto angle=[&](V3 target){V3 barrel=(point(a.gun,{0,0,1})-position(a.gun)).normalized();return std::acos(std::clamp(dot(barrel,(target-a.muzzle).normalized()),-1.f,1.f))*180/Pi;};
    const auto collect=[&]{std::vector<DrawObject> draws;for(const auto& d:g.draw())if(d.instance.objectToken==a.token||d.instance.objectToken==a.weaponToken||(a.shieldToken&&d.instance.objectToken==a.shieldToken))draws.push_back(d);return draws;};
    for(size_t appearance=0;appearance<g.appearances.size();++appearance)for(int direction=0;direction<4;++direction)for(bool carry:{false,true}){
        g.time=10;g.bullets.clear();size_t body=a.body;Impl::Actor fresh;fresh.body=body;fresh.appearance=appearance;fresh.index=g.appearances[appearance].index;fresh.scale=g.appearances[appearance].scale;
        fresh.pos={0,g.map.bounds.max.y+30,0};fresh.velocity=directions[direction];fresh.token=g.token(fresh.index);fresh.weapon=2;fresh.weaponToken=g.token(g.weapons[2].index);fresh.owned={true,true,true};fresh.ammo[2]=1;
        fresh.shield=carry?200.f:0;fresh.shieldToken=carry?g.token(13):0;fresh.target=0;a=fresh;V3 target=a.pos+V3{0,1.15f,16};g.actors[0].pos=target-V3{0,1.15f,0};
        for(int step=0;step<30;++step){g.time+=1.0/120;g.advanceGait(a,1.f/120);g.pose();}
        float lowAngle=angle(target);int before=a.ammo[2];g.fire(1,target);if(a.ammo[2]!=before||lowAngle<10)throw std::runtime_error("Enemy firing regression: low-ready jogging gun fired before raising");
        int firedAt=-1;float firedAngle=0;std::string name=std::string(carry?"carry":"normal");Camera camera{a.pos+V3{2.6f,1.45f,2.4f},a.pos+V3{0,1.02f,.22f},60};
        for(int frame=0;frame<120;++frame){g.time+=1.0/120;++g.tick;g.advanceGait(a,1.f/120);a.weaponReadyAlpha=approach(a.weaponReadyAlpha,1,8.f/120);g.pose();float beforeAngle=angle(target);before=a.ammo[2];g.fire(1,target);
            if(a.ammo[2]<before){firedAt=frame;firedAngle=beforeAngle;if(beforeAngle>10.001f)throw std::runtime_error("Enemy firing regression: barrel not aligned at discharge");}
            if(firedAt>=0&&g.time-a.lastShot<g.animations.at("MM_Shotgun_Fire").duration){g.reload(1);g.equip(1,0);if(a.reloadEnd>g.time||a.weapon!=2)throw std::runtime_error("Enemy firing regression: last round/switch cancelled its shot pose");}
            if(appearance==g.playerAppearance&&direction==0&&frame%4==0){g.renderer.render(collect(),camera,{});g.renderer.screenshot(folder/(name+"-"+std::to_string(frame/4)+".png"));}
        }
        if(firedAt<0||firedAt<3)throw std::runtime_error("Enemy firing regression: no visible raise before discharge, or unable to fire");
        checks.push_back({{"appearance",g.appearances[appearance].name},{"direction",direction},{"carriedShield",carry},{"lowReadyAngleDegrees",lowAngle},{"dischargeAngleDegrees",firedAngle},{"raiseFramesBeforeShot",firedAt}});
    }
    // Exercise the actual simulate -> AI intent -> pose -> projectile path on
    // the real map, after the isolated motion/appearance checks above.
    if(!g.restart())throw std::runtime_error("Enemy firing regression: cannot reset live encounter");g.automatic=false;g.maximumEnemyShotAngle=0;g.enemyShotgunShots=0;
    V3 candidate;bool found=false;for(int attempt=0;attempt<100;++attempt)if(g.collision.randomPoint(g.actors[0].pos,8,16,g.rng,candidate)&&!g.collision.ray(candidate+V3{0,1.5f,0},g.actors[0].pos+V3{0,1.1f,0},0,true).hit){found=true;break;}
    if(!found)throw std::runtime_error("Enemy firing regression: no visible live spawn");g.spawn(1,candidate,true);a.appearance=g.playerAppearance;a.index=g.appearances[a.appearance].index;a.scale=g.appearances[a.appearance].scale;g.objects[a.token].index=a.index;
    a.owned={true,true,true};a.weapon=2;a.weaponToken=g.token(g.weapons[2].index);a.ammo[2]=8;a.switchAt=a.nextDash=a.nextJump=a.shieldCheck=1e9;
    V3 aim=g.actors[0].pos-a.pos;a.yaw=std::atan2(aim.x,aim.z);a.target=0;a.fireAt=g.time+.1;a.localPose.clear();a.posedAt=-1;
    g.nextShield=1e9;for(int i=2;i<Impl::ActorCount;++i){g.actors[i].health=0;g.actors[i].deathAt=g.actors[i].respawnAt=1e9;g.actors[i].pos={float(i*10),-1000,0};g.collision.teleport(g.actors[i].body,g.actors[i].pos);}
    g.pose();setRecording(true);Recording recording(g.map.scene.outputDirectory(g.root,"captures")/"enemy-ready-recording",g.root/L"assets",g.seed,g.map.scene.assets);
    uint64_t movingShots=0;for(int frame=0;frame<240;++frame){for(int step=0;step<5;++step){uint64_t previousShots=g.enemyShotgunShots;advance(1.f/120);if(g.enemyShotgunShots>previousShots&&flatLength(a.velocity)>.2f)++movingShots;}recording.capture(g.renderer,*this);}
    recording.finish();setRecording(false);if(g.enemyShotgunShots==0||movingShots==0||g.maximumEnemyShotAngle>10.001f)throw std::runtime_error("Enemy firing regression: AI never performed an aligned moving shotgun shot");
    return {{"passed",true},{"cases",checks},{"weapon","Shotgun"},{"lastRoundPosePreserved",true},{"switchCannotInterruptShot",true},{"liveEnemyShotgunShots",g.enemyShotgunShots},{"liveMovingShotgunShots",movingShots},{"maximumLiveEnemyBarrelAngleDegrees",g.maximumEnemyShotAngle}};
}
Json Encounter::firingTest(){
    auto& g=*impl;auto& a=g.actors[0];auto folder=g.map.scene.outputDirectory(g.root,"captures")/"fire-regression";std::filesystem::create_directories(folder);
    const auto collect=[&]{std::vector<DrawObject> result;for(const auto& d:g.draw())if(d.instance.objectToken==a.token||d.instance.objectToken==a.weaponToken||(a.shieldToken&&d.instance.objectToken==a.shieldToken))result.push_back(d);return result;};
    const auto difference=[](const Matrix& lhs,const Matrix& rhs){float maximum=0;for(int i=0;i<8;++i){V3 corner{i&1?.5f:-.5f,i&2?.5f:-.5f,i&4?.5f:-.5f};maximum=std::max(maximum,(point(lhs,corner)-point(rhs,corner)).length());}return maximum;};
    Json cases=Json::array();const int links[][2]={{6,7},{7,8},{9,10},{10,11},{12,13},{13,14},{16,17},{17,18}};
    for(int weapon=0;weapon<3;++weapon)for(bool carry:{false,true}){
        g.time=1;g.bullets.clear();a.appearance=g.playerAppearance;const auto& appearance=g.appearances[a.appearance];a.scale=appearance.scale;a.index=appearance.index;g.objects[a.token].index=a.index;
        a.pos={0,g.map.bounds.max.y+30,0};a.velocity={};a.yaw=a.pitch=0;a.weapon=weapon;a.weaponToken=g.token(g.weapons[weapon].index);a.owned={true,true,true};a.ammo[weapon]=g.weapons[weapon].magazine;
        a.health=100;a.action.clear();a.actionEnd=a.reloadEnd=a.shieldRecovery=0;a.airAlpha=a.crouchAlpha=a.shieldAlpha=0;a.raised=false;a.grounded=true;a.lastShot=-100;a.fireAt=0;a.shield=carry?200.f:0;a.shieldToken=carry?g.token(13):0;
        a.localPose.clear();a.posedAt=-1;for(int step=0;step<36;++step){g.time+=1.0/120;g.pose();}
        Impl::Actor idle=a;int ammo=a.ammo[weapon];g.fire(0,a.pos+V3{0,1.1f,15});if(a.ammo[weapon]!=ammo-1)throw std::runtime_error("Firing regression: held/lowered shield prevented actual firing");Impl::Actor shot=a;
        const double start=g.time;float maxHand=0,maxGun=0,maxBox=0,maxBody=0,maxHead=0,maxLengthError=0,peakTime=0,settled=0;int maxPixels=0;auto peak=shot,peakIdle=idle;
        std::string name=g.weapons[weapon].name+(carry?"-carry":"-normal");
        Camera camera{a.pos+V3{2.6f,1.45f,2.4f},a.pos+V3{0,1.02f,.22f},60};
        for(int frame=0;frame<120;++frame){g.time=start+(frame+1)/120.0;++g.tick;
            a=idle;g.pose();idle=a;auto idleDraws=collect();a=shot;g.pose();shot=a;auto shotDraws=collect();
            float gun=difference(shot.gun,idle.gun);maxHand=std::max(maxHand,(shot.posedBones[11]-idle.posedBones[11]).length());maxBox=std::max(maxBox,difference(shot.boxes[3],idle.boxes[3]));
            maxBody=std::max(maxBody,difference(shot.boxes[1],idle.boxes[1]));maxHead=std::max(maxHead,difference(shot.boxes[0],idle.boxes[0]));
            if(gun>maxGun){maxGun=gun;peakTime=float(g.time-start);peak=shot;peakIdle=idle;}
            for(const auto& link:links){float length=(shot.posedBones[link[1]]-shot.posedBones[link[0]]).length(),reference=(appearance.reference[link[1]]-appearance.reference[link[0]]).length()*a.scale;maxLengthError=std::max(maxLengthError,std::abs(length/reference-1));}
            if(frame%4==0){auto base=g.renderer.captureScene(idleDraws,camera,g.objects),fired=g.renderer.captureScene(shotDraws,camera,g.objects);int changed=0;
                for(size_t i=0;i<base.color.size();i+=4)if(std::max({std::abs(int(base.color[i])-int(fired.color[i])),std::abs(int(base.color[i+1])-int(fired.color[i+1])),std::abs(int(base.color[i+2])-int(fired.color[i+2]))})>8)++changed;maxPixels=std::max(maxPixels,changed);
                g.renderer.render(shotDraws,camera,{});g.renderer.screenshot(folder/(name+"-"+std::to_string(frame/4)+".png"));}
            if(frame==119)settled=gun;
        }
        g.time=start+peakTime;
        a=peakIdle;g.pose();g.renderer.render(collect(),camera,{});g.renderer.screenshot(folder/(name+"-idle.png"));
        a=peak;g.pose();g.renderer.render(collect(),camera,{});g.renderer.screenshot(folder/(name+"-fire.png"));
        if(maxHand<.01f||maxBox<.01f||maxPixels<1000||maxLengthError>.01f||settled>.001f)throw std::runtime_error("Firing regression: missing visible motion, stretched limbs or failed recovery");
        cases.push_back({{"name",name},{"maximumRightHandDisplacementMeters",maxHand},{"maximumWeaponCornerDisplacementMeters",maxGun},{"maximumRightArmBoxDisplacementMeters",maxBox},{"maximumTorsoBoxDisplacementMeters",maxBody},{"maximumHeadBoxDisplacementMeters",maxHead},{"maximumChangedPixels",maxPixels},{"peakSeconds",peakTime},{"maximumLimbLengthRelativeError",maxLengthError},{"settledWeaponErrorMeters",settled}});
        // Holding the shield must not bypass the separate guard/fire recovery rules.
        a.fireAt=0;a.ammo[weapon]=ammo;a.shield=200;a.shieldToken=g.token(13);a.raised=true;a.shieldAlpha=1;double previous=a.lastShot;g.fire(0,a.pos+V3{0,1,15});
        if(a.ammo[weapon]!=ammo||a.lastShot!=previous)throw std::runtime_error("Firing regression: raised guard allowed fire");
    }
    for(int weapon=0;weapon<3;++weapon)for(const char* key:{"maximumRightHandDisplacementMeters","maximumWeaponCornerDisplacementMeters","maximumRightArmBoxDisplacementMeters"})
        if(std::abs(cases[weapon*2].at(key).get<float>()-cases[weapon*2+1].at(key).get<float>())>.0001f)throw std::runtime_error("Firing regression: carried shield overrides right-arm recoil");
    g.automatic=false;g.time=10;a.weapon=0;a.weaponToken=g.token(g.weapons[0].index);a.lastShot=a.previousShot=-100;a.shotWeapon=-1;a.shield=200;a.raised=true;a.shieldAlpha=1;a.fireAt=0;a.ammo[0]=12;a.localPose.clear();a.posedAt=-1;g.pose();g.shield(0,false);
    float maxLowerStep=0;bool resumed=false;auto previous=a.posedBones;
    for(int frame=0;frame<72;++frame){g.time+=1.0/120;a.shieldAlpha=approach(a.shieldAlpha,0,1.f/120/.35f);g.pose();
        for(int bone:{9,10,11})maxLowerStep=std::max(maxLowerStep,(a.posedBones[bone]-previous[bone]).length());previous=a.posedBones;
        if(!resumed){int ammo=a.ammo[0];g.fire(0,a.pos+V3{0,1,15});if(a.ammo[0]<ammo){if(a.shieldAlpha>.001f||g.time<a.shieldRecovery)throw std::runtime_error("Firing regression: shield recovery released early");resumed=true;}}}
    if(!resumed||maxLowerStep>.05f)throw std::runtime_error("Firing regression: lowering guard snaps the arm or leaves firing blocked");
    g.time=15;a.lastShot=a.previousShot=-100;a.shotWeapon=-1;a.raised=false;a.shieldAlpha=0;a.shieldRecovery=0;a.fireAt=0;a.ammo[0]=12;a.localPose.clear();a.posedAt=-1;g.pose();
    float maxRestartJump=0,maxRapidStep=0;int rapidShots=0;previous=a.posedBones;
    for(int frame=0;frame<144;++frame){g.time=15+(frame+1)/120.0;g.pose();auto before=a.posedBones;int ammo=a.ammo[0];g.fire(0,a.pos+V3{0,1,15});
        if(a.ammo[0]<ammo){++rapidShots;g.pose();for(int bone:{0,3,5,9,10,11})maxRestartJump=std::max(maxRestartJump,(a.posedBones[bone]-before[bone]).length());}
        for(int bone:{0,3,5,9,10,11})maxRapidStep=std::max(maxRapidStep,(a.posedBones[bone]-previous[bone]).length());previous=a.posedBones;}
    if(rapidShots<9||maxRestartJump>.001f||maxRapidStep>.1f)throw std::runtime_error("Firing regression: repeated shots reset/snap the pose");
    int firedWeapon=a.shotWeapon;a.weapon=1;g.pose();if(action(0)=="Fire"||a.shotWeapon!=firedWeapon)throw std::runtime_error("Firing regression: switching weapons relabeled the previous recoil");
    return {{"passed",true},{"cases",cases},{"guardBlocksFire",true},{"loweredShieldResumesFire",resumed},{"maximumLowerShieldJointStepMeters",maxLowerStep},{"rapidShots",rapidShots},{"maximumShotRestartJumpMeters",maxRestartJump},{"maximumRapidFireJointStepMeters",maxRapidStep},{"simulationHz",120},{"pixelSampleHz",30}};
}
Json Encounter::trailTest(){
    auto& g=*impl;auto directory=g.map.scene.outputDirectory(g.root,"captures")/"trail-regression";std::filesystem::create_directories(directory);
    const Camera camera{{0,0,-12},{0,0,0},60};std::vector<DrawObject> base;
    const auto fixture=[&](V3 position,V3 size,Color color){DrawObject d;d.mesh=g.box;d.instance.world=transform(position,size);d.instance.color=color;d.instance.surface=-1;d.instance.objectToken=g.token(12);d.center=position;d.radius=10;base.push_back(d);g.bounds(d.instance.objectToken,d.instance.world);};
    fixture({0,0,1},{16,9,.1f},rgb(169,169,169));fixture({0,0,-.4f},{.6f,4,.1f},rgb(95,121,148));
    g.bullets.clear();for(int weapon=0;weapon<3;++weapon){Impl::Bullet b;b.weapon=weapon;b.token=g.token(g.weapons[weapon].projectileIndex);b.pos=b.previous={4.5f,1.5f*(1-weapon),0};b.velocity={g.weapons[weapon].speed,0,0};g.bullets.push_back(b);}
    std::vector<DrawObject> after=base;
    for(const auto& d:g.draw()){
        if(d.instance.surface< -1.5)throw std::runtime_error("Projectile trail geometry is still present");
        if(std::any_of(g.bullets.begin(),g.bullets.end(),[&](const auto& b){return d.instance.objectToken==b.token;}))after.push_back(d);
    }
    if(after.size()!=base.size()+3)throw std::runtime_error("Expected exactly one projectile head per weapon");
    g.renderer.render(base,camera,{});g.renderer.screenshot(directory/L"background.png");
    g.renderer.render(after,camera,{});g.renderer.screenshot(directory/L"after.png");
    auto recorded=g.renderer.captureScene(after,camera,g.objects);
    for(const auto& b:g.bullets)if(b.token>=recorded.pixels.size()||!recorded.pixels.at(b.token))throw std::runtime_error("Projectile head is missing from clean capture");
    const std::array<const char*,3> names{"Pistol","Rifle","Shotgun"};
    std::array<std::string,3> hex;
    std::array<std::array<int,3>,3> source{},expected{};
    Json palette;std::ifstream(g.map.scene.catalogDirectory()/"CatalogManifest.json")>>palette;
    for(int w=0;w<3;++w){
        const auto value=palette.at("bboxColorsByIndex").at(std::to_string(g.weapons[w].projectileIndex)).at("srgbHex").get<std::string>();
        hex[w]="#"+value.substr(0,6);const auto raw=hexColor(value),fill=bboxFill(raw);
        source[w]={int(std::lround(raw.r*255)),int(std::lround(raw.g*255)),int(std::lround(raw.b*255))};
        expected[w]={int(std::lround(fill.r*255)),int(std::lround(fill.g*255)),int(std::lround(fill.b*255))};
    }
    const std::array<std::array<int,3>,3> castleSource{{{221,124,69},{239,118,162},{48,229,221}}};
    if(g.map.scene.id=="castle"&&source!=castleSource)throw std::runtime_error("Castle projectile colors changed");
    Json colors=Json::array();
    for(const auto& b:g.bullets){const auto& w=g.weapons[b.weapon];const auto& rgb=expected[b.weapon];
        const auto it=std::find_if(after.begin(),after.end(),[&](const auto& d){return d.instance.objectToken==b.token;});
        if(w.name!=names[b.weapon]||it==after.end())throw std::runtime_error("Projectile weapon/color mapping changed");
        const auto raw=w.bulletColor;const auto& catalog=source[b.weapon];
        if(std::lround(raw.r*255)!=catalog[0]||std::lround(raw.g*255)!=catalog[1]||std::lround(raw.b*255)!=catalog[2])
            throw std::runtime_error("Projectile catalog source color changed");
        const auto c=it->instance.color;
        if(std::lround(c.r*255)!=rgb[0]||std::lround(c.g*255)!=rgb[1]||std::lround(c.b*255)!=rgb[2]||c.a!=1||it->captureColor.a>=0)
            throw std::runtime_error("Projectile live/capture palette differs from the muted catalog color");
        size_t exact=0;for(size_t p=0;p<recorded.ids.size();++p)if(recorded.ids[p]==b.token&&recorded.color[p*4]==rgb[0]&&recorded.color[p*4+1]==rgb[1]&&recorded.color[p*4+2]==rgb[2])++exact;
        if(!exact)throw std::runtime_error("Clean capture does not contain the exact projectile RGB");
        colors.push_back({{"weapon",w.name},{"catalogHex",hex[b.weapon]},{"renderedRgb",expected[b.weapon]},{"exactCapturePixels",exact},{"visiblePixels",recorded.pixels.at(b.token)}});
    }
    std::reverse(after.begin(),after.end());g.renderer.render(after,camera,{});g.renderer.screenshot(directory/L"reordered.png");
    auto reordered=g.renderer.captureScene(after,camera,g.objects);
    if(recorded.color!=reordered.color||recorded.depth!=reordered.depth||recorded.ids!=reordered.ids)throw std::runtime_error("Projectile capture depends on draw order");
    Json outlines=Json::array();
    for(int distance:{12,24,48}){
        Camera distant{{0,0,-float(distance)},{0,0,0},60};auto view=g.renderer.captureScene(after,distant,g.objects);
        g.renderer.render(after,distant,{});g.renderer.screenshot(directory/("outlines-"+std::to_string(distance)+"m.png"));
        for(const auto& b:g.bullets){size_t black=0,fill=0;const auto& rgb=expected[b.weapon];
            for(size_t p=0;p<view.ids.size();++p)if(view.ids[p]==b.token){
                if(view.color[p*4]<=8&&view.color[p*4+1]<=8&&view.color[p*4+2]<=8)++black;
                if(view.color[p*4]==rgb[0]&&view.color[p*4+1]==rgb[1]&&view.color[p*4+2]==rgb[2])++fill;
            }
            if((!g.map.scene.fadeSmallEdges&&!black)||!fill)throw std::runtime_error("Projectile outline/fill regression at "+std::to_string(distance)+"m");
            outlines.push_back({{"distanceMeters",distance},{"weapon",names[b.weapon]},{"blackPixels",black},{"fillPixels",fill}});
        }
    }
    return {{"passed",true},{"trailSegments",0},{"projectileHeads",3},{"weapons",3},{"projectileColors",colors},{"projectileOutlines",outlines},{"captureOrderIndependent",true},{"fixtureDirectory",directory.string()}};
}
Json Encounter::coverTest(){
    auto& g=*impl;auto& a=g.actors[Impl::PlayerIndex];Json checks=Json::array();
    const auto require=[&](bool good,const char* message){if(!good)throw std::runtime_error(std::string("Cover regression: ")+message);checks.push_back(message);};
    const auto count=[&](const char* name){return g.counts[name];};
    const auto freeze=[&]{setAutopilot(false);a.velocity={};a.action.clear();a.actionEnd=a.reloadEnd=0;a.grounded=true;a.crouch=false;a.shield=0;a.shieldToken=0;a.shieldAlpha=0;a.fireAt=1e9;a.holdUntil=1e9;
        for(int i=Impl::EnemyFirst;i<Impl::ActorCount;++i){auto& other=g.actors[i];other.health=0;other.deathAt=other.respawnAt=1e9;other.pos={float(i*10),-1000,0};other.shield=0;g.collision.teleport(other.body,other.pos);g.collision.setHitBoxes(other.body,other.token,{},0,nullptr);}g.bullets.clear();g.pose();};
    freeze();
    const auto freeLocation=[&]{std::string reason;V3 p;for(int attempt=0;attempt<200;++attempt){
        if(attempt){V3 candidate;if(!g.collision.randomPoint(g.start,2,35,g.rng,candidate))continue;a.pos=candidate;g.collision.teleport(a.body,candidate);}
        for(int direction=0;direction<24;++direction){a.yaw=direction*Pi/12;a.pitch=0;g.pose();if(g.findCoverPlacement(p,reason)){a.anchor=a.pos;a.lastProgress=g.time;return;}}}throw std::runtime_error("No clear supported 4-meter cover footprint for regression");};
    freeLocation();V3 base;std::string reason;require(g.findCoverPlacement(base,reason),"supported footprint found in original scene");
    a.grounded=false;require(!g.placeCover(),"airborne placement rejected");a.grounded=true;
    a.crouch=true;require(!g.placeCover(),"crouched placement rejected");a.crouch=false;
    a.reloadEnd=g.time+1;require(!g.placeCover(),"reload placement rejected");a.reloadEnd=0;
    a.actionEnd=g.time+1;require(!g.placeCover(),"conflicting action placement rejected");a.actionEnd=0;
    require(!g.placeCover(Impl::EnemyFirst),"ordinary enemies cannot deploy cover");
    const auto size=g.coverSettings.size;const uint32_t fixture=0xfffffff0;
    Matrix obstruction=transform(base+rotate({size.x*.5f-.1f,size.y-.2f,0},a.yaw),{.3f,.3f,.3f});g.collision.setObstacle(fixture,&obstruction);
    require(!g.placeCover(),"occupied top corner rejects entire wall volume");g.collision.setObstacle(fixture,nullptr);
    auto& other=g.actors[Impl::EnemyFirst];V3 saved=other.pos;other.pos=base;other.health=100;
    require(!g.placeCover(),"nonblocking enemy capsule still rejects overlapping placement");other.pos=saved;other.health=0;
    obstruction=transform(a.pos+forward(a.yaw)*1.5f+V3{0,1.1f,0},{.4f,2,.4f},a.yaw);g.collision.setObstacle(fixture,&obstruction);
    require(!g.placeCover(),"deployment cannot cross an existing wall");g.collision.setObstacle(fixture,nullptr);
    float originalDistance=g.coverSettings.distance;g.coverSettings.distance=1000;
    require(!g.placeCover(),"unsupported remote footprint rejected");g.coverSettings.distance=originalDistance;
    require(g.coverReady==0,"failed placement consumes no cooldown");
    a.shield=200;a.shieldToken=g.token(13);a.raised=true;require(g.placeCover(),"holding shield permits instant placement");a.raised=false;a.shield=0;g.resetCover();
    const V3 origin=a.pos;CombatInput input;input.cover=true;advance(1.f/120,input);
    require(g.wall.token&&std::abs(flatLength(g.wall.base-origin)-3)<.001f,"H input places cover exactly three meters forward");
    require(g.wall.height==0,"initially buried cover has no exposed collision");require(!g.placeCover(),"one active wall rejects repeated requests");
    advance(1.f/120);require(g.wall.height>0&&g.wall.height<size.y,"cover rises with cubic ease-out");
    auto wallToken=g.wall.token;auto direction=forward(g.wall.yaw);V3 center=position(g.coverVolume());
    auto hit=g.collision.ray(center+direction*.5f,center-direction*.5f,a.token,true);
    require(hit.hit&&hit.token==wallToken,"emerged portion blocks scene traces");
    V3 above=g.wall.base+V3{0,g.wall.height+.15f,0};hit=g.collision.ray(above+direction*.5f,above-direction*.5f,a.token,true);
    require(!hit.hit||hit.token!=wallToken,"buried portion does not block space above the emerging wall");
    auto route=g.collision.path(origin,g.wall.base+direction*3);require(route.size()>2,"navigation detours around reserved footprint while wall rises");
    advance(.65f);require(std::abs(g.wall.height-size.y)<1e-5,"rise completes after UE 0.6-second duration");
    require(count("CoverPlaced")==1,"successful wall emits one placement event");
    require(g.index(wallToken)==82&&g.objects[wallToken].flags&CatalogObject::Rotation,"wall retains UE catalog identity and yaw");
    auto camera=g.camera(0);auto view=g.renderer.captureScene(g.draw(),camera,g.objects);
    require(wallToken<view.pixels.size()&&view.pixels[wallToken]>20,"same wall contributes actual BBox and depth/ID pixels");
    std::filesystem::create_directories(g.map.scene.outputDirectory(g.root,"captures")/"cover-regression");g.renderer.render(g.draw(),camera,{});g.renderer.screenshot(g.map.scene.outputDirectory(g.root,"captures")/"cover-regression/raised.png");
    bool grounded=true;for(int step=0;step<180;++step)g.collision.move(a.body,direction*2.5f+V3{0,-.1f,0},1.f/120,grounded);
    require(dot(g.collision.characterPosition(a.body)-origin,direction)<2.5f,"Jolt character capsule cannot walk through cover");g.collision.teleport(a.body,origin);
    const auto shoot=[&](int weapon){const auto& w=g.weapons[weapon];Impl::Bullet b;b.weapon=weapon;b.owner=a.token;b.index=w.projectileIndex;b.token=g.token(b.index);b.pos=b.previous=position(g.coverVolume())-forward(g.wall.yaw)*.7f;b.velocity=forward(g.wall.yaw)*w.speed;g.bullets.push_back(b);for(int i=0;i<18&&!g.bullets.empty();++i)advance(1.f/120);};
    auto shots=count("BulletHitScene"),explosions=count("Explosion"),blocks=count("ShieldBlock");shoot(0);require(std::abs(g.wall.durability-265)<.001f,"pistol impact subtracts UE 35 damage");
    shoot(2);require(std::abs(g.wall.durability-241)<.001f,"shotgun pellet subtracts UE 24 damage independently");
    shoot(1);require(std::abs(g.wall.durability-91)<.001f,"explosive rifle is intercepted as a direct 150-damage hit");shoot(1);
    require(!g.wall.token&&count("CoverBroken")==1,"zero durability breaks and removes cover exactly once");
    require(count("BulletHitScene")==shots+4&&count("ShieldBlock")==blocks&&count("Explosion")==explosions,"cover impacts are scene hits with no shield label or rifle explosion");
    require(g.events.size()>=2&&g.events[g.events.size()-2].type=="CoverBroken"&&g.events.back().type=="BulletHitScene","fatal impact preserves UE break-before-scene-hit event order");
    require(g.collision.debris().size()==4,"break creates four original UE fragment meshes with Jolt physics");
    for(const auto& d:g.draw())if(std::find(g.coverSettings.fragments.begin(),g.coverSettings.fragments.end(),d.mesh)!=g.coverSettings.fragments.end())require(d.instance.objectToken==0,"cosmetic fragments excluded from BBox capture");
    auto cleared=g.collision.path(origin,base+direction*3);require(cleared.size()==2,"breaking cover immediately releases navigation reservation");
    advance(4.1f);require(g.collision.debris().empty(),"cosmetic debris expires after four seconds");
    g.time=g.coverReady+.01;require(g.placeCover(),"cooldown permits another wall outside recording");double born=g.wall.born;
    g.time=born+20.01;g.updateCover();require(!g.wall.token&&count("CoverBroken")==1,"20-second natural expiry removes wall without break event");
    require(!g.placeCover(),"expiry does not bypass 30-second cooldown");g.time=born+30.01;require(g.placeCover(),"30-second cooldown releases placement");g.resetCover();
    setCoverProbability(1);setRecording(true);require(g.coverSelected&&!g.coverUsed,"new clip samples automatic probability once");require(g.placeCover(),"manual cover counts toward active clip quota");
    g.breakCover();g.coverReady=0;require(!g.placeCover(),"broken wall cannot restore the clip quota");g.resetCover();require(!g.placeCover(),"cleanup preserves the clip quota");
    require(g.restart(),"encounter restart completes");require(g.coverUsed,"encounter restart preserves the recording quota");freeze();freeLocation();
    setRecording(false);setCoverProbability(0);setRecording(true);require(!g.coverSelected&&!g.coverUsed,"next clip rerolls zero probability and resets quota");
    require(g.placeCover(),"manual H remains available in unselected automatic clip");g.resetCover();
    setRecording(false);setCoverProbability(1);setRecording(true);setAutopilot(true);other.health=100;other.pos=a.pos+forward(a.yaw)*10;other.holdUntil=1e9;other.fireAt=1e9;g.collision.teleport(other.body,other.pos);
    require(!g.automaticCover(Impl::EnemyFirst),"automatic cover waits three seconds after clip starts");g.time+=3.01;
    require(!g.automaticCover(-1)&&!g.automaticCover(Impl::PlayerIndex),"automatic cover requires a living hostile target");
    other.pos=a.pos+forward(a.yaw)*3;require(!g.automaticCover(Impl::EnemyFirst),"automatic placement rejects target closer than 4.5 meters");
    other.pos=a.pos+forward(a.yaw)*10+rotate({10,0,0},a.yaw);require(!g.automaticCover(Impl::EnemyFirst),"automatic placement rejects target outside wall coverage");
    other.pos=a.pos+forward(a.yaw)*10;require(g.automaticCover(Impl::EnemyFirst),"selected clip deploys cover through normal combat decision");
    setRecording(false);setRecording(true);require(g.coverSelected&&!g.coverUsed&&!g.placeCover(),"next clip keeps existing wall and cooldown");
    g.resetCover();setRecording(false);freeze();freeLocation();g.events.clear();setRecording(true);
    // Keep P2 on a real, stationary enemy next to the fixture instead of the
    // off-map enemy parking positions used for isolated collision assertions.
    V3 observer;bool observerFound=false;
    for(float side:{5.f,-5.f,7.f,-7.f}){V3 desired=a.pos+rotate({side,0,1.5f},a.yaw);if(g.collision.project(desired,observer,1,3)){observerFound=true;break;}}
    require(observerFound,"paired enemy observer has a supported position beside cover");
    other.health=100;other.pos=observer;other.velocity={};other.grounded=true;other.holdUntil=other.fireAt=1e9;other.action.clear();other.actionEnd=other.reloadEnd=0;other.localPose.clear();other.posedAt=-1;other.anchor=observer;other.lastProgress=g.time;
    V3 observerAim=a.pos+forward(a.yaw)*3-observer;other.yaw=std::atan2(observerAim.x,observerAim.z);other.pitch=0;g.collision.teleport(other.body,observer);g.nextSupply=g.nextShield=1e9;g.pose();
    Recording recording(g.map.scene.outputDirectory(g.root,"captures")/"cover-recording",g.root/L"assets",g.seed,g.map.scene.assets);
    for(int frame=0;frame<96;++frame){if(frame==0){CombatInput h;h.cover=true;advance(1.f/120,h);advance(4.f/120);}else advance(1.f/24);
        if(frame==48||frame==54){Impl::Bullet b;b.weapon=1;b.owner=a.token;b.index=g.weapons[1].projectileIndex;b.token=g.token(b.index);b.pos=b.previous=position(g.coverVolume())-forward(g.wall.yaw)*.5f;b.velocity=forward(g.wall.yaw)*g.weapons[1].speed;g.bullets.push_back(b);}
        recording.capture(g.renderer,*this);
        if(frame==20){auto p2=g.renderer.captureScene(g.draw(),g.camera(1),g.objects);require(g.wall.token<p2.pixels.size()&&p2.pixels[g.wall.token]>20,"paired enemy view records the same visible cover");}
        if(frame==4||frame==20||frame==64){g.renderer.render(g.draw(),g.camera(0),{});g.renderer.screenshot(g.map.scene.outputDirectory(g.root,"captures")/"cover-regression"/(std::to_wstring(frame)+L".png"));}}
    recording.finish();setRecording(false);
    return {{"passed",true},{"checks",checks},{"recordedFramesPerView",96},{"catalogIndex",g.coverSettings.index},{"diagnostics",diagnostics()}};
}
Json Encounter::eventTest(){
    auto& g=*impl;auto& a=g.actors[0];auto& enemy=g.actors[1];Json checks=Json::array();
    const auto require=[](bool value,const char* message){if(!value)throw std::runtime_error(std::string("Event regression: ")+message);};
    setAutopilot(false);g.nextSupply=g.nextShield=1e9;g.bullets.clear();g.events.clear();g.pickups.clear();g.traps.clear();
    a.action.clear();a.actionEnd=a.reloadEnd=0;a.shield=a.shieldAlpha=0;a.shieldToken=0;a.raised=false;a.velocity={};a.grounded=true;a.holdUntil=a.fireAt=1e9;
    for(int i=1;i<Impl::ActorCount;++i){auto& other=g.actors[i];other.health=0;other.deathAt=other.respawnAt=1e9;other.pos={float(i*10),-1000,0};other.shield=0;g.collision.teleport(other.body,other.pos);g.collision.setHitBoxes(other.body,other.token,{},0,nullptr);}
    bool found=false;V3 base;std::string reason;
    for(int attempt=0;attempt<150&&!found;++attempt){if(attempt){V3 p;if(!g.collision.randomPoint(g.start,2,35,g.rng,p))continue;a.pos=p;g.collision.teleport(a.body,p);}for(int direction=0;direction<24;++direction){a.yaw=direction*Pi/12;a.pitch=-.25f;g.pose();if(g.findCoverPlacement(base,reason)){found=true;break;}}}
    require(found,"no supported placement location");a.anchor=a.pos;a.lastProgress=g.time;
    auto folder=g.map.scene.outputDirectory(g.root,"captures")/"event-recording";setRecording(true);Recording placement(folder,g.root/L"assets",g.seed,g.map.scene.assets);
    uint64_t placedBefore=g.counts["TrapPlaced"];CombatInput plant;plant.trap=true;
    for(int frame=0;frame<72;++frame){for(int step=0;step<5;++step)advance(1.f/120,frame==0&&step==0?plant:CombatInput{});
        if(frame<23)require(g.counts["TrapPlaced"]==placedBefore,"trap was recorded before placement committed");placement.capture(g.renderer,*this);}
    placement.finish();require(g.counts["TrapPlaced"]==placedBefore+1&&g.traps.size()==1,"successful placement did not emit exactly once");
    checks.push_back("actual T input commits one TrapPlaced fact after one second and records it with visible placement gesture");
    auto trap=g.traps.front();g.spawn(1,trap.pos,true);enemy.holdUntil=enemy.fireAt=1e9;enemy.nextDash=enemy.nextJump=1e9;enemy.health=100;g.events.clear();
    Recording trigger(folder,g.root/L"assets",g.seed,g.map.scene.assets);advance(5.f/120);auto triggered=std::find_if(g.events.begin(),g.events.end(),[](const auto& e){return e.type=="TrapTriggered";});
    require(triggered!=g.events.end()&&triggered->actor==a.token&&triggered->tokens==std::vector<uint32_t>{trap.token,enemy.token},"trap trigger role is not its placer with trap/victim objects");
    trigger.capture(g.renderer,*this);for(int frame=0;frame<23;++frame){advance(5.f/120);trigger.capture(g.renderer,*this);}trigger.finish();
    require(g.counts["TrapTriggered"]==1,"trap triggered more than once");checks.push_back("one actual trap detonation belongs to the placing player, preserving trap/victim order");
    enemy.health=100;enemy.action.clear();enemy.actionEnd=enemy.reloadEnd=0;enemy.shield=200;enemy.shieldToken=g.token(13);enemy.raised=false;enemy.shieldAlpha=1;enemy.shieldRecovery=0;enemy.yaw=a.yaw;enemy.grounded=true;g.pose();g.shield(1,true);
    enemy.shieldUntil=g.time-1;enemy.shieldCheck=0;auto lowers=g.counts["ShieldLower"],raises=g.counts["ShieldRaise"];g.events.clear();
    Impl::Bullet incoming;incoming.owner=a.token;incoming.weapon=0;incoming.index=g.weapons[0].projectileIndex;incoming.token=g.token(incoming.index);incoming.pos=incoming.previous=enemy.pos+V3{0,1,0}+forward(enemy.yaw)*3;incoming.velocity=forward(enemy.yaw)*(-8);g.bullets={incoming};advance(1.f/120);
    require(enemy.raised&&enemy.shieldUntil>g.time&&g.counts["ShieldLower"]==lowers&&g.counts["ShieldRaise"]==raises,"persistent threat creates an artificial lower/raise pair");
    g.bullets.clear();enemy.shieldUntil=g.time-1;enemy.shieldCheck=0;advance(1.f/120);
    require(!enemy.raised&&g.counts["ShieldLower"]==lowers+1,"guard did not lower once after threat ended");g.bullets={incoming};enemy.shieldCheck=0;advance(1.f/120);require(!enemy.raised,"AI raised again during lower recovery");g.bullets.clear();
    checks.push_back("expired AI guard stays up while threatened, lowers once when safe, and respects recovery");
    // Stop between two 24 Hz snapshots while a real jump fact is pending.
    enemy.health=0;enemy.respawnAt=1e9;enemy.shield=0;enemy.shieldToken=0;a.action.clear();a.actionEnd=a.reloadEnd=0;a.shield=0;a.shieldToken=0;a.raised=false;a.grounded=true;g.events.clear();g.pose();
    Recording tail(folder,g.root/L"assets",g.seed,g.map.scene.assets);advance(5.f/120);tail.capture(g.renderer,*this);CombatInput jump;jump.jump=true;advance(1.f/120,jump);require(hasEvents(),"no pending final jump fact");tail.captureFinalEvents(g.renderer,*this);require(tail.frames()==2&&!hasEvents(),"final event snapshot was discarded");tail.finish();
    checks.push_back("stopping between sample ticks captures the pending jump in a real terminal frame");
    auto view=g.renderer.captureScene(g.draw(),g.camera(0),g.objects);CombatEvent invisible;invisible.type="ShieldRaise";invisible.tokens={a.token,g.token(13)};invisible.visibilityTokens={invisible.tokens[1]};
    require(!eventVisible(invisible,g.camera(0),view,a.token),"visible body incorrectly makes an invisible shield event visible");checks.push_back("equipment events require the relevant equipment to be visible");
    setRecording(false);return {{"passed",true},{"checks",checks},{"placementFacts",g.counts["TrapPlaced"]-placedBefore},{"triggerFacts",g.counts["TrapTriggered"]},{"recordingDirectory",folder.string()}};
}
Json Encounter::demoRulesTest(){
    auto require=[](bool good,const char* message){if(!good)throw std::runtime_error(message);};
    auto& g=*impl;auto& player=g.actors[Impl::PlayerIndex];Json checks=Json::array();
    player.health=100;const auto source=g.actors[1].token;
    g.damage(0,35,1,source,player.pos);require(player.health==99,"Player body hit must cost one point");
    g.damage(0,200,1,source,player.pos,true);require(player.health==98,"Player explosion must cost one point, not NPC damage");
    for(int i=0;i<98;++i)g.damage(0,24,1,source,player.pos);
    require(!playerAlive()&&player.health==0&&g.counts["Died"]==1&&g.counts["EnemyDied"]==0,"Player death event/health mismatch");
    const auto step=g.tick;const double time=g.time;advance(1.f);require(g.tick==step&&g.time==time,"Death did not freeze simulation");
    require(!std::isfinite(player.respawnAt),"Player must not respawn automatically");
    checks.push_back("one HP per bullet/blast; zero clamps health, emits one player death and freezes physics");
    HudState hud;hud.combat=true;hud.paused=true;hud.dead=true;this->hud(hud,0);g.renderer.render(g.draw(),g.camera(0),hud);
    require(g.renderer.screenshot(g.root/"captures/demo-death-menu.png"),"Cannot capture death menu");
    require(CombatUi::hitMenu(1280,720,CombatUi::menuItem(1280,720,1).x+2,CombatUi::menuItem(1280,720,1).y+2,true)==1,"Death exit hit test failed");
    require(retry()&&playerAlive()&&player.health==100&&player.owned==std::array<bool,3>{true,true,true},"Retry failed to restore player");
    for(int w=0;w<3;++w)require(player.ammo[w]==g.weapons[w].magazine,"Retry did not refill ammo");
    checks.push_back("Retry restores HP/ammo/equipment and clears the previous encounter");
    int found=0;for(int orientation=0;orientation<8;++orientation){player.yaw=orientation*Pi/4;
        for(int i=Impl::EnemyFirst;i<Impl::ActorCount;++i){const auto before=g.actors[i].pos;if(!g.frontGoal(i))continue;++found;
            const auto& enemy=g.actors[i];require(dot((enemy.goal-player.pos).normalized(),forward(player.yaw))>=.35f,"Enemy goal is behind player");
            require(enemy.path.size()>1&&(enemy.pos-before).length()==0,"Front goal bypassed navigation or teleported enemy");
            require(!g.collision.ray(enemy.goal+V3{0,1.4f,0},player.pos+V3{0,1,0},0,true).hit,"Front goal has no firing line");}}
    require(found>0,"No valid frontal NPC goal in any tested orientation");checks.push_back("NPC front goals are navigable, visible and do not teleport actors");
#ifndef _WIN32
    g.pose();const auto before=g.renderer.captureScene(g.draw(),g.camera(0),g.objects,false).color;
    const auto fast=g.renderer.captureDemoScene(g.draw(),g.camera(0));require(before==fast,"Demo RGB differs from reference-server RGB");
    hud={};hud.combat=true;hud.showHelp=false;this->hud(hud,0);const auto ui=g.renderer.captureDemoUi(hud);bool transparent=false,visible=false;
    for(size_t i=0;i<ui.size();i+=4){transparent|=ui[i+3]==0;visible|=ui[i+3]>0;require(ui[i]<=ui[i+3]+1&&ui[i+1]<=ui[i+3]+1&&ui[i+2]<=ui[i+3]+1,"UI is not premultiplied RGBA");}
    require(transparent&&visible,"UI capture is empty or opaque");
    require(before==g.renderer.captureDemoScene(g.draw(),g.camera(0)),"UI leaked into clean RGB capture");
    checks.push_back("RGB matches server capture; HUD is separate premultiplied alpha and cannot contaminate RGB");
#endif
    return {{"passed",true},{"checks",checks},{"front_goals",found}};
}
Json Encounter::loadoutTest(){
    auto require=[](bool good,const char* message){if(!good)throw std::runtime_error(message);};
    auto& g=*impl;auto& player=g.actors[0];Json checks=Json::array();
    require(g.actors.size()==3&&Impl::EnemyCount==2,"Encounter must contain one player and two enemies");
    require(player.owned==std::array<bool,3>{true,true,true},"Player must start with all weapons");
    for(int w=0;w<3;++w)require(player.ammo[w]==g.weapons[w].magazine,"Initial magazine not full");
    g.pickups.clear();g.nextSupply=g.nextShield=1e9;
    for(int i=1;i<Impl::ActorCount;++i){
        auto& enemy=g.actors[i];
        require(enemy.weapon==2&&enemy.owned==std::array<bool,3>{false,false,true},"Enemy must start with only a shotgun");
        require(enemy.ammo[2]==g.weapons[2].magazine,"Enemy shotgun magazine not full");
        enemy.holdUntil=enemy.fireAt=1e9;
        for(int w:{0,1}){
            g.pickups={{g.token(g.weapons[w].index),g.token(12),enemy.pos,w,-1}};
            g.maintenance();
            require(g.pickups.front().active&&enemy.weapon==2&&enemy.owned==std::array<bool,3>{false,false,true},"Enemy collected a forbidden weapon");
            enemy.owned[w]=true;g.equip(i,w);
            require(enemy.weapon==2,"Enemy equipped a forbidden weapon");enemy.owned[w]=false;
        }
    }
    checks.push_back("one player and two shotgun enemies; enemies cannot collect or equip pistol/rifle");
    for(int i=0;i<Impl::ActorCount;++i){
        auto& actor=g.actors[i];const auto saved=actor;
        actor.weapon=2;actor.weaponToken=g.token(g.weapons[2].index);actor.fireAt=g.time;
        g.pose();
        const auto projectiles=g.bullets.size();const auto cartridges=g.counts["WeaponFire"];
        const int ammo=actor.ammo[2],pellets=i==Impl::PlayerIndex?9:2;
        const V3 barrel=(point(actor.gun,{0,0,1})-position(actor.gun)).normalized();
        g.fire(i,actor.muzzle+barrel*20);
        require(g.bullets.size()==projectiles+pellets,"Shotgun pellet count differs for player or enemy");
        require(actor.ammo[2]==ammo-1&&g.counts["WeaponFire"]==cartridges+1,"Shotgun must consume one cartridge and emit one fire event");
        require(actor.fireAt>g.time,"Shotgun firing lost its cooldown");
        g.bullets.resize(projectiles);actor=saved;
    }g.pose();
    checks.push_back("each enemy shotgun fires two pellets; player shotgun keeps nine; one cartridge per shot");
    for(auto& actor:g.actors){actor.shield=0;actor.shieldToken=0;actor.shieldAlpha=0;actor.raised=false;}
    g.pickups={{g.token(13),g.token(12),g.actors[1].pos,3,-1}};g.maintenance();
    require(g.actors[1].shield==200&&!g.pickups.front().active,"Enemy shield pickup was disabled");
    checks.push_back("enemy shield pickup remains enabled");
    g.pickups.clear();
    setAutopilot(false);
    for(int w:{1,2,0}){CombatInput input;input.weapon=w;advance(1.f/120,input);require(player.weapon==w,"Direct weapon switch failed");}
    checks.push_back("1/2/3 switch without pickups");
    --player.ammo[0];CombatInput reload;reload.reload=true;advance(1.f/120,reload);
    require(player.reloadEnd>g.time,"Test reload did not start");
    CombatInput select;select.weapon=2;advance(1.f/120,select);
    require(player.weapon==0&&player.requestedWeapon==2,"Switch interrupted reload or request was lost");
    for(int i=0;i<1200&&player.weapon!=2;++i)advance(1.f/120);
    require(player.weapon==2&&player.requestedWeapon==-1,"Queued switch did not finish after reload");
    require(player.ammo[0]==g.weapons[0].magazine,"Reload did not refill the original weapon");
    checks.push_back("reload completes before queued weapon switch");
    require(recover(),"Loadout recovery failed");
    require(player.owned==std::array<bool,3>{true,true,true},"Recovery lost unlocked weapons");
    for(int i=1;i<Impl::ActorCount;++i){
        auto& enemy=g.actors[i];g.spawn(i,enemy.pos);
        require(enemy.weapon==2&&enemy.owned==std::array<bool,3>{false,false,true},"Respawn must retain the shotgun-only enemy loadout");
    }g.maintenance();g.pose();
    require(g.actors.size()==3,"Recovery or respawn changed the actor count");
    checks.push_back("recovery retains player weapons; enemy respawn retains shotgun-only loadout and three slots");
    int oldWidth=g.renderer.width,oldHeight=g.renderer.height;
    auto folder=g.root/"captures/loadout-tests";
    auto cleanBefore=g.renderer.captureScene(g.draw(),g.camera(0),g.objects,false);
    for(auto size:{std::pair{1280,720},std::pair{864,480},std::pair{360,240}}){
        g.renderer.resize(size.first,size.second);HudState state;hud(state,0);state.showFps=true;
        auto name=std::to_string(size.first)+"x"+std::to_string(size.second);
        g.renderer.render(g.draw(),g.camera(0),state);
        require(g.renderer.screenshot(folder/(name+"-hud.png")),"HUD screenshot failed");
        state.paused=true;state.pauseSelection=2;g.renderer.render(g.draw(),g.camera(0),state);
        require(g.renderer.screenshot(folder/(name+"-menu.png")),"Menu screenshot failed");
    }
    auto cleanAfter=g.renderer.captureScene(g.draw(),g.camera(0),g.objects,false);
    require(cleanBefore.color==cleanAfter.color,"HUD or menu leaked into clean capture");
    g.renderer.resize(oldWidth,oldHeight);
    checks.push_back("HUD/menu renders at three sizes; clean capture is unchanged");
    return {{"passed",true},{"checks",checks}};
}
Json Encounter::selfTest(){
    auto require=[](bool b,const char* message){if(!b)throw std::runtime_error(message);};
    Json checks=Json::array();for(int w=0;w<3;++w){const auto& gun=impl->weapons[w];for(float d:{10.f,30.f,60.f,100.f}){V3 target{0,0,d},v=launch({},target,gun.speed,gun.gravity);float t=d/v.z;V3 impact=v*t+V3{0,-.5f*gun.gravity*t*t,0};require((impact-target).length()<.005f,"Ballistic solver does not reach stationary target");}checks.push_back(gun.name+" ballistic arcs");}
    Json trajectories=Json::array();float maxVelocityError=0;
    for(const auto& probe:impl->ballisticProbes){const auto it=std::find_if(impl->weapons.begin(),impl->weapons.end(),[&](const auto& w){return w.name==probe.at("weapon").get<std::string>();});require(it!=impl->weapons.end(),"Unknown UE projectile probe");
        V3 from=native(probe.at("start")),target=native(probe.at("target")),expected=native(probe.at("velocity")),actual=launch(from,target,it->speed,it->gravity,it->compensate);
        float error=(actual-expected).length();maxVelocityError=std::max(maxVelocityError,error);require(error<.0001f,"Launch velocity differs from independently exported UE solver");
        float flight=flatLength(target-from)/flatLength(actual);V3 impact=from+actual*flight+V3{0,-.5f*it->gravity*flight*flight,0};require((impact-target).length()<.0001f,"UE trajectory time of flight mismatch");
        trajectories.push_back({{"weapon",it->name},{"horizontalMeters",flatLength(target-from)},{"heightMeters",target.y-from.y},{"launchSpeedMetersPerSecond",actual.length()},{"flightSeconds",flight},{"velocityErrorMetersPerSecond",error},{"customArc",probe.at("customArc")}});}
    require((launch({},{0,10,0},20,9.8f)-V3{0,20,0}).length()<.00001f,"Vertical shot changed speed");
    require((launch({},{0,0,10},20,0)-V3{0,0,20}).length()<.00001f,"Zero-gravity shot is invalid");
    require((launch({},{0,0,10},20,9.8f,false)-V3{0,0,20}).length()<.00001f,"Disabled compensation still arcs upward");
    checks.push_back("18 independently exported UE launch vectors and flight times, vertical and zero-gravity edge cases");
    std::array<double,Impl::ActorCount> savedFire;for(int i=0;i<Impl::ActorCount;++i){savedFire[i]=impl->actors[i].fireAt;impl->actors[i].fireAt=1e9;}
    float maxFlightError=0;
    for(int weapon=0;weapon<3;++weapon){const auto& w=impl->weapons[weapon];V3 origin{0,impl->map.bounds.max.y+100,0},velocity{0,0,w.speed};
        Impl::Bullet shot;shot.weapon=weapon;shot.token=impl->token(w.projectileIndex);shot.pos=shot.previous=origin;shot.velocity=velocity;impl->bullets={shot};
        for(int step=0;step<240;++step)impl->simulate(1.f/120,{});
        auto it=std::find_if(impl->bullets.begin(),impl->bullets.end(),[&](const auto& b){return b.token==shot.token;});require(it!=impl->bullets.end(),"Unobstructed projectile expired early");const auto& actual=*it;
        float error=(actual.pos-(origin+velocity*2+V3{0,-2*w.gravity,0})).length();maxFlightError=std::max(error,maxFlightError);require(error<.003f,"Actual projectile integration changed UE flight speed");
        for(float alpha:{0.f,.25f,.5f,.75f,1.f}){const auto& draws=impl->draw(alpha);auto rendered=std::find_if(draws.begin(),draws.end(),[&](const auto& d){return d.instance.objectToken==shot.token;});
            require(rendered!=draws.end()&&(position(rendered->instance.world)-(actual.previous*(1-alpha)+actual.pos*alpha)).length()<.00001f,"Projectile display interpolation changed the trajectory");
            require((impl->bullets.front().pos-(origin+velocity*2+V3{0,-2*w.gravity,0})).length()<=maxFlightError+.00001f,"Rendering advanced projectile physics");}
        for(const auto& d:impl->draw())require(d.instance.surface>=-1.5f,"Projectile emitted trail geometry");
        float range=impl->weapons[weapon].range;impl->weapons[weapon].range=.01f;impl->simulate(1.f/120,{});impl->weapons[weapon].range=range;
        require(std::none_of(impl->bullets.begin(),impl->bullets.end(),[&](const auto& b){return b.token==shot.token;}),"Range-exhausted projectile remains active");}
    impl->bullets.clear();for(int i=0;i<Impl::ActorCount;++i)impl->actors[i].fireAt=savedFire[i];
    checks.push_back("720 actual projectile integration steps, render interpolation, no trails and range expiration");
    auto& a=impl->actors[0];a.ammo[0]=0;impl->reload(0);require(a.reloadEnd>impl->time,"Empty magazine did not start reload");double end=a.reloadEnd;bool autoOld=impl->automatic;impl->automatic=false;while(impl->time<end+.02)impl->simulate(1.f/120,{});require(a.ammo[0]==impl->weapons[0].magazine,"Reload did not refill magazine");checks.push_back("magazine and reload");
    // Isolate the shield probe from AI actors and map geometry after the live simulation.
    const auto beforeShieldProbe=a;
    a.pos={0,impl->map.bounds.max.y+100,0};a.velocity={};a.yaw=a.pitch=0;
    a.action.clear();a.actionEnd=a.reloadEnd=0;a.grounded=true;a.airAlpha=a.crouchAlpha=a.shieldHit=0;
    a.localPose.clear();a.posedAt=-1;impl->collision.teleport(a.body,a.pos);
    a.shield=200;a.shieldToken=impl->token(13);impl->shield(0,true);auto before=impl->bullets.size();impl->fire(0,a.pos+V3{0,1,10});require(impl->bullets.size()==before,"Shield did not block firing");impl->shield(0,false);require(a.shieldRecovery>impl->time,"Shield recovery missing");checks.push_back("shield and fire recovery");
    impl->bullets.clear();for(auto& actor:impl->actors)actor.fireAt=impl->time+10;
    a.shieldAlpha=1;impl->shield(0,true);impl->pose();
    V3 shieldCenter=position(a.shieldMatrix),normal=forward(a.yaw);auto shieldHit=impl->collision.ray(shieldCenter+normal*.3f,shieldCenter-normal*.3f,impl->actors[2].token);
    if(!shieldHit.hit||shieldHit.token!=a.shieldToken)throw std::runtime_error("Shield surface does not intercept a bullet ray: hit="+std::to_string(shieldHit.token)+", expected="+std::to_string(a.shieldToken)+", raised="+std::to_string(a.raised)+", action="+a.action+", health="+std::to_string(a.health));
    auto ownShield=impl->collision.sweepBullet(shieldCenter+normal*.3f,shieldCenter-normal*.3f,impl->weapons[0].radius,a.token);
    require(!ownShield.hit||ownShield.token!=a.shieldToken,"Projectile hit its owner's attached shield");checks.push_back("owner and attached shield ignored during projectile sweep");
    Impl::Bullet rifle;rifle.weapon=1;rifle.index=impl->weapons[1].projectileIndex;rifle.token=impl->token(rifle.index);rifle.owner=impl->actors[2].token;rifle.pos=shieldCenter+normal*.12f;rifle.velocity=normal*(-25);impl->bullets.push_back(rifle);
    auto incoming=impl->collision.sweepBullet(rifle.pos,rifle.pos+rifle.velocity*(1.f/120),impl->weapons[1].radius,rifle.owner);
    require(incoming.hit&&incoming.token==a.shieldToken,"A body part protrudes ahead of the raised shield grip");
    auto explosions=impl->counts["Explosion"];CombatInput guard;guard.shield=true;impl->simulate(1.f/120,guard);
    require(a.shield<200,"Raised shield did not intercept the incoming rifle bullet");
    require(impl->counts["Explosion"]==explosions,"A blocked rifle bullet exploded through the shield");checks.push_back("actual shield collision absorbs rifle explosion");
    a=beforeShieldProbe;impl->collision.teleport(a.body,a.pos);impl->pose();
    {
        auto saved=a;double savedTime=impl->time;
        a.shield=0;a.shieldToken=0;a.shieldAlpha=0;a.raised=false;a.shieldRecovery=0;a.reloadEnd=0;a.actionEnd=0;a.action.clear();a.owned={true,true,true};a.velocity={};a.yaw=a.pitch=0;a.airAlpha=a.crouchAlpha=0;a.weaponReadyAlpha=1;
        for(bool automatic:{false,true}){impl->automatic=automatic;
            for(int w=0;w<3;++w){a.weapon=w;a.weaponToken=impl->token(impl->weapons[w].index);const auto& weapon=impl->weapons[w];
                for(int n=0;n<32;++n){impl->bullets.clear();a.ammo[w]=weapon.magazine;a.fireAt=impl->time;a.lastShot=a.previousShot=-100;a.shotWeapon=-1;a.localPose.clear();a.posedAt=-1;impl->pose();
                    impl->fire(0,a.pos+V3{0,1,20});double next=a.fireAt,delay=next-impl->time;size_t emitted=impl->bullets.size();
                    if(emitted!=size_t(weapon.pellets)){V3 barrel=(point(a.gun,{0,0,1})-position(a.gun)).normalized(),direction=(a.pos+V3{0,1,20}-a.muzzle).normalized();
                        std::ofstream(impl->map.scene.outputDirectory(impl->root,"captures")/"cadence-ready-failed.json")<<Json{{"weapon",w},{"automatic",automatic},{"cycle",n},{"barrel",{barrel.x,barrel.y,barrel.z}},{"aim",{direction.x,direction.y,direction.z}},{"angle",std::acos(std::clamp(dot(barrel,direction),-1.f,1.f))*180/Pi},{"action",a.action},{"health",a.health},{"reload",a.reloadEnd},{"time",impl->time},{"ready",a.weaponReadyAlpha}}.dump(2);}
                    require(emitted==size_t(weapon.pellets),"Wrong projectile count per cartridge");
                    if(automatic&&w!=1)require(delay>=impl->automaticFireMin&&delay<=impl->automaticFireMax,"Automatic cadence differs from UE range");
                    else require(std::abs(delay-weapon.cadence)<1e-6,"Manual or explosive cadence differs from configured interval");
                    impl->time=next-.0001;impl->fire(0,a.pos+V3{0,1,20});require(impl->bullets.size()==emitted,"Weapon fired before its cooldown");
                    impl->time=next;impl->fire(0,a.pos+V3{0,1,20});require(impl->bullets.size()==emitted*2,"Weapon cannot fire when its cooldown completes");
                }
            }
        }
        checks.push_back("manual and automatic fire intervals and exact projectile counts (192 cooldown cycles)");
        a.weapon=1;a.weaponToken=impl->token(impl->weapons[1].index);a.ammo[1]=30;a.fireAt=impl->time;a.lastShot=a.previousShot=-100;a.shotWeapon=-1;a.localPose.clear();a.posedAt=-1;impl->pose();impl->bullets.clear();impl->fire(0,a.pos+V3{0,1,20});
        require(impl->weapons[1].explosive&&impl->bullets.size()==1,"Explosive weapon must launch one projectile");double cooldown=a.fireAt;
        double shotTime=impl->time;impl->equip(0,0);require(a.weapon==1,"Automatic switch interrupted its shot pose");impl->time+=impl->animations.at("MM_Rifle_Fire").duration+.01;
        impl->equip(0,0);impl->equip(0,1);require(a.fireAt>=cooldown&&cooldown-shotTime==2,"Switching weapons bypassed two-second explosive cooldown");
        a.fireAt=impl->time;impl->equip(0,0);double pause=a.fireAt-impl->time;require(pause>=impl->postSwitchMin&&pause<=impl->postSwitchMax,"Automatic switch delay differs from UE");
        checks.push_back("single explosive projectile, two-second cooldown and switch preservation");
        size_t effects=impl->effects.size(),projectiles=impl->bullets.size();auto beforeExplosion=impl->counts["Explosion"];
        impl->explode(a.pos+V3{0,10,0},0,impl->bullets[0].token);
        require(impl->counts["Explosion"]==beforeExplosion+1&&impl->effects.size()==effects&&impl->bullets.size()==projectiles,"Explosion spawned artificial bullet-ring geometry");
        checks.push_back("explosion event without cosmetic projectile rings");
        a=saved;impl->time=savedTime;impl->automatic=false;impl->bullets.clear();
    }
    a.shield=0;a.shieldToken=0;a.shieldAlpha=0;a.raised=false;a.shieldRecovery=0;a.weapon=2;a.weaponToken=impl->token(impl->weapons[2].index);a.owned[2]=true;a.ammo[2]=1;a.fireAt=impl->time;
    impl->bullets.clear();auto cartridges=impl->counts["WeaponFire"];impl->fire(0,a.pos+V3{0,1,20});require(impl->bullets.size()==9&&impl->counts["WeaponFire"]==cartridges+1&&a.ammo[2]==0,"Shotgun must emit nine pellets but one cartridge event");checks.push_back("shotgun pellet and cartridge accounting");
    a.ammo[2]=8;a.fireAt=impl->time;a.action.clear();a.actionEnd=0;a.reloadEnd=0;impl->bullets.clear();impl->events.clear();impl->automatic=false;
    CombatInput movingShot;movingShot.forward=1;movingShot.lookX=125;movingShot.fire=true;impl->simulate(1.f/120,movingShot);
    auto shotEvent=std::find_if(impl->events.begin(),impl->events.end(),[&](const auto& e){return e.type=="WeaponFire"&&e.actor==a.token;});require(shotEvent!=impl->events.end(),"Moving shot did not fire");
    V3 eye=a.pos+V3{0,1.65f-.5f*a.crouchAlpha,0},muzzle=a.muzzle;auto obstruction=impl->collision.sweepBullet(eye,muzzle,impl->weapons[2].radius,a.token);
    if(obstruction.hit)muzzle=eye+(muzzle-eye)*std::max(0.f,obstruction.fraction)+obstruction.normal*.001f;
    require((shotEvent->position-muzzle).length()<.0001f,"Shot used the previous frame's hand/muzzle pose");checks.push_back("moving and turning shot uses current-frame posed muzzle and cover sweep");
    impl->bullets.clear();auto& victim=impl->actors[2];impl->traps.push_back({impl->token(1),a.token,victim.pos,impl->time-1,true});auto triggered=impl->counts["TrapTriggered"];
    impl->simulate(1.f/120,{});require(impl->counts["TrapTriggered"]==triggered+1,"Armed trap did not trigger on enemy contact");impl->simulate(1.f/120,{});require(impl->counts["TrapTriggered"]==triggered+1,"Trap detonated more than once");checks.push_back("trap contact and single detonation");
    float hp=a.health;impl->damage(0,300,2,impl->actors[2].token,a.pos);require(a.health==hp-1,"Player hit must cost one health point");checks.push_back("one health point per player hit with hit events");
    for(int i=Impl::EnemyFirst;i<Impl::ActorCount;++i)require(!impl->collision.path(impl->actors[Impl::PlayerIndex].pos,impl->actors[i].pos).empty(),"Player and enemy spawned on disconnected navigation");checks.push_back("player and two enemies use connected navigation");
    impl->actors[Impl::PlayerIndex].target=-1;impl->actors[Impl::PlayerIndex].thinkAt=0;for(int i=Impl::EnemyFirst;i<Impl::ActorCount;++i){impl->actors[i].target=i==Impl::EnemyFirst?Impl::EnemyFirst+1:Impl::EnemyFirst;impl->actors[i].thinkAt=0;}impl->think(Impl::PlayerIndex);for(int i=Impl::EnemyFirst;i<Impl::ActorCount;++i){impl->think(i);require(impl->actors[i].target<0||impl->hostile(i,impl->actors[i].target),"AI selected a friendly enemy as a target");}
    checks.push_back("enemy AI targets only the player");
    {
        auto& g=*impl;const auto savedActors=g.actors;const double savedSupply=g.nextSupply,savedShield=g.nextShield;
        const V3 origin{0,g.map.bounds.max.y+100,0};g.nextSupply=g.nextShield=1e9;g.bullets.clear();g.traps.clear();
        const auto fixture=[&](float health){
            for(int i=0;i<Impl::ActorCount;++i){auto& actor=g.actors[i];
                actor.pos=origin+V3{i==0?-10.f:i==1?0.f:2.f,0,0};actor.health=i==1?health:100;
                actor.velocity={};actor.yaw=actor.pitch=0;actor.shield=actor.shieldAlpha=0;actor.shieldToken=0;actor.raised=false;
                actor.action.clear();actor.actionEnd=actor.reloadEnd=0;actor.path.clear();actor.target=-1;
                actor.thinkAt=actor.holdUntil=actor.fireAt=1e9;actor.lastProgress=g.time;
                actor.crouch=false;actor.airAlpha=actor.crouchAlpha=0;actor.airborneTime=0;actor.grounded=true;
                actor.localPose.clear();actor.posedAt=-1;g.collision.teleport(actor.body,actor.pos);
            }g.pose();
        };
        for(float health:{250.f,100.f}){
            fixture(health);auto& direct=g.actors[1];auto& nearby=g.actors[2];nearby.health=250;
            V3 center=position(direct.boxes.at(1));
            auto hit=g.collision.ray(center+V3{0,0,2},center,g.actors[0].token);
            require(hit.hit&&hit.token==direct.token,"Direct-hit fixture missed the actor hitbox");
            Impl::Bullet bullet;bullet.weapon=1;bullet.owner=g.actors[0].token;bullet.index=g.weapons[1].projectileIndex;bullet.token=g.token(bullet.index);
            bullet.pos=bullet.previous=hit.point+hit.normal*.025f;bullet.velocity=hit.normal*(-g.weapons[1].speed);g.bullets={bullet};
            auto beforeDamage=g.counts["ExplosionDamage"],beforeDeath=g.counts["EnemyDied"];
            g.simulate(1.f/120,{});
            require(g.bullets.empty(),"Direct-hit projectile was not consumed");
            require(std::abs(direct.health-std::max(0.f,health-150))<.001f,"Direct-hit rifle damage is not exactly 150");
            require(g.counts["EnemyDied"]==beforeDeath+(health==100?1:0),"Direct hit did not kill exactly the 100 HP target");
            auto blast=std::find_if(g.events.rbegin(),g.events.rend(),[&](const auto& e){return e.type=="Explosion"&&std::find(e.tokens.begin(),e.tokens.end(),bullet.token)!=e.tokens.end();});
            require(blast!=g.events.rend(),"Direct-hit explosion event missing");
            float distance=(nearby.pos+V3{0,.9f,0}-blast->position).length();float expected=150*std::max(.1f,1-distance/8.f);
            require(expected>0&&expected<150&&std::abs(nearby.health-(250-expected))<.001f,"Nearby blast damage lost distance falloff");
            require(g.counts["ExplosionDamage"]==beforeDamage+2,"Explosion damage was applied more than once per victim");
            require(g.actors[0].health==100,"Explosion damaged shooter");
        }
        checks.push_back("actual rifle collision: 100 HP kill, exactly 150 damage at 250 HP, nearby falloff without double damage");
        fixture(250);uint32_t source=g.token(g.weapons[1].projectileIndex);
        g.explode(origin+V3{0,.9f,1},0,source,0,g.weapons[1].damage,g.weapons[1].blastRadius);
        require(std::abs(g.actors[1].health-(250-150*(1-1.f/8)))<.001f,"Indirect explosion incorrectly received full direct-hit damage");
        fixture(100);Matrix wall=transform(origin+V3{1,1,0},{.3f,4,4});const uint32_t wallToken=0xfffffff0;
        g.collision.setObstacle(wallToken,&wall);g.explode(origin+V3{0,.9f,0},0,source,0,g.weapons[1].damage,g.weapons[1].blastRadius);g.collision.setObstacle(wallToken,nullptr);
        require(g.actors[2].health==100,"Explosion splash passed through static cover");
        checks.push_back("indirect explosion keeps distance falloff and static cover blocks splash");
        for(float distance:{7.f,8.f,8.1f}){
            fixture(100);g.actors[1].pos=origin+V3{distance,0,0};
            g.explode(origin+V3{0,.9f,0},0,source,0,g.weapons[1].damage,g.weapons[1].blastRadius);
            float expected=distance<=8.f?150*std::max(.1f,1-distance/8.f):0;
            require(std::abs(g.actors[1].health-(100-expected))<.001f,"Rifle splash does not respect the enlarged eight-meter radius");
        }
        fixture(100);g.actors[1].pos=origin+V3{5,0,0};g.explode(origin+V3{0,.9f,0},0,source);
        require(g.actors[1].health==100,"Rifle radius tuning changed the trap blast radius");
        fixture(100);g.actors[1].pos=origin+V3{2,0,0};g.explode(origin+V3{0,.9f,0},0,source);
        require(std::abs(g.actors[1].health-100*2.f/4.5f)<.001f,"Rifle damage tuning changed trap splash damage");
        checks.push_back("rifle splash reaches eight meters, stops beyond eight meters, and leaves trap radius and damage unchanged");
        g.actors=savedActors;g.nextSupply=savedSupply;g.nextShield=savedShield;
        for(auto& actor:g.actors)g.collision.teleport(actor.body,actor.pos);g.pose();
    }
    impl->automatic=autoOld;require(recover(),"Encounter reset failed");require(!draws().empty(),"Recovery did not immediately restore posed draw geometry");checks.push_back("player recovery with immediate rendering");return {{"passed",true},{"checks",checks},{"ballisticReferences",trajectories},{"maximumUEVelocityErrorMetersPerSecond",maxVelocityError},{"maximumIntegratedFlightErrorMeters",maxFlightError},{"diagnostics",diagnostics()}};
}

Json Encounter::poseTest(){
    uint32_t maxProjectilePixels=0;int projectileChecks=0;
    for(int w=0;w<3;++w){Impl::Bullet bullet;bullet.weapon=w;bullet.token=impl->token(impl->weapons[w].projectileIndex);impl->bullets={bullet};
        for(float distance:{8.f,4.f,2.f,1.f,.5f,.3f,.2f,.1f,.08f,.04f,0.f,-.1f}){
            impl->bullets[0].pos={0,0,distance};std::vector<DrawObject> isolated;
            for(const auto& d:impl->draw())if(d.instance.objectToken==bullet.token)isolated.push_back(d);
            auto frame=impl->renderer.captureScene(isolated,{{0,0,0},{0,0,1},80},impl->objects);uint32_t pixels=frame.pixels.at(bullet.token);
            if(distance>=2&&pixels<100)throw std::runtime_error("Projectile disappeared before reaching the camera");
            if(pixels>1280*720*.015f)throw std::runtime_error("Projectile obscured the camera during flyby");
            maxProjectilePixels=std::max(maxProjectilePixels,pixels);++projectileChecks;
        }
    }
    impl->bullets.clear();
    struct Case {const char* name;float duration;int kind=0,weapon=0;V3 velocity{};float scale=1;};
    const Case cases[]={
        {"idle",.8f},{"walk-forward",3.5f,0,0,{0,0,2.5f}},{"walk-left",2.3f,0,0,{-2.5f,0,0}},
        {"walk-right",2.3f,0,0,{2.5f,0,0}},{"walk-backward",2.3f,0,0,{0,0,-2.5f}},
        {"rifle-walk",3.6f,0,1,{0,0,2.5f}},{"rifle-reload",2.5f,1,1},
        {"jump",1.5f,2},{"crouch",2,3},{"crouch-forward",2.4f,3,0,{0,0,1.5f}},{"crouch-backward",2.4f,3,1,{0,0,-1.5f}},
        {"pistol-fire",1.5f,9},{"rifle-fire",1.5f,9,1},{"shotgun-fire",1.5f,9,2},{"shield",2.4f,4},{"trap",2.6f,5},
        {"dash-forward",1.6f,6,1,{0,0,13}},{"dash-left",1.6f,6,1,{-13,0,0}},
        {"dash-right",1.6f,6,1,{13,0,0}},{"dash-backward",1.6f,6,1,{0,0,-13}},
        {"aim",2,8,1},{"death",1.5f,7,1},{"recover",1.2f},
        {"small-jump",1.5f,2,0,{},.65f},{"large-shield",2.4f,4,1,{},1.65f}};
    const int links[][2]={{6,7},{7,8},{9,10},{10,11},{12,13},{13,14},{16,17},{17,18}};
    auto& a=impl->actors[0];a.appearance=impl->playerAppearance;a.index=impl->appearances[a.appearance].index;impl->objects[a.token].index=a.index;a.yaw=0;a.pitch=0;a.localPose.clear();a.posedAt=-1;
    const auto& reference=impl->appearances[a.appearance].reference;std::array<float,8> lengths;
    for(int i=0;i<8;++i)lengths[i]=(reference[links[i][1]]-reference[links[i][0]]).length();
    auto folder=impl->map.scene.outputDirectory(impl->root,"captures")/"pose-regression";std::filesystem::create_directories(folder);
    Json report={{"passed",true},{"simulationHz",120},{"pixelSampleHz",30},{"cases",Json::array()}};
    std::vector<V3> previous;int captures=0;std::vector<DrawObject> actorDraws;
    const auto collect=[&]{actorDraws.clear();for(const auto& d:impl->draw())if(d.instance.objectToken==a.token||d.instance.objectToken==a.weaponToken||(a.shieldToken&&d.instance.objectToken==a.shieldToken))actorDraws.push_back(d);};
    for(const auto& test:cases){
        a.scale=test.scale;a.weapon=test.weapon;a.weaponToken=impl->token(impl->weapons[a.weapon].index);a.health=100;a.shield=0;a.shieldToken=0;a.shieldAlpha=0;a.airAlpha=0;a.crouchAlpha=0;a.reloadEnd=0;a.action.clear();a.grounded=true;a.pitch=0;a.lastShot=-100;
        a.actionStart=a.reloadStart=a.deathAt=impl->time;a.dashDirection=test.velocity.normalized();
        if(test.kind==1)a.reloadEnd=impl->time+impl->weapons[a.weapon].reload;
        if(test.kind==4){a.shield=200;a.shieldToken=impl->token(13);}
        if(test.kind==5)a.goal=a.pos+V3{0,0,.65f};
        if(test.kind==7)a.health=0;
        float maxReach=0,maxStep=0,maxLengthError=0,maxPixelChange=0,maxGripError=0;uint32_t minPixels=UINT32_MAX,previousPixels=0;
        std::array<V3,6> minimumAngles,maximumAngles;minimumAngles.fill({1e9f,1e9f,1e9f});maximumAngles.fill({-1e9f,-1e9f,-1e9f});
        Camera camera{a.pos+V3{2.4f,1.7f,-4.5f}*a.scale,a.pos+V3{0,.95f,0}*a.scale,65};
        const int count=int(std::ceil(test.duration*120));previous.clear();
        for(int frame=0;frame<count;++frame){
            constexpr float dt=1.f/120;float elapsed=(frame+1)*dt;impl->time+=dt;++impl->tick;a.velocity=test.velocity;impl->advanceGait(a,dt);
            if(test.kind==2){a.grounded=elapsed>=1.15f;a.velocity.y=elapsed<.35f?3.f:-3.f;a.action=a.grounded?"":"Jump";a.airAlpha=approach(a.airAlpha,a.grounded?0.f:1.f,dt*10);}
            if(test.kind==3)a.crouchAlpha=approach(a.crouchAlpha,elapsed<1.3f?1.f:0.f,dt*7);
            if(test.kind==4)a.shieldAlpha=approach(a.shieldAlpha,elapsed>.35f&&elapsed<1.5f?1.f:0.f,dt*4);
            if(test.kind==5)a.action=elapsed<2.35f?"Trap":"";
            if(test.kind==6){a.action=elapsed<1.321428571f?"Dash":"";if(elapsed>.892857143f)a.velocity={};}
            if(test.kind==8)a.pitch=std::sin(elapsed*Pi)*1.1f;
            if(test.kind==9&&frame==30){a.previousShot=-100;a.shotWeapon=a.weapon;a.lastShot=impl->time;}
            impl->pose();std::vector<V3> local;for(auto p:a.posedBones)local.push_back((p-a.pos)*(1/a.scale));
            for(int p=0;p<6;++p){V3 angles=bboxRotationDegreesUE(a.boxes[p]);auto& lo=minimumAngles[p];auto& hi=maximumAngles[p];
                lo={std::min(lo.x,angles.x),std::min(lo.y,angles.y),std::min(lo.z,angles.z)};hi={std::max(hi.x,angles.x),std::max(hi.y,angles.y),std::max(hi.z,angles.z)};}
            maxGripError=std::max(maxGripError,a.gripError);
            for(size_t b=0;b<local.size();++b){const auto& p=local[b];if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))throw std::runtime_error(std::string(test.name)+": nonfinite pose");
                maxReach=std::max(maxReach,flatLength(p));if(!previous.empty())maxStep=std::max(maxStep,(p-previous[b]).length());}
            for(int i=0;i<8;++i){float length=(local[links[i][1]]-local[links[i][0]]).length();maxLengthError=std::max(maxLengthError,std::abs(length/lengths[i]-1));}
            previous=std::move(local);
            if(frame%4==0){collect();auto capture=impl->renderer.captureScene(actorDraws,camera,impl->objects);uint32_t pixels=capture.pixels.at(a.token);minPixels=std::min(minPixels,pixels);++captures;
                if(previousPixels)maxPixelChange=std::max(maxPixelChange,std::abs(float(pixels)-previousPixels)/std::max(pixels,previousPixels));previousPixels=pixels;}
            if(frame==count/2){collect();impl->renderer.render(actorDraws,camera,{});if(!impl->renderer.screenshot(folder/(std::string(test.name)+".png")))throw std::runtime_error("Cannot save pose test image");}
        }
        Json result={{"name",test.name},{"frames",count},{"maxHorizontalJointReachMeters",maxReach},{"maxJointStepMeters",maxStep},{"maxLimbLengthRelativeError",maxLengthError},{"maxGripSeparationErrorMeters",maxGripError},{"minimumBodyPixels",minPixels},{"maxAdjacentPixelCountChange",maxPixelChange},{"lastAnimation",a.animation}};
        result["rotationRangesDegreesUE"]=Json::array();const char* names[]={"head","body","leftArm","rightArm","leftLeg","rightLeg"};
        for(int p=0;p<6;++p){const auto& lo=minimumAngles[p];const auto& hi=maximumAngles[p];result["rotationRangesDegreesUE"].push_back({{"name",names[p]},{"pitch",{lo.x,hi.x}},{"yaw",{lo.y,hi.y}},{"roll",{lo.z,hi.z}}});}
        report["cases"].push_back(result);
        if(maxReach>1.5f||maxStep>.25f||maxLengthError>.01f||minPixels<1200||maxPixelChange>.5f){report["passed"]=false;std::ofstream(impl->map.scene.outputDirectory(impl->root,"captures")/"pose-tests-failed.json")<<report.dump(2);throw std::runtime_error(std::string(test.name)+": pose continuity regression; see captures/pose-tests-failed.json");}
    }
    collect();Camera stable{a.pos+V3{2.4f,1.7f,-4.5f}*a.scale,a.pos+V3{0,.95f,0}*a.scale,65};
    auto baselineCapture=impl->renderer.captureScene(actorDraws,stable,impl->objects);
    for(int i=0;i<12;++i){Camera other=stable;other.eye.x-=4;impl->renderer.captureScene(actorDraws,other,impl->objects);auto actual=impl->renderer.captureScene(actorDraws,stable,impl->objects);
        if(actual.color!=baselineCapture.color||actual.depth!=baselineCapture.depth||actual.ids!=baselineCapture.ids)throw std::runtime_error("Alternating capture cameras changed a stationary frame");}
    report["renderedFrameChecks"]=captures;report["stableAlternatingViewChecks"]=12;
    report["appearances"]=Json::array();
    float maxArmTransformError=0,minWristBoxChange=1e9f;int armTransformChecks=0;
    for(size_t i=0;i<impl->appearances.size();++i){const auto& appearance=impl->appearances[i];
        a.appearance=i;a.index=appearance.index;impl->objects[a.token].index=a.index;a.scale=appearance.scale;a.health=100;a.weapon=0;a.weaponToken=impl->token(impl->weapons[0].index);
        a.velocity={};a.action.clear();a.reloadEnd=0;a.airAlpha=a.crouchAlpha=a.shieldAlpha=a.shield=0;a.shieldToken=0;a.pitch=0;a.localPose.clear();a.posedAt=-1;impl->pose();
        const auto size=[](const Matrix& m){return V3{V3{m._11,m._12,m._13}.length(),V3{m._21,m._22,m._23}.length(),V3{m._31,m._32,m._33}.length()};};
        V3 head=size(a.boxes[0]),body=size(a.boxes[1]);
        V3 expectedHead=appearance.bboxParts[0].bounds.size()*a.scale;
        const auto& bodyPart=appearance.bboxParts[1];float bodyLength=(a.posedBones[bodyPart.end]-a.posedBones[bodyPart.start]).length();
        const auto bodyEdge=[&](float v){return v<0?v*a.scale:v>bodyPart.length?bodyLength+(v-bodyPart.length)*a.scale:v*bodyLength/bodyPart.length;};
        V3 expectedBody=bodyPart.bounds.size()*a.scale;expectedBody.y=bodyEdge(bodyPart.bounds.max.y)-bodyEdge(bodyPart.bounds.min.y);
        if((head-expectedHead).length()>.0001f||(body-expectedBody).length()>.0001f)throw std::runtime_error("Character dimensions differ from UE reference-local surface bounds");
        auto twisted=a.posedRotations;twisted[0]=multiply(twisted[0],axisRotation({0,1,0},Pi*.5f));
        auto sameBody=impl->characterBoxes(a.posedBones,twisted,appearance,a.scale,a.yaw)[1];
        for(int v=0;v<8;++v){V3 corner{v&1?.5f:-.5f,v&2?.5f:-.5f,v&4?.5f:-.5f};
            if((point(sameBody,corner)-point(a.boxes[1],corner)).length()>.0001f)throw std::runtime_error("Pelvis twist rotates the segment-fitted torso");}
        for(const auto& m:a.boxes)for(int v=0;v<8;++v){V3 p=point(m,{v&1?.5f:-.5f,v&2?.5f:-.5f,v&4?.5f:-.5f});const auto& bounds=impl->objects[a.token];
            if(p.x<bounds.min.x-.001f||p.x>bounds.max.x+.001f||p.y<bounds.min.y-.001f||p.y>bounds.max.y+.001f||p.z<bounds.min.z-.001f||p.z>bounds.max.z+.001f)throw std::runtime_error("Character catalog bounds exclude a posed part");}
        V3 headCenter=position(a.boxes[0]);auto hit=impl->collision.ray(headCenter+V3{0,0,2},headCenter);
        if(!hit.hit||hit.token!=a.token||hit.part!=0)throw std::runtime_error("Cartoon head hitbox does not match visible head");
        if((position(a.gun)-a.posedBones[20]).length()>.0001f)throw std::runtime_error("Weapon detached from retargeted hand socket");
        for(int side=0;side<2;++side){int shoulder=side?9:6;
            Quat wrist=a.posedRotations[shoulder+2];Matrix original=a.boxes[2+side];auto wristPose=a.posedRotations;
            wristPose[shoulder+2]=multiply(wrist,axisRotation({0,1,0},Pi*.5f));
            Matrix turned=impl->characterBoxes(a.posedBones,wristPose,appearance,a.scale,a.yaw)[2+side];float wristChange=0;
            for(int v=0;v<8;++v){V3 corner{v&1?.5f:-.5f,v&2?.5f:-.5f,v&4?.5f:-.5f};wristChange=std::max(wristChange,(point(original,corner)-point(turned,corner)).length());}
            minWristBoxChange=std::min(minWristBoxChange,wristChange);
            if(wristChange<.01f)throw std::runtime_error("Arm BBox ignores wrist orientation");
            for(float yaw:{.4f,2.1f})for(float scale:{.65f,1.65f}){const V3 offset{17.5f,4,-8};const auto moved=[&](V3 p){return offset+rotate((p-a.pos)*scale,yaw);};
                auto movedBones=a.posedBones;auto movedRotations=a.posedRotations;for(auto& p:movedBones)p=moved(p);for(auto& q:movedRotations)q=multiply(axisRotation({0,1,0},yaw),q);
                Matrix changed=impl->characterBoxes(movedBones,movedRotations,appearance,a.scale*scale,a.yaw+yaw)[2+side];
                for(int v=0;v<8;++v){V3 corner{v&1?.5f:-.5f,v&2?.5f:-.5f,v&4?.5f:-.5f};float error=(point(changed,corner)-moved(point(original,corner))).length();
                    maxArmTransformError=std::max(maxArmTransformError,error);if(error>.0002f)throw std::runtime_error("Turning or scaling twists the arm BBox");}
                ++armTransformChecks;
            }
        }
        collect();Camera front{a.pos+V3{1.6f,1.3f,4.5f}*a.scale,a.pos+V3{0,1.f,0}*a.scale,55};
        auto frame=impl->renderer.captureScene(actorDraws,front,impl->objects);auto pixels=frame.pixels.at(a.token);
        if(pixels<1200)throw std::runtime_error("Cartoon appearance is blank");
        impl->renderer.render(actorDraws,front,{});if(!impl->renderer.screenshot(folder/("appearance-"+std::to_string(i)+".png")))throw std::runtime_error("Cannot save appearance image");
        report["appearances"].push_back({{"name",appearance.name},{"headSizeMeters",{head.x,head.y,head.z}},{"bodySizeMeters",{body.x,body.y,body.z}},{"pelvisHeightMeters",appearance.reference[0].y*a.scale},{"bodyPixels",pixels}});
        a.shield=200;a.shieldToken=impl->token(13);a.shieldAlpha=1;impl->pose();V3 shield=position(a.shieldMatrix),normal=forward(a.yaw);
        auto shieldHit=impl->collision.sweepBullet(shield+normal*.4f,shield,impl->weapons[1].radius,impl->actors[2].token);
        collect();impl->renderer.render(actorDraws,front,{});if(!impl->renderer.screenshot(folder/("appearance-shield-"+std::to_string(i)+".png")))throw std::runtime_error("Cannot save appearance shield image");
        if(!shieldHit.hit||shieldHit.token!=a.shieldToken){std::ofstream(impl->map.scene.outputDirectory(impl->root,"captures")/"shield-fit-failed.json")<<Json{{"appearance",appearance.name},{"hitToken",shieldHit.token},{"hitPart",shieldHit.part},{"expectedToken",a.shieldToken}}.dump(2);throw std::runtime_error("Cartoon surface protrudes ahead of the shield grip");}
    }
    report["weaponGripViews"]=Json::array();
    for(int weapon=0;weapon<3;++weapon)for(int aim=0;aim<3;++aim){
        a.appearance=impl->playerAppearance;const auto& appearance=impl->appearances[a.appearance];
        a.index=appearance.index;impl->objects[a.token].index=a.index;a.scale=appearance.scale;
        a.weapon=weapon;a.weaponToken=impl->token(impl->weapons[weapon].index);a.health=100;a.velocity={};a.action.clear();
        a.reloadEnd=0;a.shield=a.shieldAlpha=a.airAlpha=a.crouchAlpha=0;a.shieldToken=0;a.yaw=0;a.pitch=aim==1?.8f:aim==2?-.8f:0;
        a.localPose.clear();a.posedAt=-1;a.lastShot=-100;impl->time=.35;impl->pose();collect();
        std::string name="holding-"+impl->weapons[weapon].name+(aim==1?"-up":aim==2?"-down":"");
        const Camera views[]={{a.pos+V3{-1.2f,1.05f,2.f}*a.scale,a.pos+V3{0,.82f,.25f}*a.scale,50},
                              {a.pos+V3{-2.1f,1.1f,.35f}*a.scale,a.pos+V3{0,.82f,.25f}*a.scale,50}};
        for(int view=0;view<2;++view){impl->renderer.render(actorDraws,views[view],{});
            if(!impl->renderer.screenshot(folder/(name+(view?"-side.png":"-front.png"))))throw std::runtime_error("Cannot save weapon grip image");}
        Json joints=Json::array();for(int bone:{6,7,8,9,10,11}){V3 p=a.posedBones[bone]-a.pos;joints.push_back({p.x,p.y,p.z});}
        Json boxes=Json::array();for(int part:{2,3}){const auto& m=a.boxes[part];V3 p=position(m)-a.pos;
            boxes.push_back({{"center",{p.x,p.y,p.z}},{"axes",{{m._11,m._12,m._13},{m._21,m._22,m._23},{m._31,m._32,m._33}}}});}
        float wristError=0;Json wrists=Json::array();
        for(int side=0;side<2;++side){int bone=side?11:8;Quat actual=a.posedRotations[bone];
            Quat expectedWrist=multiply(axisRotation({1,0,0},-a.pitch*.4f),multiply(a.localRotations[bone],appearance.wristCorrection[side]));
            for(V3 axis:{V3{1,0,0},V3{0,1,0},V3{0,0,1}})wristError=std::max(wristError,(rotate(axis,actual)-rotate(axis,expectedWrist)).length());
            wrists.push_back({actual.x,actual.y,actual.z,actual.w});}
        if(a.gripError>.001f||wristError>.0001f)throw std::runtime_error("Support hand no longer follows the UE weapon grip frame");
        report["weaponGripViews"].push_back({{"name",name},{"joints",joints},{"boxes",boxes},{"wristRotations",wrists},{"wristAxisError",wristError},{"gripErrorMeters",a.gripError}});
    }
    report["armTransformChecks"]=armTransformChecks;report["maximumArmTransformErrorMeters"]=maxArmTransformError;report["minimumWristBoxChangeMeters"]=minWristBoxChange;
    report["projectileFlybyFrameChecks"]=projectileChecks;report["maximumFlybyProjectilePixels"]=maxProjectilePixels;return report;
}

Json Encounter::uePoseTest(){
    Json capture;std::ifstream input(impl->root/L"captures/ue-live-poses.json");if(!input)throw std::runtime_error("Capture live UE poses first");input>>capture;
    if(!capture.value("validatedStates",false)||capture.at("dualProtagonists").get<bool>()||capture.at("heroClass")!="/ShooterCore/Game/B_Hero_ShooterMannequin1.B_Hero_ShooterMannequin1_C"||
       capture.at("recorderClass")!="/Script/RuntimeVideoRecorder.RuntimeMergedBBoxCameraRecordingComponent")throw std::runtime_error("UE pose capture used the wrong hero/recorder");
    const auto& appearance=impl->appearances[impl->playerAppearance];
    Json report={{"passed",true},{"heroClass",capture.at("heroClass")},{"recorderClass",capture.at("recorderClass")},{"dualProtagonists",false},
        {"method","Replay actual PIE bones through native character geometry; compare with an independent UE-coordinate evaluation of the plugin's segment/limb rules"},
        {"scope","Geometry fidelity for sampled bones; this does not assert identical animation blending/timing"},{"cases",Json::array()}};
    const char* names[]={"head","body","leftArm","rightArm","leftLeg","rightLeg"};
    float maxCorner=0,maxAngle=0,maxTwist=0;int frames=0,checks=0;
    const auto axis=[](const Matrix& m,int i){return i==0?V3{m._11,m._12,m._13}:i==1?V3{m._21,m._22,m._23}:V3{m._31,m._32,m._33};};
    for(const auto& item:capture.at("cases")){
        Json summary={{"name",item.at("name")},{"samples",item.at("samples").size()},{"parts",Json::array()}};
        float partCorners[6]={},partAngles[6]={};float bodyYaw=0,bodyTilt=0,pelvisYaw=0,maxSpeed=0;
        std::array<V3,6> minimumAngles,maximumAngles;minimumAngles.fill({1e9f,1e9f,1e9f});maximumAngles.fill({-1e9f,-1e9f,-1e9f});
        for(const auto& frame:item.at("samples")){
            std::vector<V3> bones;std::vector<Quat> rotations;
            for(const auto& bone:frame.at("target")){bones.push_back(animPoint(bone.at("p")));rotations.push_back(animRotation(bone.at("q")));}
            bones.push_back(bones[11]);rotations.push_back(rotations[11]);
            auto actual=impl->characterBoxes(bones,rotations,appearance,1,0);
            const auto& references=frame.at("referenceBoxes");
            for(int p=0;p<6;++p){const auto& reference=references.at(p);V3 extent{reference.at("e")[0].get<float>()*.02f,reference.at("e")[2].get<float>()*.02f,reference.at("e")[1].get<float>()*.02f};
                if(frame.at("time").get<float>()>.5f){const auto& angles=reference.at("rotationDegreesUE");auto& lo=minimumAngles[p];auto& hi=maximumAngles[p];
                    for(int i=0;i<3;++i){(&lo.x)[i]=std::min((&lo.x)[i],angles.at(i).get<float>());(&hi.x)[i]=std::max((&hi.x)[i],angles.at(i).get<float>());}}
                Matrix expected=boneTransform(animPoint(reference.at("p")),extent,animRotation(reference.at("q")));
                for(int v=0;v<8;++v){V3 corner{v&1?.5f:-.5f,v&2?.5f:-.5f,v&4?.5f:-.5f};partCorners[p]=std::max(partCorners[p],(point(actual[p],corner)-point(expected,corner)).length());}
                for(int i=0;i<3;++i){V3 a=axis(actual[p],i).normalized(),b=axis(expected,i).normalized();float angle=std::atan2(cross(a,b).length(),dot(a,b))*180/Pi;partAngles[p]=std::max(partAngles[p],angle);}
                ++checks;
            }
            V3 x=axis(actual[1],0).normalized(),up=axis(actual[1],1).normalized();
            // Compare heading with the cached frame, whose local X may face -X.
            Quat ref=appearance.bboxParts[1].referenceFrame;V3 rx=rotate({1,0,0},ref);
            float signedYaw=std::atan2(rx.z*x.x-rx.x*x.z,rx.x*x.x+rx.z*x.z)*180/Pi;
            bodyYaw=std::max(bodyYaw,std::abs(signedYaw));
            bodyTilt=std::max(bodyTilt,std::atan2(cross(up,{0,1,0}).length(),dot(up,{0,1,0}))*180/Pi);
            V3 pelvis=rotate({1,0,0},multiply(rotations[0],inverse(appearance.referenceRotations[0])));
            pelvisYaw=std::max(pelvisYaw,std::abs(std::atan2(-pelvis.z,pelvis.x)*180/Pi));
            auto changed=rotations;changed[0]=multiply(changed[0],axisRotation({0,1,0},Pi*.5f));auto untwisted=impl->characterBoxes(bones,changed,appearance,1,0);
            for(int v=0;v<8;++v){V3 corner{v&1?.5f:-.5f,v&2?.5f:-.5f,v&4?.5f:-.5f};maxTwist=std::max(maxTwist,(point(actual[1],corner)-point(untwisted[1],corner)).length());}
            const auto& velocity=frame.at("velocity");maxSpeed=std::max(maxSpeed,V3{velocity[0],velocity[1],velocity[2]}.length()*.01f);
            ++frames;
        }
        for(int p=0;p<6;++p){maxCorner=std::max(maxCorner,partCorners[p]);maxAngle=std::max(maxAngle,partAngles[p]);const auto& lo=minimumAngles[p];const auto& hi=maximumAngles[p];
            summary["parts"].push_back({{"name",names[p]},{"maximumCornerErrorMeters",partCorners[p]},{"maximumAxisErrorDegrees",partAngles[p]},
                {"settledRotationDegreesUE",{{"pitch",{lo.x,hi.x}},{"yaw",{lo.y,hi.y}},{"roll",{lo.z,hi.z}}}}});}
        summary["maximumBodyYawFromReferenceDegrees"]=bodyYaw;summary["maximumBodyTiltDegrees"]=bodyTilt;summary["maximumPelvisBoneYawDegrees"]=pelvisYaw;summary["maximumSpeedMetersPerSecond"]=maxSpeed;
        report["cases"].push_back(summary);
    }
    report["frames"]=frames;report["boxChecks"]=checks;report["maximumCornerErrorMeters"]=maxCorner;report["maximumAxisErrorDegrees"]=maxAngle;report["pelvisTwistBodyErrorMeters"]=maxTwist;
    report["abilityEvents"]=capture.at("abilityEvents");
    if(frames<500||capture.at("cases").size()!=40||maxCorner>.0002f||maxAngle>.025f||maxTwist>.000001f){report["passed"]=false;std::ofstream(impl->map.scene.outputDirectory(impl->root,"captures")/"ue-pose-comparison-failed.json")<<report.dump(2);throw std::runtime_error("UE pose BBox comparison failed; see captures/ue-pose-comparison-failed.json");}
    return report;
}
