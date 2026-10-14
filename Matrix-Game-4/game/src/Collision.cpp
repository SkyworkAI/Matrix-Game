#include <Jolt/Jolt.h>
#include "Collision.h"
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Recast.h>
#include <DetourNavMesh.h>
#include <DetourNavMeshBuilder.h>
#include <DetourNavMeshQuery.h>
#include <map>

namespace {
JPH::Vec3 jv(V3 p){return {p.x,p.y,p.z};}
V3 v3(JPH::Vec3Arg p){return {p.GetX(),p.GetY(),p.GetZ()};}
class Broad final:public JPH::BroadPhaseLayerInterface {
public:JPH::uint GetNumBroadPhaseLayers()const override{return 2;}
JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer l)const override{return JPH::BroadPhaseLayer(l==0?0:1);}
};
class Pair final:public JPH::ObjectLayerPairFilter {public:bool ShouldCollide(JPH::ObjectLayer a,JPH::ObjectLayer b)const override{if(a==2||b==2)return a==0||b==0;return a!=0||b!=0;}};
class QueryLayers final:public JPH::ObjectLayerFilter {public:bool ShouldCollide(JPH::ObjectLayer layer)const override{return layer!=2;}};
class ObjectBroad final:public JPH::ObjectVsBroadPhaseLayerFilter {public:bool ShouldCollide(JPH::ObjectLayer,JPH::BroadPhaseLayer)const override{return true;}};
class Ignore final:public JPH::BodyFilter {
    uint32_t token,attached;
public:explicit Ignore(uint32_t t,uint32_t a=0):token(t),attached(a){}
bool ShouldCollideLocked(const JPH::Body& b)const override{uint32_t id=uint32_t(b.GetUserData());return (!token||id!=token)&&(!attached||id!=attached);}
};
void require(bool good,const char* message){if(!good)throw std::runtime_error(message);}
struct BoxFrame {V3 size,center;JPH::Quat rotation;};
BoxFrame frame(const Matrix& m){
    DirectX::XMVECTOR scale,rotation,translation;DirectX::XMMatrixDecompose(&scale,&rotation,&translation,DirectX::XMLoadFloat4x4(&m));
    DirectX::XMFLOAT3 s,p;DirectX::XMFLOAT4 q;DirectX::XMStoreFloat3(&s,scale);DirectX::XMStoreFloat3(&p,translation);DirectX::XMStoreFloat4(&q,rotation);
    return {{std::abs(s.x),std::abs(s.y),std::abs(s.z)},{p.x,p.y,p.z},JPH::Quat(q.x,q.y,q.z,q.w)};
}
bool crosses(const Matrix& matrix,V3 from,V3 to,float clearance=.48f){
    auto b=frame(matrix);V3 a=v3(b.rotation.Conjugated()*jv(from-b.center)),d=v3(b.rotation.Conjugated()*jv(to-from));
    V3 low{-.5f*b.size.x-clearance,-.5f*b.size.y-1.8f,-.5f*b.size.z-clearance};
    V3 high{.5f*b.size.x+clearance,.5f*b.size.y+.05f,.5f*b.size.z+clearance};float begin=0,end=1;
    for(int axis=0;axis<3;++axis){float origin=(&a.x)[axis],direction=(&d.x)[axis];if(std::abs(direction)<1e-6f){if(origin<(&low.x)[axis]||origin>(&high.x)[axis])return false;continue;}
        float t0=((&low.x)[axis]-origin)/direction,t1=((&high.x)[axis]-origin)/direction;if(t0>t1)std::swap(t0,t1);begin=std::max(begin,t0);end=std::min(end,t1);if(begin>end)return false;}return true;
}
}
struct Collision::Impl {
    Broad broad;Pair pairs;ObjectBroad objectBroad;
    JPH::PhysicsSystem system;
    JPH::TempAllocatorImpl temporary{16*1024*1024};
    JPH::CharacterVsCharacterCollisionSimple characterCollision;
    std::vector<JPH::Ref<JPH::CharacterVirtual>> characters;
    std::vector<std::vector<JPH::BodyID>> hitBodies;
    std::vector<std::pair<uint32_t,uint32_t>> attachments;
    uint32_t attached(uint32_t owner)const{for(const auto& [token,shield]:attachments)if(token==owner)return shield;return 0;}
    struct Obstacle {JPH::BodyID body;bool hasBody=false;Matrix nav{};bool reserved=false;};
    std::map<uint32_t,Obstacle> obstacles;
    struct Debris {JPH::BodyID body;V3 scale,localCenter;float age=0;};
    std::vector<Debris> fragments;
    JPH::JobSystemSingleThreaded jobs{1024};
    dtNavMesh* nav=nullptr;dtNavMeshQuery* query=nullptr;dtQueryFilter filter;
    int polygons=0;
    Impl(){system.Init(4096,0,4096,1024,broad,objectBroad,pairs);}
    ~Impl(){characters.clear();dtFreeNavMeshQuery(query);dtFreeNavMesh(nav);JPH::BodyIDVector ids;system.GetBodies(ids);auto& bi=system.GetBodyInterface();bi.RemoveBodies(ids.data(),int(ids.size()));bi.DestroyBodies(ids.data(),int(ids.size()));}
};
Collision::Collision(const MapData& map) {
    JPH::RegisterDefaultAllocator();JPH::Factory::sInstance=new JPH::Factory;JPH::RegisterTypes();impl=std::make_unique<Impl>();
    std::vector<float> vertices;std::vector<int> indices;
    const bool batched=map.usesBatches();
    JPH::TriangleList shared;
    const auto submit=[&](JPH::TriangleList& triangles,uint64_t user){
        if(triangles.empty())return;
        JPH::MeshShapeSettings settings(triangles);settings.mPerTriangleUserData=batched;
        auto shape=settings.Create();require(!shape.HasError(),"Cannot build scene collision");
        JPH::BodyCreationSettings s(shape.Get(),JPH::RVec3::sZero(),JPH::Quat::sIdentity(),JPH::EMotionType::Static,0);
        s.mUserData=user;s.mFriction=.8f;
        auto id=impl->system.GetBodyInterface().CreateAndAddBody(s,JPH::EActivation::DontActivate);
        require(!id.IsInvalid(),"Scene collision body capacity exceeded");triangles.clear();
    };
    for(size_t oi=0;oi<map.objects.size();++oi){
        const auto& o=map.objects[oi];JPH::TriangleList individual;
        auto& triangles=batched?shared:individual;
        const auto& nb=map.navigationBounds;
        const bool includeNav=!map.hasNavigationBounds||(o.bounds.max.x>=nb.min.x&&o.bounds.min.x<=nb.max.x&&o.bounds.max.z>=nb.min.z&&o.bounds.min.z<=nb.max.z&&o.bounds.max.y>=nb.min.y&&o.bounds.min.y<=nb.max.y);
        const int start=int(vertices.size()/3);
        if(includeNav)for(auto p:o.mesh.vertices){vertices.push_back(p.x);vertices.push_back(p.y);vertices.push_back(p.z);}
        for(size_t i=0;i<o.mesh.indices.size();i+=3){
            auto a=o.mesh.indices[i],b=o.mesh.indices[i+1],c=o.mesh.indices[i+2];
            V3 normal=cross(o.mesh.vertices[b]-o.mesh.vertices[a],o.mesh.vertices[c]-o.mesh.vertices[a]);
            if(normal.length()<1e-6f)continue;
            // Render meshes are double-sided. Give walkable faces consistent upward winding.
            if(normal.y<0)std::swap(b,c);
            triangles.emplace_back(jv(o.mesh.vertices[a]),jv(o.mesh.vertices[b]),jv(o.mesh.vertices[c]),0,uint32_t(oi+1));
            if(includeNav)indices.insert(indices.end(),{start+int(a),start+int(b),start+int(c)});
            if(batched&&triangles.size()>=8192)submit(triangles,uint64_t(1)<<63);
        }
        if(!batched)submit(triangles,oi+1);
    }
    if(batched)submit(shared,uint64_t(1)<<63);
    impl->system.OptimizeBroadPhase();
    rcContext ctx;rcConfig cfg{};cfg.cs=.30f;cfg.ch=.10f;cfg.walkableSlopeAngle=45;
    cfg.walkableHeight=18;cfg.walkableClimb=4;cfg.walkableRadius=2;cfg.maxEdgeLen=40;cfg.maxSimplificationError=1.1f;
    cfg.minRegionArea=16;cfg.mergeRegionArea=100;cfg.maxVertsPerPoly=6;cfg.detailSampleDist=1.8f;cfg.detailSampleMaxError=.1f;
    require(!indices.empty(),"Navigation region contains no geometry");
    if(map.hasNavigationBounds){const auto& b=map.navigationBounds;cfg.bmin[0]=b.min.x;cfg.bmin[1]=b.min.y;cfg.bmin[2]=b.min.z;cfg.bmax[0]=b.max.x;cfg.bmax[1]=b.max.y;cfg.bmax[2]=b.max.z;}
    else{rcCalcBounds(vertices.data(),int(vertices.size()/3),cfg.bmin,cfg.bmax);cfg.bmin[1]-=1;cfg.bmax[1]+=2;}
    rcCalcGridSize(cfg.bmin,cfg.bmax,cfg.cs,&cfg.width,&cfg.height);
    std::unique_ptr<rcHeightfield,decltype(&rcFreeHeightField)> height(rcAllocHeightfield(),rcFreeHeightField);
    require(height&&rcCreateHeightfield(&ctx,*height,cfg.width,cfg.height,cfg.bmin,cfg.bmax,cfg.cs,cfg.ch),"Nav heightfield failed");
    std::vector<unsigned char> areas(indices.size()/3);
    rcMarkWalkableTriangles(&ctx,cfg.walkableSlopeAngle,vertices.data(),int(vertices.size()/3),indices.data(),int(areas.size()),areas.data());
    require(rcRasterizeTriangles(&ctx,vertices.data(),int(vertices.size()/3),indices.data(),areas.data(),int(areas.size()),*height,cfg.walkableClimb),"Nav rasterization failed");
    rcFilterLowHangingWalkableObstacles(&ctx,cfg.walkableClimb,*height);rcFilterLedgeSpans(&ctx,cfg.walkableHeight,cfg.walkableClimb,*height);rcFilterWalkableLowHeightSpans(&ctx,cfg.walkableHeight,*height);
    std::unique_ptr<rcCompactHeightfield,decltype(&rcFreeCompactHeightfield)> compact(rcAllocCompactHeightfield(),rcFreeCompactHeightfield);
    require(rcBuildCompactHeightfield(&ctx,cfg.walkableHeight,cfg.walkableClimb,*height,*compact),"Nav compact field failed");height.reset();
    require(rcErodeWalkableArea(&ctx,cfg.walkableRadius,*compact)&&rcBuildDistanceField(&ctx,*compact)&&rcBuildRegions(&ctx,*compact,0,cfg.minRegionArea,cfg.mergeRegionArea),"Nav regions failed");
    std::unique_ptr<rcContourSet,decltype(&rcFreeContourSet)> contours(rcAllocContourSet(),rcFreeContourSet);
    require(rcBuildContours(&ctx,*compact,cfg.maxSimplificationError,cfg.maxEdgeLen,*contours),"Nav contours failed");
    std::unique_ptr<rcPolyMesh,decltype(&rcFreePolyMesh)> mesh(rcAllocPolyMesh(),rcFreePolyMesh);
    std::unique_ptr<rcPolyMeshDetail,decltype(&rcFreePolyMeshDetail)> detail(rcAllocPolyMeshDetail(),rcFreePolyMeshDetail);
    require(rcBuildPolyMesh(&ctx,*contours,cfg.maxVertsPerPoly,*mesh)&&rcBuildPolyMeshDetail(&ctx,*mesh,*compact,cfg.detailSampleDist,cfg.detailSampleMaxError,*detail),"Nav polygons failed");
    require(mesh->npolys>0,"Scene has no navigable surface");
    for(int i=0;i<mesh->npolys;++i){mesh->areas[i]=0;mesh->flags[i]=1;}
    dtNavMeshCreateParams p{};p.verts=mesh->verts;p.vertCount=mesh->nverts;p.polys=mesh->polys;p.polyAreas=mesh->areas;p.polyFlags=mesh->flags;
    p.polyCount=mesh->npolys;p.nvp=mesh->nvp;p.detailMeshes=detail->meshes;p.detailVerts=detail->verts;p.detailVertsCount=detail->nverts;p.detailTris=detail->tris;p.detailTriCount=detail->ntris;
    p.walkableHeight=1.8f;p.walkableRadius=.42f;p.walkableClimb=.4f;p.cs=cfg.cs;p.ch=cfg.ch;p.buildBvTree=true;
    memcpy(p.bmin,mesh->bmin,sizeof(p.bmin));memcpy(p.bmax,mesh->bmax,sizeof(p.bmax));
    unsigned char* data=nullptr;int dataSize=0;require(dtCreateNavMeshData(&p,&data,&dataSize),"Detour mesh creation failed");
    impl->nav=dtAllocNavMesh();require(dtStatusSucceed(impl->nav->init(data,dataSize,DT_TILE_FREE_DATA)),"Detour initialization failed");
    impl->query=dtAllocNavMeshQuery();require(dtStatusSucceed(impl->query->init(impl->nav,8192)),"Detour query failed");impl->polygons=mesh->npolys;
}
Collision::~Collision(){impl.reset();JPH::UnregisterTypes();delete JPH::Factory::sInstance;JPH::Factory::sInstance=nullptr;}
RayHit Collision::ray(V3 from,V3 to,uint32_t ignore,bool staticOnly)const{
    JPH::RRayCast ray(jv(from),jv(to-from));JPH::RayCastSettings settings;settings.mBackFaceModeTriangles=JPH::EBackFaceMode::CollideWithBackFaces;
    JPH::ClosestHitCollisionCollector<JPH::CastRayCollector> collector;Ignore filter(ignore,impl->attached(ignore));
    JPH::SpecifiedObjectLayerFilter staticFilter(0);QueryLayers all;
    impl->system.GetNarrowPhaseQuery().CastRay(ray,settings,collector,{},staticOnly?static_cast<const JPH::ObjectLayerFilter&>(staticFilter):all,filter);
    RayHit out;if(!collector.HadHit())return out;
    const auto& h=collector.mHit;out.hit=true;out.fraction=h.mFraction;out.point=from+(to-from)*h.mFraction;
    const uint64_t user=impl->system.GetBodyInterface().GetUserData(h.mBodyID);out.token=uint32_t(user);out.part=uint32_t(user>>32);
    JPH::BodyLockRead lock(impl->system.GetBodyLockInterface(),h.mBodyID);
    if(lock.Succeeded()){
        out.normal=v3(lock.GetBody().GetWorldSpaceSurfaceNormal(h.mSubShapeID2,jv(out.point)));
        if(user&(uint64_t(1)<<63)){out.token=static_cast<const JPH::MeshShape*>(lock.GetBody().GetShape())->GetTriangleUserData(h.mSubShapeID2);out.part=0;}
    }
    return out;
}
RayHit Collision::sweepBullet(V3 from,V3 to,float radius,uint32_t ignore,bool staticOnly)const{
    if(radius<=0)return ray(from,to,ignore,staticOnly);
    JPH::SphereShape sphere(radius);
    JPH::RShapeCast cast(&sphere,JPH::Vec3::sReplicate(1),JPH::RMat44::sTranslation(jv(from)),jv(to-from));
    JPH::ShapeCastSettings settings;settings.mBackFaceModeTriangles=JPH::EBackFaceMode::CollideWithBackFaces;settings.mReturnDeepestPoint=true;
    JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector> collector;Ignore filter(ignore,impl->attached(ignore));
    JPH::SpecifiedObjectLayerFilter staticFilter(0);QueryLayers all;
    impl->system.GetNarrowPhaseQuery().CastShape(cast,settings,JPH::RVec3::sZero(),collector,{},staticOnly?static_cast<const JPH::ObjectLayerFilter&>(staticFilter):all,filter);
    RayHit out;if(!collector.HadHit())return out;
    const auto& h=collector.mHit;out.hit=true;out.fraction=h.mFraction;out.point=v3(h.mContactPointOn2);out.normal=v3(-h.mPenetrationAxis.NormalizedOr(JPH::Vec3::sAxisY()));
    const auto user=impl->system.GetBodyInterface().GetUserData(h.mBodyID2);out.token=uint32_t(user);out.part=uint32_t(user>>32);
    if(user&(uint64_t(1)<<63)){
        JPH::BodyLockRead lock(impl->system.GetBodyLockInterface(),h.mBodyID2);
        if(lock.Succeeded()){out.token=static_cast<const JPH::MeshShape*>(lock.GetBody().GetShape())->GetTriangleUserData(h.mSubShapeID2);out.part=0;}
    }
    return out;
}
bool Collision::overlapBox(const Matrix& volume,uint32_t ignore)const{
    const auto b=frame(volume);JPH::BoxShape shape(jv(b.size*.5f),0);JPH::CollideShapeSettings settings;
    JPH::AnyHitCollisionCollector<JPH::CollideShapeCollector> collector;Ignore bodies(ignore,impl->attached(ignore));QueryLayers layers;
    impl->system.GetNarrowPhaseQuery().CollideShape(&shape,JPH::Vec3::sReplicate(1),JPH::RMat44::sRotationTranslation(b.rotation,jv(b.center)),settings,JPH::RVec3::sZero(),collector,{},layers,bodies);
    return collector.HadHit();
}
void Collision::setObstacle(uint32_t token,const Matrix* volume,const Matrix* navigation){
    auto& bi=impl->system.GetBodyInterface();auto it=impl->obstacles.find(token);
    if(!volume&&!navigation){if(it!=impl->obstacles.end()){if(it->second.hasBody){bi.RemoveBody(it->second.body);bi.DestroyBody(it->second.body);}impl->obstacles.erase(it);}return;}
    auto& obstacle=impl->obstacles[token];obstacle.reserved=navigation!=nullptr;if(navigation)obstacle.nav=*navigation;
    if(!volume){if(obstacle.hasBody){bi.RemoveBody(obstacle.body);bi.DestroyBody(obstacle.body);obstacle.hasBody=false;}return;}
    const auto b=frame(*volume);auto shape=new JPH::BoxShape(jv(b.size*.5f),0);
    if(!obstacle.hasBody){JPH::BodyCreationSettings s(shape,jv(b.center),b.rotation,JPH::EMotionType::Static,0);s.mUserData=token;obstacle.body=bi.CreateAndAddBody(s,JPH::EActivation::DontActivate);obstacle.hasBody=true;}
    else{bi.SetShape(obstacle.body,shape,false,JPH::EActivation::DontActivate);bi.SetPositionAndRotation(obstacle.body,jv(b.center),b.rotation,JPH::EActivation::DontActivate);}
}
void Collision::addDebris(const Matrix& meshTransform,V3 localCenter,V3 localSize,V3 velocity,V3 angularVelocity){
    auto b=frame(meshTransform);V3 scaledCenter{localCenter.x*b.size.x,localCenter.y*b.size.y,localCenter.z*b.size.z};
    V3 extent{localSize.x*b.size.x*.5f,localSize.y*b.size.y*.5f,localSize.z*b.size.z*.5f};
    JPH::BodyCreationSettings s(new JPH::BoxShape(jv(extent),0),jv(b.center)+b.rotation*jv(scaledCenter),b.rotation,JPH::EMotionType::Dynamic,2);
    s.mLinearVelocity=jv(velocity);s.mAngularVelocity=jv(angularVelocity);s.mFriction=.8f;s.mRestitution=.1f;
    auto id=impl->system.GetBodyInterface().CreateAndAddBody(s,JPH::EActivation::Activate);impl->fragments.push_back({id,b.size,scaledCenter,0});
}
void Collision::stepDebris(float dt){
    if(impl->fragments.empty())return;impl->system.Update(dt,1,&impl->temporary,&impl->jobs);
    auto& bi=impl->system.GetBodyInterface();for(auto& f:impl->fragments)f.age+=dt;
    std::erase_if(impl->fragments,[&](const auto& f){if(f.age<4)return false;bi.RemoveBody(f.body);bi.DestroyBody(f.body);return true;});
}
void Collision::clearDebris(){auto& bi=impl->system.GetBodyInterface();for(const auto& f:impl->fragments){bi.RemoveBody(f.body);bi.DestroyBody(f.body);}impl->fragments.clear();}
std::vector<Matrix> Collision::debris()const{
    std::vector<Matrix> result;auto& bi=impl->system.GetBodyInterface();for(const auto& f:impl->fragments){auto q=bi.GetRotation(f.body);V3 p=v3(bi.GetPosition(f.body))-v3(q*jv(f.localCenter));
        Matrix m;DirectX::XMStoreFloat4x4(&m,DirectX::XMMatrixScaling(f.scale.x,f.scale.y,f.scale.z)*DirectX::XMMatrixRotationQuaternion(DirectX::XMVectorSet(q.GetX(),q.GetY(),q.GetZ(),q.GetW()))*DirectX::XMMatrixTranslation(p.x,p.y,p.z));result.push_back(m);}return result;
}
size_t Collision::addCharacter(V3 feet){
    JPH::CharacterVirtualSettings s;s.mShape=new JPH::RotatedTranslatedShape(JPH::Vec3(0,.9f,0),JPH::Quat::sIdentity(),new JPH::CapsuleShape(.48f,.42f));
    s.mMaxSlopeAngle=Pi/4;s.mBackFaceMode=JPH::EBackFaceMode::CollideWithBackFaces;s.mSupportingVolume=JPH::Plane(JPH::Vec3::sAxisY(),-.42f);
    JPH::Ref<JPH::CharacterVirtual> c=new JPH::CharacterVirtual(&s,jv(feet),JPH::Quat::sIdentity(),0,&impl->system);
    c->SetCharacterVsCharacterCollision(&impl->characterCollision);impl->characterCollision.Add(c);impl->characters.push_back(c);impl->hitBodies.emplace_back();return impl->characters.size()-1;
}
void Collision::teleport(size_t i,V3 feet){auto& c=impl->characters.at(i);c->SetPosition(jv(feet));c->SetLinearVelocity(JPH::Vec3::sZero());}
V3 Collision::characterPosition(size_t i)const{return v3(impl->characters.at(i)->GetPosition());}
V3 Collision::move(size_t i,V3 velocity,float dt,bool& grounded){
    auto& c=impl->characters.at(i);c->SetLinearVelocity(jv(velocity));
    JPH::CharacterVirtual::ExtendedUpdateSettings settings;settings.mWalkStairsStepUp=JPH::Vec3(0,.42f,0);settings.mStickToFloorStepDown=JPH::Vec3(0,-.45f,0);
    if(velocity.y>0)settings.mStickToFloorStepDown=JPH::Vec3::sZero();
    c->ExtendedUpdate(dt,JPH::Vec3(0,-9.8f,0),settings,JPH::SpecifiedBroadPhaseLayerFilter(JPH::BroadPhaseLayer(0)),JPH::SpecifiedObjectLayerFilter(0),{},{},impl->temporary);
    grounded=c->GetGroundState()==JPH::CharacterBase::EGroundState::OnGround;return v3(c->GetPosition());
}
void Collision::setHitBoxes(size_t actor,uint32_t token,const std::vector<Matrix>& boxes,uint32_t shieldToken,const Matrix* shield){
    if(impl->attachments.size()<=actor)impl->attachments.resize(actor+1);impl->attachments[actor]={token,shield?shieldToken:0};
    auto& bodies=impl->hitBodies.at(actor);auto& bi=impl->system.GetBodyInterface();
    // The six posed boxes are also the native game's hit geometry, with a distinct head zone.
    size_t count=boxes.size()+(shield?1:0);
    while(bodies.size()<count){JPH::BodyCreationSettings s(new JPH::BoxShape(JPH::Vec3::sReplicate(.5f),0),JPH::RVec3(0,-1000,0),JPH::Quat::sIdentity(),JPH::EMotionType::Static,1);bodies.push_back(bi.CreateAndAddBody(s,JPH::EActivation::DontActivate));}
    for(size_t i=0;i<bodies.size();++i){
        if(i>=count){bi.SetPosition(bodies[i],JPH::RVec3(0,-1000,0),JPH::EActivation::DontActivate);continue;}
        const auto& m=i<boxes.size()?boxes[i]:*shield;
        DirectX::XMVECTOR scale,rotation,translation;DirectX::XMMatrixDecompose(&scale,&rotation,&translation,DirectX::XMLoadFloat4x4(&m));
        DirectX::XMFLOAT3 s,p;DirectX::XMFLOAT4 q;DirectX::XMStoreFloat3(&s,scale);DirectX::XMStoreFloat3(&p,translation);DirectX::XMStoreFloat4(&q,rotation);
        bi.SetShape(bodies[i],new JPH::BoxShape(JPH::Vec3(std::max(std::abs(s.x)*.5f,.005f),std::max(std::abs(s.y)*.5f,.005f),std::max(std::abs(s.z)*.5f,.005f)),0),false,JPH::EActivation::DontActivate);
        bi.SetPositionAndRotation(bodies[i],JPH::RVec3(p.x,p.y,p.z),JPH::Quat(q.x,q.y,q.z,q.w),JPH::EActivation::DontActivate);
        bi.SetUserData(bodies[i],(uint64_t(i<boxes.size()?i:6)<<32)|(i<boxes.size()?token:shieldToken));
    }
}
bool Collision::project(V3 p,V3& result,float horizontal,float vertical)const{
    const float ext[]={horizontal,vertical,horizontal};dtPolyRef ref=0;const auto s=impl->query->findNearestPoly(&p.x,ext,&impl->filter,&ref,&result.x);
    if(dtStatusFailed(s)||!ref)return false;for(const auto& [token,o]:impl->obstacles)if(o.reserved&&crosses(o.nav,result,result))return false;return true;
}
std::vector<V3> Collision::path(V3 from,V3 to)const{
    const auto staticPath=[&](V3 from,V3 to){
    const float ext[]={2,4,2};dtPolyRef a=0,b=0;V3 start,end;
    impl->query->findNearestPoly(&from.x,ext,&impl->filter,&a,&start.x);impl->query->findNearestPoly(&to.x,ext,&impl->filter,&b,&end.x);if(!a||!b)return std::vector<V3>{};
    dtPolyRef polys[512];int count=0;auto status=impl->query->findPath(a,b,&start.x,&end.x,&impl->filter,polys,&count,512);
    if(dtStatusFailed(status)||!count||polys[count-1]!=b)return std::vector<V3>{};
    std::vector<V3> points(128);int n=0;status=impl->query->findStraightPath(&start.x,&end.x,polys,count,&points[0].x,nullptr,nullptr,&n,128);
    if(dtStatusFailed(status)||dtStatusDetail(status,DT_BUFFER_TOO_SMALL))return std::vector<V3>{};points.resize(n);return points;
    };
    const auto clear=[&](const std::vector<V3>& route){if(route.empty())return false;for(const auto& [token,o]:impl->obstacles)if(o.reserved){if(crosses(o.nav,route.front(),route.front()))return false;
        for(size_t i=1;i<route.size();++i)if(crosses(o.nav,route[i-1],route[i]))return false;}return true;};
    auto direct=staticPath(from,to);if(direct.empty()||clear(direct))return direct;
    // Keep Detour's original connected floor paths. Reserve the final wall footprint
    // once and join paths around its clearance-expanded corners, without per-tick
    // Recast rebuilds or temporarily walking through the rising wall.
    std::vector<V3> nodes{direct.front(),direct.back()};
    for(const auto& [token,o]:impl->obstacles)if(o.reserved){auto b=frame(o.nav);for(int x:{-1,1})for(int z:{-1,1}){
        V3 desired=b.center+v3(b.rotation*JPH::Vec3(x*(b.size.x*.5f+.65f),-b.size.y*.5f,z*(b.size.z*.5f+.65f))),candidate;
        if(project(desired,candidate,.3f,.7f)&&(candidate-desired).length()<.8f)nodes.push_back(candidate);}}
    const size_t n=nodes.size();std::vector<std::vector<std::vector<V3>>> edges(n,std::vector<std::vector<V3>>(n));
    std::vector<float> cost(n,1e9f);std::vector<int> parent(n,-1);std::vector<bool> visited(n);cost[0]=0;
    for(size_t iteration=0;iteration<n;++iteration){size_t a=n;for(size_t i=0;i<n;++i)if(!visited[i]&&(a==n||cost[i]<cost[a]))a=i;
        if(a==n||cost[a]>=1e9f)break;if(a==1)break;visited[a]=true;
        for(size_t b=0;b<n;++b)if(!visited[b]){auto route=staticPath(nodes[a],nodes[b]);if(!clear(route))continue;
            float length=0;for(size_t i=1;i<route.size();++i)length+=(route[i]-route[i-1]).length();
            if(cost[a]+length<cost[b]){cost[b]=cost[a]+length;parent[b]=int(a);edges[a][b]=std::move(route);}}}
    if(parent[1]<0)return {};std::vector<int> chain;for(int i=1;i>=0;i=parent[i])chain.push_back(i);std::reverse(chain.begin(),chain.end());
    std::vector<V3> result{nodes[0]};for(size_t i=1;i<chain.size();++i){const auto& route=edges[chain[i-1]][chain[i]];result.insert(result.end(),route.begin()+1,route.end());}return result;
}
bool Collision::randomPoint(V3 center,float minRadius,float maxRadius,std::mt19937& rng,V3& result)const{
    std::uniform_real_distribution<float> unit(0,1);
    for(int attempt=0;attempt<24;++attempt){float angle=unit(rng)*2*Pi,dist=std::sqrt(minRadius*minRadius+unit(rng)*(maxRadius*maxRadius-minRadius*minRadius));
        V3 candidate=center+V3{std::sin(angle)*dist,0,std::cos(angle)*dist};if(!project(candidate,result,2,5))continue;
        const V3 delta=result-center;const float d=std::sqrt(delta.x*delta.x+delta.z*delta.z);
        if(d<minRadius||d>maxRadius||path(center,result).empty())continue;
        auto floor=ray(result+V3{0,.4f,0},result-V3{0,.7f,0},0,true);if(!floor.hit)continue;
        result.y=floor.point.y+.02f;return true;
    }return false;
}
V3 Collision::camera(V3 pivot,V3 desired)const{auto h=ray(pivot,desired,0,true);return h.hit?pivot+(desired-pivot)*std::max(.03f,h.fraction-.05f):desired;}
int Collision::navPolygons()const{return impl->polygons;}
