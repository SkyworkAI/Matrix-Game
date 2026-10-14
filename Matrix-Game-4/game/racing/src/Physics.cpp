#include <Jolt/Jolt.h>
#include "Physics.h"
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Vehicle/WheeledVehicleController.h>

namespace {
constexpr uint64_t GroundSurface=0xffff000000000001ull;
constexpr uint64_t BridgeSurface=0xffff000000000002ull;
JPH::Vec3 jv(V3 v){return {v.x,v.y,v.z};}
V3 v3(JPH::Vec3Arg v){return {v.GetX(),v.GetY(),v.GetZ()};}
Matrix matrix(JPH::Mat44Arg m){Matrix out;for(int i=0;i<4;++i)for(int j=0;j<4;++j)out.m[i][j]=m.GetColumn4(i)[j];return out;}
class BroadPhase final:public JPH::BroadPhaseLayerInterface {
public:JPH::uint GetNumBroadPhaseLayers()const override{return 2;}
JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer l)const override{return JPH::BroadPhaseLayer(uint8_t(l));}
};
class ObjectPair final:public JPH::ObjectLayerPairFilter {
public:bool ShouldCollide(JPH::ObjectLayer a,JPH::ObjectLayer b)const override{return a==1||b==1;}
};
class ObjectBroad final:public JPH::ObjectVsBroadPhaseLayerFilter {
public:bool ShouldCollide(JPH::ObjectLayer a,JPH::BroadPhaseLayer b)const override{return a==1||b.GetValue()==1;}
};
class GroundFilter final:public JPH::BodyFilter {
    bool streetOnly;
public:
    explicit GroundFilter(bool street=false):streetOnly(street){}
    bool ShouldCollideLocked(const JPH::Body& body)const override{return body.GetUserData()==GroundSurface||(!streetOnly&&body.GetUserData()==BridgeSurface);}
};
}
struct Physics::Impl:public JPH::ContactListener {
    BroadPhase broad;ObjectPair pairs;ObjectBroad objectBroad;
    JPH::TempAllocatorImpl temporary{32*1024*1024};
    JPH::JobSystemSingleThreaded jobs{2048};
    JPH::PhysicsSystem system;
    struct Vehicle {JPH::BodyID body;JPH::Ref<JPH::VehicleConstraint> constraint;CarModel model;bool enabled=true;float frontLateralGrip=1,rearLateralGrip=1;};
    struct Pedestrian {JPH::BodyID body;bool fallen=false;};
    std::vector<Vehicle> vehicles;
    std::vector<Pedestrian> people;
    uint32_t contacts=0;
    CollisionEvents collisionEvents;
    Impl(){system.Init(32768,0,65536,16384,broad,objectBroad,pairs);system.SetContactListener(this);}
    ~Impl(){
        for(auto& v:vehicles)if(v.enabled){system.RemoveStepListener(v.constraint);system.RemoveConstraint(v.constraint);}
        for(auto& v:vehicles)if(!v.enabled)system.GetBodyInterface().DestroyBody(v.body);
        vehicles.clear();
        JPH::BodyIDVector ids;system.GetBodies(ids);auto& bi=system.GetBodyInterface();
        bi.RemoveBodies(ids.data(),int(ids.size()));bi.DestroyBodies(ids.data(),int(ids.size()));
    }
    void track(const JPH::Body& a,const JPH::Body& b,const JPH::ContactManifold& manifold){
        const auto ua=a.GetUserData(),ub=b.GetUserData();
        if(ua==GroundSurface||ub==GroundSurface||ua==BridgeSurface||ub==BridgeSurface)return;
        if(!((ua>0&&ua<1000)||(ub>0&&ub<1000)))return;
        collisionEvents.touch({a.GetID().GetIndexAndSequenceNumber(),b.GetID().GetIndexAndSequenceNumber(),manifold.mSubShapeID1.GetValue(),manifold.mSubShapeID2.GetValue()},ua,ub,(a.GetLinearVelocity()-b.GetLinearVelocity()).Length());
    }
    void OnContactAdded(const JPH::Body& a,const JPH::Body& b,const JPH::ContactManifold& manifold,JPH::ContactSettings&)override{
        track(a,b,manifold);
        const auto ua=a.GetUserData(),ub=b.GetUserData();
        if((ua==1||ub==1)&&(a.GetLinearVelocity()-b.GetLinearVelocity()).Length()>2)++contacts;
        if((a.GetLinearVelocity()-b.GetLinearVelocity()).Length()>1.8f){
            if(ua>=1000&&ua<1000+people.size()&&ub>0&&ub<1000)people[size_t(ua-1000)].fallen=true;
            if(ub>=1000&&ub<1000+people.size()&&ua>0&&ua<1000)people[size_t(ub-1000)].fallen=true;
        }
    }
    void OnContactPersisted(const JPH::Body& a,const JPH::Body& b,const JPH::ContactManifold& m,JPH::ContactSettings&)override{track(a,b,m);}
    void OnContactRemoved(const JPH::SubShapeIDPair& p)override{
        collisionEvents.remove({p.GetBody1ID().GetIndexAndSequenceNumber(),p.GetBody2ID().GetIndexAndSequenceNumber(),p.GetSubShapeID1().GetValue(),p.GetSubShapeID2().GetValue()});
    }
};
Physics::Physics(){JPH::RegisterDefaultAllocator();JPH::Factory::sInstance=new JPH::Factory;JPH::RegisterTypes();impl=std::make_unique<Impl>();}
Physics::~Physics(){impl.reset();JPH::UnregisterTypes();delete JPH::Factory::sInstance;JPH::Factory::sInstance=nullptr;}
void Physics::addBox(V3 p,V3 extent,float yaw,bool ground){
    JPH::BodyCreationSettings settings(new JPH::BoxShape(jv(extent),std::min(.02f,std::min({extent.x,extent.y,extent.z})*.2f)),jv(p),JPH::Quat::sRotation(JPH::Vec3::sAxisY(),yaw),JPH::EMotionType::Static,0);
    settings.mFriction=.95f;settings.mUserData=ground?GroundSurface:0;impl->system.GetBodyInterface().CreateAndAddBody(settings,JPH::EActivation::DontActivate);
}
void Physics::addStaticMesh(const MeshData& mesh,const Matrix& model,bool ground,bool bridge){
    std::vector<JPH::Vec3> points;points.reserve(mesh.vertices.size());
    for(V3 v:mesh.vertices){const V3 p=transformPoint(v,model);points.emplace_back(p.x,p.y,p.z);}
    JPH::TriangleList triangles;triangles.reserve(mesh.indices.size()/3);
    for(size_t i=0;i<mesh.indices.size();i+=3){auto a=points[mesh.indices[i]],b=points[mesh.indices[i+1]],c=points[mesh.indices[i+2]];if((b-a).Cross(c-a).LengthSq()>1e-10f)triangles.emplace_back(a,b,c);}
    JPH::MeshShapeSettings shape(triangles);auto result=shape.Create();if(result.HasError())throw std::runtime_error(result.GetError().c_str());
    JPH::BodyCreationSettings settings(result.Get(),JPH::RVec3::sZero(),JPH::Quat::sIdentity(),JPH::EMotionType::Static,0);
    settings.mFriction=.8f;settings.mUserData=bridge?BridgeSurface:ground?GroundSurface:0;impl->system.GetBodyInterface().CreateAndAddBody(settings,JPH::EActivation::DontActivate);
}
size_t Physics::addCar(const CarModel& model,const MeshData& mesh,V3 pos,float heading,float longitudinalImpulseScale){
    if(!std::isfinite(longitudinalImpulseScale)||longitudinalImpulseScale<=0)throw std::runtime_error("Invalid longitudinal impulse scale");
    JPH::Array<JPH::Vec3> points;for(V3 p:mesh.vertices)points.push_back(jv(p));
    auto hull=JPH::ConvexHullShapeSettings(points).Create();if(hull.HasError())throw std::runtime_error(hull.GetError().c_str());
    auto shape=JPH::OffsetCenterOfMassShapeSettings(JPH::Vec3(0,-.35f,0),hull.Get()).Create();
    if(shape.HasError())throw std::runtime_error(shape.GetError().c_str());
    JPH::BodyCreationSettings settings(shape.Get(),jv(pos),JPH::Quat::sRotation(JPH::Vec3::sAxisY(),heading),JPH::EMotionType::Dynamic,1);
    settings.mOverrideMassProperties=JPH::EOverrideMassProperties::CalculateInertia;settings.mMassPropertiesOverride.mMass=1400;
    settings.mMotionQuality=JPH::EMotionQuality::LinearCast;settings.mFriction=.5f;settings.mRestitution=.05f;settings.mLinearDamping=.02f;settings.mAngularDamping=.25f;
    settings.mUserData=impl->vehicles.size()+1;
    auto& bi=impl->system.GetBodyInterface();JPH::Body* body=bi.CreateBody(settings);if(!body)throw std::runtime_error("Vehicle body allocation failed");bi.AddBody(body->GetID(),JPH::EActivation::Activate);
    JPH::VehicleConstraintSettings vehicle;
    vehicle.mMaxPitchRollAngle=JPH::DegreesToRadians(65);
    for(int i=0;i<4;++i){
        auto* w=new JPH::WheelSettingsWV;w->mPosition=jv(model.wheelPositions[i]+V3{0,.24f,0});
        w->mRadius=model.wheelSizes[i][0];w->mWidth=model.wheelSizes[i][1];
        w->mSuspensionMinLength=.05f;w->mSuspensionMaxLength=.32f;w->mSuspensionSpring.mFrequency=2.2f;w->mSuspensionSpring.mDamping=.8f;
        w->mMaxSteerAngle=i<2?JPH::DegreesToRadians(37):0;w->mMaxBrakeTorque=2400;w->mMaxHandBrakeTorque=i<2?0:4800;
        vehicle.mWheels.push_back(w);
    }
    auto* controller=new JPH::WheeledVehicleControllerSettings;
    controller->mEngine.mMaxTorque=480;controller->mEngine.mMaxRPM=6500;controller->mTransmission.mSwitchTime=.12f;
    controller->mTransmission.mClutchReleaseTime=.12f;controller->mDifferentials.resize(2);
    for(int i=0;i<2;++i){auto& d=controller->mDifferentials[i];d.mLeftWheel=i*2;d.mRightWheel=i*2+1;d.mEngineTorqueRatio=.5f;d.mDifferentialRatio=3.8f;}
    vehicle.mController=controller;vehicle.mAntiRollBars.resize(2);
    for(int i=0;i<2;++i){vehicle.mAntiRollBars[i].mLeftWheel=i*2;vehicle.mAntiRollBars[i].mRightWheel=i*2+1;}
    JPH::Ref<JPH::VehicleConstraint> constraint=new JPH::VehicleConstraint(*body,vehicle);
    constraint->SetVehicleCollisionTester(new JPH::VehicleCollisionTesterCastSphere(1,.1f));
    constraint->SetNumStepsBetweenCollisionTestInactive(50);
    // The default retains Jolt's sample's legacy longitudinal-impulse calibration.
    // The handling fixture can override it to distinguish tire tuning from input bugs.
    // Capture an index: adding more cars can relocate the vehicles vector.
    static_cast<JPH::WheeledVehicleController*>(constraint->GetController())->SetTireMaxImpulseCallback(
      [state=impl.get(),index=impl->vehicles.size(),longitudinalImpulseScale](JPH::uint wheel,float& longitudinal,float& lateral,float suspension,float longitudinalFriction,float lateralFriction,float,float,float){
          const auto& car=state->vehicles.at(index);
          longitudinal=longitudinalImpulseScale*longitudinalFriction*suspension;
          lateral=lateralFriction*suspension*(wheel<2?car.frontLateralGrip:car.rearLateralGrip);
      });
    impl->system.AddConstraint(constraint);impl->system.AddStepListener(constraint);
    impl->vehicles.push_back({body->GetID(),constraint,model});return impl->vehicles.size()-1;
}
size_t Physics::addPedestrian(V3 pos){
    JPH::BodyCreationSettings settings(new JPH::CapsuleShape(.52f,.24f),jv(pos),JPH::Quat::sIdentity(),JPH::EMotionType::Dynamic,1);
    settings.mOverrideMassProperties=JPH::EOverrideMassProperties::CalculateInertia;settings.mMassPropertiesOverride.mMass=72;
    settings.mUserData=1000+impl->people.size();settings.mFriction=.6f;settings.mAngularDamping=.25f;
    auto id=impl->system.GetBodyInterface().CreateAndAddBody(settings,JPH::EActivation::Activate);
    impl->people.push_back({id,false});return impl->people.size()-1;
}
void Physics::drive(size_t index,Input in,bool steeringSpeedLimit,float forwardLimitKmh){
    auto& v=impl->vehicles.at(index);auto& bi=impl->system.GetBodyInterface();
    float speed=(bi.GetRotation(v.body).Conjugated()*bi.GetLinearVelocity(v.body)).GetZ();
    if(in.throttle*speed<-.4f){in.throttle=0;in.brake=1;}
    if(forwardLimitKmh>0&&in.throttle>0)
        in.throttle*=std::clamp((forwardLimitKmh-speed*3.6f)/5.f,0.f,1.f);
    if(std::abs(speed)>42&&in.throttle*speed>0)in.throttle=0;
    if(speed< -9&&in.throttle<0){in.throttle=0;in.brake=.12f;}
    if(std::abs(in.throttle)+std::abs(in.steer)>0||((in.brake+in.handbrake)>0&&std::abs(speed)>.2f))bi.ActivateBody(v.body);
    if(steeringSpeedLimit)in.steer*=std::clamp(1.f-std::abs(speed)/55.f,.3f,1.f);
    // Native +X is screen-right with +Z forward; Jolt's right-input convention is opposite.
    static_cast<JPH::WheeledVehicleController*>(v.constraint->GetController())->SetDriverInput(in.throttle,-in.steer,in.brake,in.handbrake);
}
void Physics::setCarLateralGrip(size_t index,float front,float rear){
    if(!std::isfinite(front)||!std::isfinite(rear)||front<=0||rear<=0)throw std::runtime_error("Lateral grip scales must be finite and positive");
    auto& car=impl->vehicles.at(index);car.frontLateralGrip=front;car.rearLateralGrip=rear;
}
void Physics::setCarFrontDrive(size_t index,float frontShare){
    if(!std::isfinite(frontShare)||frontShare<0||frontShare>1)throw std::runtime_error("Front drive share must be in [0, 1]");
    auto& car=impl->vehicles.at(index);
    auto& differentials=static_cast<JPH::WheeledVehicleController*>(car.constraint->GetController())->GetDifferentials();
    for(int axle=0;axle<2;++axle){
        auto& differential=differentials.at(axle);const float share=axle==0?frontShare:1-frontShare;
        differential.mEngineTorqueRatio=share;
        // Zero torque alone still lets Jolt's center LSD feed the nominally idle axle.
        differential.mLeftWheel=share>0?axle*2:-1;
        differential.mRightWheel=share>0?axle*2+1:-1;
    }
}
void Physics::step(float dt){impl->collisionEvents.take();auto errors=impl->system.Update(dt,1,&impl->temporary,&impl->jobs);if(errors!=JPH::EPhysicsUpdateError::None)throw std::runtime_error("Physics capacity exceeded");impl->collisionEvents.step(dt);}
std::vector<CollisionEvent> Physics::takeCollisions(){return impl->collisionEvents.take();}
void Physics::setCarEnabled(size_t index,bool enabled){
    auto& v=impl->vehicles.at(index);if(v.enabled==enabled)return;
    auto& bi=impl->system.GetBodyInterface();
    if(enabled){bi.AddBody(v.body,JPH::EActivation::Activate);impl->system.AddConstraint(v.constraint);impl->system.AddStepListener(v.constraint);}
    else {impl->system.RemoveStepListener(v.constraint);impl->system.RemoveConstraint(v.constraint);bi.RemoveBody(v.body);}
    v.enabled=enabled;
}
void Physics::setCarDetail(size_t index,float distance){
    impl->vehicles.at(index).constraint->SetNumStepsBetweenCollisionTestActive(distance<40?1:distance<100?2:4);
}
void Physics::resetCar(size_t index,V3 p,float heading){
    auto& v=impl->vehicles.at(index);auto& bi=impl->system.GetBodyInterface();bi.SetPositionAndRotation(v.body,jv(p),JPH::Quat::sRotation(JPH::Vec3::sAxisY(),heading),JPH::EActivation::Activate);
    bi.SetLinearAndAngularVelocity(v.body,JPH::Vec3::sZero(),JPH::Vec3::sZero());
    for(auto* w:v.constraint->GetWheels()){w->SetAngularVelocity(0);}
    drive(index,{});
}
CarPose Physics::car(size_t index)const{
    const auto& v=impl->vehicles.at(index);const auto& bi=impl->system.GetBodyInterface();CarPose out;
    out.body=matrix(bi.GetWorldTransform(v.body));out.velocity=v3(bi.GetLinearVelocity(v.body));
    auto rotation=bi.GetRotation(v.body);out.speed=(rotation.Conjugated()*bi.GetLinearVelocity(v.body)).GetZ()*3.6f;
    out.lateralSpeed=(rotation.Conjugated()*bi.GetLinearVelocity(v.body)).GetX();out.yawRate=bi.GetAngularVelocity(v.body).GetY();
    out.steerAngle=v.constraint->GetWheel(0)->GetSteerAngle();
    const auto* controller=static_cast<const JPH::WheeledVehicleController*>(v.constraint->GetController());
    out.engineRPM=controller->GetEngine().GetCurrentRPM();out.engineThrottle=controller->GetForwardInput();
    for(int i=0;i<4;++i){float slip=static_cast<const JPH::WheelWV*>(v.constraint->GetWheel(i))->mLateralSlip;
        (i<2?out.frontSlipAngle:out.rearSlipAngle)+=slip*.5f;}
    auto forward=rotation.RotateAxisZ();out.heading=std::atan2(forward.GetX(),forward.GetZ());
    const auto linear=bi.GetLinearVelocity(v.body),angular=bi.GetAngularVelocity(v.body);
    const auto center=bi.GetCenterOfMassPosition(v.body);
    for(int i=0;i<4;++i){const auto* wheel=v.constraint->GetWheel(i);
        out.wheels[i]=matrix(v.constraint->GetWheelWorldTransform(i,JPH::Vec3::sAxisX(),JPH::Vec3::sAxisY()));out.groundedWheels+=wheel->HasContact();out.wheelAngularSpeed[i]=wheel->GetAngularVelocity();
        if(wheel->HasContact()){
            ++(i<2?out.groundedFrontWheels:out.groundedRearWheels);
            const auto relative=linear+angular.Cross(JPH::Vec3(wheel->GetContactPosition()-center))-wheel->GetContactPointVelocity();
            const float lateral=std::abs(relative.Dot(wheel->GetContactLateral()));
            auto& axleSlip=i<2?out.frontLateralSlipSpeed:out.rearLateralSlipSpeed;
            axleSlip=std::max(axleSlip,lateral);
        }
    }
    return out;
}
Matrix Physics::pedestrian(size_t i)const{return matrix(impl->system.GetBodyInterface().GetWorldTransform(impl->people.at(i).body));}
V3 Physics::carPosition(size_t i)const{return v3(impl->system.GetBodyInterface().GetPosition(impl->vehicles.at(i).body));}
bool Physics::pedestrianFallen(size_t i)const{return impl->people.at(i).fallen;}
void Physics::walk(size_t i,V3 velocity){
    auto& p=impl->people.at(i);if(p.fallen)return;auto& bi=impl->system.GetBodyInterface();
    auto current=bi.GetLinearVelocity(p.body);bi.SetLinearVelocity(p.body,JPH::Vec3(velocity.x,current.GetY(),velocity.z));
    bi.SetRotation(p.body,JPH::Quat::sRotation(JPH::Vec3::sAxisY(),std::atan2(velocity.x,velocity.z)),JPH::EActivation::DontActivate);bi.SetAngularVelocity(p.body,JPH::Vec3::sZero());
}
void Physics::resetPedestrian(size_t i,V3 p){
    auto& person=impl->people.at(i);auto& bi=impl->system.GetBodyInterface();person.fallen=false;
    bi.SetPositionAndRotation(person.body,jv(p),JPH::Quat::sIdentity(),JPH::EActivation::Activate);
    bi.SetLinearAndAngularVelocity(person.body,JPH::Vec3::sZero(),JPH::Vec3::sZero());
}
float Physics::groundHeight(V3 p,bool streetOnly)const{
    JPH::RRayCast ray(jv(p+V3{0,3,0}),JPH::Vec3(0,-10,0));JPH::RayCastResult hit;GroundFilter filter(streetOnly);
    if(impl->system.GetNarrowPhaseQuery().CastRay(ray,hit,JPH::SpecifiedBroadPhaseLayerFilter(JPH::BroadPhaseLayer(0)),JPH::SpecifiedObjectLayerFilter(0),filter))return p.y+3-hit.mFraction*10;
    return p.y;
}
std::optional<float> Physics::groundSupport(V3 p)const{
    JPH::RRayCast ray(jv(p+V3{0,2,0}),JPH::Vec3(0,-5,0));JPH::RayCastResult hit;GroundFilter filter;
    if(impl->system.GetNarrowPhaseQuery().CastRay(ray,hit,JPH::SpecifiedBroadPhaseLayerFilter(JPH::BroadPhaseLayer(0)),JPH::SpecifiedObjectLayerFilter(0),filter))return p.y+2-hit.mFraction*5;
    return {};
}
bool Physics::clearCarPlacement(size_t index,V3 p,float heading)const{
    auto& bi=impl->system.GetBodyInterface();auto id=impl->vehicles.at(index).body;auto shape=bi.GetShape(id);
    JPH::Quat rotation=JPH::Quat::sRotation(JPH::Vec3::sAxisY(),heading);
    auto center=JPH::RVec3(jv(p)+rotation*shape->GetCenterOfMass());
    JPH::AnyHitCollisionCollector<JPH::CollideShapeCollector> hits;JPH::CollideShapeSettings settings;
    settings.mMaxSeparationDistance=.08f;
    impl->system.GetNarrowPhaseQuery().CollideShape(shape,JPH::Vec3::sReplicate(1),JPH::RMat44::sRotationTranslation(rotation,center),settings,center,hits,{}, {},JPH::IgnoreSingleBodyFilter(id));
    return !hits.HadHit();
}
V3 Physics::cameraCollision(V3 target,V3 desired)const{
    JPH::RRayCast ray(jv(target),jv(desired-target));JPH::RayCastResult result;
    if(impl->system.GetNarrowPhaseQuery().CastRay(ray,result,JPH::SpecifiedBroadPhaseLayerFilter(JPH::BroadPhaseLayer(0)),JPH::SpecifiedObjectLayerFilter(0)))return target+(desired-target)*std::max(.1f,result.mFraction-.035f);
    return desired;
}
void Physics::optimize(){impl->system.OptimizeBroadPhase();}
uint32_t Physics::collisions()const{return impl->contacts;}
