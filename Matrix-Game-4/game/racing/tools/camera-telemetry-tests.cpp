#include "../src/CameraTelemetry.h"
#include <iostream>

namespace {
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void expectNear(double value,double expected,double tolerance=1e-6){require(std::abs(value-expected)<=tolerance,"Camera coordinate or rotation mismatch");}
void vectorNear(const std::array<double,3>& actual,const std::array<double,3>& expected){for(size_t i=0;i<3;++i)expectNear(actual[i],expected[i]);}
}

int main(){
    try{
        struct Fixture {V3 direction;double pitch,yaw;};
        const std::array<Fixture,8> fixtures{{
            {{0,0,1},0,0},{{1,0,0},0,90},{{0,0,-1},0,180},{{-1,0,0},0,-90},
            {{0,1,1},45,0},{{0,-1,1},-45,0},{{1,1,0},45,90},{{-1,-1,0},-45,-90}}};
        Camera camera;camera.eye={1.25f,-2.5f,3.75f};
        for(const auto& fixture:fixtures){
            camera.target=camera.eye+fixture.direction;
            const auto pose=toUeCamera(camera);
            vectorNear(pose.positionCm,{375,125,-250});
            expectNear(pose.rotationDegrees[0],fixture.pitch);expectNear(pose.rotationDegrees[1],fixture.yaw);expectNear(pose.rotationDegrees[2],0);
            const double pitch=pose.rotationDegrees[0]*std::numbers::pi/180,yaw=pose.rotationDegrees[1]*std::numbers::pi/180;
            const double sp=std::sin(pitch),cp=std::cos(pitch),sy=std::sin(yaw),cy=std::cos(yaw);
            // Rows of UE's FRotationTranslationMatrix for Roll=0.
            vectorNear(pose.forward,{cp*cy,cp*sy,sp});
            vectorNear(pose.right,{-sy,cy,0});
            vectorNear(pose.up,{-sp*cy,-sp*sy,cp});
        }
#ifdef _WIN32
        HANDLE mapping=nullptr;
#else
        int mapping=-1;
#endif
        const CameraSharedMemory* shared=nullptr;
        {
            CameraTelemetry telemetry;
#ifdef _WIN32
            const auto name=L"Local\\CitySampleBBoxCamera_"+std::to_wstring(GetCurrentProcessId());
            mapping=OpenFileMappingW(FILE_MAP_READ,FALSE,name.c_str());require(mapping!=nullptr,"Cannot open camera mapping");
            shared=static_cast<const CameraSharedMemory*>(MapViewOfFile(mapping,FILE_MAP_READ,0,0,sizeof(CameraSharedMemory)));
            require(shared!=nullptr,"Cannot read camera mapping");
#else
            mapping=shm_open(CameraTelemetry::mappingName().c_str(),O_RDONLY,0);require(mapping>=0,"Cannot open camera mapping");
            shared=static_cast<const CameraSharedMemory*>(mmap(nullptr,sizeof(CameraSharedMemory),PROT_READ,MAP_SHARED,mapping,0));
            require(shared!=MAP_FAILED,"Cannot read camera mapping");
#endif
            telemetry.publish(camera,1920,1080,1,CameraTelemetry::Active,2,1.8);
            require(shared->magic==0x50435343&&shared->version==1&&shared->bytes==208,"Camera protocol header mismatch");
            require(shared->processId==
#ifdef _WIN32
                GetCurrentProcessId()
#else
                uint32_t(getpid())
#endif
                &&shared->sequence==2,"Camera sequence mismatch");
            require(shared->sample.frame==1&&shared->sample.qpcTicks>0&&shared->sample.qpcFrequency>0,"Camera timestamp missing");
            vectorNear(shared->sample.pose.positionCm,{375,125,-250});
            expectNear(shared->sample.elapsedSeconds,2);expectNear(shared->sample.simulationSeconds,1.8);
            require(shared->sample.width==1920&&shared->sample.height==1080&&shared->sample.flags==5,"Camera viewport or flags mismatch");
            telemetry.publish(camera,1280,720,2,CameraTelemetry::Paused|CameraTelemetry::LookBack,3,1.8);
            require(shared->sample.frame==2&&shared->sequence==4&&shared->sample.flags==11,"Camera update missing");
            require(shared->sample.width==1280&&shared->sample.height==720&&shared->sample.cameraMode==2,"Camera resize or mode missing");
        }
        require(shared->sample.flags==0&&shared->sequence==6,"Closed camera stream remained live");
#ifdef _WIN32
        UnmapViewOfFile(shared);CloseHandle(mapping);
#else
        munmap(const_cast<CameraSharedMemory*>(shared),sizeof(CameraSharedMemory));close(mapping);
        require(shm_open(CameraTelemetry::mappingName().c_str(),O_RDONLY,0)<0,"Closed stream name was not unlinked");
#endif
        std::cout<<"PASS: 8 UE orientation fixtures, world centimeters, render basis, IPC layout, updates, and shutdown.\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
