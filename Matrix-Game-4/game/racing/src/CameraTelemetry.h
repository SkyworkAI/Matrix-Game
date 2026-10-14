#pragma once
#include "Common.h"
#include <cstddef>
#include <cstring>
#include <numbers>
#include <type_traits>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#endif

struct UeCameraPose {
    std::array<double,3> positionCm,rotationDegrees,forward,right,up;
};

inline UeCameraPose toUeCamera(const Camera& camera) {
    const V3 forward=(camera.target-camera.eye).normalized();
    const V3 right=cross({0,1,0},forward).normalized(),up=cross(forward,right);
    const auto axes=[](V3 v){return std::array<double,3>{v.z,v.x,v.y};};
    UeCameraPose pose;
    // Inverse of native() in tools/bbox-geometry.mjs; the map has no origin offset.
    pose.positionCm={double(camera.eye.z)*100,double(camera.eye.x)*100,double(camera.eye.y)*100};
    pose.forward=axes(forward);pose.right=axes(right);pose.up=axes(up);
    const double degrees=180/std::numbers::pi;
    // UE FVector::Rotation(), with the same world-up view used by XMMatrixLookAtLH.
    pose.rotationDegrees={std::atan2(pose.forward[2],std::hypot(pose.forward[0],pose.forward[1]))*degrees,
        std::atan2(pose.forward[1],pose.forward[0])*degrees,0};
    return pose;
}

struct CameraSample {
    uint64_t frame=0,qpcTicks=0,qpcFrequency=0;
    double elapsedSeconds=0,simulationSeconds=0;
    UeCameraPose pose{};
    double horizontalFov=0;
    uint32_t width=0,height=0,cameraMode=0,flags=0;
};

struct CameraSharedMemory {
    uint32_t magic=0x50435343,version=1,bytes=208,processId=0;
#ifdef _WIN32
    alignas(8) volatile LONG64 sequence=0;
#else
    alignas(8) uint64_t sequence=0;
#endif
    CameraSample sample;
};
static_assert(std::is_trivially_copyable_v<CameraSample>);
static_assert(sizeof(CameraSample)==184&&offsetof(CameraSample,pose)==40);
static_assert(sizeof(CameraSharedMemory)==208&&offsetof(CameraSharedMemory,sample)==24);

#ifdef _WIN32
class CameraTelemetry {
public:
    enum Flags : uint32_t { Running=1,Paused=2,Active=4,LookBack=8 };
    CameraTelemetry() {
        const auto name=L"Local\\CitySampleBBoxCamera_"+std::to_wstring(GetCurrentProcessId());
        mapping=CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,sizeof(CameraSharedMemory),name.c_str());
        if(!mapping)throw std::runtime_error("Cannot create camera shared memory");
        shared=static_cast<CameraSharedMemory*>(MapViewOfFile(mapping,FILE_MAP_ALL_ACCESS,0,0,sizeof(CameraSharedMemory)));
        if(!shared){CloseHandle(mapping);mapping=nullptr;throw std::runtime_error("Cannot map camera shared memory");}
        CameraSharedMemory initial;initial.processId=GetCurrentProcessId();
        std::memcpy(shared,&initial,sizeof(initial));
        LARGE_INTEGER counter;QueryPerformanceFrequency(&counter);frequency=uint64_t(counter.QuadPart);
    }
    CameraTelemetry(const CameraTelemetry&)=delete;
    CameraTelemetry& operator=(const CameraTelemetry&)=delete;
    ~CameraTelemetry() {
        if(shared){InterlockedIncrement64(&shared->sequence);shared->sample.flags=0;
            InterlockedIncrement64(&shared->sequence);UnmapViewOfFile(shared);}
        if(mapping)CloseHandle(mapping);
    }
    void publish(const Camera& camera,int width,int height,int mode,uint32_t flags,double elapsed,double simulation) {
        LARGE_INTEGER counter;QueryPerformanceCounter(&counter);
        CameraSample sample;
        sample.frame=++frame;sample.qpcTicks=uint64_t(counter.QuadPart);sample.qpcFrequency=frequency;
        sample.elapsedSeconds=elapsed;sample.simulationSeconds=simulation;sample.pose=toUeCamera(camera);
        sample.horizontalFov=camera.horizontalFov;sample.width=uint32_t(width);sample.height=uint32_t(height);
        sample.cameraMode=uint32_t(mode);sample.flags=flags|Running;
        // Odd sequence means a write is in progress. Interlocked operations fence both sides.
        InterlockedIncrement64(&shared->sequence);
        std::memcpy(&shared->sample,&sample,sizeof(sample));
        InterlockedIncrement64(&shared->sequence);
    }
private:
    HANDLE mapping=nullptr;
    CameraSharedMemory* shared=nullptr;
    uint64_t frame=0,frequency=0;
};

#else
// POSIX uses the same wire layout. The clock is CLOCK_MONOTONIC in nanoseconds.
class CameraTelemetry {
public:
    enum Flags : uint32_t { Running=1,Paused=2,Active=4,LookBack=8 };
    static std::string mappingName(){return "/CitySampleBBoxCamera_"+std::to_string(getpid());}
    CameraTelemetry(){
        name=mappingName();
        fd=shm_open(name.c_str(),O_CREAT|O_EXCL|O_RDWR,0600);
        if(fd<0)throw std::runtime_error("Cannot create camera shared memory: "+name);
        if(ftruncate(fd,sizeof(CameraSharedMemory))!=0){cleanup();throw std::runtime_error("Cannot size camera shared memory");}
        void* address=mmap(nullptr,sizeof(CameraSharedMemory),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        if(address==MAP_FAILED){cleanup();throw std::runtime_error("Cannot map camera shared memory");}
        shared=static_cast<CameraSharedMemory*>(address);CameraSharedMemory initial;initial.processId=uint32_t(getpid());std::memcpy(shared,&initial,sizeof(initial));
        static_assert(__atomic_always_lock_free(8,nullptr),"Camera IPC requires lock-free 64-bit atomics");
    }
    CameraTelemetry(const CameraTelemetry&)=delete;
    CameraTelemetry& operator=(const CameraTelemetry&)=delete;
    ~CameraTelemetry(){if(shared){__atomic_add_fetch(&shared->sequence,1,__ATOMIC_SEQ_CST);shared->sample.flags=0;__atomic_add_fetch(&shared->sequence,1,__ATOMIC_SEQ_CST);}cleanup();}
    void publish(const Camera& camera,int width,int height,int mode,uint32_t flags,double elapsed,double simulation){
        timespec ts{};if(clock_gettime(CLOCK_MONOTONIC,&ts)!=0)throw std::runtime_error("Cannot read monotonic clock");
        CameraSample sample;sample.frame=++frame;sample.qpcTicks=uint64_t(ts.tv_sec)*1000000000ull+uint64_t(ts.tv_nsec);sample.qpcFrequency=1000000000;
        sample.elapsedSeconds=elapsed;sample.simulationSeconds=simulation;sample.pose=toUeCamera(camera);sample.horizontalFov=camera.horizontalFov;
        sample.width=uint32_t(width);sample.height=uint32_t(height);sample.cameraMode=uint32_t(mode);sample.flags=flags|Running;
        __atomic_add_fetch(&shared->sequence,1,__ATOMIC_SEQ_CST);std::memcpy(&shared->sample,&sample,sizeof(sample));__atomic_add_fetch(&shared->sequence,1,__ATOMIC_SEQ_CST);
    }
private:
    int fd=-1;std::string name;CameraSharedMemory* shared=nullptr;uint64_t frame=0;
    void cleanup(){if(shared){munmap(shared,sizeof(CameraSharedMemory));shared=nullptr;}if(fd>=0){close(fd);fd=-1;shm_unlink(name.c_str());}}
};
#endif
