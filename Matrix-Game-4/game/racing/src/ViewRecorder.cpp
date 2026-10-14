#include "ViewRecorder.h"
#include "VideoEncoder.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

struct ViewRecorder::Impl {
    struct Job {std::vector<uint8_t> rgba;int64_t timestamp=0;};
    std::mutex mutex;std::condition_variable cv;
    std::optional<Job> pending;bool stopping=false;
    std::vector<std::vector<uint8_t>> reusable;
    std::ofstream file;
    std::atomic<uint64_t> encodedFrames{0},droppedFrames{0};
    int64_t firstStamp=-1,lastStamp=0;
    std::thread worker;

    ~Impl(){
        {std::lock_guard lock(mutex);stopping=true;}
        cv.notify_one();
        if(worker.joinable())worker.join();
    }
};

ViewRecorder::ViewRecorder(const std::filesystem::path& path,int width,int height,int fps,int bitrateKbps,int threads):impl(std::make_unique<Impl>()){
    if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());
    impl->file.open(path,std::ios::binary|std::ios::trunc);
    if(!impl->file)throw std::runtime_error("Cannot open recording file: "+path.string());
    impl->worker=std::thread([this,width,height,fps,bitrateKbps,threads]{
        try{
            VideoEncoder encoder(width,height,fps,"h264",threads,"ultrafast");encoder.setBitrate(bitrateKbps);
            for(;;){
                Impl::Job job;
                {std::unique_lock lock(impl->mutex);
                    impl->cv.wait(lock,[&]{return impl->stopping||impl->pending.has_value();});
                    if(!impl->pending)return;
                    job=std::move(*impl->pending);impl->pending.reset();}
                auto output=encoder.encode(job.rgba,job.timestamp,false);
                if(impl->firstStamp<0)impl->firstStamp=job.timestamp;impl->lastStamp=job.timestamp;
                impl->file.write(reinterpret_cast<const char*>(output.bytes.data()),std::streamsize(output.bytes.size()));
                impl->file.flush();
                {std::lock_guard lock(impl->mutex);if(impl->reusable.size()<2)impl->reusable.push_back(std::move(job.rgba));}
                impl->encodedFrames.fetch_add(1);
            }
        }catch(...){impl->file.flush();throw;}
    });
}

ViewRecorder::~ViewRecorder(){finish();}

std::vector<uint8_t> ViewRecorder::takeReusable(){
    std::lock_guard lock(impl->mutex);
    if(impl->reusable.empty())return {};
    auto buffer=std::move(impl->reusable.back());impl->reusable.pop_back();return buffer;
}

void ViewRecorder::submit(std::vector<uint8_t>&& rgba,int64_t timestampMicroseconds){
    // Publish the job only once it is complete: the worker must never observe a
    // half-filled frame, and it may already be awake from the previous submit.
    std::lock_guard lock(impl->mutex);
    if(impl->stopping||impl->pending){impl->droppedFrames.fetch_add(1);return;}
    impl->pending=Impl::Job{std::move(rgba),timestampMicroseconds};
    impl->cv.notify_one();
}

void ViewRecorder::finish(){
    if(!impl)return;
    {std::lock_guard lock(impl->mutex);impl->stopping=true;}
    impl->cv.notify_one();
    if(impl->worker.joinable())impl->worker.join();
    impl->file.flush();
}

uint64_t ViewRecorder::encoded()const{return impl->encodedFrames.load();}
uint64_t ViewRecorder::dropped()const{return impl->droppedFrames.load();}
double ViewRecorder::spanSeconds()const{return impl->lastStamp>impl->firstStamp?double(impl->lastStamp-impl->firstStamp)/1e6:0;}
