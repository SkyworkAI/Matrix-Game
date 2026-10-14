#include "DemoRecorder.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
}

namespace {
using Json=nlohmann::json;
void check(int code,const char* action){
    if(code<0){char message[AV_ERROR_MAX_STRING_SIZE];av_strerror(code,message,sizeof(message));
        throw std::runtime_error(std::string(action)+": "+message);}
}
class LosslessVideo {
    AVFormatContext* format=nullptr;
    AVCodecContext* codec=nullptr;
    AVFrame* frame=nullptr;
    AVPacket* packet=nullptr;
    AVStream* stream=nullptr;
    int width,height;
    bool finished=false;
    void release(){
        av_packet_free(&packet);av_frame_free(&frame);avcodec_free_context(&codec);
        if(format){if(format->pb)avio_closep(&format->pb);avformat_free_context(format);format=nullptr;}
    }
    void receive(){
        for(;;){int result=avcodec_receive_packet(codec,packet);
            if(result==AVERROR(EAGAIN)||result==AVERROR_EOF)break;
            check(result,"Receive lossless video");
            packet->duration=1;av_packet_rescale_ts(packet,codec->time_base,stream->time_base);
            packet->stream_index=stream->index;check(av_interleaved_write_frame(format,packet),"Write lossless video");
            av_packet_unref(packet);
        }
    }
public:
    LosslessVideo(const std::filesystem::path& path,int w,int h,int fps):width(w),height(h){
        try {
            check(avformat_alloc_output_context2(&format,nullptr,"matroska",path.string().c_str()),"Create recording container");
            const auto* encoder=avcodec_find_encoder(AV_CODEC_ID_FFV1);
            if(!encoder)throw std::runtime_error("FFV1 encoder unavailable");
            codec=avcodec_alloc_context3(encoder);frame=av_frame_alloc();packet=av_packet_alloc();
            if(!codec||!frame||!packet)throw std::bad_alloc();
            codec->width=w;codec->height=h;codec->pix_fmt=AV_PIX_FMT_BGRA;codec->time_base={1,fps};codec->framerate={fps,1};
            codec->thread_count=2;codec->thread_type=FF_THREAD_SLICE;codec->level=3;codec->gop_size=1;
            codec->color_range=AVCOL_RANGE_JPEG;codec->colorspace=AVCOL_SPC_RGB;
            codec->color_primaries=AVCOL_PRI_BT709;codec->color_trc=AVCOL_TRC_IEC61966_2_1;
            if(format->oformat->flags&AVFMT_GLOBALHEADER)codec->flags|=AV_CODEC_FLAG_GLOBAL_HEADER;
            check(avcodec_open2(codec,encoder,nullptr),"Open FFV1 encoder");
            stream=avformat_new_stream(format,nullptr);if(!stream)throw std::bad_alloc();
            stream->time_base=codec->time_base;stream->avg_frame_rate=codec->framerate;
            check(avcodec_parameters_from_context(stream->codecpar,codec),"Copy video parameters");
            check(avio_open(&format->pb,path.string().c_str(),AVIO_FLAG_WRITE),"Open video file");
            check(avformat_write_header(format,nullptr),"Write video header");
            frame->width=w;frame->height=h;frame->format=codec->pix_fmt;
            check(av_frame_get_buffer(frame,32),"Allocate recording frame");
        }catch(...){release();throw;}
    }
    ~LosslessVideo(){release();}
    void write(const std::vector<uint8_t>& rgba,uint64_t index){
        if(rgba.size()!=size_t(width)*height*4)throw std::runtime_error("Invalid offline frame size");
        check(av_frame_make_writable(frame),"Make recording frame writable");
        for(int y=0;y<height;++y){auto* dst=frame->data[0]+y*frame->linesize[0];const auto* src=rgba.data()+size_t(y)*width*4;
            for(int x=0;x<width;++x){dst[x*4]=src[x*4+2];dst[x*4+1]=src[x*4+1];dst[x*4+2]=src[x*4];dst[x*4+3]=src[x*4+3];}}
        frame->pts=int64_t(index);check(avcodec_send_frame(codec,frame),"Encode lossless frame");receive();
    }
    void finish(){if(finished)return;check(avcodec_send_frame(codec,nullptr),"Flush lossless video");receive();
        check(av_write_trailer(format),"Finalize lossless video");check(avio_closep(&format->pb),"Close video file");finished=true;}
};
class WaveFile {
    std::ofstream file;
    uint64_t samples=0;
    void u16(uint16_t n){const char b[]={char(n),char(n>>8)};file.write(b,2);}
    void u32(uint32_t n){const char b[]={char(n),char(n>>8),char(n>>16),char(n>>24)};file.write(b,4);}
    void header(){
        file.seekp(0);file.write("RIFF",4);u32(uint32_t(samples*4+36));file.write("WAVEfmt ",8);u32(16);u16(1);u16(2);
        u32(48000);u32(48000*4);u16(4);u16(16);file.write("data",4);u32(uint32_t(samples*4));
    }
public:
    explicit WaveFile(const std::filesystem::path& path){file.exceptions(std::ios::badbit|std::ios::failbit);file.open(path,std::ios::binary);header();}
    void write(const std::vector<float>& stereo){
        if((samples+stereo.size()/2)*4>UINT32_MAX-36)throw std::runtime_error("WAV recording exceeds 4 GiB");
        std::vector<char> bytes(stereo.size()*2);
        for(size_t i=0;i<stereo.size();++i){const auto value=uint16_t(int16_t(std::lround(std::clamp(stereo[i],-1.f,1.f)*32767)));
            bytes[2*i]=char(value);bytes[2*i+1]=char(value>>8);}
        file.write(bytes.data(),std::streamsize(bytes.size()));
        samples+=stereo.size()/2;
    }
    void finish(){header();file.close();}
};
void writeJson(const std::filesystem::path& path,const Json& value){
    auto temporary=path;temporary+=".tmp";std::ofstream out;out.exceptions(std::ios::badbit|std::ios::failbit);
    out.open(temporary);out<<value.dump(2)<<'\n';out.close();std::filesystem::rename(temporary,path);
}
}

struct DemoRecorder::Impl {
    struct Job {std::vector<uint8_t> scene,ui;Json state;std::vector<float> audio;uint64_t frame;};
    std::filesystem::path path;
    int width,height,fps;
    Json metadata;
    std::ofstream inputs;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Job> queue;
    std::thread worker;
    std::exception_ptr error;
    bool stopping=false,finished=false;
    uint64_t submitted=0,written=0;
    double blockedSeconds=0;
    std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    Impl(const std::filesystem::path& p,int w,int h,int f,Json data):path(p),width(w),height(h),fps(f),metadata(std::move(data)){}
    void stop(){ {std::lock_guard lock(mutex);stopping=true;}changed.notify_all();if(worker.joinable())worker.join(); }
    ~Impl(){stop();}
};
DemoRecorder::DemoRecorder(const std::filesystem::path& path,
    int width,int height,int fps,Json metadata):impl(std::make_unique<Impl>(path,width,height,fps,std::move(metadata))){
    if(width<=0||height<=0||fps<1||fps>120)throw std::invalid_argument("Invalid offline recording format");
    std::filesystem::create_directories(path.parent_path());
    if(!std::filesystem::create_directory(path))throw std::runtime_error("Recording directory already exists: "+path.string());
    auto& i=*impl;
    i.metadata.update({{"schema_version",1},{"status","recording"},{"width",width},{"height",height},{"fps",fps},
        {"frame_index_base",0},{"ui_alpha","premultiplied"},{"codec","ffv1_bgra"},{"dropped_frames",0},
        {"audio",{{"source","combat_event_mixer"},{"sample_rate",48000},{"channels",2}}}});
    writeJson(path/"session.json",i.metadata);
    i.inputs.exceptions(std::ios::badbit|std::ios::failbit);i.inputs.open(path/"inputs.jsonl");
    i.worker=std::thread([this]{auto& i=*impl;try {
        LosslessVideo scene(i.path/"reference.mkv",i.width,i.height,i.fps),ui(i.path/"ui.mkv",i.width,i.height,i.fps);
        WaveFile wave(i.path/"audio.wav");
        std::ofstream states;states.exceptions(std::ios::badbit|std::ios::failbit);states.open(i.path/"frames.jsonl");
        for(;;){Impl::Job job;
            {std::unique_lock lock(i.mutex);i.changed.wait(lock,[&]{return i.stopping||!i.queue.empty();});
                if(i.queue.empty())break;job=std::move(i.queue.front());i.queue.pop_front();}
            i.changed.notify_all();
            if(job.frame!=i.written)throw std::runtime_error("Noncontiguous offline frame");
            scene.write(job.scene,job.frame);ui.write(job.ui,job.frame);
            const uint64_t begin=job.frame*48000/i.fps,end=(job.frame+1)*48000/i.fps;
            if(job.audio.size()!=size_t(end-begin)*2)throw std::runtime_error("Audio block length mismatch");wave.write(job.audio);
            job.state.update({{"frame",job.frame},{"pts_s",double(job.frame)/i.fps},{"audio_sample_start",begin},{"audio_sample_end",end}});
            states<<job.state.dump()<<'\n';++i.written;
        }
        scene.finish();ui.finish();wave.finish();states.close();
    }catch(...){ {std::lock_guard lock(i.mutex);i.error=std::current_exception();i.stopping=true;}i.changed.notify_all();}});
}
DemoRecorder::~DemoRecorder(){if(impl&&!impl->finished){impl->stop();}}
void DemoRecorder::input(Json event){event["next_frame"]=impl->submitted;impl->inputs<<event.dump()<<'\n';}
void DemoRecorder::submit(std::vector<uint8_t> scene,std::vector<uint8_t> ui,Json state,std::vector<float> audio){
    auto& i=*impl;const auto begin=std::chrono::steady_clock::now();
    if(scene.size()!=size_t(i.width)*i.height*4||ui.size()!=scene.size())throw std::invalid_argument("Offline layer size mismatch");
    std::unique_lock lock(i.mutex);i.changed.wait(lock,[&]{return i.queue.size()<2||i.error||i.stopping;});
    i.blockedSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    if(i.error)std::rethrow_exception(i.error);if(i.stopping)throw std::runtime_error("Recorder already stopped");
    i.queue.push_back({std::move(scene),std::move(ui),std::move(state),std::move(audio),i.submitted++});lock.unlock();i.changed.notify_all();
}
void DemoRecorder::finish(const std::string& reason){
    auto& i=*impl;if(i.finished)return;const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-i.started).count();i.stop();i.inputs.close();
    i.metadata.update({{"status",i.error?"failed":"complete"},{"stop_reason",reason},{"frames",i.written},{"submitted_frames",i.submitted},
        {"duration_s",double(i.written)/i.fps},{"audio_samples",i.written*48000/i.fps},
        {"writer_blocked_s",i.blockedSeconds},{"capture_wall_s",elapsed},{"capture_fps",elapsed>0?i.submitted/elapsed:0}});
    if(i.error){try{std::rethrow_exception(i.error);}catch(const std::exception& e){i.metadata["error"]=e.what();}}
    writeJson(i.path/"session.json",i.metadata);i.finished=true;if(i.error)std::rethrow_exception(i.error);
}
uint64_t DemoRecorder::frames()const{return impl->submitted;}
const std::filesystem::path& DemoRecorder::directory()const{return impl->path;}
