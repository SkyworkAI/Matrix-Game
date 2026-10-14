#include "OfflineRecorder.h"
#include <chrono>
#include <fstream>
#include <iostream>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
void verifyVideo(const std::filesystem::path& path,bool ui){
    AVFormatContext* format=nullptr;AVCodecContext* codec=nullptr;AVFrame* frame=av_frame_alloc();AVPacket* packet=av_packet_alloc();
    struct Cleanup {AVFormatContext*& f;AVCodecContext*& c;AVFrame*& v;AVPacket*& p;
        ~Cleanup(){av_packet_free(&p);av_frame_free(&v);avcodec_free_context(&c);avformat_close_input(&f);}} cleanup{format,codec,frame,packet};
    require(avformat_open_input(&format,path.string().c_str(),nullptr,nullptr)>=0,"Cannot read encoded take");
    require(avformat_find_stream_info(format,nullptr)>=0,"Cannot read stream info");
    codec=avcodec_alloc_context3(avcodec_find_decoder(AV_CODEC_ID_FFV1));require(codec,"Missing FFV1 decoder");
    require(avcodec_parameters_to_context(codec,format->streams[0]->codecpar)>=0&&avcodec_open2(codec,nullptr,nullptr)>=0,"Cannot open decoder");
    int count=0;
    auto receive=[&](){for(;;){int result=avcodec_receive_frame(codec,frame);if(result==AVERROR(EAGAIN)||result==AVERROR_EOF)break;
        require(result>=0&&frame->format==AV_PIX_FMT_BGRA&&frame->width==64&&frame->height==36,"Wrong decoded format");
        for(int y=0;y<36;++y)for(int x=0;x<64;++x){const auto* p=frame->data[0]+y*frame->linesize[0]+x*4;const size_t i=size_t(y*64+x)*4;
            require(ui?(p[0]==8&&p[1]==16&&p[2]==32&&p[3]==64):
                (p[0]==uint8_t(i/256)&&p[1]==uint8_t(i/4%256)&&p[2]==uint8_t(count)&&p[3]==255),"Lossless RGBA roundtrip failed");}
        ++count;}};
    while(av_read_frame(format,packet)>=0){if(packet->stream_index==0){require(avcodec_send_packet(codec,packet)>=0,"Cannot decode packet");receive();}av_packet_unref(packet);}
    require(avcodec_send_packet(codec,nullptr)>=0,"Cannot flush decoder");receive();require(count==37,"Wrong decoded frame count");
}
int main(int argc,char** argv){try{
    require(argc==3,"Usage: OfflineRecorderTests AUDIO_ASSETS OUTPUT_ROOT");
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path=std::filesystem::path(argv[2])/std::to_string(stamp);
    OfflineRecorder recorder(path,argv[1],64,36,59,.6,{{"test",true}});
    bool rejected=false;try{OfflineRecorder duplicate(path,argv[1],64,36,59,.6,{});}catch(const std::exception&){rejected=true;}
    require(rejected,"Existing take was overwritten");
    rejected=false;try{recorder.submit({}, {}, {}, {});}catch(const std::exception&){rejected=true;}
    require(rejected&&recorder.frames()==0,"Bad frame accepted or consumed a frame index");
    for(uint64_t f=0;f<37;++f){
        std::vector<uint8_t> scene(64*36*4),ui(scene.size());
        for(size_t i=0;i<scene.size();i+=4){
            scene[i]=uint8_t(f);scene[i+1]=uint8_t(i/4%256);scene[i+2]=uint8_t(i/256);scene[i+3]=255;
            ui[i]=32;ui[i+1]=16;ui[i+2]=8;ui[i+3]=64;
        }
        VehicleAudioFrame audio;audio.frame=f+1;audio.rpm=2000;audio.checkpoint=f>=20?1:0;
        recorder.input({{"event","test"},{"id",f}});
        recorder.submit(std::move(scene),std::move(ui),{{"fixture",f}},audio);
    }
    recorder.finish();recorder.finish();require(recorder.frames()==37,"Dropped offline frames");
    nlohmann::json meta;std::ifstream(path/"session.json")>>meta;
    require(meta.at("status")=="complete"&&meta.at("frames")==37,"Incomplete manifest");
    require(meta.at("audio_samples")==37*48000/59,"Fractional sample rate drift");
    std::ifstream rows(path/"frames.jsonl");std::string line;uint64_t frame=0;
    while(std::getline(rows,line)){auto row=nlohmann::json::parse(line);require(row.at("frame")==frame,"Frame IDs out of order");
        require(row.at("audio_sample_start")==frame*48000/59,"Audio interval start drift");
        require(row.at("audio_sample_end")==++frame*48000/59,"Audio interval end drift");}
    require(frame==37,"Missing frame metadata");
    verifyVideo(path/"reference.mkv",false);verifyVideo(path/"ui.mkv",true);
    rejected=false;try{std::vector<uint8_t> rgba(64*36*4);recorder.submit(rgba,rgba,{},{});}catch(const std::exception&){rejected=true;}
    require(rejected,"Submit after finish accepted");
    std::cout<<"PASS: exact frame/sample counts, bounded no-drop writes, invalid input, no overwrite, idempotent finish\n"<<path<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
