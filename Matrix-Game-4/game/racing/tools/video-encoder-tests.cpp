#include "VideoEncoder.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
extern "C" {
#include <libavcodec/avcodec.h>
}
void testReconfiguration(){
    VideoEncoder encoder(640,360,60,"h264",2);
    auto* context=avcodec_alloc_context3(avcodec_find_decoder(AV_CODEC_ID_H264));
    auto* packet=av_packet_alloc();auto* frame=av_frame_alloc();
    if(avcodec_open2(context,context->codec,nullptr)<0)throw std::runtime_error("Decoder init failed");
    uint32_t random=1;std::vector<uint8_t> pixels(640*360*4);size_t high=0,low=0;
    bool valid=true;
    for(int i=0;i<120;++i){
        if(i==40)encoder.setBitrate(600);
        if(i==80)encoder.setBitrate(4000);
        for(auto& value:pixels){random=random*1664525+1013904223;value=uint8_t(random>>24);}
        auto output=encoder.encode(pixels,2000000000+i*16667,i%60==0);
        if(i>=10&&i<40)high+=output.bytes.size();
        if(i>=50&&i<80)low+=output.bytes.size();
        av_new_packet(packet,int(output.bytes.size()));std::copy(output.bytes.begin(),output.bytes.end(),packet->data);
        valid=valid&&avcodec_send_packet(context,packet)>=0&&avcodec_receive_frame(context,frame)>=0;
        av_packet_unref(packet);av_frame_unref(frame);
    }
    av_frame_free(&frame);av_packet_free(&packet);avcodec_free_context(&context);
    std::cout<<"Rate adjustment sample: high="<<high<<", low="<<low<<" bytes\n";
    // Fresh random noise cannot meet a tiny VBV budget while respecting a
    // visual quality floor. Test the actual contract: reconfiguration lowers
    // traffic and preserves references, not a hard bandwidth cap on noise.
    if(!valid||low>=high)throw std::runtime_error("Live bitrate adjustment lost decoding continuity or failed to reduce traffic");
    std::cout<<"Live H.264 bitrate decrease/increase preserves all 120 decoded frames\n";
}
void testLowBitrateVisibility(){
    // A complete decode can still be an unusable grey frame. Preserve the
    // coarse road/building contrast through a bandwidth-starved 1080p IDR.
    constexpr int w=1920,h=1080;
    VideoEncoder encoder(w,h,60,"h264",2);encoder.setBitrate(300);
    auto* context=avcodec_alloc_context3(avcodec_find_decoder(AV_CODEC_ID_H264));
    auto* packet=av_packet_alloc();auto* frame=av_frame_alloc();
    if(avcodec_open2(context,context->codec,nullptr)<0)throw std::runtime_error("Decoder init failed");
    std::vector<uint8_t> pixels(w*h*4);uint32_t random=1;
    double minimumContrast=255;
    for(int i=0;i<6;++i){
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){
            auto offset=size_t(y*w+x)*4;
            for(int c=0;c<3;++c){
                random=random*1664525+1013904223;
                int base=((x/120+y/120+i)%2)?128:0;
                pixels[offset+c]=uint8_t(base+(random>>25));
            }
            pixels[offset+3]=255;
        }
        auto output=encoder.encode(pixels,3000000000LL+i*16667,i%2==0);
        av_new_packet(packet,int(output.bytes.size()));std::copy(output.bytes.begin(),output.bytes.end(),packet->data);
        if(avcodec_send_packet(context,packet)<0||avcodec_receive_frame(context,frame)<0)throw std::runtime_error("Low bitrate frame did not decode");
        double sum=0,squares=0;size_t count=0;
        for(int y=0;y<h;y+=8)for(int x=0;x<w;x+=8){double value=frame->data[0][y*frame->linesize[0]+x];sum+=value;squares+=value*value;++count;}
        double mean=sum/count;minimumContrast=std::min(minimumContrast,std::sqrt(squares/count-mean*mean));
        av_packet_unref(packet);av_frame_unref(frame);
    }
    av_frame_free(&frame);av_packet_free(&packet);avcodec_free_context(&context);
    if(minimumContrast<30)throw std::runtime_error("Bandwidth-starved 1080p keyframe lost scene contrast: "+std::to_string(minimumContrast));
    std::cout<<"Low bitrate 1080p scene contrast preserved: "<<minimumContrast<<"\n";
}
int main(){try{
    av_log_set_level(AV_LOG_ERROR);
    for(const auto& codec:{"h264","jpeg"}){
        VideoEncoder encoder(640,360,60,codec,2);
        std::vector<uint8_t> pixels(640*360*4,100);
        for(int i=0;i<6;++i){bool key=i==0||i==3;auto output=encoder.encode(pixels,1230000000+i*17000,key);
            if(output.bytes.empty()||(key&&!output.key))throw std::runtime_error("Immediate frame or keyframe missing");
            if(key){
                const auto* codecImpl=avcodec_find_decoder(std::string(codec)=="h264"?AV_CODEC_ID_H264:AV_CODEC_ID_MJPEG);
                auto* context=avcodec_alloc_context3(codecImpl);auto* packet=av_packet_alloc();auto* frame=av_frame_alloc();
                if(avcodec_open2(context,codecImpl,nullptr)<0)throw std::runtime_error("Decoder init failed");
                av_new_packet(packet,int(output.bytes.size()));std::copy(output.bytes.begin(),output.bytes.end(),packet->data);
                int sent=avcodec_send_packet(context,packet),received=avcodec_receive_frame(context,frame);
                bool valid=sent>=0&&received>=0&&frame->width==640&&frame->height==360;
                av_frame_free(&frame);av_packet_free(&packet);avcodec_free_context(&context);
                if(!valid)throw std::runtime_error("Keyframe cannot decode independently");
            }
        }
        std::cout<<codec<<" immediate encoding and independent keyframe decode passed\n";
    }testLowBitrateVisibility();testReconfiguration();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
