#include "VideoEncoder.h"
#include <algorithm>
#include <stdexcept>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
namespace {
void checked(int code,const char* action){if(code<0){char message[AV_ERROR_MAX_STRING_SIZE];av_strerror(code,message,sizeof(message));throw std::runtime_error(std::string(action)+": "+message);}}
}
struct VideoEncoder::Impl {
    AVCodecContext* context=nullptr;AVFrame* frame=nullptr;AVPacket* packet=nullptr;SwsContext* converter=nullptr;
    int width=0,height=0;
    ~Impl(){if(converter)sws_freeContext(converter);av_packet_free(&packet);av_frame_free(&frame);avcodec_free_context(&context);}
};
VideoEncoder::VideoEncoder(int w,int h,int fps,const std::string& codec,int threads,const std::string& preset):impl(std::make_unique<Impl>()){
    bool jpeg=codec=="jpeg";const AVCodec* encoder=avcodec_find_encoder_by_name(jpeg?"mjpeg":"libx264");
    if(!encoder)throw std::runtime_error("Required encoder unavailable: "+codec);
    impl->width=w;impl->height=h;auto* c=impl->context=avcodec_alloc_context3(encoder);if(!c)throw std::bad_alloc();
    c->width=w;c->height=h;c->time_base={1,1000000};c->framerate={fps,1};c->pix_fmt=jpeg?AV_PIX_FMT_YUVJ420P:AV_PIX_FMT_YUV420P;
    c->thread_count=jpeg?1:threads;c->thread_type=FF_THREAD_SLICE;c->max_b_frames=0;c->gop_size=fps;c->bit_rate=8000000;
    if(!jpeg){c->rc_max_rate=8000000;c->rc_buffer_size=800000;}
    c->color_range=jpeg?AVCOL_RANGE_JPEG:AVCOL_RANGE_MPEG;
    c->colorspace=AVCOL_SPC_BT709;c->color_primaries=AVCOL_PRI_BT709;c->color_trc=AVCOL_TRC_BT709;
    if(jpeg){c->flags|=AV_CODEC_FLAG_QSCALE;c->global_quality=FF_QP2LAMBDA*5;}
    else {
        // Better motion prediction preserves road/car detail at the smaller
        // per-view budgets used by the five-camera browser page.
        checked(av_opt_set(c->priv_data,"preset",preset.c_str(),0),"x264 preset");checked(av_opt_set(c->priv_data,"tune","zerolatency",0),"x264 tune");
        checked(av_opt_set(c->priv_data,"profile","baseline",0),"x264 profile");checked(av_opt_set(c->priv_data,"crf","22",0),"x264 quality");
        // VFR lookahead buffers a frame even with tune=zerolatency. Media
        // timestamps still come from the snapshot; rate control uses CFR.
        // QP > 51 enables emergency denoising that can replace a 1080p
        // scene with grey slices under a small VBV budget. Preserve coarse
        // scene content; encoder credits bound the resulting packet backlog.
        checked(av_opt_set(c->priv_data,"x264-params","repeat-headers=1:annexb=1:scenecut=0:force-cfr=1:sync-lookahead=0:rc-lookahead=0:sliced-threads=1:qpmax=51",0),"x264 stream settings");
        checked(av_opt_set(c->priv_data,"forced-idr","1",0),"x264 IDR");
    }
    checked(avcodec_open2(c,encoder,nullptr),"Open video encoder");
    impl->frame=av_frame_alloc();impl->packet=av_packet_alloc();if(!impl->frame||!impl->packet)throw std::bad_alloc();
    impl->frame->width=w;impl->frame->height=h;impl->frame->format=c->pix_fmt;impl->frame->quality=c->global_quality;
    checked(av_frame_get_buffer(impl->frame,32),"Allocate YUV frame");
    impl->converter=sws_getContext(w,h,AV_PIX_FMT_RGBA,w,h,c->pix_fmt,SWS_FAST_BILINEAR,nullptr,nullptr,nullptr);
    if(!impl->converter)throw std::runtime_error("Create color converter");
    sws_setColorspaceDetails(impl->converter,sws_getCoefficients(SWS_CS_ITU709),1,sws_getCoefficients(SWS_CS_ITU709),jpeg?1:0,0,1<<16,1<<16);
}
VideoEncoder::~VideoEncoder()=default;
void VideoEncoder::setBitrate(int kbps){
    if(kbps<300||kbps>24000)throw std::runtime_error("Invalid video bitrate");
    auto* c=impl->context;
    // libavcodec's x264 wrapper reconfigures VBV on the next encode. Keep
    // reference frames intact: changing bandwidth never restarts the codec.
    if(c->codec_id==AV_CODEC_ID_H264){c->bit_rate=int64_t(kbps)*1000;c->rc_max_rate=c->bit_rate;c->rc_buffer_size=kbps*100;}
}
EncodedFrame VideoEncoder::encode(const std::vector<uint8_t>& rgba,int64_t timestamp,bool key){
    auto& i=*impl;if(rgba.size()!=size_t(i.width)*i.height*4)throw std::runtime_error("Invalid encoder input size");
    checked(av_frame_make_writable(i.frame),"Make video frame writable");const uint8_t* data[]={rgba.data()};int lines[]={i.width*4};
    sws_scale(i.converter,data,lines,0,i.height,i.frame->data,i.frame->linesize);i.frame->pts=timestamp;
    i.frame->pict_type=key?AV_PICTURE_TYPE_I:AV_PICTURE_TYPE_NONE;
    checked(avcodec_send_frame(i.context,i.frame),"Encode video frame");EncodedFrame result;
    for(;;){int code=avcodec_receive_packet(i.context,i.packet);if(code==AVERROR(EAGAIN)||code==AVERROR_EOF)break;checked(code,"Receive video frame");
        result.key=result.key||bool(i.packet->flags&AV_PKT_FLAG_KEY);result.bytes.insert(result.bytes.end(),i.packet->data,i.packet->data+i.packet->size);av_packet_unref(i.packet);
    }
    if(result.bytes.empty())throw std::runtime_error("Encoder delayed a zero-latency frame");return result;
}
