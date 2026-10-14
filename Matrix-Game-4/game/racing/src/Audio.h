#pragma once
#include <Windows.h>
#include <mmsystem.h>
#include <array>
#include <atomic>
#include <cmath>
#include <thread>

class Audio {
public:
    std::atomic<float> speed{0},throttle{0};
    std::atomic<bool> muted{false};
    Audio(){
        WAVEFORMATEX format{};format.wFormatTag=WAVE_FORMAT_PCM;format.nChannels=1;format.nSamplesPerSec=48000;
        format.wBitsPerSample=16;format.nBlockAlign=2;format.nAvgBytesPerSec=96000;
        if(waveOutOpen(&output,WAVE_MAPPER,&format,0,0,CALLBACK_NULL)!=MMSYSERR_NOERROR){output=nullptr;return;}
        for(auto& b:buffers){b.header.lpData=reinterpret_cast<LPSTR>(b.samples.data());b.header.dwBufferLength=DWORD(b.samples.size()*2);waveOutPrepareHeader(output,&b.header,sizeof(WAVEHDR));}
        running=true;thread=std::thread([this]{run();});
    }
    ~Audio(){running=false;if(thread.joinable())thread.join();if(output){waveOutReset(output);for(auto& b:buffers)waveOutUnprepareHeader(output,&b.header,sizeof(WAVEHDR));waveOutClose(output);}}
private:
    struct Buffer {std::array<int16_t,512> samples{};WAVEHDR header{};bool queued=false;};
    std::array<Buffer,3> buffers;
    HWAVEOUT output=nullptr;
    std::atomic<bool> running{false};std::thread thread;
    void run(){double phase=0;float frequency=35,gain=0;
        while(running){bool wrote=false;
            for(auto& b:buffers)if(!b.queued||(b.header.dwFlags&WHDR_DONE)){
                float target=35+std::fmod(std::abs(speed.load())*1.5f,80.f)+std::abs(throttle.load())*14;
                for(auto& sample:b.samples){frequency+=(target-frequency)*.003f;gain+=((muted?0.f:.025f+.025f*std::abs(throttle.load()))-gain)*.002f;
                    phase+=frequency*6.28318530718/48000;if(phase>6.28318530718)phase-=6.28318530718;
                    sample=int16_t((std::sin(phase)+.34*std::sin(phase*2)+.12*std::sin(phase*5))*gain*22000);}
                waveOutWrite(output,&b.header,sizeof(WAVEHDR));b.queued=true;wrote=true;
            }
            if(!wrote)Sleep(2);
        }
    }
};
