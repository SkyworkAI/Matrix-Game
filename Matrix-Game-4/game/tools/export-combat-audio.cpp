#include "CombatAudio.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

void little(std::ostream& out,uint32_t value,int bytes){
    for(int i=0;i<bytes;++i)out.put(char((value>>(i*8))&255));
}
int main(int argc,char** argv){
    try{
        if(argc!=2)throw std::runtime_error("Usage: ExportCombatAudio OUTPUT_DIRECTORY");
        std::filesystem::create_directories(argv[1]);CombatAudio audio;
        for(int i=0;i<int(CombatSound::Count);++i){
            const auto& wave=audio.waveform(CombatSound(i));
            std::ofstream out(std::filesystem::path(argv[1])/(std::to_string(i)+".wav"),std::ios::binary);
            const auto bytes=uint32_t(wave.size()*2);
            out.write("RIFF",4);little(out,36+bytes,4);out.write("WAVEfmt ",8);little(out,16,4);
            little(out,1,2);little(out,1,2);little(out,48000,4);little(out,96000,4);little(out,2,2);little(out,16,2);
            out.write("data",4);little(out,bytes,4);
            // Preserve headroom in PCM16; the browser restores this scale before mixing.
            for(float value:wave){
                if(std::abs(value)>=4)throw std::runtime_error("Sound exceeds PCM headroom");
                little(out,uint16_t(int16_t(std::lround(value*.25f*32767))),2);
            }
            if(!out)throw std::runtime_error("Could not export sound");
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
