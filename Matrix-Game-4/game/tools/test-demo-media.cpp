#include "CombatAudio.h"
#include "DemoRecorder.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

int main(){
    CombatAudio mixer;
    const auto silence=mixer.render(1000,{});for(float x:silence)assert(x==0);
    std::vector<std::vector<float>> shots;
    for(auto kind:{CombatSound::Pistol,CombatSound::Shotgun,CombatSound::Launcher,CombatSound::Explosion}){
        mixer.reset();auto pcm=mixer.render(1000,{{kind,300,1,0}});double energy=0;
        for(size_t i=0;i<pcm.size();++i){assert(std::isfinite(pcm[i])&&std::abs(pcm[i])<=1);if(i<600)assert(pcm[i]==0);energy+=pcm[i]*pcm[i];}
        assert(energy>0);shots.push_back(std::move(pcm));
    }
    for(size_t i=0;i<shots.size();++i)for(size_t j=0;j<i;++j)assert(shots[i]!=shots[j]);
    mixer.reset();auto left=mixer.render(1000,{{CombatSound::Pistol,0,1,-1}});for(size_t i=1;i<left.size();i+=2)assert(left[i]==0);
    mixer.reset();for(float x:mixer.render(1000,{}))assert(x==0);
    const auto id=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path=std::filesystem::temp_directory_path()/("castle-demo-media-"+std::to_string(id));
    {
        DemoRecorder recorder(path,16,16,48,{{"speed",2}});
        recorder.input({{"type","key"},{"key","W"},{"down",true}});
        for(int frame=0;frame<10;++frame){std::vector<uint8_t> rgb(16*16*4,255),ui(16*16*4,0);recorder.submit(rgb,ui,{{"simulation_s",(frame+1)/24.}},shots[0]);}
        recorder.finish("test");recorder.finish();assert(recorder.frames()==10);
    }
    nlohmann::json manifest;std::ifstream(path/"session.json")>>manifest;
    assert(manifest["status"]=="complete"&&manifest["frames"]==10&&manifest["audio_samples"]==10000);
    assert(std::filesystem::file_size(path/"audio.wav")==44+10000*4);
    bool rejected=false;try{DemoRecorder again(path,16,16,48,{});}catch(const std::exception&){rejected=true;}assert(rejected);
    std::filesystem::remove_all(path);
    std::cout<<"DEMO_MEDIA_TEST_PASSED\n";
}
