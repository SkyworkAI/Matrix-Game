#include "Recording.h"
#include <condition_variable>
#include <deque>
#include <iomanip>
#include <locale>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <ctime>
#ifndef _WIN32
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace {
constexpr int RecordingViews=1;
using CsvJson=nlohmann::ordered_json;
constexpr const char* CsvHeader="VideoPath,ActionList,CameraList,FPS,TotalFrames,Position,CameraPosition,CameraRotation,VisibleObjects,Event,stat,Weather";
constexpr const char* DefaultWeather="A city scene at an unspecified time of day with unspecified weather and cloud cover.";
std::string decimal(double value,int digits){if(std::abs(value)<.5*std::pow(10.,-digits))value=0;std::ostringstream stream;stream.imbue(std::locale::classic());stream<<std::fixed<<std::setprecision(digits)<<value;std::string out=stream.str();while(out.ends_with('0'))out.pop_back();if(out.ends_with('.'))out+='0';return out;}
double normalizeAngle(double value){value=std::fmod(value,360.);if(value<0)value+=360.;if(value>180)value-=360.;return value;}
bool validateCsv(const std::filesystem::path& path){if(!std::filesystem::exists(path)||std::filesystem::file_size(path)==0)return false;std::ifstream file(path,std::ios::binary);std::string header;std::getline(file,header);if(header.starts_with("\xEF\xBB\xBF"))header.erase(0,3);if(header.ends_with('\r'))header.pop_back();if(!file||header!=CsvHeader)throw std::runtime_error("Recording CSV has an incompatible header: "+path.string());return true;}
Json ue(V3 p){return Json::array({double(p.z)*100,double(p.x)*100,double(p.y)*100});}
Json rotation(const Camera& c){V3 f=(c.target-c.eye).normalized();return Json::array({0.0,double(std::asin(f.y))*180/Pi,double(std::atan2(f.x,f.z))*180/Pi});}
std::string csv(const std::string& s,bool force=false){if(!force&&s.find_first_of(",\"\r\n")==std::string::npos)return s;std::string out="\"";for(char c:s){if(c=='"')out+='"';out+=c;}return out+'"';}
std::string tuples(const Json& values){std::ostringstream s;bool first=true;for(const auto& v:values){if(!first)s<<',';first=false;s<<'('<<decimal(v[0],6)<<','<<decimal(v[1],6)<<','<<decimal(v[2],6)<<')';}return s.str();}
std::pair<std::string,std::string> inputs(const Json& positions,const Json& rotations){
    std::ostringstream movement,camera;
    constexpr int candidates[8][4]={{1,0,0,0},{1,0,0,1},{0,0,0,1},{0,1,0,1},{0,1,0,0},{0,1,1,0},{0,0,1,0},{1,0,1,0}};
    for(size_t i=0;i<positions.size();++i){size_t j=std::min(i+1,positions.size()-1);double dx=positions[j][0].get<double>()-positions[i][0].get<double>(),dy=positions[j][1].get<double>()-positions[i][1].get<double>();
        double yaw=rotations[i][2].get<double>()*Pi/180,f=(dx*std::cos(yaw)+dy*std::sin(yaw))*24,r=(-dx*std::sin(yaw)+dy*std::cos(yaw))*24;
        double relative=normalizeAngle(std::atan2(dy,dx)*180/Pi-rotations[i][2].get<double>());int sector=(int(std::floor((relative+22.5)/45))+8)%8;
        if(i){movement<<',';camera<<',';}const auto& keys=candidates[sector];movement<<'['<<int(keys[0]&&f>=20)<<','<<int(keys[1]&&f<=-20)<<','<<int(keys[2]&&r<=-20)<<','<<int(keys[3]&&r>=20)<<",0,0]";
        auto delta=[&](int axis){return std::round(normalizeAngle(rotations[j][axis].get<double>()-rotations[i][axis].get<double>())/15*100)/100;};camera<<'('<<decimal(delta(1),2)<<','<<decimal(delta(2),2)<<')';
    }return {movement.str(),camera.str()};
}
std::wstring stamp(){auto clock=std::chrono::system_clock::now();std::time_t seconds=std::chrono::system_clock::to_time_t(clock);std::tm t{};
#ifdef _WIN32
    localtime_s(&t,&seconds);
#else
    localtime_r(&seconds,&t);
#endif
    auto millis=std::chrono::duration_cast<std::chrono::milliseconds>(clock.time_since_epoch()).count()%1000;std::wostringstream s;s<<std::put_time(&t,L"%Y%m%d_%H%M%S")<<L'_'<<std::setfill(L'0')<<std::setw(3)<<millis;return s.str();}
class Encoder {
#ifdef _WIN32
    HANDLE input=nullptr,process=nullptr;
public:
    Encoder(const std::filesystem::path& output,const std::filesystem::path& executable){
        SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};HANDLE read=nullptr;
        if(!CreatePipe(&read,&input,&security,1024*1024))throw std::runtime_error("Cannot create encoder pipe");SetHandleInformation(input,HANDLE_FLAG_INHERIT,0);
        auto log=output;log+=L".encoder.log";HANDLE errors=CreateFile(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&security,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        std::wstring command=L"\""+executable.wstring()+L"\" -hide_banner -loglevel warning -y -f rawvideo -pixel_format rgba -video_size 1280x720 -framerate 24 -i pipe:0 -an -c:v libx264 -preset ultrafast -crf 16 -threads 2 -pix_fmt yuv420p -movflags +faststart \""+output.wstring()+L"\"";
        STARTUPINFO si{};si.cb=sizeof(si);si.dwFlags=STARTF_USESTDHANDLES;si.hStdInput=read;si.hStdError=errors;si.hStdOutput=errors;PROCESS_INFORMATION pi{};
        BOOL good=CreateProcess(executable.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW,nullptr,nullptr,&si,&pi);CloseHandle(read);if(errors!=INVALID_HANDLE_VALUE)CloseHandle(errors);
        if(!good){CloseHandle(input);input=nullptr;throw std::runtime_error("Cannot launch FFmpeg. Install ffmpeg or place it in bin.");}process=pi.hProcess;CloseHandle(pi.hThread);
    }
    ~Encoder(){try{finish();}catch(...){}}
    void write(const std::vector<uint8_t>& bytes){size_t offset=0;while(offset<bytes.size()){DWORD written=0;if(!WriteFile(input,bytes.data()+offset,DWORD(bytes.size()-offset),&written,nullptr)||!written)throw std::runtime_error("FFmpeg stopped accepting recording frames");offset+=written;}}
    void finish(){if(input){CloseHandle(input);input=nullptr;}if(process){DWORD waited=WaitForSingleObject(process,30000),code=1;if(waited!=WAIT_OBJECT_0){TerminateProcess(process,1);WaitForSingleObject(process,5000);}else GetExitCodeProcess(process,&code);CloseHandle(process);process=nullptr;if(code!=0)throw std::runtime_error("FFmpeg failed while finalizing a stream");}}
#else
    int input=-1;pid_t process=-1;
public:
    Encoder(const std::filesystem::path& output,const std::filesystem::path& executable){
        std::signal(SIGPIPE,SIG_IGN);int pipes[2];
#ifdef __APPLE__
        if(pipe(pipes))throw std::runtime_error("Cannot create encoder pipe");
        if(fcntl(pipes[0],F_SETFD,FD_CLOEXEC)<0||fcntl(pipes[1],F_SETFD,FD_CLOEXEC)<0){close(pipes[0]);close(pipes[1]);throw std::runtime_error("Cannot protect encoder pipe descriptors");}
#else
        if(pipe2(pipes,O_CLOEXEC))throw std::runtime_error("Cannot create encoder pipe");
#endif
        auto log=output;log+=".encoder.log";int errors=open(log.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0644);if(errors<0){close(pipes[0]);close(pipes[1]);throw std::runtime_error("Cannot open encoder log");}
        std::vector<std::string> args={executable.string(),"-hide_banner","-loglevel","warning","-y","-f","rawvideo","-pixel_format","rgba","-video_size","1280x720","-framerate","24","-i","pipe:0","-an","-c:v","libx264","-preset","ultrafast","-crf","16","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",output.string()};
        std::vector<char*> argv;for(auto& a:args)argv.push_back(a.data());argv.push_back(nullptr);
        posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);posix_spawn_file_actions_adddup2(&actions,pipes[0],STDIN_FILENO);posix_spawn_file_actions_adddup2(&actions,errors,STDERR_FILENO);posix_spawn_file_actions_adddup2(&actions,errors,STDOUT_FILENO);
        int error=posix_spawnp(&process,executable.c_str(),&actions,nullptr,argv.data(),environ);posix_spawn_file_actions_destroy(&actions);close(errors);close(pipes[0]);
        if(error){close(pipes[1]);process=-1;throw std::runtime_error("Cannot launch FFmpeg; install ffmpeg in PATH");}input=pipes[1];
    }
    ~Encoder(){try{finish();}catch(...){}}
    void write(const std::vector<uint8_t>& bytes){size_t offset=0;while(offset<bytes.size()){ssize_t written=::write(input,bytes.data()+offset,bytes.size()-offset);if(written<0&&errno==EINTR)continue;if(written<=0)throw std::runtime_error("FFmpeg stopped accepting frames");offset+=size_t(written);}}
    void finish(){if(input>=0){close(input);input=-1;}if(process>0){int status=0;pid_t result;do{result=waitpid(process,&status,0);}while(result<0&&errno==EINTR);process=-1;if(result<0||!WIFEXITED(status)||WEXITSTATUS(status)!=0)throw std::runtime_error("FFmpeg failed while finalizing a stream");}}
#endif
};
}
struct Recording::Impl {
    struct Packet {std::array<CapturedView,RecordingViews> views;};
    std::filesystem::path output,assets;std::array<std::filesystem::path,RecordingViews> videos;std::wstring stem;
    std::array<std::unique_ptr<Encoder>,RecordingViews> encoders;
    std::thread worker;std::mutex mutex;std::condition_variable changed;std::deque<Packet> queue;
    bool closing=false,finished=false,terminalFrame=false;std::string failure;int submitted=0,written=0;uint32_t seed;
    std::set<uint64_t> eventIds;
    std::vector<uint64_t> sources,steps;
    std::array<Json,RecordingViews> positions,cameras,rotations,actions;
    std::array<CsvJson,RecordingViews> visibility,events;
    std::ofstream witness;
    Impl(const std::filesystem::path& out,const std::filesystem::path& data,uint32_t s,const std::filesystem::path& sceneAssets):output(out),assets(data),stem(stamp()),seed(s){
        std::filesystem::create_directories(output);
        validateCsv(output/L"VideoData_1Dir.csv");
#ifdef _WIN32
        wchar_t found[32768]{};auto ffmpeg=assets.parent_path()/L"bin/ffmpeg.exe";
        if(!std::filesystem::exists(ffmpeg)){DWORD n=SearchPathW(nullptr,L"ffmpeg.exe",nullptr,DWORD(std::size(found)),found,nullptr);if(!n||n>=std::size(found))throw std::runtime_error("FFmpeg is required for recording");ffmpeg=found;}
#else
        auto ffmpeg=assets.parent_path()/"bin/ffmpeg";if(!std::filesystem::exists(ffmpeg))ffmpeg="ffmpeg";
#endif
        for(int p=0;p<RecordingViews;++p){positions[p]=cameras[p]=rotations[p]=actions[p]=Json::array();visibility[p]=events[p]=CsvJson::array();std::wstring base=stem+L"_P"+std::to_wstring(p+1);
            videos[p]=output/(base+L"_bounds.mp4");encoders[p]=std::make_unique<Encoder>(videos[p],ffmpeg);}
        witness.open(output/(stem+L".frames.jsonl"));if(!witness)throw std::runtime_error("Cannot write recording metadata");
        const auto catalogAssets=sceneAssets.empty()?assets:sceneAssets;
        for(const auto* name:{L"StaticMeshCatalog.csv",L"CatalogManifest.json"})std::filesystem::copy_file(catalogAssets/L"shooter"/name,output/name,std::filesystem::copy_options::overwrite_existing);
        auto images=catalogAssets/L"shooter/Images";if(std::filesystem::exists(images)){std::filesystem::create_directories(output/L"Images");for(const auto& entry:std::filesystem::directory_iterator(images))if(entry.is_regular_file())std::filesystem::copy_file(entry.path(),output/L"Images"/entry.path().filename(),std::filesystem::copy_options::overwrite_existing);}
        if(std::filesystem::exists(catalogAssets/"map.json"))std::filesystem::copy_file(catalogAssets/"map.json",output/"MapProfile.json",std::filesystem::copy_options::overwrite_existing);
        worker=std::thread([this]{run();});
    }
    ~Impl(){if(worker.joinable()){std::lock_guard lock(mutex);closing=true;changed.notify_all();}if(worker.joinable())worker.join();}
    void run(){try{for(;;){Packet packet;{std::unique_lock lock(mutex);changed.wait(lock,[&]{return closing||!queue.empty();});if(queue.empty()&&closing)break;packet=std::move(queue.front());queue.pop_front();changed.notify_all();}
                for(int p=0;p<RecordingViews;++p)encoders[p]->write(packet.views[p].color);++written;
            }for(auto& e:encoders)e->finish();
        }catch(const std::exception& e){{std::lock_guard lock(mutex);failure=e.what();closing=true;queue.clear();changed.notify_all();}for(auto& encoder:encoders)try{encoder->finish();}catch(...){}}}
    void push(Packet packet){std::unique_lock lock(mutex);changed.wait(lock,[&]{return queue.size()<3||closing;});if(!failure.empty())throw std::runtime_error(failure);if(closing)throw std::runtime_error("Recorder already closed");queue.push_back(std::move(packet));changed.notify_all();}
};
Recording::Recording(const std::filesystem::path& out,const std::filesystem::path& assets,uint32_t seed,const std::filesystem::path& sceneAssets):impl(std::make_unique<Impl>(out,assets,seed,sceneAssets)){}
Recording::~Recording(){try{finish();}catch(...){}}
void Recording::capture(Renderer& renderer,Encounter& encounter){
    if(impl->finished)throw std::runtime_error("Recorder is finished");auto& r=*impl;
    try{Impl::Packet packet;auto facts=encounter.takeEvents();const auto& draws=encounter.draws();Json witness={{"FrameIndex",r.submitted},{"SimulationStep",encounter.step()},{"TimeSeconds",encounter.seconds()},{"State",encounter.eventState()}};
    for(const auto& e:facts)if(!e.id||!r.eventIds.insert(e.id).second||e.step>encounter.step()||e.time>encounter.seconds()+1e-8||(!r.steps.empty()&&e.step<r.steps.back()))throw std::runtime_error("Gameplay event identity/frame alignment is invalid");
    for(int p=0;p<RecordingViews;++p){Camera c=encounter.camera(p);packet.views[p]=renderer.captureScene(draws,c,encounter.catalog(),false);auto& view=packet.views[p];
        r.positions[p].push_back(ue(encounter.playerPosition(p)));r.cameras[p].push_back(ue(c.eye));r.rotations[p].push_back(rotation(c));r.actions[p].push_back(encounter.action(p));
        CsvJson visible=CsvJson::array();for(const auto& o:view.visible){CsvJson object={{"Index",o.index},{"Count",o.count},{"PixelCount",o.pixels},{"Position",o.position}};if(o.flags&CatalogObject::Rotation)object["Rotation"]=o.rotation;visible.push_back(object);}
        if(r.submitted%24==0)r.visibility[p].push_back({{"FrameIndex",r.submitted},{"Objects",visible}});
        CsvJson emitted=CsvJson::array();Json audit=Json::array();for(const auto& e:facts){bool visibleEvent=encounter.eventVisible(e,c,view,encounter.playerToken(p));std::string type=e.type;
            if(type!="Explosion"&&type!="ExplosionDamage"){if(type.starts_with("Enemy"))type=type.substr(5);type=(e.actor==encounter.playerToken(0)?"Player":"Enemy")+type;}
            audit.push_back({{"EventId",e.id},{"SourceStep",e.step},{"SourceTime",e.time},{"ActorToken",e.actor},{"ObjectTokens",e.tokens},{"VisibilityTokens",e.visibilityTokens},{"Type",type},{"Position",ue(e.position)},{"Indices",e.indices},{"PointEvent",e.pointEvent},{"AllowPositionOnly",e.allowPositionOnly},{"Visible",visibleEvent}});
            if(visibleEvent){CsvJson event={{"FrameIndex",r.submitted},{"Type",type},{"Position",ue(e.position)},{"Indices",e.indices}};r.events[p].push_back(event);emitted.push_back(event);}}
        witness["P"+std::to_string(p+1)]={{"ObserverToken",encounter.playerToken(p)},{"Position",r.positions[p].back()},{"CameraPosition",r.cameras[p].back()},{"CameraRotation",r.rotations[p].back()},{"HorizontalFOV",c.horizontalFov},{"DamageState",nullptr},{"VisibleObjects",visible},{"Events",emitted},{"EventAudit",audit}};
        std::vector<uint32_t>().swap(view.ids);
    }
    r.sources.push_back(uint64_t(r.submitted));r.steps.push_back(encounter.step());r.witness<<witness.dump()<<'\n';if(!r.witness)throw std::runtime_error("Recording metadata write failed");r.push(std::move(packet));++r.submitted;
    }catch(const std::exception& e){std::lock_guard lock(r.mutex);if(r.failure.empty())r.failure=e.what();r.closing=true;r.changed.notify_all();throw;}
}
void Recording::captureFinalEvents(Renderer& renderer,Encounter& encounter){if(!impl->finished&&impl->submitted>0&&encounter.hasEvents()){impl->terminalFrame=true;capture(renderer,encounter);}}
void Recording::finish(){if(!impl||impl->finished)return;auto& r=*impl;{std::lock_guard lock(r.mutex);r.closing=true;r.changed.notify_all();}if(r.worker.joinable())r.worker.join();r.finished=true;r.witness.close();
    bool valid=r.failure.empty()&&r.written==r.submitted&&r.submitted>0;
    Json videos=Json::array();for(int p=0;p<RecordingViews;++p)videos.push_back(r.videos[p].filename().string());
    Json sidecar={{"valid",valid},{"error",r.failure},{"fps",24},{"width",1280},{"height",720},{"frames",r.submitted},{"writtenFrames",r.written},{"sourceFrames",r.sources},{"simulationSteps",r.steps},{"videos",videos},{"seed",r.seed},{"streamKind","BBox only; not Unreal RGB"},{"fixedSimulationHz",120},{"encoder","libx264 CRF 16 ultrafast"},{"hudIncluded",false}};
    sidecar["observers"]=RecordingViews;
    sidecar["recordDepth"]=false;sidecar["csvSchema"]=CsvHeader;
    sidecar["eventAuditVersion"]=1;sidecar["terminalEventFrame"]=r.terminalFrame;
    {std::ofstream f(r.output/(r.stem+L"_P1_bounds.mp4.frames.json"));f<<sidecar.dump(2);if(!f)throw std::runtime_error("Cannot write recording validity sidecar");}
    if(!valid){for(const auto& p:r.videos)if(std::filesystem::exists(p)){auto invalid=p.parent_path()/(p.stem().wstring()+L".invalid.mp4");std::filesystem::rename(p,invalid);}if(!r.submitted&&r.failure.empty())return;throw std::runtime_error(r.failure.empty()?"Incomplete recording group":r.failure);}
    auto path=r.output/L"VideoData_1Dir.csv";bool header=!validateCsv(path),separator=false;
    if(!header){std::ifstream previous(path,std::ios::binary);previous.seekg(-1,std::ios::end);char last=0;previous.get(last);if(!previous)throw std::runtime_error("Cannot read recording CSV terminator");separator=last!='\n';}
    std::ofstream csvFile(path,std::ios::binary|std::ios::app);if(!csvFile)throw std::runtime_error("Cannot write recording CSV");
    if(header)csvFile<<CsvHeader<<"\r\n";
    else if(separator)csvFile<<"\r\n";
    const std::string damageStates=CsvJson::array({CsvJson{{"FrameIndex",0},{"DamageState",nullptr}}}).dump();
    for(int p=0;p<RecordingViews;++p){auto [actions,cameraInputs]=inputs(r.positions[p],r.rotations[p]);
        csvFile<<csv(r.videos[p].filename().string())<<','<<csv(actions,true)<<','<<csv(cameraInputs,true)<<",24,"<<r.submitted<<','<<csv(tuples(r.positions[p]),true)<<','<<csv(tuples(r.cameras[p]),true)<<','<<csv(tuples(r.rotations[p]),true)<<','<<csv(r.visibility[p].dump(),true)<<','<<csv(r.events[p].dump(),true)<<','<<csv(damageStates,true)<<','<<csv(DefaultWeather,true)<<"\r\n";}
    csvFile.flush();if(!csvFile)throw std::runtime_error("CSV finalization failed");
}
int Recording::frames()const{return impl->submitted;}
const std::filesystem::path& Recording::directory()const{return impl->output;}
