#include "Map.h"
#include "Recording.h"
#include <shellapi.h>
#include <windowsx.h>
#include <iomanip>
#include <sstream>

namespace {
struct ViewController {
    V3 pivot;
    float distance=200, yaw=-.60f, pitch=-.68f, speed=20;
    V3 forward() const { return {std::sin(yaw)*std::cos(pitch),std::sin(pitch),std::cos(yaw)*std::cos(pitch)}; }
    V3 right() const { return {std::cos(yaw),0,-std::sin(yaw)}; }
    V3 up() const { return cross(forward(),right()); }
    Camera camera() const { return {pivot-forward()*distance,pivot,90}; }
    void fit(const Bounds& bounds,float aspect,bool top=false,const MapData* map=nullptr,bool structuresOnly=false) {
        pivot=bounds.center(); yaw=top?0.f:-.60f; pitch=top?-Pi*.5f+.0001f:-.68f;
        distance=1;
        // Actual geometry avoids framing empty corners of the scene-wide AABB.
        const auto include=[&](V3 point) {
            V3 relative=point-pivot;
            const float depth=dot(relative,forward());
            distance=std::max({distance,std::abs(dot(relative,right()))-depth,
                std::abs(dot(relative,up()))*aspect-depth});
        };
        if(map) {
            for(const auto& object:map->objects) if(!structuresOnly||(!object.tree&&!object.terrain&&!object.road))
                for(auto point:object.mesh.vertices) include(point);
        } else {
            for(int mask=0;mask<8;++mask)
                include({mask&1?bounds.max.x:bounds.min.x,mask&2?bounds.max.y:bounds.min.y,mask&4?bounds.max.z:bounds.min.z});
        }
        distance=distance*1.12f+1;
    }
    void look(float dx,float dy,bool orbit) {
        V3 eye=camera().eye;
        yaw=std::remainder(yaw+dx*.004f,2*Pi);
        pitch=std::clamp(pitch-dy*.004f,-Pi*.5f+.0001f,Pi*.5f-.0001f);
        if(!orbit) pivot=eye+forward()*distance;
    }
    void pan(float dx,float dy,int width) {
        const float scale=2*distance/std::max(width,1);
        pivot=pivot+right()*(-dx*scale)+up()*(dy*scale);
    }
    void move(float dt,const std::array<bool,256>& keys) {
        V3 direction=forward()*(float(keys['W'])-float(keys['S']))+
            right()*(float(keys['D'])-float(keys['A']))+V3{0,float(keys['E'])-float(keys['Q']),0};
        pivot=pivot+direction.normalized()*(dt*speed*(keys[VK_SHIFT]?4.f:1.f));
    }
};

struct Options {
    int width=1280,height=720;
    bool validate=false,visualTest=false,vsync=false,fullscreen=false;
    std::wstring view=L"all";std::string scene="castle";
    std::filesystem::path capture,benchmark;
    bool mapOnly=false,combatTest=false,poseTest=false,uePoseTest=false,coverTest=false,trailTest=false,firingTest=false,enemyFiringTest=false,eventTest=false,record=false;
    float coverProbability=.5f;
    uint32_t seed=1729;
    float duration=10;
    std::filesystem::path recordingDirectory;
};
struct App {
    ViewController view;
    Renderer* renderer=nullptr;
    bool running=true,active=true,minimized=false,fullscreen=false,capture=false;
    int pendingWidth=0,pendingHeight=0,fitRequest=-1,drag=0;
    POINT mouse{};
    RECT previousRect{};
    std::array<bool,256> keys{};
    bool gameplay=true,manual=false,paused=false,recordToggle=false,modeToggle=false,autoToggle=false;
    int player=0,weaponRequest=-1,menuSelection=0;
    float lookX=0,lookY=0;
    bool jump=false,dash=false,reload=false,trap=false,cover=false;
};
App* app=nullptr;

void clearCombatInput() {
    app->keys.fill(false);app->weaponRequest=-1;app->lookX=app->lookY=0;
    app->jump=app->dash=app->reload=app->trap=app->cover=false;
}
void chooseCombatMenu(int selection) {
    if(selection<0||selection>=int(CombatUi::MenuLabels.size()))return;
    clearCombatInput();
    if(selection==4){app->running=false;return;}
    if(selection>=1&&selection<=3){if(!app->manual){app->manual=true;app->autoToggle=true;}app->player=0;app->weaponRequest=selection-1;}
    app->paused=false;
}

void endDrag() {
    if(app->drag==2||app->drag==4) { ClipCursor(nullptr); ShowCursor(TRUE); }
    app->drag=0;
    ReleaseCapture();
}
void constrainCursor(HWND window) {
    RECT r; GetClientRect(window,&r);
    POINT a{r.left,r.top},b{r.right,r.bottom};
    ClientToScreen(window,&a); ClientToScreen(window,&b);
    r={a.x,a.y,b.x,b.y}; ClipCursor(&r);
}
void toggleFullscreen(HWND window) {
    endDrag();
    if(!app->fullscreen) {
        GetWindowRect(window,&app->previousRect);
        MONITORINFO monitor{sizeof(MONITORINFO)};
        GetMonitorInfo(MonitorFromWindow(window,MONITOR_DEFAULTTONEAREST),&monitor);
        SetWindowLongPtr(window,GWL_STYLE,WS_POPUP|WS_VISIBLE);
        SetWindowPos(window,HWND_TOP,monitor.rcMonitor.left,monitor.rcMonitor.top,
            monitor.rcMonitor.right-monitor.rcMonitor.left,monitor.rcMonitor.bottom-monitor.rcMonitor.top,SWP_FRAMECHANGED);
    } else {
        SetWindowLongPtr(window,GWL_STYLE,WS_OVERLAPPEDWINDOW|WS_VISIBLE);
        auto r=app->previousRect;
        SetWindowPos(window,nullptr,r.left,r.top,r.right-r.left,r.bottom-r.top,SWP_FRAMECHANGED|SWP_NOZORDER);
    }
    app->fullscreen=!app->fullscreen;
}

LRESULT CALLBACK windowProc(HWND window,UINT message,WPARAM w,LPARAM l) {
    if(!app) return DefWindowProc(window,message,w,l);
    switch(message) {
    case WM_CLOSE: app->running=false; return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    case WM_SIZE:
        app->minimized=w==SIZE_MINIMIZED; app->pendingWidth=LOWORD(l); app->pendingHeight=HIWORD(l);
        if(app->drag==2||app->drag==4) constrainCursor(window);
        return 0;
    case WM_GETMINMAXINFO: {
        RECT r{0,0,360,240}; AdjustWindowRectExForDpi(&r,WS_OVERLAPPEDWINDOW,FALSE,0,GetDpiForWindow(window));
        auto info=reinterpret_cast<MINMAXINFO*>(l); info->ptMinTrackSize={r.right-r.left,r.bottom-r.top}; return 0;
    }
    case WM_ACTIVATEAPP:
        app->active=w!=0;
        if(!w) { app->keys.fill(false); endDrag(); }
        return 0;
    case WM_CAPTURECHANGED:
        if(app->drag) endDrag();
        return 0;
    case WM_KEYUP:
        if(w<256) app->keys[w]=false;
        return 0;
    case WM_KEYDOWN:
        if(w<256) app->keys[w]=true;
        if(!(l&(1LL<<30))) {
            if(w==VK_F8){if(!app->paused){endDrag();app->gameplay=!app->gameplay;app->modeToggle=true;}return 0;}
            if(app->gameplay){
                if(w==VK_ESCAPE){app->paused=!app->paused;app->menuSelection=0;clearCombatInput();endDrag();return 0;}
                if(w==VK_F11){toggleFullscreen(window);return 0;}
                if(w==VK_F12){app->capture=true;return 0;}
                if(app->paused){
                    if(w==VK_UP)app->menuSelection=CombatUi::moveSelection(app->menuSelection,-1);
                    else if(w==VK_DOWN)app->menuSelection=CombatUi::moveSelection(app->menuSelection,1);
                    else if(w==VK_RETURN)chooseCombatMenu(app->menuSelection);
                    else if(w>='1'&&w<='3')chooseCombatMenu(int(w-'1')+1);
                    return 0;
                }
                if(w==VK_F7){app->manual=!app->manual;app->autoToggle=true;app->player=0;endDrag();}
                if(w==VK_F9)app->recordToggle=true;
                if(w>='1'&&w<='3'){if(!app->manual){app->manual=true;app->autoToggle=true;}app->player=0;app->weaponRequest=int(w-'1');}
                if(w==VK_SPACE)app->jump=true;
                if(w==VK_SHIFT)app->dash=true;
                if(w=='R')app->reload=true;
                if(w=='T')app->trap=true;
                if(w=='H')app->cover=true;
                if(w==VK_F11)toggleFullscreen(window);
                if(w==VK_F12)app->capture=true;
                return 0;
            }
            if(w==VK_ESCAPE) { if(app->drag) endDrag(); else app->running=false; }
            if(w==VK_HOME||w=='R'||w=='1') app->fitRequest=0;
            if(w=='F'||w=='2') app->fitRequest=1;
            if(w=='3') app->fitRequest=2;
            if(w==VK_F11) toggleFullscreen(window);
            if(w==VK_F12) app->capture=true;
        }
        return 0;
    case WM_LBUTTONDOWN: case WM_RBUTTONDOWN: case WM_MBUTTONDOWN:
        if(app->gameplay&&app->paused){if(message==WM_LBUTTONDOWN&&app->renderer)chooseCombatMenu(CombatUi::hitMenu(app->renderer->width,app->renderer->height,GET_X_LPARAM(l),GET_Y_LPARAM(l)));return 0;}
        if(app->gameplay){SetFocus(window);app->keys[message==WM_LBUTTONDOWN?VK_LBUTTON:message==WM_RBUTTONDOWN?VK_RBUTTON:VK_MBUTTON]=true;if(app->manual&&app->player==0&&!app->paused&&!app->drag){app->drag=4;constrainCursor(window);ShowCursor(FALSE);}return 0;}
        endDrag(); SetFocus(window); SetCapture(window);
        app->drag=message==WM_LBUTTONDOWN?1:message==WM_RBUTTONDOWN?2:3;
        app->mouse={GET_X_LPARAM(l),GET_Y_LPARAM(l)};
        if(app->drag==2) { constrainCursor(window); ShowCursor(FALSE); }
        return 0;
    case WM_LBUTTONUP: case WM_RBUTTONUP: case WM_MBUTTONUP:
        if(app->gameplay){app->keys[message==WM_LBUTTONUP?VK_LBUTTON:message==WM_RBUTTONUP?VK_RBUTTON:VK_MBUTTON]=false;return 0;}
        endDrag(); return 0;
    case WM_MOUSEMOVE: {
        POINT p{GET_X_LPARAM(l),GET_Y_LPARAM(l)};
        if(app->gameplay&&app->paused&&app->renderer){int item=CombatUi::hitMenu(app->renderer->width,app->renderer->height,p.x,p.y);if(item>=0)app->menuSelection=item;return 0;}
        const float dx=float(p.x-app->mouse.x),dy=float(p.y-app->mouse.y);
        if(app->drag==1) app->view.look(dx,dy,true);
        if(app->drag==3&&app->renderer) app->view.pan(dx,dy,app->renderer->width);
        app->mouse=p; return 0;
    }
    case WM_INPUT:
        if(app->drag==2||(app->gameplay&&app->manual&&app->drag==4)) {
            RAWINPUT raw{}; UINT size=sizeof(raw);
            if(GetRawInputData(reinterpret_cast<HRAWINPUT>(l),RID_INPUT,&raw,&size,sizeof(RAWINPUTHEADER))==sizeof(raw)&&raw.header.dwType==RIM_TYPEMOUSE) {
                if(!(raw.data.mouse.usFlags&MOUSE_MOVE_ABSOLUTE)) {
                    if(app->gameplay){app->lookX+=float(raw.data.mouse.lLastX);app->lookY+=float(raw.data.mouse.lLastY);}
                    else app->view.look(float(raw.data.mouse.lLastX),float(raw.data.mouse.lLastY),false);
                }
            }
        }
        return DefWindowProc(window,message,w,l);
    case WM_MOUSEWHEEL: {
        const float steps=float(GET_WHEEL_DELTA_WPARAM(w))/WHEEL_DELTA;
        if(app->drag==2) app->view.speed=std::clamp(app->view.speed*std::pow(1.2f,steps),.25f,200.f);
        else app->view.distance=std::clamp(app->view.distance*std::pow(.85f,steps),.25f,1200.f);
        return 0;
    }
    }
    return DefWindowProc(window,message,w,l);
}

Options readOptions() {
    Options o;
    int argc=0; LPWSTR* argv=CommandLineToArgvW(GetCommandLineW(),&argc);
    if(!argv) throw std::runtime_error("Cannot parse command line");
    try {
        for(int i=1;i<argc;++i) {
            std::wstring arg=argv[i];
            auto next=[&]() { if(i+1>=argc) throw std::runtime_error("Missing option value"); return std::wstring(argv[++i]); };
            if(arg==L"--width") o.width=std::stoi(next());
            else if(arg==L"--height") o.height=std::stoi(next());
            else if(arg==L"--validate") o.validate=true;
            else if(arg==L"--visual-test") o.visualTest=true;
            else if(arg==L"--vsync") o.vsync=true;
            else if(arg==L"--fullscreen") o.fullscreen=true;
            else if(arg==L"--benchmark") o.benchmark=next();
            else if(arg==L"--capture") o.capture=next();
            else if(arg==L"--scene") {const auto value=next();o.scene=std::string(value.begin(),value.end());}
            else if(arg==L"--view") o.view=next();
            else if(arg==L"--map") o.mapOnly=true;
            else if(arg==L"--combat-test") o.combatTest=true;
            else if(arg==L"--pose-test") o.poseTest=true;
            else if(arg==L"--ue-pose-test") o.uePoseTest=true;
            else if(arg==L"--cover-test") o.coverTest=true;
            else if(arg==L"--trail-test") o.trailTest=true;
            else if(arg==L"--fire-test") o.firingTest=true;
            else if(arg==L"--enemy-fire-test") o.enemyFiringTest=true;
            else if(arg==L"--event-test") o.eventTest=true;
            else if(arg==L"--cover-probability") o.coverProbability=std::stof(next());
            else if(arg==L"--record") o.record=true;
            else if(arg==L"--recording-dir") o.recordingDirectory=next();
            else if(arg==L"--duration") o.duration=std::stof(next());
            else if(arg==L"--seed") o.seed=uint32_t(std::stoul(next()));
            else throw std::runtime_error("Unknown command-line option");
        }
    } catch(...) { LocalFree(argv); throw; }
    LocalFree(argv);
    if(o.width<360||o.width>7680||o.height<240||o.height>4320) throw std::runtime_error("Unsupported window size");
    if(o.view!=L"all"&&o.view!=L"castle"&&o.view!=L"top") throw std::runtime_error("View must be all, castle or top");
    if(!std::isfinite(o.duration)||o.duration<=0||o.duration>3600)throw std::runtime_error("Duration must be 0-3600 seconds");
    if(!std::isfinite(o.coverProbability)||o.coverProbability<0||o.coverProbability>1)throw std::runtime_error("Cover probability must be 0-1");
    if(o.mapOnly&&(o.record||o.combatTest))throw std::runtime_error("Recording and combat tests require encounter mode");
    return o;
}

void writeValidation(const std::filesystem::path& root,const MapData& map) {
    std::filesystem::create_directories(root/L"captures");
    std::ofstream out(root/L"captures/validation.json");
    if(!out) throw std::runtime_error("Cannot write validation report");
    out<<"{\n  \"passed\": true,\n  \"objects\": "<<map.objects.size()<<",\n  \"primitives\": "<<map.primitiveCount
       <<",\n  \"trees\": "<<map.treeCount<<",\n  \"structures\": "<<map.objects.size()-map.treeCount-map.terrainCount-map.roadCount
       <<",\n  \"triangles\": "<<map.triangleCount<<",\n  \"terrain\": "<<(map.terrainCount?"true":"false")
       <<",\n  \"terrainObjects\": "<<map.terrainCount<<",\n  \"roadObjects\": "<<map.roadCount<<",\n  \"gameplay\": false\n}\n";
}

struct PerformanceSamples {
    std::vector<double> frame,cpu,gpu,wait,present;
};
void writeBenchmark(const Options& opt,const Renderer& renderer,const PerformanceSamples& samples) {
    if(samples.frame.empty()||samples.gpu.empty()) throw std::runtime_error("No benchmark samples collected");
    auto destination=std::filesystem::absolute(opt.benchmark);
    std::filesystem::create_directories(destination.parent_path());
    std::ofstream out(destination);
    if(!out) throw std::runtime_error("Cannot write benchmark report");
    const double mean=std::accumulate(samples.frame.begin(),samples.frame.end(),0.0)/samples.frame.size();
    out<<std::fixed<<std::setprecision(4)<<"{\n  \"gpu\": "<<std::quoted(renderer.adapterName)
       <<",\n  \"width\": "<<renderer.width<<",\n  \"height\": "<<renderer.height
       <<",\n  \"vsync\": "<<(opt.vsync?"true":"false")<<",\n  \"fullscreen\": "<<(opt.fullscreen?"true":"false")
       <<",\n  \"view\": "<<std::quoted(opt.mapOnly?std::string(opt.view.begin(),opt.view.end()):"P1")
       <<",\n  \"mode\": "<<std::quoted(opt.mapOnly?"map":"shooting")
       <<",\n  \"recording\": "<<(opt.record?"true":"false")
       <<",\n  \"tearingSupported\": "<<(renderer.tearing?"true":"false")
       <<",\n  \"frames\": "<<samples.frame.size()<<",\n  \"averageFps\": "<<1000/mean
       <<",\n  \"framesUnder3msPercent\": "<<100.0*std::count_if(samples.frame.begin(),samples.frame.end(),[](double v){return v<=3;})/samples.frame.size()
       <<",\n  \"visibleObjects\": "<<renderer.visibleCount<<",\n  \"drawCalls\": "<<renderer.drawCalls;
    const auto metric=[&](const char* name,std::vector<double> values) {
        std::sort(values.begin(),values.end());
        out<<",\n  \""<<name<<"\": {\"mean\": "<<std::accumulate(values.begin(),values.end(),0.0)/values.size()
           <<", \"p95\": "<<values[size_t(std::ceil(values.size()*.95))-1]
           <<", \"p99\": "<<values[size_t(std::ceil(values.size()*.99))-1]<<", \"max\": "<<values.back()<<'}';
    };
    metric("frameMs",samples.frame);metric("cpuRenderMs",samples.cpu);metric("gpuMs",samples.gpu);
    metric("frameWaitMs",samples.wait);metric("presentMs",samples.present);
    out<<"\n}\n";
}

void visualTest(const std::filesystem::path& root,Renderer& renderer,const MapData& map,const std::vector<DrawObject>& draws) {
    struct TestView { const char* name; int width,height,mode; };
    const TestView views[]={{"overview",1440,900,0},{"castle",1440,900,1},{"top",1440,900,2},
        {"portrait",720,1080,0},{"wide",1920,720,0},{"courtyard",1440,900,3},
        {"roads",1440,900,4},{"road-meshes",1440,900,5}};
    std::ofstream report(root/L"captures/visual-tests.json");
    if(!report) throw std::runtime_error("Cannot write visual test report");
    report<<"{\n  \"views\": [\n";
    bool first=true;
    for(const auto& v:views) {
        renderer.resize(v.width,v.height);
        ViewController view;
        view.fit(v.mode==1?map.castleBounds:map.bounds,float(v.width)/v.height,v.mode==2,&map,v.mode==1);
        if(v.mode==3) {
            auto tower=std::find_if(map.objects.begin(),map.objects.end(),[](const auto& o) { return o.catalogIndex==74; });
            view.fit(tower!=map.objects.end()?tower->bounds:map.castleBounds,float(v.width)/v.height);
            view.distance*=1.8f;
        }
        if(v.mode>=4) {
            Bounds roadBounds{{1e9f,1e9f,1e9f},{-1e9f,-1e9f,-1e9f}};
            for(const auto& o:map.objects) if(o.road) {
                roadBounds.min={std::min(roadBounds.min.x,o.bounds.min.x),std::min(roadBounds.min.y,o.bounds.min.y),std::min(roadBounds.min.z,o.bounds.min.z)};
                roadBounds.max={std::max(roadBounds.max.x,o.bounds.max.x),std::max(roadBounds.max.y,o.bounds.max.y),std::max(roadBounds.max.z,o.bounds.max.z)};
            }
            view.fit(map.roadCount?roadBounds:map.castleBounds,float(v.width)/v.height);
            if(v.mode==4) { view.yaw=Pi*.5f; view.pitch=-1.35f; view.distance*=1.6f; }
        }
        std::vector<DrawObject> roadDraws;
        if(v.mode==5) for(size_t i=0;i<map.objects.size();++i) if(map.objects[i].road) roadDraws.push_back(draws[i]);
        renderer.render(v.mode==5&&!roadDraws.empty()?roadDraws:draws,view.camera(),{});
        if(!renderer.visibleCount) throw std::runtime_error("Visual test culled the entire map");
        if(v.mode==5&&map.roadCount&&renderer.visibleCount!=map.roadCount) throw std::runtime_error("Road preview is missing instances");
        renderer.screenshot(root/L"captures"/(std::string(v.name)+".png"));
        renderer.present(false);
        if(!first) report<<",\n";
        first=false;
        const auto c=view.camera();
        report<<"    {\"name\": \""<<v.name<<"\", \"width\": "<<v.width<<", \"height\": "<<v.height
              <<", \"visibleObjects\": "<<renderer.visibleCount<<", \"drawCalls\": "<<renderer.drawCalls
              <<", \"eye\": ["<<c.eye.x<<','<<c.eye.y<<','<<c.eye.z<<"], \"target\": ["<<c.target.x<<','<<c.target.y<<','<<c.target.z<<"]}";
    }
    report<<"\n  ],\n  \"passed\": true\n}\n";
}
}

int WINAPI wWinMain(HINSTANCE instance,HINSTANCE,LPWSTR,int) {
    std::filesystem::path root;
    HWND window=nullptr;
    App state; app=&state;
    bool automated=false,comInitialized=false;
    try {
        wchar_t executable[32768];
        const auto chars=GetModuleFileName(nullptr,executable,DWORD(std::size(executable)));
        if(!chars||chars>=std::size(executable)) throw std::runtime_error("Cannot locate executable");
        root=std::filesystem::path(executable).parent_path().parent_path();
        auto opt=readOptions();
        automated=opt.validate||opt.visualTest||opt.combatTest||opt.poseTest||opt.uePoseTest||opt.coverTest||opt.trailTest||opt.firingTest||opt.enemyFiringTest||opt.eventTest||opt.record||!opt.capture.empty()||!opt.benchmark.empty();
        const auto scene=SceneConfig::load(root,opt.scene);
        MapData map(scene);
        writeValidation(root,map);
        if(opt.validate) return 0;
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        check(CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED),"Initialize COM"); comInitialized=true;
        WNDCLASS wc{}; wc.style=CS_HREDRAW|CS_VREDRAW; wc.lpfnWndProc=windowProc; wc.hInstance=instance;
        wc.hCursor=LoadCursor(nullptr,IDC_ARROW); wc.lpszClassName=L"MedievalCastleBBoxMap";
        if(!RegisterClass(&wc)) throw std::runtime_error("Cannot register map window");
        RECT r{0,0,opt.width,opt.height};
        AdjustWindowRectExForDpi(&r,WS_OVERLAPPEDWINDOW,FALSE,0,GetDpiForSystem());
        std::wstring title=L"Medieval Castle BBox | CF_01_Demo_Scene_CPY | "+std::to_wstring(map.objects.size())+
            L" objects | "+std::to_wstring(map.primitiveCount)+L" primitives";
        window=CreateWindow(wc.lpszClassName,title.c_str(),WS_OVERLAPPEDWINDOW,CW_USEDEFAULT,CW_USEDEFAULT,
            r.right-r.left,r.bottom-r.top,nullptr,nullptr,instance,nullptr);
        if(!window) throw std::runtime_error("Cannot create map window");
        RAWINPUTDEVICE mouse{0x01,0x02,0,window};
        if(!RegisterRawInputDevices(&mouse,1,sizeof(mouse))) throw std::runtime_error("Cannot register mouse input");
        {
            Renderer renderer(window,opt.width,opt.height); state.renderer=&renderer;
            auto draws=map.upload(renderer);
            std::ofstream log(root/L"runtime.log");
            log<<"GPU: "<<renderer.adapterName<<"\nObjects: "<<map.objects.size()<<"\nPrimitives: "<<map.primitiveCount
               <<"\nTriangles: "<<map.triangleCount<<"\nTerrain objects: "<<map.terrainCount<<"\nRoad objects: "<<map.roadCount
               <<"\nVSync: "<<(opt.vsync?"on":"off")<<"\nTearing supported: "<<renderer.tearing<<"\nGameplay: "<<(opt.mapOnly?"map only":"Solo Encounter BBox")<<'\n'; log.flush();
            state.pendingWidth=state.pendingHeight=0;state.gameplay=!opt.mapOnly;
            state.view.fit(opt.view==L"castle"?map.castleBounds:map.bounds,float(opt.width)/opt.height,opt.view==L"top",&map,opt.view==L"castle");
            if(opt.visualTest) visualTest(root,renderer,map,draws);
            else if(opt.eventTest) {
                Encounter encounter(root,map,renderer,opt.seed);
                auto report=encounter.eventTest();std::ofstream(root/L"captures/event-tests.json")<<report.dump(2);
            } else if(opt.uePoseTest) {
                Encounter encounter(root,map,renderer,opt.seed);
                std::ofstream(root/L"captures/ue-pose-comparison.json")<<encounter.uePoseTest().dump(2);
            } else if(opt.enemyFiringTest) {
                Encounter encounter(root,map,renderer,opt.seed);
                std::ofstream(root/L"captures/enemy-fire-tests.json")<<encounter.enemyFiringTest().dump(2);
            } else if(opt.firingTest) {
                Encounter encounter(root,map,renderer,opt.seed);
                std::ofstream(root/L"captures/fire-tests.json")<<encounter.firingTest().dump(2);
            } else if(opt.trailTest) {
                Encounter encounter(root,map,renderer,opt.seed);
                std::ofstream(root/L"captures/trail-tests.json")<<encounter.trailTest().dump(2);
            } else if(opt.coverTest) {
                Encounter encounter(root,map,renderer,opt.seed);
                std::ofstream(root/L"captures/cover-tests.json")<<encounter.coverTest().dump(2);
            } else if(opt.poseTest) {
                Encounter encounter(root,map,renderer,opt.seed);
                std::ofstream(root/L"captures/pose-tests.json")<<encounter.poseTest().dump(2);
            } else if(!opt.capture.empty()) {
                renderer.render(draws,state.view.camera(),{});
                renderer.screenshot(std::filesystem::absolute(opt.capture));
            } else {
                std::unique_ptr<Encounter> encounter;
                if(!opt.mapOnly)encounter=std::make_unique<Encounter>(root,map,renderer,opt.seed);
                if(encounter)encounter->setCoverProbability(opt.coverProbability);
                if(opt.combatTest){auto report=encounter->selfTest();std::ofstream(root/L"captures/combat-tests.json")<<report.dump(2);}
                std::unique_ptr<Recording> recording;
                std::filesystem::path recordingDir=opt.recordingDirectory.empty()?scene.outputDirectory(root,"recordings"):std::filesystem::absolute(opt.recordingDirectory);
                const auto startRecording=[&]{if(!encounter)encounter=std::make_unique<Encounter>(root,map,renderer,opt.seed);encounter->setCoverProbability(opt.coverProbability);encounter->prepareClip();encounter->takeEvents();recording=std::make_unique<Recording>(recordingDir,root/L"assets",opt.seed,scene.assets);encounter->setRecording(true);};
                const auto stopRecording=[&]{if(recording){recording->captureFinalEvents(renderer,*encounter);recording->finish();recording.reset();encounter->setRecording(false);}};
                if(opt.record)startRecording();
                ShowWindow(window,SW_SHOW); UpdateWindow(window);
                SetProp(window,L"MedievalCastleBBoxReady",reinterpret_cast<HANDLE>(1));
                if(opt.fullscreen) {
                    toggleFullscreen(window);
                    state.fitRequest=opt.view==L"castle"?1:opt.view==L"top"?2:0;
                }
                double previous=now();
                double fpsStart=previous;
                const double benchmarkStart=previous;
                double previousFrameEnd=previous;
                PerformanceSamples timings;
                uint32_t fpsFrames=0;
                HudState hud; hud.showFps=true;
                double accumulator=0,simulated=0,nextRecoveryAttempt=0;int recordedFrames=0,recordTicks=0,completedClips=0;bool resumeRecording=false;
                while(state.running) {
                    MSG message;
                    while(PeekMessage(&message,nullptr,0,0,PM_REMOVE)) {
                        if(message.message==WM_QUIT) state.running=false;
                        TranslateMessage(&message); DispatchMessage(&message);
                    }
                    if(!state.running) break;
                    if((state.minimized||!state.active)&&opt.benchmark.empty()&&!opt.combatTest&&!opt.record&&!recording) {
                        stopRecording();
                        WaitMessage(); previous=now();
                        fpsStart=previous; fpsFrames=0; hud.fps=0;
                        previousFrameEnd=previous;
                        continue;
                    }
                    if(state.pendingWidth>0&&state.pendingHeight>0) {
                        renderer.resize(state.pendingWidth,state.pendingHeight);
                        state.pendingWidth=state.pendingHeight=0;
                    }
                    if(state.fitRequest>=0) {
                        state.view.fit(state.fitRequest==1?map.castleBounds:map.bounds,float(renderer.width)/renderer.height,state.fitRequest==2,&map,state.fitRequest==1);
                        state.fitRequest=-1;
                    }
                    const double waitMs=renderer.waitForFrame(!opt.vsync);
                    const double current=now();
                    const double elapsed=std::min(current-previous,.1);previous=current;
                    if(state.autoToggle){if(encounter)encounter->setAutopilot(!state.manual);state.autoToggle=false;}
                    if(state.modeToggle){if(state.gameplay&&!encounter)encounter=std::make_unique<Encounter>(root,map,renderer,opt.seed);state.modeToggle=false;}
                    if(state.recordToggle){if(recording)stopRecording();else{startRecording();recordTicks=0;}state.recordToggle=false;resumeRecording=false;}
                    if(state.paused)stopRecording();
                    if(encounter&&!state.paused){
                        accumulator+=opt.combatTest?1.0/24:elapsed;
                        CombatInput input;input.forward=float(state.keys['W'])-float(state.keys['S']);input.right=float(state.keys['D'])-float(state.keys['A']);
                        input.fire=state.keys[VK_LBUTTON];input.ads=state.keys[VK_RBUTTON];input.shield=state.keys['F'];input.crouch=state.keys['C'];
                        input.weapon=state.weaponRequest;input.jump=state.jump;input.dash=state.dash;input.reload=state.reload;input.trap=state.trap;input.cover=state.cover;input.lookX=state.lookX;input.lookY=state.lookY;
                        const bool requestCover=input.cover;
                        if(!state.gameplay||state.player!=0||!state.manual||!state.active)input={};
                        if(state.gameplay&&state.player==0&&state.active)input.cover=requestCover;
                        const int physicsTicks=int(accumulator/(1.0/120));
                        const float lookX=CameraInput::pixelsPerTick(input.lookX,physicsTicks,1.f/120);
                        const float lookY=CameraInput::pixelsPerTick(input.lookY,physicsTicks,1.f/120);
                        if(physicsTicks){state.weaponRequest=-1;state.jump=state.dash=state.reload=state.trap=state.cover=false;state.lookX=state.lookY=0;}
                        for(int tick=0;tick<physicsTicks;++tick){input.lookX=lookX;input.lookY=lookY;encounter->advance(1.f/120,input);accumulator-=1.0/120;simulated+=1.0/120;input.lookX=input.lookY=0;input.jump=input.dash=input.reload=input.trap=input.cover=false;input.weapon=-1;
                            if(recording&&++recordTicks==5){recording->capture(renderer,*encounter);recordTicks=0;++recordedFrames;
                                if(opt.record&&recordedFrames>=int(std::round(opt.duration*24))){stopRecording();state.running=false;break;}
                                if(recording->frames()>=240){stopRecording();++completedClips;if(completedClips%20==0&&!encounter->recover()){resumeRecording=true;nextRecoveryAttempt=encounter->seconds()+2;}else{startRecording();recordTicks=0;}}}
                            else if(!recording)encounter->takeEvents();
                            if((encounter->recoveryRequested()||resumeRecording)&&encounter->seconds()>=nextRecoveryAttempt){resumeRecording=resumeRecording||bool(recording);stopRecording();if(encounter->recover()){if(resumeRecording){startRecording();recordTicks=-24;resumeRecording=false;}}else nextRecoveryAttempt=encounter->seconds()+2;}
                        }
                        if(opt.combatTest&&simulated>=opt.duration)state.running=false;
                    }else accumulator=0;
                    if(!state.gameplay)state.view.move(float(elapsed),state.keys);
                    hud.combat=state.gameplay;hud.paused=state.paused&&state.gameplay;hud.pauseSelection=state.menuSelection;if(state.gameplay&&encounter)encounter->hud(hud,state.player);
                    if(state.paused)hud.combatAction="PAUSED";
                    hud.recording=bool(recording);hud.recordFrames=recording?recording->frames():0;
                    renderer.render(state.gameplay&&encounter?encounter->draws(state.paused?1.f:float(accumulator*120)):draws,state.gameplay&&encounter?encounter->camera(state.player):state.view.camera(),hud);
                    if(state.capture) {
                        SYSTEMTIME t; GetLocalTime(&t);
                        std::wostringstream name;
                        name<<L"map-"<<t.wYear<<L'-'<<t.wMonth<<L'-'<<t.wDay<<L'-'<<t.wHour<<L'-'<<t.wMinute<<L'-'<<t.wSecond<<L'-'<<t.wMilliseconds<<L".png";
                        auto screenshot=root/L"captures"/name.str();renderer.screenshot(screenshot);
                        Json snapshot={{"gameplay",state.gameplay},{"player",state.player+1},{"paused",state.paused},{"recording",bool(recording)},{"recordFrames",recording?recording->frames():0},{"width",renderer.width},{"height",renderer.height}};
                        if(encounter)snapshot["encounter"]=encounter->diagnostics();
                        std::ofstream(screenshot.wstring()+L".json")<<snapshot.dump(2);state.capture=false;
                    }
                    const double cpuMs=(now()-current)*1000;
                    const double presentMs=renderer.present(opt.vsync);
                    const double frameEnd=now();
                    if(!opt.benchmark.empty()&&frameEnd-benchmarkStart>=1) {
                        timings.frame.push_back((frameEnd-previousFrameEnd)*1000);
                        timings.cpu.push_back(cpuMs);timings.wait.push_back(waitMs);timings.present.push_back(presentMs);
                        if(renderer.newGpuSample) timings.gpu.push_back(renderer.gpuMs());
                        if(frameEnd-benchmarkStart>=6) state.running=false;
                    }
                    previousFrameEnd=frameEnd;
                    ++fpsFrames;
                    // Include presentation/frame pacing; only camera movement clamps its delta.
                    if(frameEnd-fpsStart>=.25) {
                        hud.fps=fpsFrames/(frameEnd-fpsStart);
                        fpsStart=frameEnd; fpsFrames=0;
                    }
                }
                stopRecording();
                if(encounter){std::ofstream(root/L"captures/encounter-last.json")<<encounter->diagnostics().dump(2);if(opt.combatTest)renderer.screenshot(root/L"captures/combat.png");}
                if(!opt.benchmark.empty()) writeBenchmark(opt,renderer,timings);
            }
            state.renderer=nullptr;
        }
        endDrag(); DestroyWindow(window); CoUninitialize(); return 0;
    } catch(const std::exception& e) {
        endDrag();
        if(!root.empty()) { std::ofstream log(root/L"runtime.log",std::ios::app); log<<"ERROR: "<<e.what()<<'\n'; }
        if(!automated) MessageBoxA(window,e.what(),"Medieval Castle BBox Map",MB_OK|MB_ICONERROR);
        if(window) DestroyWindow(window);
        if(comInitialized) CoUninitialize();
        return 1;
    }
}
