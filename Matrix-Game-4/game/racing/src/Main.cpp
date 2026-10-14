#include "Renderer.h"
#include "World.h"
#include "Audio.h"
#include "Gamepad.h"
#include "CameraTelemetry.h"
#include <Xinput.h>
#include <shellapi.h>
#include <sstream>
#include <iomanip>
#include <numeric>
#include "GameRuntime.h"
#include "KeyboardSteering.h"

namespace {
struct App {
    Renderer* renderer=nullptr;
    bool running=true,paused=false,active=true,minimized=false,fullscreen=false;
    bool reset=false,capture=false,lookBack=false,hud=false,performance=false;
    int cameraMode=1,pauseSelection=0;
    int pendingWidth=0,pendingHeight=0;
    bool benchmark=false;
    std::array<bool,256> keys{};
    RECT previousRect{};
};
App* app=nullptr;
void toggleFullscreen(HWND window){
    if(!app->fullscreen){GetWindowRect(window,&app->previousRect);MONITORINFO monitor{sizeof(MONITORINFO)};GetMonitorInfo(MonitorFromWindow(window,MONITOR_DEFAULTTONEAREST),&monitor);
        SetWindowLongPtr(window,GWL_STYLE,WS_POPUP|WS_VISIBLE);SetWindowPos(window,HWND_TOP,monitor.rcMonitor.left,monitor.rcMonitor.top,monitor.rcMonitor.right-monitor.rcMonitor.left,monitor.rcMonitor.bottom-monitor.rcMonitor.top,SWP_FRAMECHANGED);}
    else{SetWindowLongPtr(window,GWL_STYLE,WS_OVERLAPPEDWINDOW|WS_VISIBLE);auto r=app->previousRect;SetWindowPos(window,nullptr,r.left,r.top,r.right-r.left,r.bottom-r.top,SWP_FRAMECHANGED|SWP_NOZORDER);}
    app->fullscreen=!app->fullscreen;
}
void menuCommand(int selection){if(selection==0)app->paused=false;else if(selection==1){app->reset=true;app->paused=false;}else app->running=false;}
LRESULT CALLBACK windowProc(HWND window,UINT message,WPARAM w,LPARAM l){
    if(!app)return DefWindowProc(window,message,w,l);
    switch(message){
    case WM_CLOSE:app->running=false;return 0;
    case WM_DESTROY:PostQuitMessage(0);return 0;
    case WM_SIZE:app->minimized=w==SIZE_MINIMIZED;app->pendingWidth=LOWORD(l);app->pendingHeight=HIWORD(l);return 0;
    case WM_GETMINMAXINFO:{
        RECT minimum{0,0,640,360};AdjustWindowRectExForDpi(&minimum,WS_OVERLAPPEDWINDOW,FALSE,0,GetDpiForWindow(window));
        auto* info=reinterpret_cast<MINMAXINFO*>(l);info->ptMinTrackSize={minimum.right-minimum.left,minimum.bottom-minimum.top};return 0;
    }
    case WM_ACTIVATEAPP:app->active=w!=0;if(!w){app->keys.fill(false);if(!app->benchmark)app->paused=true;}return 0;
    case WM_KEYUP:case WM_SYSKEYUP:if(w<256)app->keys[w]=false;return 0;
    case WM_KEYDOWN:case WM_SYSKEYDOWN:
        if(w<256)app->keys[w]=true;
        if((l&(1<<30))==0){
            if(w==VK_ESCAPE)app->paused=!app->paused;
            if(w=='R')app->reset=true;
            if(w=='C')app->cameraMode=(app->cameraMode+1)%3;
            if(w==VK_TAB)app->hud=!app->hud;
            if(w==VK_F8)app->performance=!app->performance;
            if(w==VK_F11)toggleFullscreen(window);
            if(w==VK_F12)app->capture=true;
            if(app->paused&&app->hud){if(w==VK_UP||w=='W')app->pauseSelection=(app->pauseSelection+2)%3;if(w==VK_DOWN||w=='S')app->pauseSelection=(app->pauseSelection+1)%3;if(w==VK_RETURN)menuCommand(app->pauseSelection);}
        }
        if(w==VK_F4&&(GetKeyState(VK_MENU)&0x8000))app->running=false;
        return 0;
    case WM_MOUSEMOVE:case WM_LBUTTONUP:
        if(app->paused&&app->hud&&app->renderer){
            float x=float(short(LOWORD(l))),y=float(short(HIWORD(l)));float mx=app->renderer->width*.5f-160,my=app->renderer->height*.5f-59;
            for(int i=0;i<3;++i)if(x>=mx&&x<mx+305&&y>=my+i*48&&y<my+i*48+38){app->pauseSelection=i;if(message==WM_LBUTTONUP)menuCommand(i);}
        }return 0;
    }
    return DefWindowProc(window,message,w,l);
}
Options options(const std::filesystem::path& root){
    Options o;auto ini=root/L"settings.ini";
    o.width=GetPrivateProfileInt(L"Display",L"Width",1920,ini.c_str());o.height=GetPrivateProfileInt(L"Display",L"Height",1080,ini.c_str());
    o.traffic=GetPrivateProfileInt(L"World",L"Traffic",64,ini.c_str());o.pedestrians=GetPrivateProfileInt(L"World",L"Pedestrians",160,ini.c_str());
    o.vsync=GetPrivateProfileInt(L"Display",L"VSync",0,ini.c_str())!=0;o.noHud=GetPrivateProfileInt(L"Display",L"HUD",0,ini.c_str())==0;
    o.lowLatency=GetPrivateProfileInt(L"Display",L"LowLatencyWait",0,ini.c_str())!=0;
    o.physicsHz=float(GetPrivateProfileInt(L"Simulation",L"PhysicsHz",500,ini.c_str()));
    int argc=0;LPWSTR* argv=CommandLineToArgvW(GetCommandLineW(),&argc);
    if(!argv)throw std::runtime_error("Cannot parse command line");
    try{for(int i=1;i<argc;++i){std::wstring arg=argv[i];auto next=[&](){if(i+1>=argc)throw std::runtime_error("Missing command-line value");return std::wstring(argv[++i]);};
        if(arg==L"--width")o.width=std::stoi(next());else if(arg==L"--height")o.height=std::stoi(next());
        else if(arg==L"--traffic")o.traffic=std::stoi(next());else if(arg==L"--pedestrians")o.pedestrians=std::stoi(next());
        else if(arg==L"--player-steering-limit"){auto value=next();if(value!=L"on"&&value!=L"off")throw std::runtime_error("--player-steering-limit must be on or off");o.playerSteeringLimit=value==L"on";}
        else if(arg==L"--player-front-grip")o.playerFrontGrip=std::stof(next());
        else if(arg==L"--player-rear-grip")o.playerRearGrip=std::stof(next());
        else if(arg==L"--player-front-drive"){auto value=next();size_t used=0;o.playerFrontDrive=std::stof(value,&used);if(used!=value.size())throw std::runtime_error("Invalid --player-front-drive number");}
        else if(arg==L"--benchmark"){o.benchmark=std::stof(next());o.autodrive=true;o.noAudio=true;}
        else if(arg==L"--tag"){auto t=next();if(t.find_first_not_of(L"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")!=std::wstring::npos)throw std::runtime_error("Invalid benchmark tag");o.tag.clear();for(wchar_t c:t)o.tag.push_back(static_cast<char>(c));}
        else if(arg==L"--self-test")o.selfTest=true;else if(arg==L"--visual-test")o.visualTest=true;else if(arg==L"--autodrive")o.autodrive=true;
        else if(arg==L"--map-test")o.mapTest=true;
        else if(arg==L"--underpass-test")o.underpassTest=true;
        else if(arg==L"--ramp-ground-test")o.rampGroundTest=true;
        else if(arg==L"--all-ramps-test")o.allRampsTest=true;
        else if(arg==L"--no-audio")o.noAudio=true;else if(arg==L"--no-hud")o.noHud=true;
        else if(arg==L"--vsync")o.vsync=true;else throw std::runtime_error("Unknown command-line option");
    }}catch(...){LocalFree(argv);throw;}LocalFree(argv);
    if(o.width<640||o.width>7680||o.height<360||o.height>4320||o.traffic<0||o.traffic>100||o.pedestrians<0||o.pedestrians>200||o.physicsHz<100||o.physicsHz>1000||!std::isfinite(o.benchmark)||o.benchmark<0||o.benchmark>600)throw std::runtime_error("Settings outside supported range");
    if(!std::isfinite(o.playerFrontGrip)||!std::isfinite(o.playerRearGrip)||o.playerFrontGrip<=0||o.playerFrontGrip>3||o.playerRearGrip<=0||o.playerRearGrip>3)throw std::runtime_error("Player lateral grip scales must be in (0, 3]");
    if(!std::isfinite(o.playerFrontDrive)||o.playerFrontDrive<0||o.playerFrontDrive>1)throw std::runtime_error("--player-front-drive must be in [0, 1]");
    return o;
}
Input readInput(const Gamepad& gamepad,bool& keyboardSteering){
    auto key=[](int k){return app->keys[size_t(k)];};
    Input input;input.throttle=float(key('W')||key(VK_UP))-float(key('S')||key(VK_DOWN));
    input.steer=float(key('D')||key(VK_RIGHT))-float(key('A')||key(VK_LEFT));input.handbrake=key(VK_SPACE)?1.f:0;
    Input pad=gamepad.read();keyboardSteering=std::abs(pad.steer)<=.01f;if(!keyboardSteering)input.steer=pad.steer;
    if(std::abs(pad.throttle)>.01f)input.throttle=pad.throttle;input.handbrake=std::max(input.handbrake,pad.handbrake);
    return input;
}

}
int WINAPI wWinMain(HINSTANCE instance,HINSTANCE,LPWSTR,int){
    std::filesystem::path root;
    HWND window=nullptr;App state;app=&state;
    try{
        wchar_t executable[32768];GetModuleFileName(nullptr,executable,DWORD(std::size(executable)));
        root=std::filesystem::path(executable).parent_path();
        while(!std::filesystem::exists(root/L"assets/bbox-assets.bin")){auto parent=root.parent_path();if(parent==root)throw std::runtime_error("Cannot find game assets");root=parent;}
        Options opt=options(root);state.benchmark=opt.benchmark>0||opt.selfTest||opt.visualTest||opt.mapTest||opt.underpassTest||opt.rampGroundTest||opt.allRampsTest;
        std::ofstream startup(root/L"runtime.log");auto log=[&](const char* text){startup<<text<<std::endl;};log("Options loaded");
        if(opt.selfTest)return selfTest(root)?0:2;
        check(CoInitializeEx(nullptr,COINIT_MULTITHREADED),"COM initialization");SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);timeBeginPeriod(1);
        state.hud=!opt.noHud;state.performance=opt.benchmark>0;
        WNDCLASS cls{};cls.lpfnWndProc=windowProc;cls.hInstance=instance;cls.lpszClassName=L"CitySampleBBoxNative";cls.hCursor=LoadCursor(nullptr,IDC_ARROW);
        RegisterClass(&cls);RECT rect{0,0,opt.width,opt.height};AdjustWindowRect(&rect,WS_OVERLAPPEDWINDOW,FALSE);
        window=CreateWindow(cls.lpszClassName,L"CitySample BBox Drive",WS_OVERLAPPEDWINDOW,40,40,rect.right-rect.left,rect.bottom-rect.top,nullptr,nullptr,instance,nullptr);
        if(!window)throw std::runtime_error("Cannot create game window");
        log("Window created");
        Renderer renderer(window,opt.width,opt.height);state.renderer=&renderer;
        log("Renderer created");
        World world(root/L"assets/bbox-assets.bin",opt.traffic,opt.pedestrians,opt.visualTest);world.upload(renderer);
        log("World and meshes ready");
        if(auto result=runSceneTests(root,opt,renderer,world)){
            state.renderer=nullptr;DestroyWindow(window);window=nullptr;timeEndPeriod(1);CoUninitialize();return *result;
        }
        world.setPlayerSteeringLimit(opt.playerSteeringLimit);
        world.setPlayerLateralGrip(opt.playerFrontGrip,opt.playerRearGrip);
        world.setPlayerFrontDrive(opt.playerFrontDrive);
        log(opt.playerSteeringLimit?"Player steering speed limit: on":"Player steering speed limit: off");
        startup<<"Player lateral grip: front "<<(opt.autodrive?1:opt.playerFrontGrip)<<", rear "<<(opt.autodrive?1:opt.playerRearGrip)<<std::endl;
        startup<<"Player nominal front drive share: "<<(opt.autodrive?.5f:opt.playerFrontDrive)<<std::endl;
        std::unique_ptr<Audio> audio;if(!opt.noAudio)audio=std::make_unique<Audio>();
        Gamepad gamepad;
        KeyboardSteering steering;
        CameraTelemetry cameraTelemetry;
        SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_ABOVE_NORMAL);
        // Consume STARTUPINFO's show override, then explicitly show the interactive viewport.
        ShowWindow(window,SW_SHOWDEFAULT);ShowWindow(window,state.benchmark?SW_SHOWNOACTIVATE:SW_SHOW);UpdateWindow(window);
        log("Viewport shown");
        if(!state.benchmark){SetForegroundWindow(window);SetFocus(window);state.active=true;}
        state.paused=false;state.pendingWidth=0;state.pendingHeight=0;
        std::vector<Sample> samples;samples.reserve(100000);
        double beginning=now(),previous=beginning,accumulator=0,lastFrame=beginning,cpuAverage=0,frameAverage=3.0;
        bool captured=false,previousScreenshot=false;uint64_t screenshotIndex=0;Input currentInput;
        double previousCpu=0,previousPresent=0;
        while(state.running){
            double messageStart=now();
            MSG message{};while(PeekMessage(&message,nullptr,0,0,PM_REMOVE)){if(message.message==WM_QUIT)state.running=false;TranslateMessage(&message);DispatchMessage(&message);}
            double messageMs=(now()-messageStart)*1000,resizeStart=now();
            if(!state.running)break;
            if(state.pendingWidth>0&&state.pendingHeight>0){renderer.resize(state.pendingWidth,state.pendingHeight);state.pendingWidth=state.pendingHeight=0;}
            double resizeMs=(now()-resizeStart)*1000;
            if(state.minimized){steering.reset();Sleep(20);previous=now();lastFrame=previous;continue;}
            double waitMs=renderer.waitForFrame(opt.lowLatency&&!state.paused&&!opt.vsync),frameStart=now();double frameInterval=(frameStart-lastFrame)*1000;lastFrame=frameStart;
            // Drain input after the GPU queue wait so each frame consumes the most recent events.
            while(PeekMessage(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessage(&message);}
            double dt=std::min(frameStart-previous,.05);previous=frameStart;
            if(state.reset){world.reset();steering.reset();state.reset=false;accumulator=0;}
            if(state.paused||!state.active)steering.reset();
            bool keyboardSteering=true;currentInput=state.paused?Input{}:readInput(gamepad,keyboardSteering);
            double physicsStart=now();int updates=0;
            if(!state.paused){accumulator+=dt;const double step=1.0/opt.physicsHz;
                while(accumulator>=step&&updates<25){world.update(float(step),opt.autodrive?currentInput:steering.apply(currentInput,step,keyboardSteering),opt.autodrive);accumulator-=step;++updates;}
                if(updates==25)accumulator=0;
            }else accumulator=0;
            double physicsMs=(now()-physicsStart)*1000;auto player=world.player();
            if(audio){audio->speed=player.speed;audio->throttle=currentInput.throttle;audio->muted=state.paused;}
            HudState hud;hud.speed=player.speed;hud.location=position(player.body);hud.heading=player.heading;hud.visible=state.hud;
            hud.performance=state.performance;hud.paused=state.paused;hud.pauseSelection=state.pauseSelection;hud.cameraMode=state.cameraMode;
            hud.roads=&world.mapLines();hud.mapCenter=world.mapCenter();hud.mapHalfSize=world.mapHalfSize();
            hud.cpuMs=cpuAverage;hud.gpuMs=renderer.gpuMs();hud.frameMs=frameAverage;
            const Camera camera=world.camera(state.cameraMode,state.keys['Q']);
            renderer.render(world.drawObjects(),camera,hud);
            const uint32_t cameraFlags=(state.paused?CameraTelemetry::Paused:0u)|(state.active?CameraTelemetry::Active:0u)|(state.keys['Q']?CameraTelemetry::LookBack:0u);
            cameraTelemetry.publish(camera,renderer.width,renderer.height,state.cameraMode,cameraFlags,frameStart-beginning,world.simulationTime);
            double cpuMs=(now()-frameStart)*1000;cpuAverage=cpuAverage*.97+cpuMs*.03;frameAverage=frameAverage*.97+frameInterval*.03;
            bool screenshotFrame=false;
            if(state.capture||(state.benchmark&&!captured&&frameStart-beginning>2)){
                const auto name=state.benchmark?opt.tag:("drive-"+std::to_string(GetTickCount64())+"-"+std::to_string(screenshotIndex++));
                renderer.screenshot(root/L"captures"/(name+".png"));state.capture=false;captured=true;screenshotFrame=true;
            }
            double presentMs=renderer.present(opt.vsync);
            // The interval ends at this frame's start, so it includes the preceding frame's work.
            double unaccountedMs=std::max(0.0,frameInterval-previousCpu-previousPresent-waitMs-messageMs-resizeMs);
            if(state.benchmark&&frameStart-beginning>4&&!screenshotFrame&&!previousScreenshot)samples.push_back({frameInterval,cpuMs,renderer.newGpuSample?renderer.gpuMs():-1,physicsMs,presentMs,waitMs,messageMs,resizeMs,unaccountedMs,updates});
            previousCpu=cpuMs;previousPresent=presentMs;previousScreenshot=screenshotFrame;
            if(state.benchmark&&frameStart-beginning>opt.benchmark+4)state.running=false;
            if(state.paused&&!state.benchmark)Sleep(12);
        }
        if(state.benchmark)report(root,opt,renderer,world,samples);
        log("Normal shutdown");
        audio.reset();state.renderer=nullptr;DestroyWindow(window);window=nullptr;timeEndPeriod(1);CoUninitialize();return 0;
    }catch(const std::exception& error){
        if(!root.empty()){std::ofstream file(root/L"error.log",std::ios::app);file<<error.what()<<'\n';}
        if(!state.benchmark)MessageBoxA(window,error.what(),"CitySample BBox Drive",MB_OK|MB_ICONERROR);
        if(window)DestroyWindow(window);return 1;
    }
}
