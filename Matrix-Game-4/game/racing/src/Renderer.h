#pragma once
#include "Common.h"
#include "Navigation.h"
#include "HandlingMenu.h"
#include "CaptionTypes.h"
#include <optional>
#ifdef _WIN32
#include <d3d11.h>
#include <dxgi1_5.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#else
struct SDL_Window;
#endif

struct HudState {
    float speed=0,heading=0; V3 location;
    double frameMs=0,cpuMs=0,gpuMs=0;
    bool visible=false,performance=false,paused=false;
    int pauseSelection=0,cameraMode=0;
    const std::vector<MapLine>* roads=nullptr;
    V3 mapCenter;
    float mapHalfSize=1000;
    bool race=false,finished=false;
    bool pursuit=false,captureEligible=false;
    V3 suspectPosition;
    float suspectHeading=0,suspectDistance=0;
    double captureSeconds=0;
    double elapsed=0,penalty=0;
    int nextCheckpoint=0,totalCheckpoints=0,playerNumber=1;
    RouteNavigation navigation;
    std::string racePhase,notice;
    const RouteGuide* routeGuide=nullptr;
    std::string menuTitle;
    std::string menuDetail;
    std::vector<std::string> menu;
    std::optional<PlayerHandling> handling;
    std::vector<std::string> latencyStatus;
};
struct GateVisual {V3 center,forward;float halfWidth=5.5f,height=3.5f;Color color{1,.77f,.16f,.9f};};
struct FrameBundle {int width=0,height=0;std::vector<uint8_t> scene,ui,display;};
class Renderer {
public:
#ifdef _WIN32
    Renderer(HWND window,int width,int height);
#else
    Renderer(SDL_Window* window,int width,int height);
#endif
    ~Renderer();
    void makeCurrent();
    bool readbackPending(size_t slot)const;
    void beginReadback(size_t slot);
    uint32_t addMesh(const MeshData& mesh);
    void resize(int width,int height);
    double waitForFrame(bool lowLatency);
    void render(const std::vector<DrawObject>& objects,const Camera& camera,const HudState& hud,const std::vector<GateVisual>& gates={});
    double present(bool vsync);
    bool screenshot(const std::filesystem::path& path);
    std::vector<uint8_t> readBack();
    bool savePng(const std::filesystem::path& path,const std::vector<uint8_t>& rgba);
#ifndef _WIN32
    std::vector<captions::PixelStats> captionVisibility(const std::vector<DrawObject>& objects,const Camera& camera,int sampleWidth=432,float aspect=0);
    std::vector<uint8_t> readBackResized(int outputWidth,int outputHeight);
    // Replay a fixed-resolution, top-down RGBA reference/output frame, with local HUD only.
    void displayPixels(const std::vector<uint8_t>& rgba,const HudState& hud);
    // Redirect drawing into this window's own offscreen target and back to the
    // window. Overlapping X11 windows alias their back buffers, so a window
    // readback only returns the pixels of the last window drawn.
    void bindOffscreen();
    void blitToWindow();
    FrameBundle renderLayers(const std::vector<DrawObject>& objects,const Camera& camera,const HudState& hud,const std::vector<GateVisual>& gates,bool readLayers=false,int readbackSlot=-1,int outputWidth=0,int outputHeight=0);
    std::vector<uint8_t> finishReadback(size_t slot,std::vector<uint8_t> buffer={});
    static std::vector<uint8_t> png(const std::vector<uint8_t>& rgba,int width,int height);
#endif
    double gpuMs() const { return latestGpuMs; }
    bool newGpuSample=false;
    uint32_t visibleCount=0,drawCalls=0;
    std::string adapterName;
    int width=0,height=0;
    bool tearing=false;
    float outlineWidth=1.2f;
    float hudScale=1.f;
    bool buildingStripes=true;
private:
    struct UiVertex { float x,y; Color color; };
    std::vector<UiVertex> ui;
    double latestGpuMs=0;
#ifdef _WIN32
    struct GpuMesh { ComPtr<ID3D11Buffer> buffer; UINT count=0; };
    struct Timer { ComPtr<ID3D11Query> disjoint,begin,end; bool pending=false; };
    HWND window=nullptr;
    HANDLE frameWait=nullptr;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain2> swap;
    ComPtr<ID3D11RenderTargetView> target;
    ComPtr<ID3D11DepthStencilView> depth;
    ComPtr<ID3D11VertexShader> vertexShader,uiVertexShader;
    ComPtr<ID3D11PixelShader> pixelShader,uiPixelShader;
    ComPtr<ID3D11InputLayout> layout,uiLayout;
    ComPtr<ID3D11Buffer> constants,instances,uiBuffer;
    ComPtr<ID3D11RasterizerState> raster;
    ComPtr<ID3D11DepthStencilState> depthOn,depthOff;
    ComPtr<ID3D11BlendState> blend;
    std::vector<GpuMesh> meshes;
    std::vector<std::vector<Instance>> batches;
    std::array<Timer,8> timers;
    uint64_t frame=0;
#else
    struct GlState;
    std::unique_ptr<GlState> gl;
    std::vector<uint8_t> readTextureResized(uint32_t texture,int outputWidth,int outputHeight);
#endif
    void buildHud(const HudState& hud);
    void buildHudLayout(const HudState& hud,int width,int height);
    void drawMinimap(const HudState& hud,int width,int height);
    void rectangle(float x,float y,float w,float h,Color color);
    void line(float x1,float y1,float x2,float y2,float thickness,Color color);
    void text(float x,float y,const std::string& value,float scale,Color color);
    void drawGates(const std::vector<GateVisual>& gates,const Camera& camera);
    void drawHud(const HudState& hud);
};
