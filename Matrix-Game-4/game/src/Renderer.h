#pragma once
#include "Common.h"
#include "VisibleObjects.h"
#include "CombatUi.h"
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
    double fps=0;
    bool showFps=false;
    bool visible=false,performance=false,paused=false;
    int pauseSelection=0,cameraMode=0;
    const std::vector<MapLine>* roads=nullptr;
    V3 mapCenter;
    float mapHalfSize=1000;
    bool combat=false,automatic=true,raised=false,recording=false,dead=false,showHelp=true;
    int player=1,ammo=0,weapon=0,kills=0,recordFrames=0;
    float health=100,shield=0,trapCooldown=0;
    float coverDurability=0;
    std::string coverStatus,coverHint;
    std::array<bool,3> owned{};
    std::string combatAction;
};
struct CapturedView {std::vector<uint8_t> color,depth;std::vector<uint32_t> pixels,ids;std::vector<VisibleObject> visible;};
class Renderer {
public:
#ifdef _WIN32
    Renderer(HWND window,int width,int height);
#else
    Renderer(SDL_Window* window,int width,int height);
#endif
    ~Renderer();
    uint32_t addMesh(const MeshData& mesh);
    void resize(int width,int height);
    double waitForFrame(bool lowLatency);
    void render(const std::vector<DrawObject>& objects,const Camera& camera,const HudState& hud);
    std::vector<VisibleObject> captureVisibleObjects(const std::vector<CatalogObject>& objects,std::vector<uint32_t>* ids=nullptr);
    CapturedView captureScene(const std::vector<DrawObject>& objects,const Camera& camera,const std::vector<CatalogObject>& catalog,bool includeDepth=true);
#ifndef _WIN32
    std::vector<uint8_t> captureDemoScene(const std::vector<DrawObject>& objects,const Camera& camera);
    std::vector<uint8_t> captureDemoUi(const HudState& hud);
#endif
    int idWidth=0,idHeight=0;
    double present(bool vsync);
    bool screenshot(const std::filesystem::path& path);
    double gpuMs() const { return latestGpuMs; }
    bool newGpuSample=false;
    uint32_t visibleCount=0,drawCalls=0;
    std::string adapterName;
    int width=0,height=0;
    bool tearing=false;
    float outlineWidth=1.0f;
    bool fadeSmallEdges=false;
private:
#ifdef _WIN32
    struct GpuMesh { ComPtr<ID3D11Buffer> buffer; UINT count=0; };
    struct Timer { ComPtr<ID3D11Query> disjoint,begin,end; bool pending=false; };
    struct UiVertex { float x,y; Color color; };
    HWND window=nullptr;
    HANDLE frameWait=nullptr;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain2> swap;
    ComPtr<ID3D11RenderTargetView> target;
    ComPtr<ID3D11DepthStencilView> depth;
    ComPtr<ID3D11VertexShader> vertexShader,uiVertexShader;
    ComPtr<ID3D11PixelShader> pixelShader,uiPixelShader,idPixelShader,depthPixelShader;
    std::array<ComPtr<ID3D11Texture2D>,2> captureTextures,captureReadback;
    std::array<ComPtr<ID3D11RenderTargetView>,2> captureTargets;
    ComPtr<ID3D11DepthStencilView> captureDepth;
    std::vector<uint32_t> visibilityPixels;
    ComPtr<ID3D11Texture2D> idTexture,idReadback;
    ComPtr<ID3D11RenderTargetView> idTarget;
    ComPtr<ID3D11DepthStencilView> idDepth;
    ComPtr<ID3D11InputLayout> layout,uiLayout;
    ComPtr<ID3D11Buffer> constants,instances,uiBuffer;
    ComPtr<ID3D11RasterizerState> raster,effectRaster;
    ComPtr<ID3D11DepthStencilState> depthOn,depthOff,effectDepth;
    ComPtr<ID3D11BlendState> blend,additiveBlend;
    std::vector<GpuMesh> meshes;
    std::vector<std::vector<Instance>> batches,effectBatches;
    std::vector<UiVertex> ui;
    std::array<Timer,8> timers;
    uint64_t frame=0;
    double latestGpuMs=0;
#else
    struct Impl;std::unique_ptr<Impl> impl;
    struct UiVertex {float x,y;Color color;};
    std::vector<UiVertex> ui;
    double latestGpuMs=0;
#endif
    void rectangle(float x,float y,float w,float h,Color color);
    void line(float x1,float y1,float x2,float y2,float thickness,Color color);
    void text(float x,float y,const std::string& value,float scale,Color color);
    void drawHud(const HudState& hud);
    void drawCombatOverlay(const HudState& hud);
#ifndef _WIN32
    void submitHud();
#endif
};
