#include "Renderer.h"
#include <d3dcompiler.h>
#include <wincodec.h>
#include <map>
#include <sstream>
#include <iomanip>

namespace {
constexpr UINT MaxInstances=65536,MaxUiVertices=100000;
const char* Shader=R"(
cbuffer Frame : register(b0) { row_major float4x4 viewProjection; float2 viewport; float outlineWidth; float padding; };
struct Input { float3 p:POSITION; float3 bary:TEXCOORD0; float3 edge:TEXCOORD1; float4 meshColor:COLOR1;
  float4 m0:INSTANCE0; float4 m1:INSTANCE1; float4 m2:INSTANCE2; float4 m3:INSTANCE3;
  float4 color:COLOR0; float surface:TEXCOORD2; };
struct Pixel { float4 p:SV_POSITION; float3 bary:TEXCOORD0;
  nointerpolation float3 edge:TEXCOORD1; nointerpolation float4 color:COLOR0;
  float3 world:TEXCOORD3; nointerpolation float surface:TEXCOORD2; };
Pixel VS(Input v) {
  Pixel o; float4 w=mul(float4(v.p,1),float4x4(v.m0,v.m1,v.m2,v.m3));
  o.p=mul(w,viewProjection); o.bary=v.bary; o.edge=v.edge; o.color=v.color*v.meshColor;
  o.world=w.xyz; o.surface=v.surface; return o;
}
float4 PS(Pixel p):SV_TARGET {
  float3 c=p.color.rgb;
  if(p.surface>0.5) {
    // Match the OpenGL world-space grain profile; fade subpixel grains.
    uint2 q=(uint2)(int2)floor(p.world.xz*20.0);
    uint h=q.x*1664525u+q.y*1013904223u;
    h^=h>>16;h*=0x7feb352du;h^=h>>15;h*=0x846ca68bu;h^=h>>16;
    float n=float(h>>8)/16777216.0;
    float footprint=max(length(ddx(p.world.xz)),length(ddy(p.world.xz)))*20.0;
    float fade=1.0-smoothstep(.5,1.5,footprint);
    c+=(n-.5)*.045*fade;
  } else if(p.surface>-.5) {
    float3 gradient=max(fwidth(p.bary),.00001);
    float3 d=abs(p.bary)/gradient;
    d=lerp(float3(10000,10000,10000),d,p.edge);
    // Keep full-contrast outlines on small projected faces, as in the control videos.
    if(outlineWidth>0 && min(d.x,min(d.y,d.z))<outlineWidth) c=0;
  }
  return float4(c,1);
}
struct UiIn {float2 p:POSITION;float4 c:COLOR0;};
struct UiOut {float4 p:SV_POSITION;float4 c:COLOR0;};
UiOut UiVS(UiIn v) {UiOut o;o.p=float4(v.p.x/viewport.x*2-1,1-v.p.y/viewport.y*2,0,1);o.c=v.c;return o;}
float4 UiPS(UiOut p):SV_TARGET {return p.c;}
)";
ComPtr<ID3DBlob> compile(const char* entry,const char* profile) {
    ComPtr<ID3DBlob> code,error;
    HRESULT hr=D3DCompile(Shader,strlen(Shader),"bbox.hlsl",nullptr,nullptr,entry,profile,
        D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error);
    if(FAILED(hr)) throw std::runtime_error(error?static_cast<char*>(error->GetBufferPointer()):"Shader compilation failed");
    return code;
}
struct Vertex { V3 p,bary,edge; uint32_t color; };
uint64_t edgeKey(uint32_t a,uint32_t b) { return (uint64_t(std::min(a,b))<<32)|std::max(a,b); }
}

Renderer::Renderer(HWND inWindow,int w,int h):window(inWindow) {
    ComPtr<IDXGIFactory2> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"Create DXGI factory");
    ComPtr<IDXGIFactory5> factory5;
    if(SUCCEEDED(factory.As(&factory5))) {
        BOOL allow=FALSE;
        if(SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING,&allow,sizeof(allow)))) tearing=allow!=FALSE;
    }
    ComPtr<IDXGIAdapter1> selected;
    SIZE_T largest=0;
    for(UINT i=0;;++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if(factory->EnumAdapters1(i,&adapter)==DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc{};adapter->GetDesc1(&desc);
        if(!(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&desc.DedicatedVideoMemory>=largest) { selected=adapter;largest=desc.DedicatedVideoMemory; }
    }
    if(!selected) throw std::runtime_error("A hardware Direct3D 11 GPU is required.");
    DXGI_ADAPTER_DESC1 description{};selected->GetDesc1(&description);
    char name[256];WideCharToMultiByte(CP_UTF8,0,description.Description,-1,name,256,nullptr,nullptr);adapterName=name;
    D3D_FEATURE_LEVEL requested=D3D_FEATURE_LEVEL_11_0,actual;
    check(D3D11CreateDevice(selected.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,&requested,1,D3D11_SDK_VERSION,&device,&actual,&context),"Create D3D11 device");
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width=w;desc.Height=h;desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.SampleDesc.Count=1;
    desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.BufferCount=2;desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.Flags=DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT|(tearing?DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING:0);
    ComPtr<IDXGISwapChain1> initial;
    check(factory->CreateSwapChainForHwnd(device.Get(),window,&desc,nullptr,nullptr,&initial),"Create flip swap chain");
    check(initial.As(&swap),"Query swap chain");swap->SetMaximumFrameLatency(1);
    frameWait=swap->GetFrameLatencyWaitableObject();factory->MakeWindowAssociation(window,DXGI_MWA_NO_ALT_ENTER);
    auto vs=compile("VS","vs_5_0"),ps=compile("PS","ps_5_0"),uvs=compile("UiVS","vs_5_0"),ups=compile("UiPS","ps_5_0");
    check(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertexShader),"Vertex shader");
    check(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixelShader),"Pixel shader");
    check(device->CreateVertexShader(uvs->GetBufferPointer(),uvs->GetBufferSize(),nullptr,&uiVertexShader),"UI vertex shader");
    check(device->CreatePixelShader(ups->GetBufferPointer(),ups->GetBufferSize(),nullptr,&uiPixelShader),"UI pixel shader");
    D3D11_INPUT_ELEMENT_DESC elements[]={
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",1,DXGI_FORMAT_R32G32B32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"COLOR",1,DXGI_FORMAT_R8G8B8A8_UNORM,0,36,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"INSTANCE",0,DXGI_FORMAT_R32G32B32A32_FLOAT,1,0,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"INSTANCE",1,DXGI_FORMAT_R32G32B32A32_FLOAT,1,16,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"INSTANCE",2,DXGI_FORMAT_R32G32B32A32_FLOAT,1,32,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"INSTANCE",3,DXGI_FORMAT_R32G32B32A32_FLOAT,1,48,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,1,64,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"TEXCOORD",2,DXGI_FORMAT_R32_FLOAT,1,80,D3D11_INPUT_PER_INSTANCE_DATA,1}};
    check(device->CreateInputLayout(elements,UINT(std::size(elements)),vs->GetBufferPointer(),vs->GetBufferSize(),&layout),"Input layout");
    D3D11_INPUT_ELEMENT_DESC uie[]={
        {"POSITION",0,DXGI_FORMAT_R32G32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,8,D3D11_INPUT_PER_VERTEX_DATA,0}};
    check(device->CreateInputLayout(uie,2,uvs->GetBufferPointer(),uvs->GetBufferSize(),&uiLayout),"UI input layout");
    auto buffer=[&](UINT size,UINT bind,ComPtr<ID3D11Buffer>& output) {
        D3D11_BUFFER_DESC d{};d.ByteWidth=size;d.Usage=D3D11_USAGE_DYNAMIC;d.BindFlags=bind;d.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
        check(device->CreateBuffer(&d,nullptr,&output),"Dynamic buffer");
    };
    buffer(80,D3D11_BIND_CONSTANT_BUFFER,constants);
    buffer(MaxInstances*sizeof(Instance),D3D11_BIND_VERTEX_BUFFER,instances);
    buffer(MaxUiVertices*sizeof(UiVertex),D3D11_BIND_VERTEX_BUFFER,uiBuffer);
    D3D11_RASTERIZER_DESC rd{};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;rd.DepthClipEnable=TRUE;
    check(device->CreateRasterizerState(&rd,&raster),"Raster state");
    D3D11_DEPTH_STENCIL_DESC dd{};dd.DepthEnable=TRUE;dd.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;dd.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
    check(device->CreateDepthStencilState(&dd,&depthOn),"Depth state");
    dd.DepthEnable=FALSE;dd.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
    check(device->CreateDepthStencilState(&dd,&depthOff),"UI depth state");
    D3D11_BLEND_DESC bd{};auto& rt=bd.RenderTarget[0];rt.BlendEnable=TRUE;rt.SrcBlend=D3D11_BLEND_SRC_ALPHA;
    rt.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOp=D3D11_BLEND_OP_ADD;rt.SrcBlendAlpha=D3D11_BLEND_ONE;
    rt.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOpAlpha=D3D11_BLEND_OP_ADD;rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
    check(device->CreateBlendState(&bd,&blend),"Blend state");
    for(auto& timer:timers) {
        D3D11_QUERY_DESC q{D3D11_QUERY_TIMESTAMP_DISJOINT,0};check(device->CreateQuery(&q,&timer.disjoint),"GPU timer");
        q.Query=D3D11_QUERY_TIMESTAMP;check(device->CreateQuery(&q,&timer.begin),"GPU timestamp");check(device->CreateQuery(&q,&timer.end),"GPU timestamp");
    }
    resize(w,h);
}
Renderer::~Renderer() { if(context) context->ClearState(); if(frameWait) CloseHandle(frameWait); }
void Renderer::resize(int w,int h) {
    if(w<=0||h<=0) return;
    context->OMSetRenderTargets(0,nullptr,nullptr);target.Reset();depth.Reset();
    check(swap->ResizeBuffers(0,w,h,DXGI_FORMAT_UNKNOWN,DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT|(tearing?DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING:0)),"Resize swap chain");
    width=w;height=h;
    ComPtr<ID3D11Texture2D> back;
    check(swap->GetBuffer(0,IID_PPV_ARGS(&back)),"Swap chain buffer");
    check(device->CreateRenderTargetView(back.Get(),nullptr,&target),"Render target");
    D3D11_TEXTURE2D_DESC d{};d.Width=w;d.Height=h;d.MipLevels=1;d.ArraySize=1;d.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;
    d.SampleDesc.Count=1;d.BindFlags=D3D11_BIND_DEPTH_STENCIL;
    ComPtr<ID3D11Texture2D> texture;check(device->CreateTexture2D(&d,nullptr,&texture),"Depth buffer");
    check(device->CreateDepthStencilView(texture.Get(),nullptr,&depth),"Depth view");
}
uint32_t Renderer::addMesh(const MeshData& mesh) {
    if(!context) throw std::runtime_error("Renderer is not current");
    if(mesh.vertices.empty()||mesh.indices.empty()) throw std::runtime_error("Empty mesh");
    if(!mesh.colors.empty()&&mesh.colors.size()!=mesh.vertices.size())throw std::runtime_error("Invalid mesh colors");
    std::map<uint64_t,bool> edges;
    for(auto e:mesh.edges) edges[edgeKey(e[0],e[1])]=true;
    std::vector<Vertex> data;data.reserve(mesh.indices.size());
    for(size_t i=0;i+2<mesh.indices.size();i+=3) {
        uint32_t a=mesh.indices[i],b=mesh.indices[i+1],c=mesh.indices[i+2];
        if(std::max({a,b,c})>=mesh.vertices.size()) throw std::runtime_error("Invalid mesh index");
        V3 flags{float(edges.contains(edgeKey(b,c))),float(edges.contains(edgeKey(a,c))),float(edges.contains(edgeKey(a,b)))};
        auto color=[&](uint32_t index){return mesh.colors.empty()?0xffffffffu:mesh.colors[index];};
        data.push_back({mesh.vertices[a],{1,0,0},flags,color(a)});data.push_back({mesh.vertices[b],{0,1,0},flags,color(b)});data.push_back({mesh.vertices[c],{0,0,1},flags,color(c)});
    }
    GpuMesh gpu;gpu.count=UINT(data.size());
    D3D11_BUFFER_DESC d{};d.ByteWidth=UINT(data.size()*sizeof(Vertex));d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA initial{data.data(),0,0};check(device->CreateBuffer(&d,&initial,&gpu.buffer),"Mesh buffer");
    meshes.push_back(gpu);batches.emplace_back();return uint32_t(meshes.size()-1);
}
double Renderer::waitForFrame(bool lowLatency) {
    double t=now();
    if(frameWait){
        if(lowLatency){
            // Bound active waiting to one frame budget; keep the flip queue at one frame.
            while(now()-t<.003){
                if(WaitForSingleObjectEx(frameWait,0,TRUE)==WAIT_OBJECT_0)return (now()-t)*1000;
                YieldProcessor();
            }
        }
        WaitForSingleObjectEx(frameWait,100,TRUE);
    }
    return (now()-t)*1000;
}
void Renderer::render(const std::vector<DrawObject>& objects,const Camera& camera,const HudState& hud,const std::vector<GateVisual>& gates) {
    (void)gates;
    using namespace DirectX;
    newGpuSample=false;
    for(auto& t:timers) if(t.pending) {
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dis{};UINT64 start=0,end=0;
        if(context->GetData(t.disjoint.Get(),&dis,sizeof(dis),D3D11_ASYNC_GETDATA_DONOTFLUSH)==S_OK&&
            context->GetData(t.begin.Get(),&start,sizeof(start),D3D11_ASYNC_GETDATA_DONOTFLUSH)==S_OK&&
            context->GetData(t.end.Get(),&end,sizeof(end),D3D11_ASYNC_GETDATA_DONOTFLUSH)==S_OK) {
            if(!dis.Disjoint&&dis.Frequency) { latestGpuMs=double(end-start)/double(dis.Frequency)*1000;newGpuSample=true; }t.pending=false;
        }
    }
    Timer& timer=timers[frame%timers.size()];bool measure=!timer.pending;
    if(measure) {context->Begin(timer.disjoint.Get());context->End(timer.begin.Get());}
    const float black[4]={0,0,0,1};context->ClearRenderTargetView(target.Get(),black);
    context->ClearDepthStencilView(depth.Get(),D3D11_CLEAR_DEPTH,1,0);
    ID3D11RenderTargetView* rtv=target.Get();context->OMSetRenderTargets(1,&rtv,depth.Get());
    D3D11_VIEWPORT viewport{0,0,float(width),float(height),0,1};context->RSSetViewports(1,&viewport);
    context->RSSetState(raster.Get());context->OMSetDepthStencilState(depthOn.Get(),0);context->OMSetBlendState(nullptr,nullptr,0xffffffff);
    context->IASetInputLayout(layout.Get());context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(vertexShader.Get(),nullptr,0);context->PSSetShader(pixelShader.Get(),nullptr,0);
    auto view=XMMatrixLookAtLH(XMVectorSet(camera.eye.x,camera.eye.y,camera.eye.z,1),XMVectorSet(camera.target.x,camera.target.y,camera.target.z,1),XMVectorSet(0,1,0,0));
    float aspect=float(width)/height;
    float vertical=2*std::atan(std::tan(camera.horizontalFov*Pi/360)/aspect);
    auto projection=XMMatrixPerspectiveFovLH(vertical,aspect,.08f,2000);
    struct Constants { Matrix viewProjection;float w,h,outline,pad; } c{};
    XMStoreFloat4x4(&c.viewProjection,view*projection);c.w=float(width);c.h=float(height);c.outline=outlineWidth;
    D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(constants.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped),"Map constants");
    memcpy(mapped.pData,&c,sizeof(c));context->Unmap(constants.Get(),0);
    ID3D11Buffer* cb=constants.Get();context->VSSetConstantBuffers(0,1,&cb);context->PSSetConstantBuffers(0,1,&cb);
    V3 forward=(camera.target-camera.eye).normalized();
    V3 right=cross({0,1,0},forward).normalized();V3 up=cross(forward,right);
    const float tx=std::tan(camera.horizontalFov*Pi/360),ty=tx/aspect;
    visibleCount=0;drawCalls=0;for(auto& b:batches)b.clear();
    for(const auto& object:objects) {
        if(object.instance.surface< -1.5f&&!buildingStripes)continue;
        V3 diff=object.center-camera.eye;float z=dot(diff,forward),r=object.radius;
        if(z+r<.08f||diff.length()>object.range+r||std::abs(dot(diff,right))>z*tx+r*(1+tx)||std::abs(dot(diff,up))>z*ty+r*(1+ty)) continue;
        if(visibleCount==MaxInstances) break;
        batches.at(object.mesh).push_back(object.instance);++visibleCount;
    }
    if(visibleCount) {
        check(context->Map(instances.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped),"Map instances");
        auto output=static_cast<Instance*>(mapped.pData);for(auto& b:batches) {memcpy(output,b.data(),b.size()*sizeof(Instance));output+=b.size();}
        context->Unmap(instances.Get(),0);
        UINT instanceOffset=0;
        for(size_t i=0;i<meshes.size();++i) if(!batches[i].empty()) {
            ID3D11Buffer* buffers[]={meshes[i].buffer.Get(),instances.Get()};UINT strides[]={sizeof(Vertex),sizeof(Instance)},offsets[]={0,0};
            context->IASetVertexBuffers(0,2,buffers,strides,offsets);
            context->DrawInstanced(meshes[i].count,UINT(batches[i].size()),0,instanceOffset);
            instanceOffset+=UINT(batches[i].size());++drawCalls;
        }
    }
    drawHud(hud);
    if(measure) {context->End(timer.end.Get());context->End(timer.disjoint.Get());timer.pending=true;}
    ++frame;
}
double Renderer::present(bool vsync) {
    double t=now();HRESULT hr=swap->Present(vsync?1:0,(!vsync&&tearing)?DXGI_PRESENT_ALLOW_TEARING:0);
    check(hr,"Present");return (now()-t)*1000;
}
void Renderer::drawHud(const HudState& hud) {
    buildHud(hud);
    if(ui.empty()) return;
    if(ui.size()>MaxUiVertices)throw std::runtime_error("HUD buffer exceeded");
    D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(uiBuffer.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped),"Map HUD");
    memcpy(mapped.pData,ui.data(),ui.size()*sizeof(UiVertex));context->Unmap(uiBuffer.Get(),0);
    context->OMSetDepthStencilState(depthOff.Get(),0);context->OMSetBlendState(blend.Get(),nullptr,0xffffffff);
    context->IASetInputLayout(uiLayout.Get());context->VSSetShader(uiVertexShader.Get(),nullptr,0);context->PSSetShader(uiPixelShader.Get(),nullptr,0);
    ID3D11Buffer* buffer=uiBuffer.Get();UINT stride=sizeof(UiVertex),offset=0;context->IASetVertexBuffers(0,1,&buffer,&stride,&offset);
    context->Draw(UINT(ui.size()),0);
}
bool Renderer::screenshot(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    auto temporary=path;temporary+=L".tmp";
    ComPtr<ID3D11Texture2D> back,staging;check(swap->GetBuffer(0,IID_PPV_ARGS(&back)),"Screenshot source");
    D3D11_TEXTURE2D_DESC d{};back->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;d.MiscFlags=0;
    check(device->CreateTexture2D(&d,nullptr,&staging),"Screenshot staging");context->CopyResource(staging.Get(),back.Get());
    D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),"Screenshot readback");
    std::vector<BYTE> pixels(size_t(width)*height*4);
    for(int y=0;y<height;++y)memcpy(pixels.data()+size_t(y)*width*4,static_cast<BYTE*>(mapped.pData)+size_t(y)*mapped.RowPitch,size_t(width)*4);
    for(size_t i=0;i<pixels.size();i+=4)std::swap(pixels[i],pixels[i+2]);
    context->Unmap(staging.Get(),0);
    ComPtr<IWICImagingFactory> factory;check(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory)),"PNG factory");
    ComPtr<IWICStream> stream;check(factory->CreateStream(&stream),"PNG stream");check(stream->InitializeFromFilename(temporary.c_str(),GENERIC_WRITE),"PNG file");
    ComPtr<IWICBitmapEncoder> encoder;check(factory->CreateEncoder(GUID_ContainerFormatPng,nullptr,&encoder),"PNG encoder");check(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache),"PNG initialize");
    ComPtr<IWICBitmapFrameEncode> output;check(encoder->CreateNewFrame(&output,nullptr),"PNG frame");check(output->Initialize(nullptr),"PNG frame initialize");check(output->SetSize(width,height),"PNG size");
    WICPixelFormatGUID format=GUID_WICPixelFormat32bppBGRA;check(output->SetPixelFormat(&format),"PNG format");
    if(format!=GUID_WICPixelFormat32bppBGRA)throw std::runtime_error("PNG encoder changed pixel format");
    check(output->WritePixels(height,width*4,UINT(pixels.size()),pixels.data()),"PNG pixels");check(output->Commit(),"PNG frame commit");check(encoder->Commit(),"PNG commit");
    output.Reset();encoder.Reset();stream.Reset();
    if(!MoveFileEx(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Cannot finalize screenshot");
    return true;
}
std::vector<uint8_t> Renderer::readBack(){
    ComPtr<ID3D11Texture2D> back,staging;check(swap->GetBuffer(0,IID_PPV_ARGS(&back)),"Readback source");
    D3D11_TEXTURE2D_DESC d{};back->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;d.MiscFlags=0;
    check(device->CreateTexture2D(&d,nullptr,&staging),"Readback staging");context->CopyResource(staging.Get(),back.Get());
    D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),"Readback map");
    std::vector<uint8_t> pixels(size_t(width)*height*4);
    for(int y=0;y<height;++y)memcpy(pixels.data()+size_t(y)*width*4,static_cast<BYTE*>(mapped.pData)+size_t(y)*mapped.RowPitch,size_t(width)*4);
    for(size_t i=0;i<pixels.size();i+=4)std::swap(pixels[i],pixels[i+2]);
    context->Unmap(staging.Get(),0);
    return pixels;
}
bool Renderer::savePng(const std::filesystem::path& path,const std::vector<uint8_t>& rgba){
    if(rgba.size()!=size_t(width)*height*4)throw std::runtime_error("Invalid PNG pixel buffer");
    std::vector<uint8_t> bgra=rgba;for(size_t i=0;i<bgra.size();i+=4)std::swap(bgra[i],bgra[i+2]);
    std::filesystem::create_directories(path.parent_path());
    auto temporary=path;temporary+=L".tmp";
    ComPtr<IWICImagingFactory> factory;check(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory)),"PNG factory");
    ComPtr<IWICStream> stream;check(factory->CreateStream(&stream),"PNG stream");check(stream->InitializeFromFilename(temporary.c_str(),GENERIC_WRITE),"PNG file");
    ComPtr<IWICBitmapEncoder> encoder;check(factory->CreateEncoder(GUID_ContainerFormatPng,nullptr,&encoder),"PNG encoder");check(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache),"PNG initialize");
    ComPtr<IWICBitmapFrameEncode> output;check(encoder->CreateNewFrame(&output,nullptr),"PNG frame");check(output->Initialize(nullptr),"PNG frame initialize");check(output->SetSize(width,height),"PNG size");
    WICPixelFormatGUID format=GUID_WICPixelFormat32bppBGRA;check(output->SetPixelFormat(&format),"PNG format");
    check(output->WritePixels(height,width*4,UINT(bgra.size()),bgra.data()),"PNG pixels");check(output->Commit(),"PNG frame commit");check(encoder->Commit(),"PNG commit");
    output.Reset();encoder.Reset();stream.Reset();
    if(!MoveFileEx(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Cannot finalize PNG");
    return true;
}
