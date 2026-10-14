#include "Renderer.h"
#include <d3dcompiler.h>
#include <wincodec.h>
#include <map>
#include <sstream>
#include <iomanip>

namespace {
constexpr UINT MaxInstances=65536,MaxUiVertices=100000;
const char* Shader=R"(
cbuffer Frame : register(b0) { row_major float4x4 viewProjection; float2 viewport; float outlineWidth; float fadeSmallEdges; };
struct Input { float3 p:POSITION; float3 bary:TEXCOORD0; float3 edge:TEXCOORD1; float4 meshColor:COLOR1;
  float4 m0:INSTANCE0; float4 m1:INSTANCE1; float4 m2:INSTANCE2; float4 m3:INSTANCE3;
  float4 color:COLOR0; float surface:TEXCOORD2; uint token:TEXCOORD4; uint instanceToken:TEXCOORD5; };
struct Pixel { float4 p:SV_POSITION; float3 bary:TEXCOORD0;
  nointerpolation float3 edge:TEXCOORD1; nointerpolation float4 color:COLOR0;
  float3 world:TEXCOORD3; nointerpolation float surface:TEXCOORD2; nointerpolation uint token:TEXCOORD4;
  nointerpolation float3 effectAxis:TEXCOORD6; };
Pixel VS(Input v) {
  Pixel o; float4 w=mul(float4(v.p,1),float4x4(v.m0,v.m1,v.m2,v.m3));
  o.p=mul(w,viewProjection); o.bary=v.bary; o.edge=v.edge; o.color=v.color*v.meshColor;
  o.world=w.xyz; o.surface=v.surface; o.token=v.token!=0?v.token:v.instanceToken; o.effectAxis=v.m1.xyz; return o;
}
uint IdPS(Pixel p):SV_TARGET { return p.token; }
float4 DepthPS(Pixel p):SV_TARGET {
  float meters=.08*2000/(2000-p.p.z*(2000-.08));
  return float4(saturate(meters/200).xxx,1);
}
float checker(float2 uv) {
  // Integrate the checker over the pixel footprint to avoid distant grid shimmer.
  float2 width=max(fwidth(uv),.0001);
  float2 integral=2*(abs(frac((uv-.5*width)*.5)-.5)-abs(frac((uv+.5*width)*.5)-.5))/width;
  return saturate(.5-.5*integral.x*integral.y);
}
float4 PS(Pixel p):SV_TARGET {
  float3 c=p.color.rgb;
  // Cosmetic projectile trails preserve hue and fade their additive contribution.
  // The effect pass reads scene depth and never writes its own opaque depth.
  if(p.surface< -1.5) {
    float3 n=normalize(cross(ddx(p.world),ddy(p.world)));
    // UE suppresses each cylinder's cap, avoiding bright rings at segment joins.
    float cap=1-saturate(abs(dot(n,normalize(p.effectAxis)))*10);
    return float4(c,saturate(p.color.a)*cap);
  }
  if(p.surface>1.5) {
    float3 n=normalize(cross(ddx(p.world),ddy(p.world)));
    float2 uv=p.world.xz;
    // Road risers and side walls also get square cells, without stretching.
    if(p.surface==3.0||p.surface==5.0) {
      float3 a=abs(n);
      if(a.x>max(a.y,a.z)) uv=p.world.zy;
      else if(a.z>a.y) uv=p.world.xy;
    }
    // Two aligned scales: 1 m cells, each containing 10x10 cells of 0.1 m.
    float coarse=checker(uv),fine=checker(uv*10);
    float gray=lerp(.66,.71,coarse)+(fine-.5)*.02;
    // Egypt: retain 0.5 m small cells, grouped into 4 m blocks (8x8 cells).
    if(p.surface>3.5) gray=lerp(.70,.73,checker(uv*.25))+(checker(uv*2)-.5)*.012;
    float lighting=.95+.05*abs(dot(n,normalize(float3(-.5,1,-.35))));
    c=gray.xxx*lighting;
  } else if(p.surface>0.5) {
    float2 q=floor(p.world.xz*30);
    float n=frac(sin(dot(q,float2(12.9898,78.233)))*43758.5453);
    float patch=sin(p.world.x*.31+sin(p.world.z*.42))*sin(p.world.z*.22);
    float fade=saturate(1-length(ddx(p.world)+ddy(p.world))*3);
    c+=((n-.5)*.034*fade+patch*.014);
  } else if(p.surface>-.5) {
    float3 gradient=max(fwidth(p.bary),.00001);
    float3 d=abs(p.bary)/gradient;
    d=lerp(float3(10000,10000,10000),d,p.edge);
    // Only filter the stroke boundary; small/far triangles retain black edges.
    if(fadeSmallEdges>0){float3 altitude=1/gradient;float strength=smoothstep(2.5,9,min(altitude.x,min(altitude.y,altitude.z)));if(outlineWidth>0&&min(d.x,min(d.y,d.z))<outlineWidth)c*=1-strength;}
    else if(outlineWidth>0) c*=smoothstep(max(0,outlineWidth-.5),outlineWidth+.5,min(d.x,min(d.y,d.z)));
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
struct Vertex { V3 p,bary,edge; uint32_t color,token; };
uint64_t edgeKey(uint32_t a,uint32_t b) { return (uint64_t(std::min(a,b))<<32)|std::max(a,b); }
const std::map<char,std::array<uint8_t,7>> Font={
 {'0',{14,17,19,21,25,17,14}},{'1',{4,12,4,4,4,4,14}},{'2',{14,17,1,2,4,8,31}},
 {'3',{30,1,1,14,1,1,30}},{'4',{2,6,10,18,31,2,2}},{'5',{31,16,16,30,1,1,30}},
 {'6',{6,8,16,30,17,17,14}},{'7',{31,1,2,4,8,8,8}},{'8',{14,17,17,14,17,17,14}},
 {'9',{14,17,17,15,1,2,12}},{'A',{14,17,17,31,17,17,17}},{'B',{30,17,17,30,17,17,30}},
 {'C',{14,17,16,16,16,17,14}},{'D',{30,17,17,17,17,17,30}},{'E',{31,16,16,30,16,16,31}},
 {'F',{31,16,16,30,16,16,16}},{'G',{14,17,16,23,17,17,15}},{'H',{17,17,17,31,17,17,17}},
 {'I',{14,4,4,4,4,4,14}},{'J',{7,2,2,2,2,18,12}},{'K',{17,18,20,24,20,18,17}},
 {'L',{16,16,16,16,16,16,31}},{'M',{17,27,21,21,17,17,17}},{'N',{17,25,21,19,17,17,17}},
 {'O',{14,17,17,17,17,17,14}},{'P',{30,17,17,30,16,16,16}},{'Q',{14,17,17,17,21,18,13}},
 {'R',{30,17,17,30,20,18,17}},{'S',{15,16,16,14,1,1,30}},{'T',{31,4,4,4,4,4,4}},
 {'U',{17,17,17,17,17,17,14}},{'V',{17,17,17,17,17,10,4}},{'W',{17,17,17,21,21,27,17}},
 {'X',{17,17,10,4,10,17,17}},{'Y',{17,17,10,4,4,4,4}},{'Z',{31,1,2,4,8,16,31}},
 {'.',{0,0,0,0,0,6,6}},{'-',{0,0,0,31,0,0,0}},{'/',{1,1,2,4,8,16,16}},
 {':',{0,6,6,0,6,6,0}},{'%',{17,2,4,4,8,16,17}}
};
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
    check(initial.As(&swap),"Query swap chain");check(swap->SetMaximumFrameLatency(1),"Set frame latency");
    frameWait=swap->GetFrameLatencyWaitableObject();factory->MakeWindowAssociation(window,DXGI_MWA_NO_ALT_ENTER);
    auto vs=compile("VS","vs_5_0"),ps=compile("PS","ps_5_0"),uvs=compile("UiVS","vs_5_0"),ups=compile("UiPS","ps_5_0");
    check(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertexShader),"Vertex shader");
    check(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixelShader),"Pixel shader");
    auto ids=compile("IdPS","ps_5_0");check(device->CreatePixelShader(ids->GetBufferPointer(),ids->GetBufferSize(),nullptr,&idPixelShader),"ID pixel shader");
    auto ds=compile("DepthPS","ps_5_0");check(device->CreatePixelShader(ds->GetBufferPointer(),ds->GetBufferSize(),nullptr,&depthPixelShader),"Depth pixel shader");
    check(device->CreateVertexShader(uvs->GetBufferPointer(),uvs->GetBufferSize(),nullptr,&uiVertexShader),"UI vertex shader");
    check(device->CreatePixelShader(ups->GetBufferPointer(),ups->GetBufferSize(),nullptr,&uiPixelShader),"UI pixel shader");
    D3D11_INPUT_ELEMENT_DESC elements[]={
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",1,DXGI_FORMAT_R32G32B32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"COLOR",1,DXGI_FORMAT_R8G8B8A8_UNORM,0,36,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",4,DXGI_FORMAT_R32_UINT,0,40,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"INSTANCE",0,DXGI_FORMAT_R32G32B32A32_FLOAT,1,0,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"INSTANCE",1,DXGI_FORMAT_R32G32B32A32_FLOAT,1,16,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"INSTANCE",2,DXGI_FORMAT_R32G32B32A32_FLOAT,1,32,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"INSTANCE",3,DXGI_FORMAT_R32G32B32A32_FLOAT,1,48,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,1,64,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"TEXCOORD",2,DXGI_FORMAT_R32_FLOAT,1,80,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"TEXCOORD",5,DXGI_FORMAT_R32_UINT,1,84,D3D11_INPUT_PER_INSTANCE_DATA,1}};
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
    rd.CullMode=D3D11_CULL_BACK;
    check(device->CreateRasterizerState(&rd,&effectRaster),"Projectile effect raster state");
    D3D11_DEPTH_STENCIL_DESC dd{};dd.DepthEnable=TRUE;dd.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;dd.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
    check(device->CreateDepthStencilState(&dd,&depthOn),"Depth state");
    dd.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
    check(device->CreateDepthStencilState(&dd,&effectDepth),"Projectile effect depth state");
    dd.DepthEnable=FALSE;dd.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
    check(device->CreateDepthStencilState(&dd,&depthOff),"UI depth state");
    D3D11_BLEND_DESC bd{};auto& rt=bd.RenderTarget[0];rt.BlendEnable=TRUE;rt.SrcBlend=D3D11_BLEND_SRC_ALPHA;
    rt.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOp=D3D11_BLEND_OP_ADD;rt.SrcBlendAlpha=D3D11_BLEND_ONE;
    rt.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOpAlpha=D3D11_BLEND_OP_ADD;rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
    check(device->CreateBlendState(&bd,&blend),"Blend state");
    rt.DestBlend=D3D11_BLEND_ONE;rt.SrcBlendAlpha=D3D11_BLEND_ZERO;rt.DestBlendAlpha=D3D11_BLEND_ONE;
    check(device->CreateBlendState(&bd,&additiveBlend),"Projectile additive blend state");
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
    idTexture.Reset();idReadback.Reset();idTarget.Reset();idDepth.Reset();idWidth=0;idHeight=0;
    ComPtr<ID3D11Texture2D> back;
    check(swap->GetBuffer(0,IID_PPV_ARGS(&back)),"Swap chain buffer");
    check(device->CreateRenderTargetView(back.Get(),nullptr,&target),"Render target");
    D3D11_TEXTURE2D_DESC d{};d.Width=w;d.Height=h;d.MipLevels=1;d.ArraySize=1;d.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;
    d.SampleDesc.Count=1;d.BindFlags=D3D11_BIND_DEPTH_STENCIL;
    ComPtr<ID3D11Texture2D> texture;check(device->CreateTexture2D(&d,nullptr,&texture),"Depth buffer");
    check(device->CreateDepthStencilView(texture.Get(),nullptr,&depth),"Depth view");
}
uint32_t Renderer::addMesh(const MeshData& mesh) {
    if(mesh.vertices.empty()||mesh.indices.empty()) throw std::runtime_error("Empty mesh");
    if(!mesh.colors.empty()&&mesh.colors.size()!=mesh.vertices.size())throw std::runtime_error("Invalid mesh colors");
    if(!mesh.tokens.empty()&&mesh.tokens.size()!=mesh.vertices.size())throw std::runtime_error("Invalid mesh object IDs");
    std::map<uint64_t,bool> edges;
    for(auto e:mesh.edges) edges[edgeKey(e[0],e[1])]=true;
    std::vector<Vertex> data;data.reserve(mesh.indices.size());
    for(size_t i=0;i+2<mesh.indices.size();i+=3) {
        uint32_t a=mesh.indices[i],b=mesh.indices[i+1],c=mesh.indices[i+2];
        if(std::max({a,b,c})>=mesh.vertices.size()) throw std::runtime_error("Invalid mesh index");
        V3 flags{float(edges.contains(edgeKey(b,c))),float(edges.contains(edgeKey(a,c))),float(edges.contains(edgeKey(a,b)))};
        auto color=[&](uint32_t index){return mesh.colors.empty()?0xffffffffu:mesh.colors[index];};
        auto token=[&](uint32_t index){return mesh.tokens.empty()?0u:mesh.tokens[index];};
        if(token(a)!=token(b)||token(a)!=token(c))throw std::runtime_error("Triangle spans different logical objects");
        data.push_back({mesh.vertices[a],{1,0,0},flags,color(a),token(a)});data.push_back({mesh.vertices[b],{0,1,0},flags,color(b),token(b)});data.push_back({mesh.vertices[c],{0,0,1},flags,color(c),token(c)});
    }
    GpuMesh gpu;gpu.count=UINT(data.size());
    D3D11_BUFFER_DESC d{};d.ByteWidth=UINT(data.size()*sizeof(Vertex));d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA initial{data.data(),0,0};check(device->CreateBuffer(&d,&initial,&gpu.buffer),"Mesh buffer");
    meshes.push_back(gpu);batches.emplace_back();effectBatches.emplace_back();return uint32_t(meshes.size()-1);
}
std::vector<VisibleObject> Renderer::captureVisibleObjects(const std::vector<CatalogObject>& objects,std::vector<uint32_t>* ids) {
    if(!idTexture){
        idWidth=1280;idHeight=720;
        D3D11_TEXTURE2D_DESC d{};d.Width=UINT(idWidth);d.Height=UINT(idHeight);d.MipLevels=1;d.ArraySize=1;
        d.Format=DXGI_FORMAT_R32_UINT;d.SampleDesc.Count=1;d.BindFlags=D3D11_BIND_RENDER_TARGET;
        check(device->CreateTexture2D(&d,nullptr,&idTexture),"ID target texture");
        check(device->CreateRenderTargetView(idTexture.Get(),nullptr,&idTarget),"ID target view");
        d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        check(device->CreateTexture2D(&d,nullptr,&idReadback),"ID readback texture");
        d.Usage=D3D11_USAGE_DEFAULT;d.CPUAccessFlags=0;d.BindFlags=D3D11_BIND_DEPTH_STENCIL;d.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;
        ComPtr<ID3D11Texture2D> depthTexture;check(device->CreateTexture2D(&d,nullptr,&depthTexture),"ID depth texture");
        check(device->CreateDepthStencilView(depthTexture.Get(),nullptr,&idDepth),"ID depth view");
    }
    const float clear[4]{};context->ClearRenderTargetView(idTarget.Get(),clear);context->ClearDepthStencilView(idDepth.Get(),D3D11_CLEAR_DEPTH,1,0);
    ID3D11RenderTargetView* rtv=idTarget.Get();context->OMSetRenderTargets(1,&rtv,idDepth.Get());
    D3D11_VIEWPORT viewport{0,0,float(idWidth),float(idHeight),0,1};context->RSSetViewports(1,&viewport);
    context->RSSetState(raster.Get());context->OMSetDepthStencilState(depthOn.Get(),0);context->OMSetBlendState(nullptr,nullptr,0xffffffff);
    context->IASetInputLayout(layout.Get());context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(vertexShader.Get(),nullptr,0);context->PSSetShader(idPixelShader.Get(),nullptr,0);
    ID3D11Buffer* cb=constants.Get();context->VSSetConstantBuffers(0,1,&cb);
    UINT instanceOffset=0;
    for(size_t i=0;i<meshes.size();++i)if(!batches[i].empty()){
        ID3D11Buffer* buffers[]={meshes[i].buffer.Get(),instances.Get()};UINT strides[]={sizeof(Vertex),sizeof(Instance)},offsets[]={0,0};
        context->IASetVertexBuffers(0,2,buffers,strides,offsets);
        context->DrawInstanced(meshes[i].count,UINT(batches[i].size()),0,instanceOffset);instanceOffset+=UINT(batches[i].size());
    }
    context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(idReadback.Get(),idTexture.Get());
    D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(idReadback.Get(),0,D3D11_MAP_READ,0,&mapped),"Read visible object IDs");
    std::vector<uint32_t> pixels(objects.size());bool invalid=false;
    if(ids)ids->resize(size_t(idWidth)*idHeight);
    for(int y=0;y<idHeight;++y){const auto* row=reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(mapped.pData)+size_t(y)*mapped.RowPitch);
        if(ids)std::copy_n(row,idWidth,ids->data()+size_t(y)*idWidth);
        for(int x=0;x<idWidth;++x){if(row[x]<pixels.size())++pixels[row[x]];else invalid=true;}}
    context->Unmap(idReadback.Get(),0);
    rtv=target.Get();context->OMSetRenderTargets(1,&rtv,depth.Get());viewport.Width=float(width);viewport.Height=float(height);context->RSSetViewports(1,&viewport);
    if(invalid)throw std::runtime_error("GPU returned an invalid catalog token");
    visibilityPixels=pixels;
    return reduceVisibleObjects(objects,pixels,double(width)*height/(double(idWidth)*idHeight));
}

CapturedView Renderer::captureScene(const std::vector<DrawObject>& objects,const Camera& camera,const std::vector<CatalogObject>& catalog,bool includeDepth) {
    if(!captureTextures[0]) {
        D3D11_TEXTURE2D_DESC d{};d.Width=1280;d.Height=720;d.MipLevels=1;d.ArraySize=1;d.Format=DXGI_FORMAT_R8G8B8A8_UNORM;d.SampleDesc.Count=1;
        for(int i=0;i<2;++i){d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_RENDER_TARGET;d.CPUAccessFlags=0;
            check(device->CreateTexture2D(&d,nullptr,&captureTextures[i]),"Capture texture");check(device->CreateRenderTargetView(captureTextures[i].Get(),nullptr,&captureTargets[i]),"Capture target");
            d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;check(device->CreateTexture2D(&d,nullptr,&captureReadback[i]),"Capture readback");}
        d.Usage=D3D11_USAGE_DEFAULT;d.CPUAccessFlags=0;d.BindFlags=D3D11_BIND_DEPTH_STENCIL;d.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;
        ComPtr<ID3D11Texture2D> texture;check(device->CreateTexture2D(&d,nullptr,&texture),"Capture depth texture");check(device->CreateDepthStencilView(texture.Get(),nullptr,&captureDepth),"Capture depth view");
    }
    auto savedTarget=target;auto savedDepth=depth;int savedWidth=width,savedHeight=height;
    const auto restore=[&]{target=savedTarget;depth=savedDepth;width=savedWidth;height=savedHeight;context->OMSetRenderTargets(0,nullptr,nullptr);};
    CapturedView result;
    try {
        target=captureTargets[0];depth=captureDepth;width=1280;height=720;
        std::vector<DrawObject> boundsObjects;boundsObjects.reserve(objects.size());
        for(const auto& object:objects)if(object.instance.objectToken){boundsObjects.push_back(object);if(object.captureColor.a>=0)boundsObjects.back().instance.color=object.captureColor;}
        render(boundsObjects,camera,{});
        context->CopyResource(captureReadback[0].Get(),captureTextures[0].Get());
        if(includeDepth){
        const float clear[4]={0,0,0,1};context->ClearRenderTargetView(captureTargets[1].Get(),clear);context->ClearDepthStencilView(captureDepth.Get(),D3D11_CLEAR_DEPTH,1,0);
        ID3D11RenderTargetView* rtv=captureTargets[1].Get();context->OMSetRenderTargets(1,&rtv,captureDepth.Get());context->PSSetShader(depthPixelShader.Get(),nullptr,0);
        UINT instanceOffset=0;for(size_t i=0;i<meshes.size();++i)if(!batches[i].empty()){
            ID3D11Buffer* buffers[]={meshes[i].buffer.Get(),instances.Get()};UINT strides[]={sizeof(Vertex),sizeof(Instance)},offsets[]={0,0};context->IASetVertexBuffers(0,2,buffers,strides,offsets);
            context->DrawInstanced(meshes[i].count,UINT(batches[i].size()),0,instanceOffset);instanceOffset+=UINT(batches[i].size());}
        context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(captureReadback[1].Get(),captureTextures[1].Get());
        }
        result.visible=captureVisibleObjects(catalog,&result.ids);result.pixels=visibilityPixels;
        for(int i=0;i<(includeDepth?2:1);++i){auto& bytes=i==0?result.color:result.depth;bytes.resize(1280*720*4);D3D11_MAPPED_SUBRESOURCE m{};check(context->Map(captureReadback[i].Get(),0,D3D11_MAP_READ,0,&m),"Capture readback map");for(int y=0;y<720;++y)memcpy(bytes.data()+size_t(y)*1280*4,static_cast<const uint8_t*>(m.pData)+size_t(y)*m.RowPitch,1280*4);context->Unmap(captureReadback[i].Get(),0);}
    }catch(...){restore();throw;}
    restore();return result;
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
void Renderer::render(const std::vector<DrawObject>& objects,const Camera& camera,const HudState& hud) {
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
    XMStoreFloat4x4(&c.viewProjection,view*projection);c.w=float(width);c.h=float(height);c.outline=outlineWidth;c.pad=fadeSmallEdges?1.f:0.f;
    D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(constants.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped),"Map constants");
    memcpy(mapped.pData,&c,sizeof(c));context->Unmap(constants.Get(),0);
    ID3D11Buffer* cb=constants.Get();context->VSSetConstantBuffers(0,1,&cb);context->PSSetConstantBuffers(0,1,&cb);
    V3 forward=(camera.target-camera.eye).normalized();
    V3 right=cross({0,1,0},forward).normalized();V3 up=cross(forward,right);
    const float tx=std::tan(camera.horizontalFov*Pi/360),ty=tx/aspect;
    visibleCount=0;drawCalls=0;for(auto& b:batches)b.clear();for(auto& b:effectBatches)b.clear();
    for(const auto& object:objects) {
        V3 diff=object.center-camera.eye;float z=dot(diff,forward),r=object.radius;
        if(z+r<.08f||diff.length()>object.range+r||std::abs(dot(diff,right))>z*tx+r*(1+tx)||std::abs(dot(diff,up))>z*ty+r*(1+ty)) continue;
        if(visibleCount==MaxInstances) break;
        Instance instance=object.instance;
        if(object.maxScreenDiameter>0){
            if(z<=.08f)continue;
            // Keep enlarged projectile visuals from filling the view when they cross the lens.
            // Color, depth and ID passes all consume this same adjusted instance.
            if(instance.surface< -1.5){
                // Trail thickness is the safety envelope. Scaling the segment's
                // length independently opens gaps between adjacent path samples.
                const auto& m=instance.world;
                float transverse=.5f*std::sqrt(m._11*m._11+m._12*m._12+m._13*m._13+m._31*m._31+m._32*m._32+m._33*m._33);
                float closest=z-.5f*std::abs(dot({m._21,m._22,m._23},forward));
                float scale=std::min(1.f,object.maxScreenDiameter*ty*std::max(.001f,closest-.08f)/std::max(.001f,transverse));
                for(int row:{0,2})for(int column=0;column<3;++column)instance.world.m[row][column]*=scale;
            }else{
                float scale=std::min(1.f,object.maxScreenDiameter*ty*(z-.08f)/std::max(.001f,r));
                for(int row=0;row<3;++row)for(int column=0;column<3;++column)instance.world.m[row][column]*=scale;
            }
        }
        (instance.surface< -1.5?effectBatches:batches).at(object.mesh).push_back(instance);++visibleCount;
    }
    if(visibleCount) {
        check(context->Map(instances.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped),"Map instances");
        auto output=static_cast<Instance*>(mapped.pData);for(auto& b:batches) {memcpy(output,b.data(),b.size()*sizeof(Instance));output+=b.size();}
        for(auto& b:effectBatches) {memcpy(output,b.data(),b.size()*sizeof(Instance));output+=b.size();}
        context->Unmap(instances.Get(),0);
        UINT instanceOffset=0;
        for(size_t i=0;i<meshes.size();++i) if(!batches[i].empty()) {
            ID3D11Buffer* buffers[]={meshes[i].buffer.Get(),instances.Get()};UINT strides[]={sizeof(Vertex),sizeof(Instance)},offsets[]={0,0};
            context->IASetVertexBuffers(0,2,buffers,strides,offsets);
            context->DrawInstanced(meshes[i].count,UINT(batches[i].size()),0,instanceOffset);
            instanceOffset+=UINT(batches[i].size());++drawCalls;
        }
        // Draw glow after all opaque geometry. Addition is order independent;
        // depth testing keeps walls/characters in front, without solid dark tails.
        if(instanceOffset<visibleCount){
        context->RSSetState(effectRaster.Get());context->OMSetDepthStencilState(effectDepth.Get(),0);context->OMSetBlendState(additiveBlend.Get(),nullptr,0xffffffff);
        for(size_t i=0;i<meshes.size();++i)if(!effectBatches[i].empty()) {
            ID3D11Buffer* buffers[]={meshes[i].buffer.Get(),instances.Get()};UINT strides[]={sizeof(Vertex),sizeof(Instance)},offsets[]={0,0};
            context->IASetVertexBuffers(0,2,buffers,strides,offsets);
            context->DrawInstanced(meshes[i].count,UINT(effectBatches[i].size()),0,instanceOffset);
            instanceOffset+=UINT(effectBatches[i].size());++drawCalls;
        }
        context->RSSetState(raster.Get());context->OMSetDepthStencilState(depthOn.Get(),0);context->OMSetBlendState(nullptr,nullptr,0xffffffff);
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
void Renderer::rectangle(float x,float y,float w,float h,Color c) {
    ui.insert(ui.end(),{{x,y,c},{x+w,y,c},{x+w,y+h,c},{x,y,c},{x+w,y+h,c},{x,y+h,c}});
}
void Renderer::line(float x1,float y1,float x2,float y2,float thickness,Color c) {
    float dx=x2-x1,dy=y2-y1,l=std::sqrt(dx*dx+dy*dy);if(l<.01f)return;
    float ox=-dy/l*thickness*.5f,oy=dx/l*thickness*.5f;
    ui.insert(ui.end(),{{x1+ox,y1+oy,c},{x2+ox,y2+oy,c},{x2-ox,y2-oy,c},{x1+ox,y1+oy,c},{x2-ox,y2-oy,c},{x1-ox,y1-oy,c}});
}
void Renderer::text(float x,float y,const std::string& value,float scale,Color c) {
    for(char ch:value) { auto it=Font.find(char(std::toupper(static_cast<unsigned char>(ch))));if(it!=Font.end())
        for(int row=0;row<7;++row)for(int col=0;col<5;++col)if(it->second[row]&(1<<(4-col)))rectangle(x+col*scale,y+row*scale,scale,scale,c);
        x+=6*scale;
    }
}
void Renderer::drawHud(const HudState& hud) {
    if(!hud.visible&&!hud.showFps&&!hud.combat)return;
    ui.clear();Color white=rgb(231,236,232),green=rgb(94,215,159);
    if(hud.showFps) {
        std::ostringstream label;
        label<<"FPS ";
        if(hud.fps>0&&std::isfinite(hud.fps)) label<<std::fixed<<std::setprecision(1)<<hud.fps;
        else label<<"--";
        rectangle(12,12,156,56,{.025f,.025f,.025f,.85f});
        text(22,22,label.str(),2,white);
        std::ostringstream timing;
        if(hud.fps>0&&std::isfinite(hud.fps)) timing<<std::fixed<<std::setprecision(2)<<1000/hud.fps<<" MS";
        else timing<<"-- MS";
        text(22,44,timing.str(),2,hud.fps>=1000.0/3?green:white);
    }
    if(hud.combat) {
        float cx=width*.5f,cy=height*.5f;
        line(cx-12,cy,cx-5,cy,1.5f,white);line(cx+5,cy,cx+12,cy,1.5f,white);
        line(cx,cy-12,cx,cy-5,1.5f,white);line(cx,cy+5,cx,cy+12,1.5f,white);
        float right=float(width)-(width<600?168:240);
        text(right,22,"P"+std::to_string(hud.player)+(hud.automatic?"  AUTO":"  MANUAL"),2,white);
        if(hud.recording){rectangle(right,48,8,8,rgb(255,73,73));text(right+16,46,"REC "+std::to_string(hud.recordFrames),2,rgb(255,100,100));}
        float y=float(height)-80-CombatUi::footerHeight(width);bool compact=width<600;
        float leftWidth=compact?136.f:210.f,weaponWidth=compact?float(width)-172:246.f,weaponX=float(width)-weaponWidth-12;
        rectangle(12,y,leftWidth,62,{.025f,.025f,.025f,.8f});text(22,y+10,"HP "+std::to_string(int(hud.health)),compact?1.5f:2.f,white);
        rectangle(22,y+30,leftWidth-20,5,{.2f,.2f,.2f,1});rectangle(22,y+30,(leftWidth-20)*std::clamp(hud.health/100,0.f,1.f),5,green);
        text(22,y+44,"SHIELD "+std::to_string(int(hud.shield)),compact?1.2f:1.5f,hud.raised?rgb(105,177,255):white);
        rectangle(weaponX,y,weaponWidth,62,{.025f,.025f,.025f,.8f});const char* names[]={"PISTOL","RIFLE","SHOTGUN"};
        text(weaponX+12,y+10,names[hud.weapon],compact?1.5f:2.f,white);text(float(width)-76,y+10,std::to_string(hud.ammo),compact?2.f:3.f,white);
        text(weaponX+12,y+36,hud.combatAction,1.5f,green);
        text(weaponX+12,y+50,"TRAP "+(hud.trapCooldown>0?std::to_string(int(std::ceil(hud.trapCooldown)))+" S":"READY"),1.2f,white);
        rectangle(weaponX,y-30,weaponWidth,26,{.025f,.025f,.025f,.8f});
        text(weaponX+12,y-22,"H COVER "+hud.coverStatus,1.3f,hud.coverDurability>0?green:white);
        if(!hud.coverHint.empty())text(weaponX-24,y-45,hud.coverHint,1.2f,rgb(255,200,120));
        if(width>=850)for(int i=0;i<3;++i){float x=cx-60+i*42;rectangle(x,y+25,34,3,hud.owned[i]?i==hud.weapon?white:green:rgb(66,66,66));text(x+12,y+8,std::to_string(i+1),1.6f,hud.owned[i]?white:rgb(88,88,88));}
    }
    if(hud.visible) {
        float x=28,y=float(height)-86;
        rectangle(x-12,y-12,202,69,{.03f,.045f,.045f,.82f});
        std::ostringstream speed;speed<<std::setfill('0')<<std::setw(3)<<int(std::abs(hud.speed));
        text(x,y,speed.str(),5,white);text(x+105,y+20,"KM/H",2,white);
        text(x+169,y+7,hud.speed<-.5?"R":"D",3,green);
        float mx=float(width)-153,my=float(height)-153;
        rectangle(mx-73,my-73,146,146,{.025f,.04f,.04f,.86f});
        float mapScale=62/std::max(100.f,hud.mapHalfSize);
        if(hud.roads)for(const auto& road:*hud.roads){
            line(mx+(road.a.x-hud.mapCenter.x)*mapScale,my-(road.a.z-hud.mapCenter.z)*mapScale,mx+(road.b.x-hud.mapCenter.x)*mapScale,my-(road.b.z-hud.mapCenter.z)*mapScale,road.freeway?1.7f:.8f,road.freeway?rgb(164,149,99):rgb(99,107,105));}
        float px=mx+std::clamp((hud.location.x-hud.mapCenter.x)*mapScale,-62.f,62.f),py=my-std::clamp((hud.location.z-hud.mapCenter.z)*mapScale,-62.f,62.f);
        float dx=std::sin(hud.heading),dy=-std::cos(hud.heading);
        line(px-dx*3,py-dy*3,px+dx*6,py+dy*6,3,green);rectangle(px-2,py-2,4,4,green);
        text(mx-62,my-64,"N",1,white);
    }
    if(hud.performance) {
        auto ms=[](double v) {std::ostringstream s;s<<std::fixed<<std::setprecision(2)<<v;return s.str();};
        rectangle(16,16,352,74,{.015f,.025f,.025f,.9f});
        text(28,27,"FRAME "+ms(hud.frameMs)+" MS",2,white);
        text(28,49,"CPU "+ms(hud.cpuMs)+"  GPU "+ms(hud.gpuMs),2,white);
        text(28,71,"OBJECTS "+std::to_string(visibleCount),1,green);
    }
    if(hud.combat)drawCombatOverlay(hud);
    if(hud.paused&&!hud.combat) {
        rectangle(0,0,float(width),float(height),{0,0,0,.66f});
        float px=width*.5f-150,py=height*.5f-112;
        text(px,py,"PAUSED",4,white);
        const char* labels[]={"RESUME","RESET VEHICLE","EXIT"};
        for(int i=0;i<3;++i) {
            float y=py+65+i*48;
            if(i==hud.pauseSelection)rectangle(px-10,y-12,305,38,{.15f,.25f,.21f,1});
            text(px,y,labels[i],2,i==hud.pauseSelection?green:white);
        }
    }
    if(ui.empty()) return;
    if(ui.size()>MaxUiVertices)throw std::runtime_error("HUD buffer exceeded");
    D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(uiBuffer.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped),"Map HUD");
    memcpy(mapped.pData,ui.data(),ui.size()*sizeof(UiVertex));context->Unmap(uiBuffer.Get(),0);
    context->OMSetDepthStencilState(depthOff.Get(),0);context->OMSetBlendState(blend.Get(),nullptr,0xffffffff);
    context->IASetInputLayout(uiLayout.Get());context->VSSetShader(uiVertexShader.Get(),nullptr,0);context->PSSetShader(uiPixelShader.Get(),nullptr,0);
    ID3D11Buffer* buffer=uiBuffer.Get();UINT stride=sizeof(UiVertex),offset=0;context->IASetVertexBuffers(0,1,&buffer,&stride,&offset);
    context->Draw(UINT(ui.size()),0);
}
#include "RendererCombatHud.inc"
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
