#include "Renderer.h"
#include "HeadlessGL.h"
#include <SDL.h>
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GL/glew.h>
#endif
#include <map>
#include <cstring>
#include <cctype>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace {
#include "BitmapFont.h"
constexpr size_t MaxInstances=65536,MaxUiVertices=100000;
struct Vertex {V3 p,bary,edge;uint32_t color,token;};
const char* VertexShader=R"(#version 330 core
layout(location=0) in vec3 pos;layout(location=1) in vec3 bary;layout(location=2) in vec3 edge;
layout(location=3) in vec4 meshColor;layout(location=4) in uint meshToken;
layout(location=5) in vec4 m0;layout(location=6) in vec4 m1;layout(location=7) in vec4 m2;layout(location=8) in vec4 m3;
layout(location=9) in vec4 instanceColor;layout(location=10) in float instanceSurface;layout(location=11) in uint instanceToken;
uniform mat4 viewProjection;out vec3 vbary;flat out vec3 vedge;flat out vec4 color;out vec3 world;
flat out float surface;flat out uint token;flat out vec3 effectAxis;
void main(){vec4 p=mat4(m0,m1,m2,m3)*vec4(pos,1);gl_Position=viewProjection*p;gl_Position.z=2*gl_Position.z-gl_Position.w;
vbary=bary;vedge=edge;color=instanceColor*meshColor;world=p.xyz;surface=instanceSurface;token=meshToken!=0u?meshToken:instanceToken;effectAxis=m1.xyz;})";
const char* FragmentShader=R"(#version 330 core
in vec3 vbary;flat in vec3 vedge;flat in vec4 color;in vec3 world;flat in float surface;flat in uint token;flat in vec3 effectAxis;
uniform float outlineWidth;uniform bool fadeSmallEdges;out vec4 pixel;
float checker(vec2 uv){vec2 w=max(fwidth(uv),vec2(.0001));vec2 integral=2*(abs(fract((uv-.5*w)*.5)-.5)-abs(fract((uv+.5*w)*.5)-.5))/w;return clamp(.5-.5*integral.x*integral.y,0,1);}
void main(){vec3 c=color.rgb;if(surface< -1.5){vec3 n=normalize(cross(dFdx(world),dFdy(world)));float cap=1-clamp(abs(dot(n,normalize(effectAxis)))*10,0,1);pixel=vec4(c,clamp(color.a,0,1)*cap);return;}
if(surface>1.5){vec3 n=normalize(cross(dFdx(world),dFdy(world)));vec2 uv=world.xz;if(surface==3.0||surface==5.0){vec3 a=abs(n);if(a.x>max(a.y,a.z))uv=world.zy;else if(a.z>a.y)uv=world.xy;}
// Two aligned scales: 1 m cells, each containing 10x10 cells of 0.1 m.
float coarse=checker(uv),fine=checker(uv*10);float gray=mix(.66,.71,coarse)+(fine-.5)*.02;
// Egypt: retain 0.5 m small cells, grouped into 4 m blocks (8x8 cells).
if(surface>3.5)gray=mix(.70,.73,checker(uv*.25))+(checker(uv*2)-.5)*.012;
c=vec3(gray)*(.95+.05*abs(dot(n,normalize(vec3(-.5,1,-.35)))));}
else if(surface>.5){vec2 q=floor(world.xz*30);float n=fract(sin(dot(q,vec2(12.9898,78.233)))*43758.5453);float patch=sin(world.x*.31+sin(world.z*.42))*sin(world.z*.22);float fade=clamp(1-length(dFdx(world)+dFdy(world))*3,0,1);c+=(n-.5)*.034*fade+patch*.014;}
else if(surface>-.5){vec3 gradient=max(fwidth(vbary),vec3(.00001));vec3 d=mix(vec3(10000),abs(vbary)/gradient,vedge);
// Only filter the stroke boundary; small/far triangles retain black edges.
if(fadeSmallEdges){vec3 altitude=1/gradient;float strength=smoothstep(2.5,9,min(altitude.x,min(altitude.y,altitude.z)));if(outlineWidth>0&&min(d.x,min(d.y,d.z))<outlineWidth)c*=1-strength;}
else if(outlineWidth>0)c*=smoothstep(max(0.0,outlineWidth-.5),outlineWidth+.5,min(d.x,min(d.y,d.z)));}
pixel=vec4(c,1);})";
const char* IdShader=R"(#version 330 core
flat in uint token;layout(location=0) out uint id;void main(){id=token;})";
const char* DepthShader=R"(#version 330 core
out vec4 pixel;void main(){float meters=.08*2000/(2000-gl_FragCoord.z*(2000-.08));pixel=vec4(vec3(clamp(meters/200,0,1)),1);})";
const char* UiVertexShader=R"(#version 330 core
layout(location=0) in vec2 pos;layout(location=1) in vec4 color;uniform vec2 viewport;out vec4 shade;
void main(){gl_Position=vec4(pos.x/viewport.x*2-1,1-pos.y/viewport.y*2,0,1);shade=color;})";
const char* UiFragmentShader=R"(#version 330 core
in vec4 shade;out vec4 pixel;void main(){pixel=shade;})";
GLuint program(const char* vs,const char* fs){auto compile=[](GLenum kind,const char* source){GLuint shader=glCreateShader(kind);glShaderSource(shader,1,&source,nullptr);glCompileShader(shader);GLint ok;glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);if(!ok){char log[8192];glGetShaderInfoLog(shader,sizeof(log),nullptr,log);glDeleteShader(shader);throw std::runtime_error(log);}return shader;};
    GLuint v=compile(GL_VERTEX_SHADER,vs),f=compile(GL_FRAGMENT_SHADER,fs),p=glCreateProgram();glAttachShader(p,v);glAttachShader(p,f);glLinkProgram(p);glDeleteShader(v);glDeleteShader(f);GLint ok;glGetProgramiv(p,GL_LINK_STATUS,&ok);if(!ok){char log[8192];glGetProgramInfoLog(p,sizeof(log),nullptr,log);throw std::runtime_error(log);}return p;}
struct Target {GLuint fbo=0,color=0,depth=0;};
Target target(bool integer){Target t;glGenFramebuffers(1,&t.fbo);glBindFramebuffer(GL_FRAMEBUFFER,t.fbo);glGenTextures(1,&t.color);glBindTexture(GL_TEXTURE_2D,t.color);glTexImage2D(GL_TEXTURE_2D,0,integer?GL_R32UI:GL_RGBA8,1280,720,0,integer?GL_RED_INTEGER:GL_RGBA,integer?GL_UNSIGNED_INT:GL_UNSIGNED_BYTE,nullptr);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t.color,0);
glGenRenderbuffers(1,&t.depth);glBindRenderbuffer(GL_RENDERBUFFER,t.depth);glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,1280,720);glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_STENCIL_ATTACHMENT,GL_RENDERBUFFER,t.depth);if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)throw std::runtime_error("OpenGL capture framebuffer incomplete");return t;}
void destroy(Target& t){glDeleteFramebuffers(1,&t.fbo);glDeleteTextures(1,&t.color);glDeleteRenderbuffers(1,&t.depth);}
template<class T>void flip(std::vector<T>& bytes,int w,int h,int channels){for(int y=0;y<h/2;++y)for(int x=0;x<w*channels;++x)std::swap(bytes[size_t(y)*w*channels+x],bytes[size_t(h-1-y)*w*channels+x]);}
uint64_t edgeKey(uint32_t a,uint32_t b){return (uint64_t(std::min(a,b))<<32)|std::max(a,b);}
}
struct Renderer::Impl {
    struct Mesh {GLuint vao=0,vbo=0;GLsizei count=0;};SDL_Window* window;SDL_GLContext context=nullptr;std::unique_ptr<HeadlessGL> headless;
    GLuint colorProgram=0,idProgram=0,depthProgram=0,uiProgram=0,instanceBuffer=0,uiVao=0,uiBuffer=0;Target colorTarget,depthTarget,idTarget;
    struct Timer {GLuint query=0;bool pending=false;};std::array<Timer,8> timers;uint64_t frame=0;
    std::vector<Mesh> meshes;std::vector<std::vector<Instance>> batches,effects;Matrix viewProjection{};GLuint destination=0;Camera camera;std::vector<uint32_t> pixels;
    Impl(SDL_Window* w):window(w){if(w){context=SDL_GL_CreateContext(w);if(!context)throw std::runtime_error(SDL_GetError());}else headless=std::make_unique<HeadlessGL>();
#ifndef __APPLE__
        glewExperimental=GL_TRUE;auto glewStatus=glewInit();if(glewStatus!=GLEW_OK&&!(headless&&glewStatus==GLEW_ERROR_NO_GLX_DISPLAY))throw std::runtime_error("GLEW initialization failed");glGetError();if(!GLEW_VERSION_3_3)throw std::runtime_error("OpenGL 3.3 required");
#endif
        colorProgram=program(VertexShader,FragmentShader);idProgram=program(VertexShader,IdShader);depthProgram=program(VertexShader,DepthShader);uiProgram=program(UiVertexShader,UiFragmentShader);
        for(auto& t:timers)glGenQueries(1,&t.query);glGenBuffers(1,&instanceBuffer);glGenVertexArrays(1,&uiVao);glGenBuffers(1,&uiBuffer);colorTarget=target(false);depthTarget=target(false);idTarget=target(true);glBindFramebuffer(GL_FRAMEBUFFER,0);glDisable(GL_CULL_FACE);glEnable(GL_DEPTH_TEST);glDepthFunc(GL_LESS);SDL_GL_SetSwapInterval(0);
    }
    ~Impl(){for(auto& m:meshes){glDeleteVertexArrays(1,&m.vao);glDeleteBuffers(1,&m.vbo);}for(auto& t:timers)glDeleteQueries(1,&t.query);destroy(colorTarget);destroy(depthTarget);destroy(idTarget);for(auto p:{colorProgram,idProgram,depthProgram,uiProgram})glDeleteProgram(p);glDeleteBuffers(1,&instanceBuffer);glDeleteBuffers(1,&uiBuffer);glDeleteVertexArrays(1,&uiVao);if(context)SDL_GL_DeleteContext(context);}
    void draw(GLuint shader,bool effectsToo=false){glUseProgram(shader);glUniformMatrix4fv(glGetUniformLocation(shader,"viewProjection"),1,GL_FALSE,&viewProjection.m[0][0]);
        auto pass=[&](const std::vector<std::vector<Instance>>& groups){for(size_t i=0;i<groups.size();++i)if(!groups[i].empty()){glBindVertexArray(meshes[i].vao);glBindBuffer(GL_ARRAY_BUFFER,instanceBuffer);glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(groups[i].size()*sizeof(Instance)),groups[i].data(),GL_STREAM_DRAW);glDrawArraysInstanced(GL_TRIANGLES,0,meshes[i].count,GLsizei(groups[i].size()));}};
        pass(batches);if(effectsToo){glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA,GL_ONE);glDepthMask(GL_FALSE);pass(effects);glDepthMask(GL_TRUE);glDisable(GL_BLEND);}}
};
Renderer::Renderer(SDL_Window* window,int w,int h):width(w),height(h),impl(std::make_unique<Impl>(window)){adapterName=reinterpret_cast<const char*>(glGetString(GL_RENDERER));tearing=true;}
Renderer::~Renderer()=default;
void Renderer::resize(int w,int h){if(w>0&&h>0){width=w;height=h;}}
uint32_t Renderer::addMesh(const MeshData& mesh){if(mesh.vertices.empty()||mesh.indices.empty())throw std::runtime_error("Empty mesh");std::map<uint64_t,bool> edges;for(auto e:mesh.edges)edges[edgeKey(e[0],e[1])]=true;std::vector<Vertex> data;
    for(size_t i=0;i+2<mesh.indices.size();i+=3){uint32_t a=mesh.indices[i],b=mesh.indices[i+1],c=mesh.indices[i+2];if(std::max({a,b,c})>=mesh.vertices.size())throw std::runtime_error("Mesh index out of range");V3 flags{float(edges.contains(edgeKey(b,c))),float(edges.contains(edgeKey(a,c))),float(edges.contains(edgeKey(a,b)))};
        auto col=[&](uint32_t x){return mesh.colors.empty()?0xffffffffu:mesh.colors.at(x);};auto tok=[&](uint32_t x){return mesh.tokens.empty()?0u:mesh.tokens.at(x);};data.push_back({mesh.vertices[a],{1,0,0},flags,col(a),tok(a)});data.push_back({mesh.vertices[b],{0,1,0},flags,col(b),tok(b)});data.push_back({mesh.vertices[c],{0,0,1},flags,col(c),tok(c)});}
    Impl::Mesh m;glGenVertexArrays(1,&m.vao);glBindVertexArray(m.vao);glGenBuffers(1,&m.vbo);glBindBuffer(GL_ARRAY_BUFFER,m.vbo);glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(data.size()*sizeof(Vertex)),data.data(),GL_STATIC_DRAW);m.count=GLsizei(data.size());
    for(int i=0;i<3;++i){glEnableVertexAttribArray(i);glVertexAttribPointer(i,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(size_t(i)*sizeof(V3)));}glEnableVertexAttribArray(3);glVertexAttribPointer(3,4,GL_UNSIGNED_BYTE,GL_TRUE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,color)));glEnableVertexAttribArray(4);glVertexAttribIPointer(4,1,GL_UNSIGNED_INT,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,token)));
    glBindBuffer(GL_ARRAY_BUFFER,impl->instanceBuffer);for(int row=0;row<4;++row){glEnableVertexAttribArray(5+row);glVertexAttribPointer(5+row,4,GL_FLOAT,GL_FALSE,sizeof(Instance),reinterpret_cast<void*>(size_t(row)*16));glVertexAttribDivisor(5+row,1);}glEnableVertexAttribArray(9);glVertexAttribPointer(9,4,GL_FLOAT,GL_FALSE,sizeof(Instance),reinterpret_cast<void*>(offsetof(Instance,color)));glVertexAttribDivisor(9,1);glEnableVertexAttribArray(10);glVertexAttribPointer(10,1,GL_FLOAT,GL_FALSE,sizeof(Instance),reinterpret_cast<void*>(offsetof(Instance,surface)));glVertexAttribDivisor(10,1);glEnableVertexAttribArray(11);glVertexAttribIPointer(11,1,GL_UNSIGNED_INT,sizeof(Instance),reinterpret_cast<void*>(offsetof(Instance,objectToken)));glVertexAttribDivisor(11,1);
    impl->meshes.push_back(m);impl->batches.emplace_back();impl->effects.emplace_back();return uint32_t(impl->meshes.size()-1);}
void Renderer::render(const std::vector<DrawObject>& objects,const Camera& camera,const HudState& hud){using namespace DirectX;newGpuSample=false;for(auto& t:impl->timers)if(t.pending){GLint ready=0;glGetQueryObjectiv(t.query,GL_QUERY_RESULT_AVAILABLE,&ready);if(ready){GLuint64 ns=0;glGetQueryObjectui64v(t.query,GL_QUERY_RESULT,&ns);latestGpuMs=double(ns)/1e6;newGpuSample=true;t.pending=false;}}
    auto& timer=impl->timers[impl->frame++%impl->timers.size()];bool measure=!timer.pending;if(measure)glBeginQuery(GL_TIME_ELAPSED,timer.query);
    impl->camera=camera;glBindFramebuffer(GL_FRAMEBUFFER,impl->destination);glViewport(0,0,width,height);glClearColor(0,0,0,1);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);glEnable(GL_DEPTH_TEST);glDepthMask(GL_TRUE);glDisable(GL_BLEND);
    float aspect=float(width)/height,tx=std::tan(camera.horizontalFov*Pi/360),ty=tx/aspect;auto view=XMMatrixLookAtLH(XMVectorSet(camera.eye.x,camera.eye.y,camera.eye.z,1),XMVectorSet(camera.target.x,camera.target.y,camera.target.z,1),XMVectorSet(0,1,0,0));auto projection=XMMatrixPerspectiveFovLH(2*std::atan(ty),aspect,.08f,2000);XMStoreFloat4x4(&impl->viewProjection,view*projection);
    V3 forward=(camera.target-camera.eye).normalized(),right=cross({0,1,0},forward).normalized(),up=cross(forward,right);visibleCount=drawCalls=0;for(auto& b:impl->batches)b.clear();for(auto& b:impl->effects)b.clear();
    for(const auto& object:objects){V3 delta=object.center-camera.eye;float z=dot(delta,forward),radius=object.radius;if(z+radius<.08f||delta.length()>object.range+radius||std::abs(dot(delta,right))>z*tx+radius*(1+tx)||std::abs(dot(delta,up))>z*ty+radius*(1+ty))continue;if(visibleCount>=MaxInstances)break;Instance instance=object.instance;
        if(object.maxScreenDiameter>0){if(z<=.08f)continue;if(instance.surface< -1.5f){const auto& m=instance.world;float transverse=.5f*std::sqrt(m._11*m._11+m._12*m._12+m._13*m._13+m._31*m._31+m._32*m._32+m._33*m._33);float closest=z-.5f*std::abs(dot({m._21,m._22,m._23},forward));float scale=std::min(1.f,object.maxScreenDiameter*ty*std::max(.001f,closest-.08f)/std::max(.001f,transverse));for(int row:{0,2})for(int col=0;col<3;++col)instance.world.m[row][col]*=scale;}else{float scale=std::min(1.f,object.maxScreenDiameter*ty*(z-.08f)/std::max(.001f,radius));for(int row=0;row<3;++row)for(int col=0;col<3;++col)instance.world.m[row][col]*=scale;}}
        (instance.surface< -1.5f?impl->effects:impl->batches).at(object.mesh).push_back(instance);++visibleCount;}
    glUseProgram(impl->colorProgram);glUniform1f(glGetUniformLocation(impl->colorProgram,"outlineWidth"),outlineWidth);glUniform1i(glGetUniformLocation(impl->colorProgram,"fadeSmallEdges"),fadeSmallEdges);impl->draw(impl->colorProgram,true);for(size_t i=0;i<impl->batches.size();++i)drawCalls+=unsigned(!impl->batches[i].empty())+unsigned(!impl->effects[i].empty());drawHud(hud);if(measure){glEndQuery(GL_TIME_ELAPSED);timer.pending=true;}}
std::vector<VisibleObject> Renderer::captureVisibleObjects(const std::vector<CatalogObject>& catalog,std::vector<uint32_t>* ids){glBindFramebuffer(GL_FRAMEBUFFER,impl->idTarget.fbo);glViewport(0,0,1280,720);GLuint clear[4]{};glClearBufferuiv(GL_COLOR,0,clear);glClear(GL_DEPTH_BUFFER_BIT);glDisable(GL_BLEND);glEnable(GL_DEPTH_TEST);glDepthMask(GL_TRUE);impl->draw(impl->idProgram);std::vector<uint32_t> data(1280*720);glReadPixels(0,0,1280,720,GL_RED_INTEGER,GL_UNSIGNED_INT,data.data());flip(data,1280,720,1);impl->pixels.assign(catalog.size(),0);for(auto id:data){if(id>=catalog.size())throw std::runtime_error("Invalid OpenGL object ID");++impl->pixels[id];}if(ids)*ids=std::move(data);idWidth=1280;idHeight=720;glBindFramebuffer(GL_FRAMEBUFFER,impl->destination);glViewport(0,0,width,height);return reduceVisibleObjects(catalog,impl->pixels,double(width)*height/(1280.*720));}
CapturedView Renderer::captureScene(const std::vector<DrawObject>& objects,const Camera& camera,const std::vector<CatalogObject>& catalog,bool includeDepth){int oldWidth=width,oldHeight=height;GLuint oldTarget=impl->destination;auto restore=[&]{width=oldWidth;height=oldHeight;impl->destination=oldTarget;glBindFramebuffer(GL_FRAMEBUFFER,oldTarget);glViewport(0,0,width,height);};CapturedView result;
    try{width=1280;height=720;impl->destination=impl->colorTarget.fbo;std::vector<DrawObject> filtered;for(auto object:objects)if(object.instance.objectToken){if(object.captureColor.a>=0)object.instance.color=object.captureColor;filtered.push_back(object);}render(filtered,camera,{});result.color.resize(1280*720*4);glReadPixels(0,0,1280,720,GL_RGBA,GL_UNSIGNED_BYTE,result.color.data());flip(result.color,1280,720,4);
        if(includeDepth){glBindFramebuffer(GL_FRAMEBUFFER,impl->depthTarget.fbo);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);impl->draw(impl->depthProgram);result.depth.resize(1280*720*4);glReadPixels(0,0,1280,720,GL_RGBA,GL_UNSIGNED_BYTE,result.depth.data());flip(result.depth,1280,720,4);}result.visible=captureVisibleObjects(catalog,&result.ids);result.pixels=impl->pixels;
    }catch(...){restore();throw;}restore();return result;}
std::vector<uint8_t> Renderer::captureDemoScene(const std::vector<DrawObject>& objects,const Camera& camera){
    const int oldWidth=width,oldHeight=height;const auto oldTarget=impl->destination;
    auto restore=[&]{width=oldWidth;height=oldHeight;impl->destination=oldTarget;glBindFramebuffer(GL_FRAMEBUFFER,oldTarget);glViewport(0,0,width,height);};
    try{width=1280;height=720;impl->destination=impl->colorTarget.fbo;
        std::vector<DrawObject> filtered;for(auto object:objects)if(object.instance.objectToken){if(object.captureColor.a>=0)object.instance.color=object.captureColor;filtered.push_back(object);}
        render(filtered,camera,{});std::vector<uint8_t> pixels(1280*720*4);
        glReadPixels(0,0,1280,720,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());flip(pixels,1280,720,4);restore();return pixels;
    }catch(...){restore();throw;}
}
std::vector<uint8_t> Renderer::captureDemoUi(const HudState& hud){
    const int oldWidth=width,oldHeight=height;const auto oldTarget=impl->destination;
    auto restore=[&]{width=oldWidth;height=oldHeight;impl->destination=oldTarget;glBindFramebuffer(GL_FRAMEBUFFER,oldTarget);glViewport(0,0,width,height);};
    try{width=1280;height=720;impl->destination=impl->colorTarget.fbo;glBindFramebuffer(GL_FRAMEBUFFER,impl->destination);glViewport(0,0,width,height);
        glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);drawHud(hud);std::vector<uint8_t> pixels(1280*720*4);
        glReadPixels(0,0,1280,720,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());flip(pixels,1280,720,4);restore();return pixels;
    }catch(...){restore();throw;}
}
double Renderer::waitForFrame(bool){return 0;}
double Renderer::present(bool vsync){double start=now();SDL_GL_SetSwapInterval(vsync?1:0);SDL_GL_SwapWindow(impl->window);return (now()-start)*1000;}
bool Renderer::screenshot(const std::filesystem::path& path){std::filesystem::create_directories(path.parent_path());std::vector<uint8_t> image(size_t(width)*height*4);glBindFramebuffer(GL_FRAMEBUFFER,impl->destination);glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,image.data());flip(image,width,height,4);return stbi_write_png(path.c_str(),width,height,4,image.data(),width*4)!=0;}
void Renderer::submitHud(){glDisable(GL_DEPTH_TEST);glEnable(GL_BLEND);glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);glUseProgram(impl->uiProgram);glUniform2f(glGetUniformLocation(impl->uiProgram,"viewport"),float(width),float(height));glBindVertexArray(impl->uiVao);glBindBuffer(GL_ARRAY_BUFFER,impl->uiBuffer);glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(ui.size()*sizeof(UiVertex)),ui.data(),GL_STREAM_DRAW);glEnableVertexAttribArray(0);glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(UiVertex),nullptr);glEnableVertexAttribArray(1);glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,sizeof(UiVertex),reinterpret_cast<void*>(offsetof(UiVertex,color)));glDrawArrays(GL_TRIANGLES,0,GLsizei(ui.size()));glDisable(GL_BLEND);glEnable(GL_DEPTH_TEST);}
#include "RendererHudLinux.inc"
