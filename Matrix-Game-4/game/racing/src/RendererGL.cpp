#include "Renderer.h"
#include "HeadlessGL.h"
#include <SDL.h>
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#else
#define GL_GLEXT_PROTOTYPES 1
#include <SDL_opengl.h>
#endif
#include <zlib.h>
#include <map>
#include <cstring>
#include <sstream>
namespace {
thread_local HeadlessGL* activeHeadless=nullptr;
void* resolveGl(const char* name){return activeHeadless?activeHeadless->function(name):SDL_GL_GetProcAddress(name);}
}
#define SDL_GL_GetProcAddress resolveGl

// Resolve the GL entry points through the active SDL context; libGL need not
// export extension symbols (notably 64-bit timer queries on Mesa).
namespace glapi {
#define FRAMEBUFFER_FUNCTIONS(X) \
 X(glGenFramebuffers) X(glBindFramebuffer) X(glDeleteFramebuffers) X(glFramebufferTexture2D) X(glCheckFramebufferStatus) \
 X(glGenTextures) X(glBindTexture) X(glTexImage2D) X(glTexParameteri) X(glDeleteTextures) X(glActiveTexture) X(glUniform1i) X(glBlendFuncSeparate) X(glMapBufferRange) X(glUnmapBuffer) X(glBlitFramebuffer) X(glClearBufferuiv) X(glGetIntegerv)
#define DECLARE_GL(name) inline decltype(&::name) name=nullptr;
FRAMEBUFFER_FUNCTIONS(DECLARE_GL)
#undef DECLARE_GL
inline decltype(&::glAttachShader) glAttachShader=nullptr;
inline decltype(&::glBeginQuery) glBeginQuery=nullptr;
inline decltype(&::glBindBuffer) glBindBuffer=nullptr;
inline decltype(&::glBindVertexArray) glBindVertexArray=nullptr;
inline decltype(&::glBlendFunc) glBlendFunc=nullptr;
inline decltype(&::glBufferData) glBufferData=nullptr;
inline decltype(&::glClear) glClear=nullptr;
inline decltype(&::glClearColor) glClearColor=nullptr;
inline decltype(&::glClientWaitSync) glClientWaitSync=nullptr;
inline decltype(&::glCompileShader) glCompileShader=nullptr;
inline decltype(&::glCreateProgram) glCreateProgram=nullptr;
inline decltype(&::glCreateShader) glCreateShader=nullptr;
inline decltype(&::glDeleteBuffers) glDeleteBuffers=nullptr;
inline decltype(&::glDeleteProgram) glDeleteProgram=nullptr;
inline decltype(&::glDeleteQueries) glDeleteQueries=nullptr;
inline decltype(&::glDeleteShader) glDeleteShader=nullptr;
inline decltype(&::glDeleteSync) glDeleteSync=nullptr;
inline decltype(&::glDeleteVertexArrays) glDeleteVertexArrays=nullptr;
inline decltype(&::glDepthFunc) glDepthFunc=nullptr;
inline decltype(&::glDepthMask) glDepthMask=nullptr;
inline decltype(&::glDisable) glDisable=nullptr;
inline decltype(&::glDrawArrays) glDrawArrays=nullptr;
inline decltype(&::glDrawArraysInstanced) glDrawArraysInstanced=nullptr;
inline decltype(&::glEnable) glEnable=nullptr;
inline decltype(&::glEnableVertexAttribArray) glEnableVertexAttribArray=nullptr;
inline decltype(&::glEndQuery) glEndQuery=nullptr;
inline decltype(&::glFenceSync) glFenceSync=nullptr;
inline decltype(&::glFlush) glFlush=nullptr;
inline decltype(&::glGenBuffers) glGenBuffers=nullptr;
inline decltype(&::glGenQueries) glGenQueries=nullptr;
inline decltype(&::glGenVertexArrays) glGenVertexArrays=nullptr;
inline decltype(&::glGetError) glGetError=nullptr;
inline decltype(&::glGetProgramInfoLog) glGetProgramInfoLog=nullptr;
inline decltype(&::glGetProgramiv) glGetProgramiv=nullptr;
inline decltype(&::glGetQueryObjectiv) glGetQueryObjectiv=nullptr;
inline decltype(&::glGetQueryObjectui64v) glGetQueryObjectui64v=nullptr;
inline decltype(&::glGetShaderInfoLog) glGetShaderInfoLog=nullptr;
inline decltype(&::glGetShaderiv) glGetShaderiv=nullptr;
inline decltype(&::glGetString) glGetString=nullptr;
inline decltype(&::glGetUniformLocation) glGetUniformLocation=nullptr;
inline decltype(&::glLinkProgram) glLinkProgram=nullptr;
inline decltype(&::glPixelStorei) glPixelStorei=nullptr;
inline decltype(&::glReadBuffer) glReadBuffer=nullptr;
inline decltype(&::glReadPixels) glReadPixels=nullptr;
inline decltype(&::glShaderSource) glShaderSource=nullptr;
inline decltype(&::glUniform1f) glUniform1f=nullptr;
inline decltype(&::glUniform2f) glUniform2f=nullptr;
inline decltype(&::glUniformMatrix4fv) glUniformMatrix4fv=nullptr;
inline decltype(&::glUseProgram) glUseProgram=nullptr;
inline decltype(&::glVertexAttribDivisor) glVertexAttribDivisor=nullptr;
inline decltype(&::glVertexAttribPointer) glVertexAttribPointer=nullptr;
inline decltype(&::glViewport) glViewport=nullptr;
void load(){
#define LOAD_GL(name) name=reinterpret_cast<decltype(name)>(SDL_GL_GetProcAddress(#name));if(!name)throw std::runtime_error("Missing OpenGL entry point: " #name);
    FRAMEBUFFER_FUNCTIONS(LOAD_GL)
#undef LOAD_GL
    glAttachShader=reinterpret_cast<decltype(glAttachShader)>(SDL_GL_GetProcAddress("glAttachShader"));if(!glAttachShader)throw std::runtime_error("Missing OpenGL entry point: glAttachShader");
    glBeginQuery=reinterpret_cast<decltype(glBeginQuery)>(SDL_GL_GetProcAddress("glBeginQuery"));if(!glBeginQuery)throw std::runtime_error("Missing OpenGL entry point: glBeginQuery");
    glBindBuffer=reinterpret_cast<decltype(glBindBuffer)>(SDL_GL_GetProcAddress("glBindBuffer"));if(!glBindBuffer)throw std::runtime_error("Missing OpenGL entry point: glBindBuffer");
    glBindVertexArray=reinterpret_cast<decltype(glBindVertexArray)>(SDL_GL_GetProcAddress("glBindVertexArray"));if(!glBindVertexArray)throw std::runtime_error("Missing OpenGL entry point: glBindVertexArray");
    glBlendFunc=reinterpret_cast<decltype(glBlendFunc)>(SDL_GL_GetProcAddress("glBlendFunc"));if(!glBlendFunc)throw std::runtime_error("Missing OpenGL entry point: glBlendFunc");
    glBufferData=reinterpret_cast<decltype(glBufferData)>(SDL_GL_GetProcAddress("glBufferData"));if(!glBufferData)throw std::runtime_error("Missing OpenGL entry point: glBufferData");
    glClear=reinterpret_cast<decltype(glClear)>(SDL_GL_GetProcAddress("glClear"));if(!glClear)throw std::runtime_error("Missing OpenGL entry point: glClear");
    glClearColor=reinterpret_cast<decltype(glClearColor)>(SDL_GL_GetProcAddress("glClearColor"));if(!glClearColor)throw std::runtime_error("Missing OpenGL entry point: glClearColor");
    glClientWaitSync=reinterpret_cast<decltype(glClientWaitSync)>(SDL_GL_GetProcAddress("glClientWaitSync"));if(!glClientWaitSync)throw std::runtime_error("Missing OpenGL entry point: glClientWaitSync");
    glCompileShader=reinterpret_cast<decltype(glCompileShader)>(SDL_GL_GetProcAddress("glCompileShader"));if(!glCompileShader)throw std::runtime_error("Missing OpenGL entry point: glCompileShader");
    glCreateProgram=reinterpret_cast<decltype(glCreateProgram)>(SDL_GL_GetProcAddress("glCreateProgram"));if(!glCreateProgram)throw std::runtime_error("Missing OpenGL entry point: glCreateProgram");
    glCreateShader=reinterpret_cast<decltype(glCreateShader)>(SDL_GL_GetProcAddress("glCreateShader"));if(!glCreateShader)throw std::runtime_error("Missing OpenGL entry point: glCreateShader");
    glDeleteBuffers=reinterpret_cast<decltype(glDeleteBuffers)>(SDL_GL_GetProcAddress("glDeleteBuffers"));if(!glDeleteBuffers)throw std::runtime_error("Missing OpenGL entry point: glDeleteBuffers");
    glDeleteProgram=reinterpret_cast<decltype(glDeleteProgram)>(SDL_GL_GetProcAddress("glDeleteProgram"));if(!glDeleteProgram)throw std::runtime_error("Missing OpenGL entry point: glDeleteProgram");
    glDeleteQueries=reinterpret_cast<decltype(glDeleteQueries)>(SDL_GL_GetProcAddress("glDeleteQueries"));if(!glDeleteQueries)throw std::runtime_error("Missing OpenGL entry point: glDeleteQueries");
    glDeleteShader=reinterpret_cast<decltype(glDeleteShader)>(SDL_GL_GetProcAddress("glDeleteShader"));if(!glDeleteShader)throw std::runtime_error("Missing OpenGL entry point: glDeleteShader");
    glDeleteSync=reinterpret_cast<decltype(glDeleteSync)>(SDL_GL_GetProcAddress("glDeleteSync"));if(!glDeleteSync)throw std::runtime_error("Missing OpenGL entry point: glDeleteSync");
    glDeleteVertexArrays=reinterpret_cast<decltype(glDeleteVertexArrays)>(SDL_GL_GetProcAddress("glDeleteVertexArrays"));if(!glDeleteVertexArrays)throw std::runtime_error("Missing OpenGL entry point: glDeleteVertexArrays");
    glDepthFunc=reinterpret_cast<decltype(glDepthFunc)>(SDL_GL_GetProcAddress("glDepthFunc"));if(!glDepthFunc)throw std::runtime_error("Missing OpenGL entry point: glDepthFunc");
    glDepthMask=reinterpret_cast<decltype(glDepthMask)>(SDL_GL_GetProcAddress("glDepthMask"));if(!glDepthMask)throw std::runtime_error("Missing OpenGL entry point: glDepthMask");
    glDisable=reinterpret_cast<decltype(glDisable)>(SDL_GL_GetProcAddress("glDisable"));if(!glDisable)throw std::runtime_error("Missing OpenGL entry point: glDisable");
    glDrawArrays=reinterpret_cast<decltype(glDrawArrays)>(SDL_GL_GetProcAddress("glDrawArrays"));if(!glDrawArrays)throw std::runtime_error("Missing OpenGL entry point: glDrawArrays");
    glDrawArraysInstanced=reinterpret_cast<decltype(glDrawArraysInstanced)>(SDL_GL_GetProcAddress("glDrawArraysInstanced"));if(!glDrawArraysInstanced)throw std::runtime_error("Missing OpenGL entry point: glDrawArraysInstanced");
    glEnable=reinterpret_cast<decltype(glEnable)>(SDL_GL_GetProcAddress("glEnable"));if(!glEnable)throw std::runtime_error("Missing OpenGL entry point: glEnable");
    glEnableVertexAttribArray=reinterpret_cast<decltype(glEnableVertexAttribArray)>(SDL_GL_GetProcAddress("glEnableVertexAttribArray"));if(!glEnableVertexAttribArray)throw std::runtime_error("Missing OpenGL entry point: glEnableVertexAttribArray");
    glEndQuery=reinterpret_cast<decltype(glEndQuery)>(SDL_GL_GetProcAddress("glEndQuery"));if(!glEndQuery)throw std::runtime_error("Missing OpenGL entry point: glEndQuery");
    glFenceSync=reinterpret_cast<decltype(glFenceSync)>(SDL_GL_GetProcAddress("glFenceSync"));if(!glFenceSync)throw std::runtime_error("Missing OpenGL entry point: glFenceSync");
    glFlush=reinterpret_cast<decltype(glFlush)>(SDL_GL_GetProcAddress("glFlush"));if(!glFlush)throw std::runtime_error("Missing OpenGL entry point: glFlush");
    glGenBuffers=reinterpret_cast<decltype(glGenBuffers)>(SDL_GL_GetProcAddress("glGenBuffers"));if(!glGenBuffers)throw std::runtime_error("Missing OpenGL entry point: glGenBuffers");
    glGenQueries=reinterpret_cast<decltype(glGenQueries)>(SDL_GL_GetProcAddress("glGenQueries"));if(!glGenQueries)throw std::runtime_error("Missing OpenGL entry point: glGenQueries");
    glGenVertexArrays=reinterpret_cast<decltype(glGenVertexArrays)>(SDL_GL_GetProcAddress("glGenVertexArrays"));if(!glGenVertexArrays)throw std::runtime_error("Missing OpenGL entry point: glGenVertexArrays");
    glGetError=reinterpret_cast<decltype(glGetError)>(SDL_GL_GetProcAddress("glGetError"));if(!glGetError)throw std::runtime_error("Missing OpenGL entry point: glGetError");
    glGetProgramInfoLog=reinterpret_cast<decltype(glGetProgramInfoLog)>(SDL_GL_GetProcAddress("glGetProgramInfoLog"));if(!glGetProgramInfoLog)throw std::runtime_error("Missing OpenGL entry point: glGetProgramInfoLog");
    glGetProgramiv=reinterpret_cast<decltype(glGetProgramiv)>(SDL_GL_GetProcAddress("glGetProgramiv"));if(!glGetProgramiv)throw std::runtime_error("Missing OpenGL entry point: glGetProgramiv");
    glGetQueryObjectiv=reinterpret_cast<decltype(glGetQueryObjectiv)>(SDL_GL_GetProcAddress("glGetQueryObjectiv"));if(!glGetQueryObjectiv)throw std::runtime_error("Missing OpenGL entry point: glGetQueryObjectiv");
    glGetQueryObjectui64v=reinterpret_cast<decltype(glGetQueryObjectui64v)>(SDL_GL_GetProcAddress("glGetQueryObjectui64v"));if(!glGetQueryObjectui64v)throw std::runtime_error("Missing OpenGL entry point: glGetQueryObjectui64v");
    glGetShaderInfoLog=reinterpret_cast<decltype(glGetShaderInfoLog)>(SDL_GL_GetProcAddress("glGetShaderInfoLog"));if(!glGetShaderInfoLog)throw std::runtime_error("Missing OpenGL entry point: glGetShaderInfoLog");
    glGetShaderiv=reinterpret_cast<decltype(glGetShaderiv)>(SDL_GL_GetProcAddress("glGetShaderiv"));if(!glGetShaderiv)throw std::runtime_error("Missing OpenGL entry point: glGetShaderiv");
    glGetString=reinterpret_cast<decltype(glGetString)>(SDL_GL_GetProcAddress("glGetString"));if(!glGetString)throw std::runtime_error("Missing OpenGL entry point: glGetString");
    glGetUniformLocation=reinterpret_cast<decltype(glGetUniformLocation)>(SDL_GL_GetProcAddress("glGetUniformLocation"));if(!glGetUniformLocation)throw std::runtime_error("Missing OpenGL entry point: glGetUniformLocation");
    glLinkProgram=reinterpret_cast<decltype(glLinkProgram)>(SDL_GL_GetProcAddress("glLinkProgram"));if(!glLinkProgram)throw std::runtime_error("Missing OpenGL entry point: glLinkProgram");
    glPixelStorei=reinterpret_cast<decltype(glPixelStorei)>(SDL_GL_GetProcAddress("glPixelStorei"));if(!glPixelStorei)throw std::runtime_error("Missing OpenGL entry point: glPixelStorei");
    glReadBuffer=reinterpret_cast<decltype(glReadBuffer)>(SDL_GL_GetProcAddress("glReadBuffer"));if(!glReadBuffer)throw std::runtime_error("Missing OpenGL entry point: glReadBuffer");
    glReadPixels=reinterpret_cast<decltype(glReadPixels)>(SDL_GL_GetProcAddress("glReadPixels"));if(!glReadPixels)throw std::runtime_error("Missing OpenGL entry point: glReadPixels");
    glShaderSource=reinterpret_cast<decltype(glShaderSource)>(SDL_GL_GetProcAddress("glShaderSource"));if(!glShaderSource)throw std::runtime_error("Missing OpenGL entry point: glShaderSource");
    glUniform1f=reinterpret_cast<decltype(glUniform1f)>(SDL_GL_GetProcAddress("glUniform1f"));if(!glUniform1f)throw std::runtime_error("Missing OpenGL entry point: glUniform1f");
    glUniform2f=reinterpret_cast<decltype(glUniform2f)>(SDL_GL_GetProcAddress("glUniform2f"));if(!glUniform2f)throw std::runtime_error("Missing OpenGL entry point: glUniform2f");
    glUniformMatrix4fv=reinterpret_cast<decltype(glUniformMatrix4fv)>(SDL_GL_GetProcAddress("glUniformMatrix4fv"));if(!glUniformMatrix4fv)throw std::runtime_error("Missing OpenGL entry point: glUniformMatrix4fv");
    glUseProgram=reinterpret_cast<decltype(glUseProgram)>(SDL_GL_GetProcAddress("glUseProgram"));if(!glUseProgram)throw std::runtime_error("Missing OpenGL entry point: glUseProgram");
    glVertexAttribDivisor=reinterpret_cast<decltype(glVertexAttribDivisor)>(SDL_GL_GetProcAddress("glVertexAttribDivisor"));if(!glVertexAttribDivisor)throw std::runtime_error("Missing OpenGL entry point: glVertexAttribDivisor");
    glVertexAttribPointer=reinterpret_cast<decltype(glVertexAttribPointer)>(SDL_GL_GetProcAddress("glVertexAttribPointer"));if(!glVertexAttribPointer)throw std::runtime_error("Missing OpenGL entry point: glVertexAttribPointer");
    glViewport=reinterpret_cast<decltype(glViewport)>(SDL_GL_GetProcAddress("glViewport"));if(!glViewport)throw std::runtime_error("Missing OpenGL entry point: glViewport");
}
}
#define glAttachShader glapi::glAttachShader
#define glBeginQuery glapi::glBeginQuery
#define glBindBuffer glapi::glBindBuffer
#define glBindVertexArray glapi::glBindVertexArray
#define glBlendFunc glapi::glBlendFunc
#define glBufferData glapi::glBufferData
#define glClear glapi::glClear
#define glClearColor glapi::glClearColor
#define glClientWaitSync glapi::glClientWaitSync
#define glCompileShader glapi::glCompileShader
#define glCreateProgram glapi::glCreateProgram
#define glCreateShader glapi::glCreateShader
#define glDeleteBuffers glapi::glDeleteBuffers
#define glDeleteProgram glapi::glDeleteProgram
#define glDeleteQueries glapi::glDeleteQueries
#define glDeleteShader glapi::glDeleteShader
#define glDeleteSync glapi::glDeleteSync
#define glDeleteVertexArrays glapi::glDeleteVertexArrays
#define glDepthFunc glapi::glDepthFunc
#define glDepthMask glapi::glDepthMask
#define glDisable glapi::glDisable
#define glDrawArrays glapi::glDrawArrays
#define glDrawArraysInstanced glapi::glDrawArraysInstanced
#define glEnable glapi::glEnable
#define glEnableVertexAttribArray glapi::glEnableVertexAttribArray
#define glEndQuery glapi::glEndQuery
#define glFenceSync glapi::glFenceSync
#define glFlush glapi::glFlush
#define glGenBuffers glapi::glGenBuffers
#define glGenQueries glapi::glGenQueries
#define glGenVertexArrays glapi::glGenVertexArrays
#define glGetError glapi::glGetError
#define glGetProgramInfoLog glapi::glGetProgramInfoLog
#define glGetProgramiv glapi::glGetProgramiv
#define glGetQueryObjectiv glapi::glGetQueryObjectiv
#define glGetQueryObjectui64v glapi::glGetQueryObjectui64v
#define glGetShaderInfoLog glapi::glGetShaderInfoLog
#define glGetShaderiv glapi::glGetShaderiv
#define glGetString glapi::glGetString
#define glGetUniformLocation glapi::glGetUniformLocation
#define glLinkProgram glapi::glLinkProgram
#define glPixelStorei glapi::glPixelStorei
#define glReadBuffer glapi::glReadBuffer
#define glReadPixels glapi::glReadPixels
#define glShaderSource glapi::glShaderSource
#define glUniform1f glapi::glUniform1f
#define glUniform2f glapi::glUniform2f
#define glUniformMatrix4fv glapi::glUniformMatrix4fv
#define glUseProgram glapi::glUseProgram
#define glVertexAttribDivisor glapi::glVertexAttribDivisor
#define glVertexAttribPointer glapi::glVertexAttribPointer
#define glViewport glapi::glViewport
#define glGenFramebuffers glapi::glGenFramebuffers
#define glBindFramebuffer glapi::glBindFramebuffer
#define glDeleteFramebuffers glapi::glDeleteFramebuffers
#define glFramebufferTexture2D glapi::glFramebufferTexture2D
#define glCheckFramebufferStatus glapi::glCheckFramebufferStatus
#define glGenTextures glapi::glGenTextures
#define glBindTexture glapi::glBindTexture
#define glTexImage2D glapi::glTexImage2D
#define glTexParameteri glapi::glTexParameteri
#define glDeleteTextures glapi::glDeleteTextures
#define glActiveTexture glapi::glActiveTexture
#define glUniform1i glapi::glUniform1i
#define glBlendFuncSeparate glapi::glBlendFuncSeparate
#define glMapBufferRange glapi::glMapBufferRange
#define glUnmapBuffer glapi::glUnmapBuffer
#define glBlitFramebuffer glapi::glBlitFramebuffer
#define glClearBufferuiv glapi::glClearBufferuiv
#define glGetIntegerv glapi::glGetIntegerv

namespace {
constexpr uint32_t MaxInstances=65536;
struct Vertex {V3 p,bary,edge;uint32_t color;float captionToken=0;};
const char* captionVertexSource=R"(#version 330 core
layout(location=0) in vec3 p;
layout(location=4) in mat4 model;
layout(location=10) in float meshToken;
layout(location=11) in float instanceToken;
uniform mat4 viewProjection;
flat out uint token;
void main(){gl_Position=viewProjection*model*vec4(p,1);token=uint((instanceToken>0?instanceToken:meshToken)+0.5);}
)";
const char* captionFragmentSource=R"(#version 330 core
flat in uint token;
layout(location=0) out uint objectId;
void main(){objectId=token;}
)";
uint64_t edgeKey(uint32_t a,uint32_t b){return (uint64_t(std::min(a,b))<<32)|std::max(a,b);}
const char* vertexSource=R"(#version 330 core
layout(location=0) in vec3 p;
layout(location=1) in vec3 bary;
layout(location=2) in vec3 edge;
layout(location=3) in vec4 meshColor;
layout(location=4) in mat4 model;
layout(location=8) in vec4 color;
layout(location=9) in float surface;
uniform mat4 viewProjection;
out vec3 vBary;flat out vec3 vEdge;flat out vec4 vColor;
out vec3 vWorld;flat out float vSurface;
void main(){vec4 w=model*vec4(p,1);gl_Position=viewProjection*w;
vBary=bary;vEdge=edge;vColor=color*meshColor;vWorld=w.xyz;vSurface=surface;}
)";
const char* fragmentSource=R"(#version 330 core
in vec3 vBary;flat in vec3 vEdge;flat in vec4 vColor;
in vec3 vWorld;flat in float vSurface;
uniform float outlineWidth;
out vec4 outputColor;
void main(){vec3 c=vColor.rgb;
if(vSurface>0.5){
// Stable 5 cm grains, with no periodic brightness layer or floating sine hash.
uvec2 q=uvec2(ivec2(floor(vWorld.xz*20.0)));
uint h=q.x*1664525u+q.y*1013904223u;
h^=h>>16;h*=0x7feb352du;h^=h>>15;h*=0x846ca68bu;h^=h>>16;
float n=float(h>>8)/16777216.0;
float footprint=max(length(dFdx(vWorld.xz)),length(dFdy(vWorld.xz)))*20.0;
float fade=1.0-smoothstep(.5,1.5,footprint);
c+=(n-.5)*.045*fade;
}else if(vSurface>-.5){
vec3 gradient=max(fwidth(vBary),vec3(.00001));
vec3 d=mix(vec3(10000),abs(vBary)/gradient,vEdge);
// Keep full-contrast outlines on small projected faces, as in the control videos.
if(outlineWidth>0.0&&min(d.x,min(d.y,d.z))<outlineWidth)c=vec3(0);
}outputColor=vec4(c,1);}
)";
const char* uiVertexSource=R"(#version 330 core
layout(location=0) in vec2 p;layout(location=1) in vec4 c;
uniform vec2 viewport;out vec4 color;
void main(){gl_Position=vec4(p.x/viewport.x*2.0-1.0,1.0-p.y/viewport.y*2.0,0,1);color=c;}
)";
const char* uiFragmentSource=R"(#version 330 core
in vec4 color;out vec4 outputColor;void main(){outputColor=color;}
)";
const char* gateVertexSource=R"(#version 330 core
layout(location=0) in vec3 p;layout(location=1) in vec4 c;
uniform mat4 viewProjection;out vec4 color;
void main(){gl_Position=viewProjection*vec4(p,1);color=c;}
)";
const char* compositeVertexSource=R"(#version 330 core
out vec2 uv;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(p*2.0-1.0,0,1);}
)";
const char* compositeFragmentSource=R"(#version 330 core
in vec2 uv;uniform sampler2D sceneTexture;uniform sampler2D uiTexture;out vec4 color;
void main(){vec4 overlay=texture(uiTexture,uv);color=vec4(overlay.rgb+texture(sceneTexture,uv).rgb*(1.0-overlay.a),1);}
)";
const char* resizeFragmentSource=R"(#version 330 core
uniform sampler2D sourceTexture;
uniform int horizontal;
uniform float scaleFactor;
uniform float cropOffset;
out vec4 color;
void main(){
ivec2 dst=ivec2(gl_FragCoord.xy),size=textureSize(sourceTexture,0);
float center=(float(horizontal!=0?dst.x:dst.y)+cropOffset+.5)*scaleFactor;
float support=max(scaleFactor,1.0);
int first=max(0,int(floor(center-support+.5)));
int last=min(horizontal!=0?size.x:size.y,int(floor(center+support+.5)));
vec4 sum=vec4(0);float total=0;
for(int i=first;i<last;++i){
float weight=max(0.0,1.0-abs((float(i)+.5-center)/support));
ivec2 pixel=horizontal!=0?ivec2(i,dst.y):ivec2(dst.x,i);
sum+=texelFetch(sourceTexture,pixel,0)*weight;total+=weight;
}
// Pillow rounds the uint8 intermediate after each separable resize pass.
color=floor(clamp(sum/total,0.0,1.0)*255.0+.5)/255.0;
}
)";
GLuint compile(GLenum type,const char* source){
    GLuint shader=glCreateShader(type);glShaderSource(shader,1,&source,nullptr);glCompileShader(shader);
    GLint ok=0;glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
    if(!ok){char message[4096]{};glGetShaderInfoLog(shader,sizeof(message),nullptr,message);glDeleteShader(shader);throw std::runtime_error(std::string("GL shader: ")+message);}return shader;
}
GLuint program(const char* vs,const char* fs){
    GLuint v=compile(GL_VERTEX_SHADER,vs),f=0,p=0;
    try{f=compile(GL_FRAGMENT_SHADER,fs);p=glCreateProgram();glAttachShader(p,v);glAttachShader(p,f);glLinkProgram(p);
        GLint ok=0;glGetProgramiv(p,GL_LINK_STATUS,&ok);if(!ok){char message[4096]{};glGetProgramInfoLog(p,sizeof(message),nullptr,message);throw std::runtime_error(std::string("GL link: ")+message);}}
    catch(...){glDeleteShader(v);if(f)glDeleteShader(f);if(p)glDeleteProgram(p);throw;}
    glDeleteShader(v);glDeleteShader(f);return p;
}
Matrix viewProjection(const Camera& camera,float aspect){
    const V3 f=(camera.target-camera.eye).normalized(),r=cross({0,1,0},f).normalized(),u=cross(f,r);
    Matrix v{};v.m[0][0]=r.x;v.m[1][0]=r.y;v.m[2][0]=r.z;v.m[3][0]=-dot(camera.eye,r);
    v.m[0][1]=u.x;v.m[1][1]=u.y;v.m[2][1]=u.z;v.m[3][1]=-dot(camera.eye,u);
    v.m[0][2]=f.x;v.m[1][2]=f.y;v.m[2][2]=f.z;v.m[3][2]=-dot(camera.eye,f);v.m[3][3]=1;
    const float n=.08f,far=2000,tx=std::tan(camera.horizontalFov*Pi/360);
    Matrix p{};p.m[0][0]=1/tx;p.m[1][1]=aspect/tx;p.m[2][2]=(far+n)/(far-n);p.m[2][3]=1;p.m[3][2]=-2*far*n/(far-n);
    return multiply(v,p);
}
void bigEndian(std::ostream& out,uint32_t value){const char b[]={char(value>>24),char(value>>16),char(value>>8),char(value)};out.write(b,4);}
void chunk(std::ostream& out,const char* name,const std::vector<uint8_t>& bytes){
    bigEndian(out,uint32_t(bytes.size()));out.write(name,4);if(!bytes.empty())out.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));
    uLong crc=crc32(0,reinterpret_cast<const Bytef*>(name),4);if(!bytes.empty())crc=crc32(crc,bytes.data(),uInt(bytes.size()));bigEndian(out,uint32_t(crc));
}
}
struct Renderer::GlState {
    SDL_Window* window=nullptr;SDL_GLContext context=nullptr;bool functionsLoaded=false;
    std::unique_ptr<HeadlessGL> headless;
    GLuint scene=0,hud=0,instances=0,uiVao=0,uiBuffer=0;
    struct Mesh{GLuint vao=0,buffer=0;GLsizei count=0;};
    struct Timer{GLuint query=0;bool pending=false;};
    std::vector<Mesh> meshes;std::vector<std::vector<Instance>> batches;
    std::array<Timer,8> timers{};uint64_t frame=0;int swapInterval=-2;
    GLsync fence=nullptr;
    std::array<GLuint,3> framebuffers{};std::array<GLuint,4> textures{};
    GLuint gate=0,composite=0,gateVao=0,gateBuffer=0,compositeVao=0;
    struct Readback {GLuint buffer=0;size_t bytes=0;bool pending=false;};
    std::array<Readback,5> readbacks{};
    int layerWidth=0,layerHeight=0;
    // Each window draws into its own offscreen target and is then blitted to
    // the window. Reading the window's own back buffer is unreliable when
    // several X11 windows overlap on one screen: the driver aliases them, so
    // every window yields the pixels of the last one drawn. The offscreen
    // target is per-context and therefore always holds this view's frame.
    GLuint offscreenFbo=0,offscreenTexture=0,offscreenDepth=0;
    int offscreenWidth=0,offscreenHeight=0;
    bool layerMode=false;
    GLuint captionProgram=0,captionFbo=0,captionTexture=0,captionDepth=0;
    int captionWidth=0,captionHeight=0;
    GLuint resizeProgram=0,resizeVao=0;
    std::array<GLuint,2> resizeFbos{},resizeTextures{};
    int resizeWidth=0,resizeHeight=0,resizeSourceHeight=0;
    ~GlState(){if(!context&&!headless)return;
        if(headless)headless->makeCurrent();else SDL_GL_MakeCurrent(window,context);
        if(!functionsLoaded){if(context)SDL_GL_DeleteContext(context);activeHeadless=nullptr;return;}
        if(fence)glDeleteSync(fence);
        for(auto t:timers)if(t.query)glDeleteQueries(1,&t.query);
        for(auto m:meshes){glDeleteBuffers(1,&m.buffer);glDeleteVertexArrays(1,&m.vao);}
        glDeleteBuffers(1,&instances);glDeleteBuffers(1,&uiBuffer);glDeleteVertexArrays(1,&uiVao);
        glDeleteFramebuffers(3,framebuffers.data());glDeleteTextures(4,textures.data());
        if(captionProgram)glDeleteProgram(captionProgram);
        if(captionFbo)glDeleteFramebuffers(1,&captionFbo);
        if(captionTexture)glDeleteTextures(1,&captionTexture);
        if(captionDepth)glDeleteTextures(1,&captionDepth);
        if(resizeProgram)glDeleteProgram(resizeProgram);
        if(resizeVao)glDeleteVertexArrays(1,&resizeVao);
        glDeleteFramebuffers(2,resizeFbos.data());glDeleteTextures(2,resizeTextures.data());
        if(offscreenFbo)glDeleteFramebuffers(1,&offscreenFbo);
        if(offscreenTexture)glDeleteTextures(1,&offscreenTexture);
        if(offscreenDepth)glDeleteTextures(1,&offscreenDepth);
        for(auto& r:readbacks)if(r.buffer)glDeleteBuffers(1,&r.buffer);
        glDeleteBuffers(1,&gateBuffer);glDeleteVertexArrays(1,&gateVao);glDeleteVertexArrays(1,&compositeVao);
        if(gate)glDeleteProgram(gate);if(composite)glDeleteProgram(composite);
        if(scene)glDeleteProgram(scene);if(hud)glDeleteProgram(hud);if(context)SDL_GL_DeleteContext(context);activeHeadless=nullptr;
    }
};
Renderer::Renderer(SDL_Window* window,int w,int h):gl(std::make_unique<GlState>()){
    gl->window=window;
    if(window){activeHeadless=nullptr;gl->context=SDL_GL_CreateContext(window);
        if(!gl->context)throw std::runtime_error(std::string("OpenGL 3.3 context: ")+SDL_GetError());
        if(SDL_GL_MakeCurrent(window,gl->context)!=0)throw std::runtime_error(SDL_GetError());
    }else {gl->headless=std::make_unique<HeadlessGL>();activeHeadless=gl->headless.get();}
    glapi::load();gl->functionsLoaded=true;
    const auto* name=glGetString(GL_RENDERER);adapterName=name?reinterpret_cast<const char*>(name):"OpenGL";
    gl->scene=program(vertexSource,fragmentSource);gl->hud=program(uiVertexSource,uiFragmentSource);
    glGenBuffers(1,&gl->instances);glGenVertexArrays(1,&gl->uiVao);glGenBuffers(1,&gl->uiBuffer);
    glBindVertexArray(gl->uiVao);glBindBuffer(GL_ARRAY_BUFFER,gl->uiBuffer);
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(UiVertex),nullptr);
    glEnableVertexAttribArray(1);glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,sizeof(UiVertex),reinterpret_cast<void*>(offsetof(UiVertex,color)));
    for(auto& t:gl->timers)glGenQueries(1,&t.query);
    glDisable(GL_CULL_FACE);glEnable(GL_DEPTH_TEST);glDepthFunc(GL_LEQUAL);resize(w,h);
}
Renderer::~Renderer()=default;
void Renderer::makeCurrent(){if(gl->headless)gl->headless->makeCurrent();else SDL_GL_MakeCurrent(gl->window,gl->context);}
bool Renderer::readbackPending(size_t slot)const{return gl->readbacks.at(slot).pending;}
void Renderer::beginReadback(size_t slot){
    auto& r=gl->readbacks.at(slot);
    if(r.pending)throw std::runtime_error("Previous readback has not been collected");
    if(gl->headless)throw std::runtime_error("Readback needs a window");
    if(!r.buffer)glGenBuffers(1,&r.buffer);
    // Read the offscreen target, which keeps this view's frame. The window
    // buffer is only a blit destination and is shared between overlapping
    // windows.
    if(gl->offscreenFbo)glBindFramebuffer(GL_FRAMEBUFFER,gl->offscreenFbo);else glBindFramebuffer(GL_FRAMEBUFFER,0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER,r.buffer);const size_t bytes=size_t(width)*height*4;
    if(r.bytes!=bytes){glBufferData(GL_PIXEL_PACK_BUFFER,GLsizeiptr(bytes),nullptr,GL_STREAM_READ);r.bytes=bytes;}
    glReadBuffer(gl->offscreenFbo?GL_COLOR_ATTACHMENT0:GL_BACK);glPixelStorei(GL_PACK_ALIGNMENT,1);
    glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glBindBuffer(GL_PIXEL_PACK_BUFFER,0);glFlush();r.pending=true;
}
void Renderer::resize(int w,int h){if(w>0&&h>0){width=w;height=h;glViewport(0,0,w,h);}}
void Renderer::bindOffscreen(){
    // EGL's tiny pbuffer is not an image target. Both windowed and headless
    // scene rendering need the full-resolution FBO; layer rendering owns its own.
    if((!gl->headless&&!gl->context)||gl->layerMode)return;
    if(gl->offscreenWidth!=width||gl->offscreenHeight!=height){
        if(gl->offscreenFbo)glDeleteFramebuffers(1,&gl->offscreenFbo);
        if(gl->offscreenTexture)glDeleteTextures(1,&gl->offscreenTexture);
        if(gl->offscreenDepth)glDeleteTextures(1,&gl->offscreenDepth);
        glGenFramebuffers(1,&gl->offscreenFbo);glGenTextures(1,&gl->offscreenTexture);glGenTextures(1,&gl->offscreenDepth);
        glBindTexture(GL_TEXTURE_2D,gl->offscreenTexture);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
        glBindTexture(GL_TEXTURE_2D,gl->offscreenDepth);
        glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,width,height,0,GL_DEPTH_COMPONENT,GL_UNSIGNED_INT,nullptr);
        glBindFramebuffer(GL_FRAMEBUFFER,gl->offscreenFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,gl->offscreenTexture,0);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,gl->offscreenDepth,0);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)throw std::runtime_error("Incomplete render target");
        gl->offscreenWidth=width;gl->offscreenHeight=height;
    }
    glBindFramebuffer(GL_FRAMEBUFFER,gl->offscreenFbo);
}
void Renderer::blitToWindow(){
    if(gl->headless||!gl->context||gl->layerMode||!gl->offscreenFbo)return;
    glBindFramebuffer(GL_READ_FRAMEBUFFER,gl->offscreenFbo);glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);
    glBlitFramebuffer(0,0,width,height,0,0,width,height,GL_COLOR_BUFFER_BIT,GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
}
uint32_t Renderer::addMesh(const MeshData& mesh){
    if(mesh.vertices.empty()||mesh.indices.empty()||mesh.indices.size()%3)throw std::runtime_error("Invalid mesh");
    if(!mesh.colors.empty()&&mesh.colors.size()!=mesh.vertices.size())throw std::runtime_error("Invalid mesh colors");
    if(!mesh.captionTokens.empty()&&mesh.captionTokens.size()!=mesh.indices.size()/3)throw std::runtime_error("Invalid caption triangles");
    std::map<uint64_t,bool> edges;for(auto e:mesh.edges)edges[edgeKey(e[0],e[1])]=true;
    std::vector<Vertex> data;data.reserve(mesh.indices.size());
    for(size_t i=0;i<mesh.indices.size();i+=3){uint32_t a=mesh.indices[i],b=mesh.indices[i+1],c=mesh.indices[i+2];
        if(std::max({a,b,c})>=mesh.vertices.size())throw std::runtime_error("Invalid mesh index");
        V3 flags{float(edges.contains(edgeKey(b,c))),float(edges.contains(edgeKey(a,c))),float(edges.contains(edgeKey(a,b)))};
        auto color=[&](uint32_t k){return mesh.colors.empty()?0xffffffffu:mesh.colors[k];};
        float token=mesh.captionTokens.empty()?0:float(mesh.captionTokens[i/3]);
        data.push_back({mesh.vertices[a],{1,0,0},flags,color(a),token});data.push_back({mesh.vertices[b],{0,1,0},flags,color(b),token});data.push_back({mesh.vertices[c],{0,0,1},flags,color(c),token});
    }
    GlState::Mesh m;glGenVertexArrays(1,&m.vao);glGenBuffers(1,&m.buffer);m.count=GLsizei(data.size());
    glBindVertexArray(m.vao);glBindBuffer(GL_ARRAY_BUFFER,m.buffer);glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(data.size()*sizeof(Vertex)),data.data(),GL_STATIC_DRAW);
    for(int i=0;i<3;++i){glEnableVertexAttribArray(i);glVertexAttribPointer(i,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(size_t(i)*sizeof(V3)));}
    glEnableVertexAttribArray(3);glVertexAttribPointer(3,4,GL_UNSIGNED_BYTE,GL_TRUE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,color)));
    glEnableVertexAttribArray(10);glVertexAttribPointer(10,1,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,captionToken)));
    gl->meshes.push_back(m);gl->batches.emplace_back();return uint32_t(gl->meshes.size()-1);
}
double Renderer::waitForFrame(bool){
    const double start=now();if(gl->fence){GLenum result;
        do{result=glClientWaitSync(gl->fence,GL_SYNC_FLUSH_COMMANDS_BIT,1000000);}while(result==GL_TIMEOUT_EXPIRED);
        glDeleteSync(gl->fence);gl->fence=nullptr;if(result==GL_WAIT_FAILED)throw std::runtime_error("GL frame fence wait failed");}
    return (now()-start)*1000;
}
void Renderer::render(const std::vector<DrawObject>& objects,const Camera& camera,const HudState& hud,const std::vector<GateVisual>& gates){
    newGpuSample=false;
    for(auto& t:gl->timers)if(t.pending){GLint ready=0;glGetQueryObjectiv(t.query,GL_QUERY_RESULT_AVAILABLE,&ready);if(ready){GLuint64 ns=0;glGetQueryObjectui64v(t.query,GL_QUERY_RESULT,&ns);latestGpuMs=double(ns)/1e6;newGpuSample=true;t.pending=false;}}
    auto& timer=gl->timers[gl->frame++%gl->timers.size()];bool measure=!timer.pending;if(measure)glBeginQuery(GL_TIME_ELAPSED,timer.query);
    bindOffscreen();
    glViewport(0,0,width,height);glEnable(GL_DEPTH_TEST);glDepthMask(GL_TRUE);glDisable(GL_BLEND);glClearColor(0,0,0,1);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glUseProgram(gl->scene);auto matrix=viewProjection(camera,float(width)/height);
    glUniformMatrix4fv(glGetUniformLocation(gl->scene,"viewProjection"),1,GL_FALSE,&matrix.m[0][0]);glUniform1f(glGetUniformLocation(gl->scene,"outlineWidth"),outlineWidth);
    const V3 f=(camera.target-camera.eye).normalized(),right=cross({0,1,0},f).normalized(),up=cross(f,right);
    float tx=std::tan(camera.horizontalFov*Pi/360),ty=tx*height/width;
    visibleCount=0;drawCalls=0;for(auto& b:gl->batches)b.clear();
    for(const auto& o:objects){V3 d=o.center-camera.eye;float z=dot(d,f),r=o.radius;
        if(o.instance.surface< -1.5f&&!buildingStripes)continue;
        if(z+r<.08f||d.length()>o.range+r||std::abs(dot(d,right))>z*tx+r*(1+tx)||std::abs(dot(d,up))>z*ty+r*(1+ty))continue;
        if(visibleCount>=MaxInstances)throw std::runtime_error("Visible instance capacity exceeded");
        gl->batches.at(o.mesh).push_back(o.instance);++visibleCount;}
    std::vector<Instance> all;all.reserve(visibleCount);for(const auto& b:gl->batches)all.insert(all.end(),b.begin(),b.end());
    glBindBuffer(GL_ARRAY_BUFFER,gl->instances);glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(all.size()*sizeof(Instance)),all.data(),GL_STREAM_DRAW);
    size_t offset=0;
    for(size_t i=0;i<gl->meshes.size();++i){const auto& batch=gl->batches[i];if(batch.empty())continue;const auto& m=gl->meshes[i];glBindVertexArray(m.vao);glBindBuffer(GL_ARRAY_BUFFER,gl->instances);
        for(int j=0;j<4;++j){glEnableVertexAttribArray(4+j);glVertexAttribPointer(4+j,4,GL_FLOAT,GL_FALSE,sizeof(Instance),reinterpret_cast<void*>(offset+size_t(j)*16));glVertexAttribDivisor(4+j,1);}
        glEnableVertexAttribArray(8);glVertexAttribPointer(8,4,GL_FLOAT,GL_FALSE,sizeof(Instance),reinterpret_cast<void*>(offset+offsetof(Instance,color)));glVertexAttribDivisor(8,1);
        glEnableVertexAttribArray(9);glVertexAttribPointer(9,1,GL_FLOAT,GL_FALSE,sizeof(Instance),reinterpret_cast<void*>(offset+offsetof(Instance,surface)));glVertexAttribDivisor(9,1);
        glDrawArraysInstanced(GL_TRIANGLES,0,m.count,GLsizei(batch.size()));offset+=batch.size()*sizeof(Instance);++drawCalls;}
    if(!gates.empty())drawGates(gates,camera);
    drawHud(hud);if(measure){glEndQuery(GL_TIME_ELAPSED);timer.pending=true;}
    blitToWindow();
    GLenum error=glGetError();if(error!=GL_NO_ERROR)throw std::runtime_error("OpenGL draw error: "+std::to_string(error));
}
std::vector<captions::PixelStats> Renderer::captionVisibility(const std::vector<DrawObject>& objects,const Camera& camera,int sampleWidth,float aspect){
    if(sampleWidth<64||sampleWidth>2048)throw std::runtime_error("Invalid caption visibility resolution");
    if(aspect==0)aspect=float(width)/height;
    if(!std::isfinite(aspect)||aspect<.25f||aspect>4)throw std::runtime_error("Invalid caption aspect ratio");
    GLint previousDraw=0,previousRead=0;glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&previousDraw);glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING,&previousRead);
    const int w=sampleWidth,h=std::max(1,int(std::lround(w/aspect)));
    if(!gl->captionProgram){gl->captionProgram=program(captionVertexSource,captionFragmentSource);
        glGenFramebuffers(1,&gl->captionFbo);glGenTextures(1,&gl->captionTexture);glGenTextures(1,&gl->captionDepth);}
    glBindFramebuffer(GL_FRAMEBUFFER,gl->captionFbo);
    if(gl->captionWidth!=w||gl->captionHeight!=h){
        glBindTexture(GL_TEXTURE_2D,gl->captionTexture);glTexImage2D(GL_TEXTURE_2D,0,GL_R32UI,w,h,0,GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,gl->captionTexture,0);
        glBindTexture(GL_TEXTURE_2D,gl->captionDepth);glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,w,h,0,GL_DEPTH_COMPONENT,GL_UNSIGNED_INT,nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,gl->captionDepth,0);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)throw std::runtime_error("Incomplete caption target");
        gl->captionWidth=w;gl->captionHeight=h;
    }
    glViewport(0,0,w,h);glDisable(GL_BLEND);glEnable(GL_DEPTH_TEST);glDepthMask(GL_TRUE);
    const GLuint zero[4]={};glClearBufferuiv(GL_COLOR,0,zero);glClear(GL_DEPTH_BUFFER_BIT);glUseProgram(gl->captionProgram);
    auto matrix=viewProjection(camera,aspect);glUniformMatrix4fv(glGetUniformLocation(gl->captionProgram,"viewProjection"),1,GL_FALSE,&matrix.m[0][0]);
    const V3 f=(camera.target-camera.eye).normalized(),right=cross({0,1,0},f).normalized(),up=cross(f,right);
    float tx=std::tan(camera.horizontalFov*Pi/360),ty=tx/aspect;
    for(auto& batch:gl->batches)batch.clear();size_t count=0;
    // Unknown token 0 must still write depth, otherwise captions see through it.
    for(const auto& o:objects){V3 d=o.center-camera.eye;float z=dot(d,f),r=o.radius;
        // Cosmetic stripes and junction paint must not occlude caption entities;
        // their original walls/ground still participate in this depth pass.
        if(o.instance.surface< -1.5f||o.instance.surface==2.f)continue;
        if(z+r<.08f||d.length()>o.range+r||std::abs(dot(d,right))>z*tx+r*(1+tx)||std::abs(dot(d,up))>z*ty+r*(1+ty))continue;
        if(++count>MaxInstances)throw std::runtime_error("Caption instance capacity exceeded");gl->batches.at(o.mesh).push_back(o.instance);}
    std::vector<Instance> all;all.reserve(count);for(const auto& batch:gl->batches)all.insert(all.end(),batch.begin(),batch.end());
    glBindBuffer(GL_ARRAY_BUFFER,gl->instances);glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(all.size()*sizeof(Instance)),all.data(),GL_STREAM_DRAW);
    size_t offset=0;
    for(size_t i=0;i<gl->meshes.size();++i){const auto& batch=gl->batches[i];if(batch.empty())continue;const auto& mesh=gl->meshes[i];
        glBindVertexArray(mesh.vao);glBindBuffer(GL_ARRAY_BUFFER,gl->instances);
        for(int j=0;j<4;++j){glEnableVertexAttribArray(4+j);glVertexAttribPointer(4+j,4,GL_FLOAT,GL_FALSE,sizeof(Instance),reinterpret_cast<void*>(offset+size_t(j)*16));glVertexAttribDivisor(4+j,1);}
        glEnableVertexAttribArray(11);glVertexAttribPointer(11,1,GL_FLOAT,GL_FALSE,sizeof(Instance),reinterpret_cast<void*>(offset+offsetof(Instance,pad)));glVertexAttribDivisor(11,1);
        glDrawArraysInstanced(GL_TRIANGLES,0,mesh.count,GLsizei(batch.size()));offset+=batch.size()*sizeof(Instance);
    }
    std::vector<uint32_t> pixels(size_t(w)*h);glReadBuffer(GL_COLOR_ATTACHMENT0);glBindBuffer(GL_PIXEL_PACK_BUFFER,0);glPixelStorei(GL_PACK_ALIGNMENT,1);
    glReadPixels(0,0,w,h,GL_RED_INTEGER,GL_UNSIGNED_INT,pixels.data());
    struct Sum{uint32_t n=0;uint64_t x=0,y=0;};std::map<uint32_t,Sum> sums;
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){auto& s=sums[pixels[size_t(y)*w+x]];++s.n;s.x+=x;s.y+=y;}
    std::vector<captions::PixelStats> result;for(const auto& [token,s]:sums)
        result.push_back({token,s.n,float((double(s.x)/s.n+.5)/w),float(1-(double(s.y)/s.n+.5)/h)});
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER,GLuint(previousDraw));glBindFramebuffer(GL_READ_FRAMEBUFFER,GLuint(previousRead));glViewport(0,0,width,height);
    GLenum error=glGetError();if(error!=GL_NO_ERROR)throw std::runtime_error("OpenGL caption error: "+std::to_string(error));
    return result;
}
std::vector<uint8_t> Renderer::readBackResized(int outputWidth,int outputHeight){
    if(outputWidth==width&&outputHeight==height)return readBack();
    if(!gl->offscreenFbo)throw std::runtime_error("Render reference frame before resizing");
    return readTextureResized(gl->offscreenTexture,outputWidth,outputHeight);
}
std::vector<uint8_t> Renderer::readTextureResized(uint32_t texture,int outputWidth,int outputHeight){
    if(outputWidth<=0||outputHeight<=0||outputWidth>width||outputHeight>height)
        throw std::runtime_error("Reference resize only supports positive downsample dimensions");
    const double ratio=std::max(double(outputWidth)/width,double(outputHeight)/height);
    const int scaledWidth=int(std::nearbyint(width*ratio)),scaledHeight=int(std::nearbyint(height*ratio));
    const int cropX=int(std::nearbyint((scaledWidth-outputWidth)*.5)),cropY=int(std::nearbyint((scaledHeight-outputHeight)*.5));
    // Current reference profile only crops vertically. Reject another aspect
    // instead of silently changing the horizontal camera/prompt convention.
    if(cropX||scaledWidth!=outputWidth)throw std::runtime_error("Reference resize requires uncropped horizontal field of view");
    GLint previousDraw=0,previousRead=0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&previousDraw);glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING,&previousRead);
    if(!gl->resizeProgram){gl->resizeProgram=program(compositeVertexSource,resizeFragmentSource);
        glGenVertexArrays(1,&gl->resizeVao);glGenFramebuffers(2,gl->resizeFbos.data());glGenTextures(2,gl->resizeTextures.data());}
    if(gl->resizeWidth!=outputWidth||gl->resizeHeight!=outputHeight||gl->resizeSourceHeight!=height){
        for(int i=0;i<2;++i){
            glBindTexture(GL_TEXTURE_2D,gl->resizeTextures[i]);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,outputWidth,i?outputHeight:height,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
            glBindFramebuffer(GL_FRAMEBUFFER,gl->resizeFbos[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,gl->resizeTextures[i],0);
            if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)throw std::runtime_error("Incomplete reference resize target");
        }
        gl->resizeWidth=outputWidth;gl->resizeHeight=outputHeight;gl->resizeSourceHeight=height;
    }
    GLint dithering=0;glGetIntegerv(GL_DITHER,&dithering);
    glDisable(GL_DEPTH_TEST);glDisable(GL_BLEND);glDisable(GL_DITHER);
    glUseProgram(gl->resizeProgram);glBindVertexArray(gl->resizeVao);glActiveTexture(GL_TEXTURE0);
    glUniform1i(glGetUniformLocation(gl->resizeProgram,"sourceTexture"),0);
    for(int i=0;i<2;++i){
        glBindFramebuffer(GL_FRAMEBUFFER,gl->resizeFbos[i]);glViewport(0,0,outputWidth,i?outputHeight:height);
        glBindTexture(GL_TEXTURE_2D,i?gl->resizeTextures[0]:texture);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glUniform1i(glGetUniformLocation(gl->resizeProgram,"horizontal"),i==0);
        glUniform1f(glGetUniformLocation(gl->resizeProgram,"scaleFactor"),i?float(height)/scaledHeight:float(width)/scaledWidth);
        glUniform1f(glGetUniformLocation(gl->resizeProgram,"cropOffset"),i?float(cropY):0.f);
        glDrawArrays(GL_TRIANGLES,0,3);
    }
    std::vector<uint8_t> pixels(size_t(outputWidth)*outputHeight*4),flipped(pixels.size());
    glBindBuffer(GL_PIXEL_PACK_BUFFER,0);glPixelStorei(GL_PACK_ALIGNMENT,1);glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0,0,outputWidth,outputHeight,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    const size_t stride=size_t(outputWidth)*4;
    for(int y=0;y<outputHeight;++y)std::memcpy(flipped.data()+size_t(y)*stride,pixels.data()+size_t(outputHeight-1-y)*stride,stride);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER,GLuint(previousDraw));glBindFramebuffer(GL_READ_FRAMEBUFFER,GLuint(previousRead));
    glViewport(0,0,width,height);glEnable(GL_DEPTH_TEST);
    if(dithering)glEnable(GL_DITHER);
    if(glGetError()!=GL_NO_ERROR)throw std::runtime_error("OpenGL reference resize error");
    return flipped;
}
void Renderer::drawGates(const std::vector<GateVisual>& gates,const Camera& camera){
    if(gates.empty())return;
    if(!gl->gate){gl->gate=program(gateVertexSource,uiFragmentSource);glGenVertexArrays(1,&gl->gateVao);glGenBuffers(1,&gl->gateBuffer);}
    struct GateVertex {V3 p;Color color;};std::vector<GateVertex> vertices;
    for(const auto& gate:gates){
        V3 right=cross({0,1,0},gate.forward).normalized();std::vector<std::array<float,2>> points;
        points.push_back({-gate.halfWidth,.06f});
        const float shoulder=std::max(1.f,gate.height-1.2f);
        for(int i=0;i<=20;++i){float a=Pi-float(i)*Pi/20;points.push_back({std::cos(a)*gate.halfWidth,shoulder+std::sin(a)*1.2f});}
        points.push_back({gate.halfWidth,.06f});
        for(size_t i=1;i<points.size();++i){auto a=points[i-1],b=points[i];float dx=b[0]-a[0],dy=b[1]-a[1],len=std::hypot(dx,dy);if(len<.001f)continue;
            float ox=-dy/len*.14f,oy=dx/len*.14f;
            auto v=[&](float x,float y){return GateVertex{gate.center+right*x+V3{0,y,0},gate.color};};
            auto p=v(a[0]+ox,a[1]+oy),q=v(b[0]+ox,b[1]+oy),r=v(b[0]-ox,b[1]-oy),s=v(a[0]-ox,a[1]-oy);vertices.insert(vertices.end(),{p,q,r,p,r,s});
        }
    }
    if(vertices.empty())return;
    glEnable(GL_DEPTH_TEST);glDepthMask(GL_FALSE);glEnable(GL_BLEND);glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(gl->gate);auto vp=viewProjection(camera,float(width)/height);glUniformMatrix4fv(glGetUniformLocation(gl->gate,"viewProjection"),1,GL_FALSE,&vp.m[0][0]);
    glBindVertexArray(gl->gateVao);glBindBuffer(GL_ARRAY_BUFFER,gl->gateBuffer);glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(vertices.size()*sizeof(GateVertex)),vertices.data(),GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(GateVertex),nullptr);glEnableVertexAttribArray(1);glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,sizeof(GateVertex),reinterpret_cast<void*>(offsetof(GateVertex,color)));
    glDrawArrays(GL_TRIANGLES,0,GLsizei(vertices.size()));
    glDepthMask(GL_TRUE);glDisable(GL_BLEND);
}
void Renderer::drawHud(const HudState& hud){
    buildHud(hud);if(ui.empty())return;
    glDisable(GL_DEPTH_TEST);glEnable(GL_BLEND);glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(gl->hud);glUniform2f(glGetUniformLocation(gl->hud,"viewport"),float(width),float(height));
    glBindVertexArray(gl->uiVao);glBindBuffer(GL_ARRAY_BUFFER,gl->uiBuffer);glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(ui.size()*sizeof(UiVertex)),ui.data(),GL_STREAM_DRAW);glDrawArrays(GL_TRIANGLES,0,GLsizei(ui.size()));
}
double Renderer::present(bool vsync){
    const double start=now();const int interval=vsync?1:0;
    if(gl->headless){glFlush();return (now()-start)*1000;}
    if(gl->swapInterval!=interval){if(SDL_GL_SetSwapInterval(interval)!=0&&vsync)throw std::runtime_error(std::string("VSync: ")+SDL_GetError());gl->swapInterval=interval;}
    if(gl->fence)glDeleteSync(gl->fence);gl->fence=glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);glFlush();SDL_GL_SwapWindow(gl->window);return (now()-start)*1000;
}
std::vector<uint8_t> Renderer::readBack(){
    std::vector<uint8_t> pixels(size_t(width)*height*4);
    // The offscreen target holds this view's frame; the window's own buffer is
    // only a blit destination and is aliased across overlapping windows.
    if(gl->offscreenFbo)glBindFramebuffer(GL_FRAMEBUFFER,gl->offscreenFbo);
    glReadBuffer(gl->offscreenFbo?GL_COLOR_ATTACHMENT0:GL_BACK);glPixelStorei(GL_PACK_ALIGNMENT,1);glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    if(glGetError()!=GL_NO_ERROR)throw std::runtime_error("Screenshot readback failed");
    std::vector<uint8_t> flipped(pixels.size());const size_t stride=size_t(width)*4;
    for(int y=0;y<height;++y)std::memcpy(flipped.data()+size_t(y)*stride,pixels.data()+size_t(height-1-y)*stride,stride);
    return flipped;
}
bool Renderer::screenshot(const std::filesystem::path& path){return savePng(path,readBack());}
void Renderer::displayPixels(const std::vector<uint8_t>& rgba,const HudState& hud){
    if(rgba.size()!=size_t(width)*height*4)throw std::runtime_error("Invalid reference pixel buffer");
    makeCurrent();bindOffscreen();
    const size_t stride=size_t(width)*4;std::vector<uint8_t> flipped(rgba.size());
    for(int y=0;y<height;++y)std::memcpy(flipped.data()+size_t(y)*stride,rgba.data()+size_t(height-1-y)*stride,stride);
    glBindTexture(GL_TEXTURE_2D,gl->offscreenTexture);glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,flipped.data());
    glViewport(0,0,width,height);drawHud(hud);
    int w=width,h=height;SDL_GL_GetDrawableSize(gl->window,&w,&h);
    glBindFramebuffer(GL_READ_FRAMEBUFFER,gl->offscreenFbo);glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);
    glBlitFramebuffer(0,0,width,height,0,0,w,h,GL_COLOR_BUFFER_BIT,GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
}
bool Renderer::savePng(const std::filesystem::path& path,const std::vector<uint8_t>& pixels){
    if(pixels.size()!=size_t(width)*height*4)throw std::runtime_error("Invalid PNG pixel buffer");
    std::vector<uint8_t> rows((size_t(width)*4+1)*height);
    const size_t stride=size_t(width)*4;for(int y=0;y<height;++y)memcpy(rows.data()+size_t(y)*(stride+1)+1,pixels.data()+size_t(y)*stride,stride);
    uLongf size=compressBound(rows.size());std::vector<uint8_t> compressed(size);
    if(compress2(compressed.data(),&size,rows.data(),rows.size(),Z_BEST_SPEED)!=Z_OK)throw std::runtime_error("PNG compression failed");compressed.resize(size);
    if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());auto temporary=path;temporary+=".tmp";
    std::ofstream out(temporary,std::ios::binary);out.exceptions(std::ios::badbit|std::ios::failbit);out.write("\x89PNG\r\n\x1a\n",8);
    std::vector<uint8_t> header(13);for(int i=0;i<4;++i){header[i]=uint8_t(uint32_t(width)>>(24-i*8));header[4+i]=uint8_t(uint32_t(height)>>(24-i*8));}header[8]=8;header[9]=6;
    chunk(out,"IHDR",header);chunk(out,"IDAT",compressed);chunk(out,"IEND",{});out.close();std::filesystem::rename(temporary,path);return true;
}
FrameBundle Renderer::renderLayers(const std::vector<DrawObject>& objects,const Camera& camera,const HudState& hud,const std::vector<GateVisual>& gates,bool readLayers,int readbackSlot,int outputWidth,int outputHeight){
    const bool resized=outputWidth>0||outputHeight>0;
    if(resized&&(outputWidth<=0||outputHeight<=0||readbackSlot>=0))throw std::runtime_error("Invalid resized layer capture");
    if(gl->layerWidth!=width||gl->layerHeight!=height){
        glDeleteFramebuffers(3,gl->framebuffers.data());glDeleteTextures(4,gl->textures.data());
        glGenFramebuffers(3,gl->framebuffers.data());glGenTextures(4,gl->textures.data());
        for(size_t i=0;i<4;++i){
            glBindTexture(GL_TEXTURE_2D,gl->textures[i]);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D,0,i==3?GL_DEPTH_COMPONENT24:GL_RGBA8,width,height,0,i==3?GL_DEPTH_COMPONENT:GL_RGBA,i==3?GL_UNSIGNED_INT:GL_UNSIGNED_BYTE,nullptr);
        }
        for(size_t i=0;i<3;++i){
            glBindFramebuffer(GL_FRAMEBUFFER,gl->framebuffers[i]);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,gl->textures[i],0);
            if(i<2)glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,gl->textures[3],0);
            if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)throw std::runtime_error("Incomplete race framebuffer");
        }
        gl->layerWidth=width;gl->layerHeight=height;
    }
    if(!gl->composite){gl->composite=program(compositeVertexSource,compositeFragmentSource);glGenVertexArrays(1,&gl->compositeVao);}
    FrameBundle bundle;bundle.width=resized?outputWidth:width;bundle.height=resized?outputHeight:height;
    auto read=[&](std::vector<uint8_t>& out,GLuint texture){
        if(resized){out=readTextureResized(texture,outputWidth,outputHeight);return;}
        const size_t stride=size_t(width)*4;out.resize(stride*height);std::vector<uint8_t> row(stride);
        glReadBuffer(GL_COLOR_ATTACHMENT0);glPixelStorei(GL_PACK_ALIGNMENT,1);glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,out.data());
        for(int y=0;y<height/2;++y){auto* a=out.data()+size_t(y)*stride;auto* b=out.data()+size_t(height-1-y)*stride;std::memcpy(row.data(),a,stride);std::memcpy(a,b,stride);std::memcpy(b,row.data(),stride);}
    };
    glBindFramebuffer(GL_FRAMEBUFFER,gl->framebuffers[0]);gl->layerMode=true;render(objects,camera,{});gl->layerMode=false;if(readLayers)read(bundle.scene,gl->textures[0]);
    // UI shares the scene depth attachment, but clearing its color leaves that
    // depth intact. Arches therefore disappear behind buildings and bridges.
    glBindFramebuffer(GL_FRAMEBUFFER,gl->framebuffers[1]);glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);
    drawGates(gates,camera);drawHud(hud);if(readLayers)read(bundle.ui,gl->textures[1]);
    glBindFramebuffer(GL_FRAMEBUFFER,gl->framebuffers[2]);glDisable(GL_DEPTH_TEST);glDisable(GL_BLEND);
    glUseProgram(gl->composite);glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,gl->textures[0]);glUniform1i(glGetUniformLocation(gl->composite,"sceneTexture"),0);
    glActiveTexture(GL_TEXTURE1);glBindTexture(GL_TEXTURE_2D,gl->textures[1]);glUniform1i(glGetUniformLocation(gl->composite,"uiTexture"),1);
    glBindVertexArray(gl->compositeVao);glDrawArrays(GL_TRIANGLES,0,3);
    if(readbackSlot<0)read(bundle.display,gl->textures[2]);
    else {
        auto& r=gl->readbacks.at(size_t(readbackSlot));
        if(r.pending)throw std::runtime_error("Previous player frame has not been read");
        if(!r.buffer)glGenBuffers(1,&r.buffer);
        glBindBuffer(GL_PIXEL_PACK_BUFFER,r.buffer);const size_t bytes=size_t(width)*height*4;
        if(r.bytes!=bytes){glBufferData(GL_PIXEL_PACK_BUFFER,GLsizeiptr(bytes),nullptr,GL_STREAM_READ);r.bytes=bytes;}
        glReadBuffer(GL_COLOR_ATTACHMENT0);glPixelStorei(GL_PACK_ALIGNMENT,1);
        glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
        glBindBuffer(GL_PIXEL_PACK_BUFFER,0);glFlush();r.pending=true;
    }
    glBindFramebuffer(GL_FRAMEBUFFER,0);glActiveTexture(GL_TEXTURE0);glDepthMask(GL_TRUE);
    if(glGetError()!=GL_NO_ERROR)throw std::runtime_error("Race layer rendering failed");return bundle;
}
std::vector<uint8_t> Renderer::finishReadback(size_t slot,std::vector<uint8_t> out){
    auto& r=gl->readbacks.at(slot);
    if(!r.pending)throw std::runtime_error("No player frame pending");
    // Keep the size of a recycled buffer: resize then avoids clearing 8 MiB
    // per 1080p view even though the following copy overwrites every pixel.
    out.resize(r.bytes);const size_t stride=size_t(width)*4;
    glBindBuffer(GL_PIXEL_PACK_BUFFER,r.buffer);
    const auto* pixels=static_cast<const uint8_t*>(glMapBufferRange(GL_PIXEL_PACK_BUFFER,0,GLsizeiptr(r.bytes),GL_MAP_READ_BIT));
    if(!pixels){glBindBuffer(GL_PIXEL_PACK_BUFFER,0);throw std::runtime_error("Player frame readback failed");}
    for(int y=0;y<height;++y)std::memcpy(out.data()+size_t(y)*stride,pixels+size_t(height-1-y)*stride,stride);
    bool valid=glUnmapBuffer(GL_PIXEL_PACK_BUFFER)==GL_TRUE;glBindBuffer(GL_PIXEL_PACK_BUFFER,0);r.pending=false;
    if(!valid)throw std::runtime_error("Corrupted player frame readback");return out;
}
std::vector<uint8_t> Renderer::png(const std::vector<uint8_t>& rgba,int w,int h){
    const size_t stride=size_t(w)*4;if(rgba.size()!=stride*h)throw std::runtime_error("Invalid RGBA frame");
    std::vector<uint8_t> rows((stride+1)*h);for(int y=0;y<h;++y)std::memcpy(rows.data()+size_t(y)*(stride+1)+1,rgba.data()+size_t(y)*stride,stride);
    uLongf size=compressBound(rows.size());std::vector<uint8_t> compressed(size);if(compress2(compressed.data(),&size,rows.data(),rows.size(),Z_BEST_SPEED)!=Z_OK)throw std::runtime_error("PNG compression failed");compressed.resize(size);
    std::ostringstream out(std::ios::binary);out.write("\x89PNG\r\n\x1a\n",8);std::vector<uint8_t> header(13);
    for(int i=0;i<4;++i){header[i]=uint8_t(uint32_t(w)>>(24-i*8));header[4+i]=uint8_t(uint32_t(h)>>(24-i*8));}header[8]=8;header[9]=6;
    chunk(out,"IHDR",header);chunk(out,"IDAT",compressed);chunk(out,"IEND",{});auto bytes=out.str();return {bytes.begin(),bytes.end()};
}
