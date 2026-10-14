#include "HeadlessGL.h"
#include <stdexcept>
#include <string>
#ifdef CITYSAMPLE_HAVE_EGL
#include <EGL/egl.h>
#include <EGL/eglext.h>
struct HeadlessGL::Impl {
    EGLDisplay display=EGL_NO_DISPLAY;EGLContext context=EGL_NO_CONTEXT;EGLSurface surface=EGL_NO_SURFACE;
    ~Impl(){if(display!=EGL_NO_DISPLAY){eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);if(context!=EGL_NO_CONTEXT)eglDestroyContext(display,context);if(surface!=EGL_NO_SURFACE)eglDestroySurface(display,surface);eglTerminate(display);}}
};
HeadlessGL::HeadlessGL():impl(std::make_unique<Impl>()){
    auto query=reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
    auto platform=reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if(!query||!platform)throw std::runtime_error("EGL device extensions unavailable");
    EGLDeviceEXT devices[16];EGLint count=0;if(!query(16,devices,&count))throw std::runtime_error("Cannot enumerate EGL devices");
    for(int i=0;i<count;++i){auto display=platform(EGL_PLATFORM_DEVICE_EXT,devices[i],nullptr);if(display==EGL_NO_DISPLAY)continue;
        if(eglInitialize(display,nullptr,nullptr)){impl->display=display;break;}
    }
    if(impl->display==EGL_NO_DISPLAY)throw std::runtime_error("No EGL device initialized; verify matching graphics libraries and device access");
    if(!eglBindAPI(EGL_OPENGL_API))throw std::runtime_error("EGL desktop OpenGL unavailable");
    const EGLint configAttributes[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_BIT,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_DEPTH_SIZE,24,EGL_NONE};
    EGLConfig config;EGLint configs=0;if(!eglChooseConfig(impl->display,configAttributes,&config,1,&configs)||!configs)throw std::runtime_error("No EGL OpenGL config");
    const EGLint surfaceAttributes[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};impl->surface=eglCreatePbufferSurface(impl->display,config,surfaceAttributes);
    const EGLint contextAttributes[]={EGL_CONTEXT_MAJOR_VERSION_KHR,3,EGL_CONTEXT_MINOR_VERSION_KHR,3,EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR,EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,EGL_NONE};
    impl->context=eglCreateContext(impl->display,config,EGL_NO_CONTEXT,contextAttributes);
    if(impl->surface==EGL_NO_SURFACE||impl->context==EGL_NO_CONTEXT)throw std::runtime_error("Cannot create EGL OpenGL 3.3 context");makeCurrent();
}
void HeadlessGL::makeCurrent(){if(!eglMakeCurrent(impl->display,impl->surface,impl->surface,impl->context))throw std::runtime_error("Cannot activate EGL context");}
void* HeadlessGL::function(const char* name)const{return reinterpret_cast<void*>(eglGetProcAddress(name));}
#else
struct HeadlessGL::Impl {};
HeadlessGL::HeadlessGL(){throw std::runtime_error("EGL rendering is available in Linux builds only");}
void HeadlessGL::makeCurrent(){}
void* HeadlessGL::function(const char*)const{return nullptr;}
#endif
HeadlessGL::~HeadlessGL()=default;
