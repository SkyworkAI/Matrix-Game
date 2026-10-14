#include "Renderer.h"
#include <SDL.h>
#include <iostream>
#include <map>

void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    if(SDL_Init(SDL_INIT_VIDEO)!=0)throw std::runtime_error(SDL_GetError());
    struct Lifetime {~Lifetime(){SDL_Quit();}} lifetime;
#ifdef __APPLE__
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,4);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
#else
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
#endif
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
    std::unique_ptr<SDL_Window,decltype(&SDL_DestroyWindow)> window(SDL_CreateWindow("Caption visibility test",0,0,864,480,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN),SDL_DestroyWindow);
    if(!window)throw std::runtime_error(SDL_GetError());Renderer renderer(window.get(),864,480);
    MeshData mesh;mesh.vertices={{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0}};mesh.indices={0,1,2,0,2,3};mesh.captionTokens={1,1};
    auto tagged=renderer.addMesh(mesh);mesh.captionTokens.clear();auto plain=renderer.addMesh(mesh);
    const auto object=[](uint32_t m,V3 p,float token,float scale=1){return DrawObject{m,{transform(p,{scale,scale,scale}),{},0,{token,0,0}},p,2*scale,100};};
    Camera camera{{0,0,0},{0,0,1},90};
    std::vector<DrawObject> objects={object(tagged,{0,0,3},0),object(plain,{0,0,6},2),object(plain,{-3,0,6},3),object(plain,{3,0,6},4),object(plain,{100,0,6},5)};
    renderer.render(objects,camera,{});const auto rgb=renderer.readBack();
    auto stats=renderer.captionVisibility(objects,camera);std::map<uint32_t,captions::PixelStats> byToken;uint64_t total=0;
    for(auto s:stats){byToken[s.token]=s;total+=s.pixels;}
    require(total==432*240,"visibility resolution/aspect");require(byToken.contains(1)&&byToken.contains(3)&&byToken.contains(4),"mesh and instance tokens present");
    require(!byToken.contains(2)&&!byToken.contains(5),"occluded and offscreen objects excluded");
    require(byToken.at(3).x<.5&&byToken.at(4).x>.5,"screen directions preserved");
    require(rgb==renderer.readBack(),"ID pass cannot contaminate RGB");
    objects.insert(objects.begin(),object(plain,{0,0,2},0));renderer.render(objects,camera,{});stats=renderer.captionVisibility(objects,camera);byToken.clear();for(auto s:stats)byToken[s.token]=s;
    require(!byToken.contains(1)&&!byToken.contains(2),"unknown geometry occludes known objects");
    renderer.resize(640,360);renderer.render(objects,camera,{});renderer.captionVisibility(objects,camera,320);
    std::cout<<"PASS: integer ID pass, static/dynamic tokens, occlusion, unknown occluders, frustum, RGB preservation, resize\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
