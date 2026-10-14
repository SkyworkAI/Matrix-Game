#include "Renderer.h"
#include "World.h"
#include "ReferenceCaptureProfile.h"
#include <SDL.h>
#include <nlohmann/json.hpp>
#include <iostream>

void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void save(const std::filesystem::path& path,const std::vector<uint8_t>& rgba,int width,int height){
    auto bytes=Renderer::png(rgba,width,height);std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));require(bool(f),"PNG write failed");
}
void sameVisibility(const std::vector<captions::PixelStats>& a,const std::vector<captions::PixelStats>& b){
    require(a.size()==b.size(),"Appearance overlays changed visible entity count");
    for(size_t i=0;i<a.size();++i)require(a[i].token==b[i].token&&a[i].pixels==b[i].pixels&&a[i].x==b[i].x&&a[i].y==b[i].y,
        "Appearance overlays changed caption IDs, area or screen position");
}
int main(int argc,char** argv){try{
    require(argc==3,"Usage: ReferenceRenderTests <game-root> <output-dir>");
    const std::filesystem::path root=argv[1],output=argv[2];std::filesystem::create_directories(output);
    const bool headless=std::getenv("CITYSAMPLE_EGL")&&std::string(std::getenv("CITYSAMPLE_EGL"))=="1";
    require(SDL_Init(headless?SDL_INIT_TIMER:SDL_INIT_VIDEO)==0,"SDL init failed");
    struct Lifetime{~Lifetime(){SDL_Quit();}} lifetime;
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef __APPLE__
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
#endif
    constexpr int renderWidth=reference::capture::RenderWidth,renderHeight=reference::capture::RenderHeight;
    static_assert(renderWidth==1920&&renderHeight==1080,"Reference capture must render at 1080p");
    std::unique_ptr<SDL_Window,decltype(&SDL_DestroyWindow)> window(headless?nullptr:SDL_CreateWindow("Reference render test",0,0,renderWidth,renderHeight,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN),SDL_DestroyWindow);
    require(headless||bool(window),"SDL window failed");Renderer renderer(window.get(),renderWidth,renderHeight);
    World world(root/"assets/bbox-assets.bin",16,40);world.enableCaptions(root);world.upload(renderer);world.reset();
    const auto camera=world.camera(1,false);const auto& objects=world.drawObjects();
    nlohmann::json report={{"renderer",renderer.adapterName}};
    for(const auto& name:{"legacy","resize_only","dataset"}){
        const bool legacy=std::string(name)=="legacy";renderer.resize(legacy?864:renderWidth,legacy?480:renderHeight);
        renderer.outlineWidth=legacy?1.2f:reference::capture::OutlineWidth;
        renderer.buildingStripes=std::string(name)=="dataset";
        renderer.render(objects,camera,{});
        auto native=renderer.readBack();auto resized=renderer.readBackResized(864,480);
        save(output/(std::string(name)+"-native.png"),native,renderer.width,renderer.height);
        save(output/(std::string(name)+".png"),resized,864,480);
        require(renderer.readBack()==native,"Resize contaminated source RGB");
        auto visible=renderer.captionVisibility(objects,camera,432,864.f/480);uint64_t area=0;for(auto s:visible)area+=s.pixels;
        require(area==432*240,"Caption viewport does not match final crop");
        require(renderer.readBackResized(864,480)==resized,"Caption pass contaminated resized RGB");
        const double start=now();for(int i=0;i<20;++i){renderer.render(objects,camera,{});renderer.readBackResized(864,480);}
        report[name]={{"capture_mean_ms",(now()-start)*1000/20},{"bytes",resized.size()},
            {"render_width",renderer.width},{"render_height",renderer.height}};
    }
    renderer.resize(renderWidth,renderHeight);renderer.hudScale=reference::capture::HudScale;renderer.buildingStripes=false;
    std::vector<DrawObject> base;size_t stripeCount=0;
    for(const auto& o:objects)if(o.instance.surface< -1.5f)++stripeCount;else base.push_back(o);
    require(stripeCount==614,"Missing authored building fronts");
    renderer.render(objects,camera,{});const auto noStripes=renderer.readBackResized(864,480);
    const auto baseVisibility=renderer.captionVisibility(base,camera,432,864.f/480);
    renderer.buildingStripes=true;renderer.render(base,camera,{});
    require(renderer.readBackResized(864,480)==noStripes,"Stripe switch changed base rendering");
    renderer.render(objects,camera,{});const auto withStripes=renderer.readBackResized(864,480);
    HudState recordHud;recordHud.visible=true;recordHud.speed=42;
    const auto layers=renderer.renderLayers(objects,camera,recordHud,{},true,-1,864,480);
    require(layers.scene==withStripes,"Offline scene differs from reference server resize/crop");
    require(layers.ui.size()==layers.scene.size()&&layers.display.size()==layers.scene.size(),"Offline layer dimensions differ");
    size_t transparent=0,visibleUi=0;
    for(size_t i=0;i<layers.ui.size();i+=4){
        const int alpha=layers.ui[i+3];transparent+=alpha==0;visibleUi+=alpha>0;
        for(int c=0;c<3;++c)require(layers.ui[i+c]<=alpha+1,"UI is not premultiplied alpha");
    }
    require(transparent>10000&&visibleUi>100,"UI transparency or HUD capture missing");
    save(output/"offline-ui.png",layers.ui,864,480);save(output/"offline-display.png",layers.display,864,480);
    recordHud.race=true;recordHud.pursuit=true;recordHud.roads=&world.mapLines();
    recordHud.location=position(world.player().body);recordHud.heading=world.player().heading;
    recordHud.mapCenter=world.mapCenter();recordHud.mapHalfSize=world.mapHalfSize();
    recordHud.suspectPosition=recordHud.mapCenter;recordHud.suspectDistance=900;recordHud.racePhase="PURSUIT";
    std::vector<uint8_t> legacyUi;
    for(auto size:{std::array<int,2>{1280,720},std::array<int,2>{renderWidth,renderHeight},std::array<int,2>{640,360}}){
        const int outputWidth=std::min(864,size[0]),outputHeight=std::min(480,size[1]);
        renderer.hudScale=size[0]==renderWidth?reference::capture::HudScale:1.f;
        renderer.resize(size[0],size[1]);renderer.render(objects,camera,{});
        auto clean=renderer.readBackResized(outputWidth,outputHeight);
        auto chase=renderer.renderLayers(objects,camera,recordHud,{},true,-1,outputWidth,outputHeight);
        require(chase.scene==clean,"Pursuit HUD contaminated clean model input");
        size_t red=0,green=0;
        for(size_t i=0;i<chase.ui.size();i+=4){
            red+=chase.ui[i]>120&&chase.ui[i]>chase.ui[i+1]*1.5f;
            green+=chase.ui[i+1]>120&&chase.ui[i+1]>chase.ui[i]*1.5f;
        }
        require(red>15&&green>15,"Pursuit map markers missing");
        if(size[0]==1280)legacyUi=chase.ui;
        if(size[0]==renderWidth){
            size_t mismatch=0;
            for(size_t i=3;i<chase.ui.size();i+=4)mismatch+=(chase.ui[i]>20)!=(legacyUi[i]>20);
            require(mismatch<864*480/100,"1080p capture changed HUD panel footprints");
            report["hud"]={{"logical_width",reference::capture::HudWidth},{"logical_height",reference::capture::HudHeight},
                {"render_scale",renderer.hudScale},{"footprint_mismatch_pixels",mismatch}};
            save(output/"pursuit-ui-1080.png",chase.ui,outputWidth,outputHeight);
        }
        save(output/("pursuit-"+std::to_string(size[0])+".png"),chase.display,outputWidth,outputHeight);
    }
    renderer.resize(renderWidth,renderHeight);renderer.hudScale=reference::capture::HudScale;
    renderer.render(objects,camera,{});require(renderer.readBackResized(864,480)==withStripes,"Layer capture contaminated server rendering");
    size_t stripePixels=0;for(size_t i=0;i<noStripes.size();i+=4)
        stripePixels+=!std::equal(noStripes.begin()+i,noStripes.begin()+i+3,withStripes.begin()+i);
    require(stripePixels>100,"Authored fronts did not render");
    sameVisibility(baseVisibility,renderer.captionVisibility(objects,camera,432,864.f/480));
    renderer.buildingStripes=false;
    sameVisibility(baseVisibility,renderer.captionVisibility(objects,camera,432,864.f/480));
    report["building_stripes"]={{"geometry_chunks",stripeCount},{"changed_pixels",stripePixels},{"caption_unchanged",true}};
    for(int mode=0;mode<2;++mode){
        const auto view=world.camera(mode,false);renderer.render(objects,view,{});
        save(output/("camera-mode-"+std::to_string(mode)+".png"),renderer.readBackResized(864,480),864,480);
        report["cameras"][std::to_string(mode)]={{"eye",{view.eye.x,view.eye.y,view.eye.z}},
            {"target",{view.target.x,view.target.y,view.target.z}},{"horizontal_fov",view.horizontalFov}};
    }
    const auto offset=camera.eye-position(world.player().body),direction=camera.target-camera.eye;
    require(std::abs(std::hypot(offset.x,offset.z)-6.f)<.001f,"Default camera distance changed");
    require(std::abs(offset.y-2.1f)<.001f,"Default camera must be 0.30 m above the dataset-matched view");
    require(std::abs(camera.horizontalFov-90.f)<.001f,"Default camera FOV changed");
    require(std::abs(std::atan2(-direction.y,std::hypot(direction.x,direction.z))*180/Pi-13.15f)<.001f,
        "Default camera pitch changed");
    std::vector<DrawObject> ego;
    for(auto object:objects){const auto token=size_t(object.instance.pad[0]);
        if(token&&world.captionEntities().at(token).ego){object.instance.color={1,1,1,1};ego.push_back(object);}}
    require(ego.size()==5,"Missing ego body or wheels in framing test");
    renderer.render(ego,camera,{});const auto mask=renderer.readBackResized(864,480);
    int left=864,top=480,right=-1,bottom=-1;
    for(int y=0;y<480;++y)for(int x=0;x<864;++x)if(mask[(y*864+x)*4]>128){
        left=std::min(left,x);right=std::max(right,x);top=std::min(top,y);bottom=std::max(bottom,y);}
    require(left>=320&&right<=545&&right-left>=170&&top>=185&&top<=240&&bottom>=350&&bottom<=410,
        "Default ego framing no longer matches the control-video composition");
    report["ego_framing"]={{"bounds",{left,top,right,bottom}},{"bottom_margin_px",479-bottom}};
    save(output/"camera-ego-mask.png",mask,864,480);
    MeshData wall;wall.vertices={{-3,-3,0},{3,-3,0},{3,3,0},{-3,3,0}};wall.indices={0,1,2,0,2,3};
    const auto building=renderer.addMesh(wall);
    MeshData ground;ground.vertices={{-3,0,-3},{3,0,-3},{3,0,3},{-3,0,3}};ground.indices=wall.indices;
    const auto horizontal=renderer.addMesh(ground);
    const Camera view{{0,2,0},{0,0,6},90};
    for(auto mesh:{building,horizontal}){
        std::vector<DrawObject> fixture={{mesh,{transform({0,0,6}),{.6f,.4f,.5f,1},0}, {0,0,6},5,100}};
        renderer.buildingStripes=false;renderer.render(fixture,view,{});auto plain=renderer.readBackResized(864,480);
        renderer.buildingStripes=true;renderer.render(fixture,view,{});auto striped=renderer.readBackResized(864,480);
        require(plain==striped,"Stripe toggle changed a base surface without overlay geometry");
    }
    std::vector<DrawObject> captionFixture={{building,{transform({0,0,6}),{.6f,.4f,.5f,1},0,{1,0,0}}, {0,0,6},5,100}};
    const auto wallVisibility=renderer.captionVisibility(captionFixture,view);
    captionFixture.push_back({building,{transform({0,0,5.99f}),{.37f,.39f,.41f,1},-2}, {0,0,5.99f},5,100});
    sameVisibility(wallVisibility,renderer.captionVisibility(captionFixture,view));
    renderer.buildingStripes=false;
    std::vector<DrawObject> road={{horizontal,{transform({0,0,6}),{.6f,.6f,.6f,1},-1}, {0,0,6},5,100}};
    renderer.render(road,view,{});auto flat=renderer.readBackResized(864,480);
    road[0].instance.surface=1;renderer.render(road,view,{});auto grains=renderer.readBackResized(864,480);
    size_t changed=0;int maxDelta=0;
    for(size_t i=0;i<grains.size();i+=4){
        require(grains[i]==grains[i+1]&&grains[i]==grains[i+2],"Road noise must be achromatic");
        auto delta=std::abs(int(grains[i])-int(flat[i]));changed+=delta>0;maxDelta=std::max(maxDelta,delta);
    }
    require(changed>100&&maxDelta<=6,"Road grain absent or exceeds intended amplitude");
    save(output/"ground-grains.png",grains,864,480);
    for(int i=0;i<8;++i){auto moved=view;moved.eye.x+=i*.02f;moved.target.x+=i*.02f;
        renderer.render(road,moved,{});save(output/("ground-motion-"+std::to_string(i)+".png"),renderer.readBackResized(864,480),864,480);}
    renderer.render(road,view,{});require(renderer.readBackResized(864,480)==grains,"Ground texture depends on frame history");
    report["ground_grains"]={{"changed_pixels",changed},{"max_delta_from_flat",maxDelta},{"world_locked",true}};
    road[0].instance.surface=2;renderer.render(road,view,{});
    require(renderer.readBackResized(864,480)==grains,"Junction asphalt must use exactly the road grain shader");
    MeshData pole;pole.vertices={{-.25f,-3,0},{.25f,-3,0},{.25f,3,0},{-.25f,3,0}};
    pole.indices={0,1,2,0,2,3};pole.edges={{0,1},{1,2},{2,3},{3,0}};
    const auto poleMesh=renderer.addMesh(pole);const float outline=renderer.outlineWidth;
    const Camera poleView{{0,0,0},{0,0,1},90};
    for(float distance:{50.f,100.f,200.f}){
        std::vector<DrawObject> fixture={{poleMesh,{transform({0,0,distance}),{1,1,1,1},0,{1,0,0}},
            {0,0,distance},4,1000}};
        renderer.outlineWidth=0;renderer.render(fixture,poleView,{});const auto unlined=renderer.readBack();
        const auto visibility=renderer.captionVisibility(fixture,poleView);
        renderer.outlineWidth=outline;renderer.render(fixture,poleView,{});const auto lined=renderer.readBack();
        size_t facePixels=0,blackPixels=0;
        for(size_t i=0;i<unlined.size();i+=4)if(unlined[i]>200){
            ++facePixels;
            if(lined[i]!=unlined[i]){
                require(lined[i]==0&&lined[i+1]==0&&lined[i+2]==0,"Small-face outlines must not fade");
                ++blackPixels;
            }
        }
        require(facePixels>0&&blackPixels>0,"Thin distant fixture or its outline disappeared");
        sameVisibility(visibility,renderer.captionVisibility(fixture,poleView));
        report["outlines"][std::to_string(int(distance))]={{"face_pixels",facePixels},{"black_pixels",blackPixels}};
    }
    report["outlines"]["width_internal_px"]=outline;
    const std::array<Camera,3> junctionViews={
        Camera{{-325.93f,3.414f,363.007f},{-325.92f,.611f,375.007f},90},
        Camera{{-421.08f,3.417f,-119.16f},{-418.80f,.614f,-107.38f},90},
        Camera{{-324,95,394},{-324,.6f,395},90}};
    renderer.buildingStripes=true;
    std::vector<DrawObject> withoutJunctions;size_t junctionCount=0;
    for(const auto& o:objects)if(o.instance.surface==2){
        ++junctionCount;
        require(o.instance.pad[0]==0,"Junction overlay must not have a caption entity");
        require(std::abs(o.instance.color.r-132.f/255)<.00001f&&std::abs(o.instance.color.g-134.f/255)<.00001f&&
            std::abs(o.instance.color.b-132.f/255)<.00001f,"Junction/road color mismatch");
    }else withoutJunctions.push_back(o);
    require(junctionCount==19,"Missing junction material patches");
    for(size_t i=0;i<junctionViews.size();++i){
        renderer.render(objects,junctionViews[i],{});
        save(output/("junction-"+std::to_string(i)+".png"),renderer.readBackResized(864,480),864,480);
        sameVisibility(renderer.captionVisibility(objects,junctionViews[i]),renderer.captionVisibility(withoutJunctions,junctionViews[i]));
    }
    report["junction_surfaces"]={{"chunks",junctionCount},{"caption_unchanged",true},{"road_rgb",{132,134,132}},{"sidewalk_rgb",{172,173,169}}};
    size_t patchPixels=0;bool savedPatch=false;
    for(const auto& patch:objects)if(patch.instance.surface==2){
        Camera inspection{patch.center+V3{0,65,0},patch.center+V3{0,0,1},90};
        renderer.render(withoutJunctions,inspection,{});auto before=renderer.readBackResized(864,480);
        renderer.render(objects,inspection,{});auto after=renderer.readBackResized(864,480);
        size_t changed=0;for(size_t i=0;i<before.size();i+=4)changed+=!std::equal(before.begin()+i,before.begin()+i+3,after.begin()+i);
        if(changed>100&&!savedPatch){save(output/"junction-patch-before.png",before,864,480);save(output/"junction-patch-after.png",after,864,480);savedPatch=true;}
        patchPixels+=changed;
        sameVisibility(renderer.captionVisibility(objects,inspection),renderer.captionVisibility(withoutJunctions,inspection));
        renderer.render(objects,inspection,{});require(renderer.readBackResized(864,480)==after,"Junction patches are not deterministic");
    }
    require(patchPixels>100,"Junction material patches are not visible");
    report["junction_surfaces"]["changed_pixels"]=patchPixels;
    std::ofstream(output/"timings.json")<<report.dump(2)<<'\n';
    std::cout<<report.dump(2)<<"\nPASS: camera framing, resize/crop, authored front geometry, unchanged caption visibility, bounded world-locked road grain, full-contrast distant outlines\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
