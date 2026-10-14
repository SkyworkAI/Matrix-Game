#include "SceneConfig.h"
#include <stdexcept>

SceneConfig SceneConfig::load(const std::filesystem::path& root,const std::string& id){
    SceneConfig scene;
    scene.id=id;
    if(id=="castle"){
        scene.name="Medieval Castle";
        scene.assets=root/"assets";
        scene.keepCourtyardTrees=true;
        scene.groundPattern=GroundPattern::CastleNested;
        scene.fadeSmallEdges=false;
        scene.cameraDistance=4.8f;
        scene.cameraUp=.20f;
        scene.outlineWidth=1.f;
        scene.prompt="a third-person shooter game in a medieval castle with stone walls, towers and courtyards, stylized cartoon characters firing glowing projectiles, trees and grass, snowy mountains in the distance, realistic AAA game rendering, physically based materials, natural daylight, detailed vegetation and terrain";
    }else if(id=="ancient-egypt"){
        scene.name="Ancient Egypt";
        scene.assets=root/"assets/maps/ancient-egypt";
        scene.fadeSmallEdges=false;
        scene.outlineWidth=1.f;
        scene.prompt="a third-person shooter game in an ancient Egyptian desert city with a great pyramid, sandstone temples and colonnades, palm trees and sand, stylized cartoon characters firing glowing projectiles, dust and haze, realistic AAA game rendering, physically based materials, natural daylight, detailed sand and stone textures";
    }else throw std::runtime_error("Scene must be castle or ancient-egypt: "+id);
    for(const auto& path:{scene.assets/"castle-map.bin",scene.catalogDirectory()/"CatalogManifest.json"})
        if(!std::filesystem::is_regular_file(path))throw std::runtime_error("Missing scene asset: "+path.string());
    if(id!="castle"&&!std::filesystem::is_regular_file(scene.assets/"map.json"))
        throw std::runtime_error("Missing scene profile: "+(scene.assets/"map.json").string());
    return scene;
}

std::filesystem::path SceneConfig::outputDirectory(const std::filesystem::path& root,const char* category) const{
    return id=="castle"?root/category:root/category/id;
}
