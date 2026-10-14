#include "SceneConfig.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>

int main(int argc,char** argv){
    const auto root=std::filesystem::temp_directory_path()/("matrix-scene-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {std::filesystem::path root;~Cleanup(){std::filesystem::remove_all(root);}} cleanup{root};
    for(const auto* folder:{"assets","assets/maps/ancient-egypt"}){
        std::filesystem::create_directories(root/folder/"shooter");
        std::ofstream(root/folder/"castle-map.bin").put('0');
        std::ofstream(root/folder/"map.json")<<"{}";
        std::ofstream(root/folder/"shooter/CatalogManifest.json")<<"{}";
    }
    const auto castle=SceneConfig::load(root),egypt=SceneConfig::load(root,"ancient-egypt");
    assert(castle.keepCourtyardTrees&&castle.groundPattern==SceneConfig::GroundPattern::CastleNested&&!castle.fadeSmallEdges);
    assert(castle.cameraDistance==4.8f&&castle.cameraUp==.2f&&castle.outlineWidth==1.f);
    assert(!castle.prompt.empty());
    assert(!egypt.keepCourtyardTrees&&egypt.groundPattern==SceneConfig::GroundPattern::EgyptNested&&!egypt.fadeSmallEdges);
    assert(egypt.cameraDistance==4.2f&&egypt.cameraUp==.15f&&egypt.outlineWidth==1.f);
    assert(!egypt.prompt.empty()&&egypt.prompt!=castle.prompt);
    assert(egypt.prompt.find("realistic AAA game rendering")!=std::string::npos);
    assert(castle.outputDirectory(root,"offline-recordings")==root/"offline-recordings");
    assert(egypt.outputDirectory(root,"offline-recordings")==root/"offline-recordings/ancient-egypt");
    assert(castle.catalogDirectory()!=egypt.catalogDirectory());
    for(const auto* id:{"desert","../castle",""}){
        bool rejected=false;try{SceneConfig::load(root,id);}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
    std::cout<<"PASS scene defaults, isolated outputs, catalogs and invalid scene rejection\n";
}
