#pragma once
#include <filesystem>
#include <string>

// Shared by the desktop player, offline recorder and reference server.
struct SceneConfig {
    enum class GroundPattern { CastleNested, EgyptNested };
    std::string id, name, prompt;
    std::filesystem::path assets;
    bool keepCourtyardTrees=false;
    GroundPattern groundPattern=GroundPattern::EgyptNested;
    bool fadeSmallEdges=true;
    float cameraDistance=4.2f, cameraUp=.15f, outlineWidth=.72f;

    static SceneConfig load(const std::filesystem::path& root,const std::string& id="castle");
    std::filesystem::path outputDirectory(const std::filesystem::path& root,const char* category) const;
    std::filesystem::path catalogDirectory() const { return assets/"shooter"; }
};
