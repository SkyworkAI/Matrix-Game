#pragma once
#include <memory>
// An offscreen desktop OpenGL context on an EGL device (no X server).
class HeadlessGL {
public:
    HeadlessGL();~HeadlessGL();
    void makeCurrent();
    void* function(const char* name)const;
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
