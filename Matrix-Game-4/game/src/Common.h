#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <iterator>
#include <utility>
#include <map>
#include <set>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <locale>
#include <iomanip>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <DirectXMath.h>

constexpr float Pi = 3.14159265358979323846f;
struct V3 {
    float x = 0, y = 0, z = 0;
    V3 operator+(V3 b) const { return {x+b.x,y+b.y,z+b.z}; }
    V3 operator-(V3 b) const { return {x-b.x,y-b.y,z-b.z}; }
    V3 operator*(float s) const { return {x*s,y*s,z*s}; }
    float length() const { return std::sqrt(x*x+y*y+z*z); }
    V3 normalized() const { float l=length(); return l>0.00001f ? *this*(1/l) : V3{}; }
};
inline float dot(V3 a,V3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline V3 cross(V3 a,V3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
struct Color { float r=1,g=1,b=1,a=1; };
inline Color rgb(int r,int g,int b) { return {r/255.f,g/255.f,b/255.f,1}; }
// Apply once to raw catalog colors; castle static colors are already calibrated.
inline Color bboxFill(Color c){return {.23f+.36f*c.r,.23f+.36f*c.g,.23f+.36f*c.b,c.a};}
using Matrix = DirectX::XMFLOAT4X4;
inline Matrix transform(V3 p,V3 scale={1,1,1},float yaw=0,float roll=0) {
    Matrix m;
    DirectX::XMStoreFloat4x4(&m,DirectX::XMMatrixScaling(scale.x,scale.y,scale.z)*
        DirectX::XMMatrixRotationRollPitchYaw(0,yaw,roll)*DirectX::XMMatrixTranslation(p.x,p.y,p.z));
    return m;
}
inline Matrix multiply(const Matrix& a,const Matrix& b) {
    Matrix m; DirectX::XMStoreFloat4x4(&m,DirectX::XMLoadFloat4x4(&a)*DirectX::XMLoadFloat4x4(&b)); return m;
}
inline V3 position(const Matrix& m) { return {m._41,m._42,m._43}; }
#ifdef _WIN32
inline void check(HRESULT hr,const char* action) {
    if(FAILED(hr)) throw std::runtime_error(std::string(action)+" (HRESULT "+std::to_string(uint32_t(hr))+")");
}
#endif
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct MeshData { std::vector<V3> vertices; std::vector<uint32_t> indices; std::vector<std::array<uint32_t,2>> edges; std::vector<uint32_t> colors,tokens; };
struct MapLine { V3 a,b; bool freeway=false; };
struct Instance { Matrix world; Color color; float surface=0; uint32_t objectToken=0; float pad[2]{}; };
struct DrawObject { uint32_t mesh=0; Instance instance; V3 center; float radius=1; float range=300; Color captureColor{0,0,0,-1}; float maxScreenDiameter=0; };
struct Camera { V3 eye{0,3,-7},target{0,1,0}; float horizontalFov=90; };
struct Input { float throttle=0,steer=0,brake=0,handbrake=0; };
