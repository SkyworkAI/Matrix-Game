#pragma once
#ifdef _WIN32
#include <Windows.h>
#include <DirectXMath.h>
#endif
#include <chrono>
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
#ifdef _WIN32
using Matrix = DirectX::XMFLOAT4X4;
#else
// Row-vector matrices, identical byte layout to the original DirectX matrices.
struct Matrix { float m[4][4]{}; };
#endif
inline Matrix multiply(const Matrix& a,const Matrix& b) {
    Matrix out{};
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<4;++k)out.m[i][j]+=a.m[i][k]*b.m[k][j];
    return out;
}
inline Matrix transform(V3 p,V3 scale={1,1,1},float yaw=0,float roll=0) {
    const float cy=std::cos(yaw),sy=std::sin(yaw),cr=std::cos(roll),sr=std::sin(roll);
    Matrix m{};
    m.m[0][0]=scale.x*cr*cy;m.m[0][1]=scale.x*sr;m.m[0][2]=-scale.x*cr*sy;
    m.m[1][0]=-scale.y*sr*cy;m.m[1][1]=scale.y*cr;m.m[1][2]=scale.y*sr*sy;
    m.m[2][0]=scale.z*sy;m.m[2][2]=scale.z*cy;
    m.m[3][0]=p.x;m.m[3][1]=p.y;m.m[3][2]=p.z;m.m[3][3]=1;
    return m;
}
inline V3 position(const Matrix& m) { return {m.m[3][0],m.m[3][1],m.m[3][2]}; }
inline V3 transformPoint(V3 v,const Matrix& m) {
    return {v.x*m.m[0][0]+v.y*m.m[1][0]+v.z*m.m[2][0]+m.m[3][0],
            v.x*m.m[0][1]+v.y*m.m[1][1]+v.z*m.m[2][1]+m.m[3][1],
            v.x*m.m[0][2]+v.y*m.m[1][2]+v.z*m.m[2][2]+m.m[3][2]};
}
#ifdef _WIN32
inline void check(HRESULT hr,const char* action) {
    if(FAILED(hr)) throw std::runtime_error(std::string(action)+" (HRESULT "+std::to_string(uint32_t(hr))+")");
}
#endif
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct MeshData { std::vector<V3> vertices; std::vector<uint32_t> indices; std::vector<std::array<uint32_t,2>> edges; std::vector<uint32_t> colors; std::vector<uint32_t> captionTokens; };
struct MapLine { V3 a,b; bool freeway=false; };
struct Instance { Matrix world; Color color; float surface=0; float pad[3]{}; };
struct DrawObject { uint32_t mesh=0; Instance instance; V3 center; float radius=1; float range=300; };
struct Camera { V3 eye{0,3,-7},target{0,1,0}; float horizontalFov=90; };
struct Input { float throttle=0,steer=0,brake=0,handbrake=0; };
