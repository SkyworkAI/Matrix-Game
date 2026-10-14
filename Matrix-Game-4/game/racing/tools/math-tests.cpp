#include "../src/Common.h"
#include <iostream>
int main(){try{
    auto near=[](V3 a,V3 b){if((a-b).length()>1e-5f)throw std::runtime_error("Transform convention mismatch");};
    near(transformPoint({1,2,3},transform({10,20,30},{2,3,4})),{12,26,42});
    near(transformPoint({0,0,1},transform({0,0,0},{1,1,1},Pi/2)),{1,0,0});
    near(transformPoint({1,0,0},transform({0,0,0},{1,1,1},0,Pi/2)),{0,1,0});
    auto a=transform({2,3,4},{2,1,3},.7f,.3f),b=transform({-3,2,1},{1,4,2},-.2f,.8f);
    near(transformPoint({1,-2,3},multiply(a,b)),transformPoint(transformPoint({1,-2,3},a),b));
#ifdef _WIN32
    Matrix expected;DirectX::XMStoreFloat4x4(&expected,DirectX::XMMatrixScaling(2,1,3)*DirectX::XMMatrixRotationRollPitchYaw(0,.7f,.3f)*DirectX::XMMatrixTranslation(2,3,4));
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)if(std::abs(a.m[i][j]-expected.m[i][j])>1e-5f)throw std::runtime_error("DirectX transform regression");
#endif
    std::cout<<"PASS: scale, translation, yaw, roll, composition\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
