#include "../src/ReferencePacing.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    for(int fps : {24,30,40,60}) {
        const double period=1.0/fps;
        assert(resumeReferenceTick(10,10-period/2,fps)==10);
        assert(resumeReferenceTick(10,10+period/2,fps)==10);
        assert(std::abs(resumeReferenceTick(10,12,fps)-(12+1.0/120))<1e-12);
        double next=100;
        for(int chunk=0;chunk<1000;++chunk) {
            const double before=next;
            next=resumeReferenceTick(next,next+period/4,fps);
            assert(next==before);
            for(int tick=0;tick<4*120/fps;++tick)next+=1.0/120;
        }
        assert(std::abs(next-(100+4000.0/fps))<1e-8);
    }
    std::cout<<"REFERENCE_PACING_TEST_PASSED\n";
}
