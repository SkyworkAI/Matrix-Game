#pragma once

// Keep ordinary capture/IPC overhead on the existing clock. After real
// backpressure, rebase instead of accumulating an unbounded catch-up burst.
inline double resumeReferenceTick(double nextTick, double current, int fps) {
    return current - nextTick > 1.0 / fps ? current + 1.0 / 120 : nextTick;
}
