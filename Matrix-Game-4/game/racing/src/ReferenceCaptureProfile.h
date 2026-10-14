#pragma once

namespace reference::capture {
constexpr int RenderWidth=1920,RenderHeight=1080;
constexpr int HudWidth=1280,HudHeight=720;
constexpr float HudScale=float(RenderWidth)/HudWidth;
// Keep the final contour thickness unchanged when increasing supersampling.
constexpr float OutlineWidth=1.2f*HudScale;
}
