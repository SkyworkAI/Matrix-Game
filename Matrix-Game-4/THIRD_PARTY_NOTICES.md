# Third-Party Notices

The project-level Apache-2.0 license does not replace the licenses below.
Original notices in source files are retained. All listed components are used
as source libraries or external runtime dependencies, not rebranded assets.

| Component | Location/use | License / provenance |
| --- | --- | --- |
| MG4CF model implementation | `matrix_game/inference/wan`, wrappers, loading, scheduler, decoder, rollout | Apache-2.0; extracted from MG4CF `d5b0307`, based on Wan and Causal Forcing; `licenses/MG4CF-LICENSE` |
| Wan model/codec code | Same model modules | Original Alibaba/Wan notices retained; project snapshot uses Apache-2.0 |
| TAEHV | `matrix_game/inference/taehv.py` | MIT; Ollin Boer Bohan, [upstream](https://github.com/madebyollin/taehv), commit `011dfc2112197741c540e0bdd5b7b67bcc930771`; `licenses/TAEHV-LICENSE` |
| MediaMTX reader | `matrix_game/web/mediamtx-reader.js` | MIT; adjacent `mediamtx-reader.LICENSE` |
| Jolt Physics | `game/third_party/JoltPhysics` | MIT; included `LICENSE` |
| Recast/Detour | `game/third_party/recastnavigation` | zlib; included `License.txt` |
| DirectXMath and SAL compatibility | `game/third_party/DirectXMath` | Included `LICENSE` and `SAL-LICENSE.txt` |
| stb | `game/third_party/stb` | Included dual-license text |
| nlohmann/json | `game/third_party/json.hpp` | MIT notice embedded in the header |
| Racing engine, tires and impacts | Separate `racing/assets/audio` bundle | CC0 and CC BY 3.0; retain `audio/CREDITS.md`, including tire attribution and adaptation details |

Changes to the extracted Python modules include package-qualified imports,
removal of unrelated training/data entrypoints, safe checkpoint loading and
removal of private deployment defaults. The full backbone is retained to avoid
changing its parameter structure during release packaging. TAEHV has the
existing Wan2.2 streaming adapter; the model math is not a new implementation.

Project-owned streaming integration and optimization code is covered by the
root Apache-2.0 `LICENSE`. This does not replace the third-party notices above.

MediaMTX, GStreamer, x264, FFmpeg, SDL2, PyTorch and other installed dependencies
are not bundled as binaries here. Their own licenses apply. Review the concrete
build and codec configuration before distributing binary installers.

Game runtime assets are distributed separately by Skywork AI, with their own
license/attribution documents. The code repository does not contain an Unreal
Engine installation or proprietary GPU driver libraries. Base-model and codec
weight licenses accompany their upstream distributions.
