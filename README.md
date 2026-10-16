<h1 align="center">Matrix-Game 4.0</h1>
<h3 align="center">Skywork AI</h3>
<p align="center">Streaming Neural Rendering for Playable Games</p>

<p align="center">
  <a href="https://matrix-game-v4-0.github.io/page/">Project Website</a> |
  <a href="Matrix-Game-4/docs/Matrix_Game_4_0.pdf">Technical Report</a> |
  <a href="https://huggingface.co/Skywork/Matrix-Game-4.0">Models &amp; Game Assets</a> |
  <a href="Matrix-Game-4">Code &amp; Quick Start</a> |
  <a href="#previous-versions">Previous Versions</a>
</p>

**Matrix-Game** is Skywork AI's series of open-source interactive world models.
**Matrix-Game 4.0** combines executable game engines with a causal video model:
the engine handles gameplay, physics and player input, while the model transforms
low-poly RGB frames into detailed visuals through streaming neural rendering.

https://github.com/user-attachments/assets/4ded08da-1309-43c9-9662-6e2f2b3fbb41

## News

- **October 16, 2026:** Optimized inference for [RTX 5080](Matrix-Game-4/README.md#rtx-5080) and [RTX 5070 Ti](Matrix-Game-4/README.md#rtx-5070-ti) is now available. With fixed prompts and SageAttention, model throughput reaches up to **67 FPS on RTX 5080** and **58 FPS on RTX 5070 Ti** at 864 x 480 (steady state, excluding video I/O).
- **October 16, 2026:** The Matrix-Game 4.0 [technical report](Matrix-Game-4/docs/Matrix_Game_4_0.pdf) is now available.
- **October 15, 2026:** [RTX 5090 optimized inference](Matrix-Game-4/README.md#rtx-5090) is now available, with FP8 linear layers, fused kernels, CUDA graphs and an optional SageAttention backend.
- **October 15, 2026:** Released **Matrix-Game 4.0**. [Inference, browser demo and game source code](Matrix-Game-4) are now available, with model weights and game assets for Medieval Castle, Ancient Egypt and City Racing on [Hugging Face](https://huggingface.co/Skywork/Matrix-Game-4.0).
- **March 27, 2026:** Released [Matrix-Game 3.0](Matrix-Game-3), a real-time streaming interactive world model with long-horizon memory.
- **August 12, 2025:** Released [Matrix-Game 2.0](Matrix-Game-2), an interactive world foundation model for real-time long video generation.
- **May 12, 2025:** Released [Matrix-Game 1.0](Matrix-Game-1), the first open-source release of the Matrix-Game series.

## Matrix-Game 4.0

- **Engine-grounded gameplay:** game logic and physical state remain in an executable engine, separate from neural rendering.
- **Streaming generation:** a causal, few-step video model renders successive chunks conditioned on low-poly RGB and text prompts.
- **Three scene models:** Medieval Castle and Ancient Egypt for third-person shooting, and City Racing for driving.

Start with the [v4 README](Matrix-Game-4/README.md) for installation, resource
downloads, offline rendering and the online browser demo. The release includes
the three native games and single-GPU inference paths for A100, RTX 5090,
RTX 5080 and RTX 5070 Ti.

| Scene | Model Weights |
| --- | --- |
| Medieval Castle | [fps-castle](https://huggingface.co/Skywork/Matrix-Game-4.0/tree/main/fps-castle) |
| Ancient Egypt | [fps-egypt](https://huggingface.co/Skywork/Matrix-Game-4.0/tree/main/fps-egypt) |
| City Racing | [driving_game](https://huggingface.co/Skywork/Matrix-Game-4.0/tree/main/driving_game) |

Game assets, sink images and inference configuration files are included in the
[Hugging Face repository](https://huggingface.co/Skywork/Matrix-Game-4.0).
See the [project website](https://matrix-game-v4-0.github.io/page/) for demos and
the broader project overview.

## Previous Versions

Earlier implementations remain available in their original directories:

- [Matrix-Game 3.0](Matrix-Game-3): streaming interaction with long-horizon memory.
- [Matrix-Game 2.0](Matrix-Game-2): real-time interactive long video generation.
- [Matrix-Game 1.0](Matrix-Game-1): the first Matrix-Game world model.

## License

Matrix-Game 4.0 is released under **Apache-2.0**; see the
[v4 license](Matrix-Game-4/LICENSE).
Third-party components and assets retain their respective licenses.
Earlier versions retain their existing license terms; see the repository's
[MIT license](LICENSE) and the notices within each version directory.
