<h1 align="center">Matrix-Game 4.0</h1>
<h3 align="center">Skywork AI</h3>
<p align="center">Streaming Neural Rendering for Playable Games</p>

<p align="center">
  <a href="https://matrix-game-v4-0.github.io/page/">Project Website</a> |
  <a href="https://huggingface.co/Skywork/Matrix-Game-4.0">Models &amp; Game Assets</a> |
  <a href="#previous-versions">Previous Versions</a>
</p>

**Matrix-Game** is Skywork AI's series of open-source interactive world models.
**Matrix-Game 4.0** combines executable game engines with a causal video model:
the engine handles gameplay, physics and player input, while the model transforms
low-poly RGB frames into detailed visuals through streaming neural rendering.

https://github.com/user-attachments/assets/53a0cfb7-4193-42c8-9819-7c023389a3d7

## News

- **October 14, 2026:** Introducing **Matrix-Game 4.0**. Model weights and game assets for Medieval Castle, Ancient Egypt and City Racing are available on [Hugging Face](https://huggingface.co/Skywork/Matrix-Game-4.0). Inference and game source code will follow in this repository.
- **March 27, 2026:** Released [Matrix-Game 3.0](Matrix-Game-3), a real-time streaming interactive world model with long-horizon memory.
- **August 12, 2025:** Released [Matrix-Game 2.0](Matrix-Game-2), an interactive world foundation model for real-time long video generation.
- **May 12, 2025:** Released [Matrix-Game 1.0](Matrix-Game-1), the first open-source release of the Matrix-Game series.

## Matrix-Game 4.0

- **Engine-grounded gameplay:** game logic and physical state remain in an executable engine, separate from neural rendering.
- **Streaming generation:** a causal, few-step video model renders successive chunks conditioned on low-poly RGB and text prompts.
- **Three scene models:** Medieval Castle and Ancient Egypt for third-person shooting, and City Racing for driving.

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
[v4 license](https://huggingface.co/Skywork/Matrix-Game-4.0/blob/main/LICENSE).
Third-party components and assets retain their respective licenses.
Earlier versions retain their existing license terms; see the repository's
[MIT license](LICENSE) and the notices within each version directory.
