<div align="center">
<h1 align="center">Matrix-Game 3.0</h1>
<h3 align="center">Real-Time and Streaming Interactive World Model with Long-Horizon Memory</h3>
</div>

<font size=7><div align='center' >  [[🤗 HuggingFace](https://huggingface.co/Skywork/Matrix-Game-3.0)] [[📖 Technical Report](assets/pdf/report.pdf)] [[🚀 Project Website](https://matrix-game-v3.github.io/)] </div></font>


https://github.com/user-attachments/assets/5b95bb21-bc77-4bb5-bc2b-7b12de2d3f21

## 📝 Overview
**Matrix-Game-3.0** is an open-sourced, memory-augmented interactive world model designed for 720p real-time long-form video generation.
- **Upgraded Data Engine**: Combines Unreal Engine-based synthetic data, large-scale automated AAA game data, and real-world video augmentation to generate high-quality Video–Pose–Action–Prompt data. 
- **Long-horizon Memory & Consistency**: Uses prediction residuals and frame re-injection for self-correction, while camera-aware memory ensures long-term spatiotemporal consistency. 
- **Real-Time Interactivity & Open Access**: It employs a multi-segment autoregressive distillation strategy based on Distribution Matching Distillation (DMD), combined with model quantization and VAE decoder distillation to support [40fps] real-time generation at 720p resolution with a 5B model, while maintaining stable memory consistency over minute-long sequence.
- **Scale Up 28B-MoE Model**: Scaling up to a 2×14B model further improves generation quality, dynamics, and generalization. 

## 🤗 Matrix-Game-3.0 Model
We provide two pretrained 5B model weights, including the base model and the distilled model, for first-person generation in unreal scenes. These resources are available on our HuggingFace page. 

In addition, the model trained on a combination of unreal and real-world data, as well as the 28B large model, will be released soon! 🚀🚀

## Requirements
It supports one gpu or multi-gpu inference. We tested this repo on the following setup:
* NVIDIA A/H series GPUs are tested.
* AMD Instinct MI300X GPUs are also supported (ROCm 7.x + AITER).
* Linux operating system.
* 64 GB RAM.

## ⚙️ Quick Start
### Installation (NVIDIA)
Create a conda environment and install dependencies:
```
conda create -n matrix-game-3.0 python=3.12 -y
conda activate matrix-game-3.0
# install FlashAttention
# Our project also depends on [FlashAttention](https://github.com/Dao-AILab/flash-attention)
git clone https://github.com/SkyworkAI/Matrix-Game-3.0.git
cd Matrix-Game-3.0
pip install -r requirements.txt
```

### Installation (AMD ROCm)
For AMD GPUs (e.g. MI300X) with ROCm 7.x:
```bash
conda create -n matrix-game-3.0 python=3.10 -y
conda activate matrix-game-3.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
git clone https://github.com/SkyworkAI/Matrix-Game-3.0.git
cd Matrix-Game-3.0
grep -v -E "^torch|flash.attn" requirements.txt | pip install -r /dev/stdin
pip install opencv-python-headless
# Install AITER (AMD flash attention CK backend)
pip install aiter  # or: git clone https://github.com/ROCm/aiter && cd aiter && git submodule update --init 3rdparty/composable_kernel && pip install .
```

### Model Download
```
pip install "huggingface_hub[cli]"
huggingface-cli download Matrix-Game-3.0 --local-dir Matrix-Game-3.0
```
### Inference
Before running inference, you need to prepare:
- Input image
- Text prompt

After downloading pretrained models, you can use the following command to generate an interactive video with random actions:
``` sh
torchrun --nproc_per_node=$NUM_GPUS generate.py --size 704*1280 --dit_fsdp --t5_fsdp --ckpt_dir Matrix-Game-3.0 --fa_version 3 --use_int8 --num_iterations 12 --num_inference_steps 3 --image demo_images/001/image.png --prompt "A colorful, animated cityscape with a gas station and various buildings." --save_name test --seed 42 --compile_vae --lightvae_pruning_rate 0.5 --vae_type mg_lightvae --output_dir ./output
# "num_iterations" refers to the number of iterations you want to generate. The total number of frames generated is given by:57 + (num_iterations - 1) * 40 
```
Tips: 
If you want to use the base model, you can use `--use_base_model --num_inference_steps 50`. To run with your own input actions, use `--interactive`.
For LightVAE, use `--vae_type mg_lightvae` with `--lightvae_pruning_rate 0.5`, or `--vae_type mg_lightvae_v2` with `--lightvae_pruning_rate 0.75`. `mg_lightvae_v2` is faster than `mg_lightvae` while keeping quality close to the latter.
With multiple GPUs, you can pass `--use_async_vae --async_vae_warmup_iters 1` to speed up inference (see [`test.sh`](test.sh)).

## ⭐ Acknowledgements
- [Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their excellent work
- [GameFactory](https://github.com/KwaiVGI/GameFactory) for their idea of action control module
- [LightX2V](https://github.com/ModelTC/lightx2v) for their excellent quantization framework
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) for their strong base model
- [lingbot-world](https://github.com/Robbyant/lingbot-world) for their context parallel framework 
## 📜 License
This project is licensed under the Apache License, Version 2.0 — see [LICENSE.txt](LICENSE.txt).

## 📖 Citation
If you find this work useful for your research, please kindly cite our paper:

```
  @misc{2026matrix,
    title={Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory},
    author={{Skywork AI Matrix-Game Team}},
    year={2026},
    howpublished={Technical report},
    url={https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/assets/pdf/report.pdf}
  }
```
