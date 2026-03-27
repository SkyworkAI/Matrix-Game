---
license: apache-2.0
language:
- en
base_model:
- Wan-AI/Wan2.2-TI2V-5B
pipeline_tag: image-text-to-video
library_name: diffusers
---
# Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory
<div style="display: flex; justify-content: center; gap: 10px;">
  <a href="https://github.com/SkyworkAI/Matrix-Game">
    <img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/assets/pdf/report.pdf">
    <img src="https://img.shields.io/badge/Technical Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="report">
  </a>
  <a href="https://matrix-game-v3.github.io/">
    <img src="https://img.shields.io/badge/Project%20Page-grey?style=flat&logo=huggingface&color=FFA500" alt="Project Page">
  </a>

  
</div>

## 📝 Overview
**Matrix-Game-3.0** is an open-sourced, memory-augmented interactive world model designed for 720p real-time long-form video generation.

## Framework Overview
Our framework unifies three stages into an end-to-end pipeline: 
- Data Engine — an industrial-scale infinite data engine integrating Unreal Engine synthetic scenes, large-scale automated AAA game collection,and real-world video augmentation to produce high-quality Video-Pose-Action-Prompt quadruplets at scale; 
- Model Training — a memory-augmented Diffusion Transformer (DiT) with an error buffer that learns action-conditioned generation with memory-enhanced long-horizon consistency; 
- Inference Deployment — few-step sampling, INT8 quantization, and model distillation achieving 720p@40FPS real-time generation with a 5B model.

![Model Overview](./framework.png)

## ✨ Key Features
- 🚀 **Feature 1**: **Upgraded Data Engine**: Combines Unreal Engine-based synthetic data, large-scale automated AAA game data, and real-world video augmentation to generate high-quality Video–Pose–Action–Prompt data. 
- 🖱️ **Feature 2**: **Long-horizon Memory & Consistency**: Uses prediction residuals and frame re-injection for self-correction, while camera-aware memory ensures long-term spatiotemporal consistency. 
- 🎬 **Feature 3**: **Real-Time Interactivity & Open Access**: It employs a multi-segment autoregressive distillation strategy based on Distribution Matching Distillation (DMD), combined with model quantization and VAE decoder distillation to support [40fps] real-time generation at 720p resolution with a 5B model, while maintaining stable memory consistency over minute-long sequence.
- 👍 **Feature 3**: **Scale Up 28B-MoE Model**: Scaling up to a 2×14B model further improves generation quality, dynamics, and generalization.

## 🔥 Latest Updates

* [2026-03] 🎉 Initial release of Matrix-Game-3.0 Model

## 🚀 Quick Start
### Installation
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

### Model Download
```
pip install "huggingface_hub[cli]"
huggingface-cli download Matrix-Game-3.0 --local-dir Matrix-Game-3.0
```
### Inference
Before running inference, you need to prepare:
- Input image
- Text prompt

**Single GPU** (`python`, no distributed flags):
``` sh
python generate.py --size 704*1280 --ckpt_dir Matrix-Game-3.0 --fa_version 3 --use_int8 --num_iterations 12 --num_inference_steps 3 --image demo_images/000/image.png --prompt "a vintage gas station with a classic car parked under a canopy, set against a desert landscape." --save_name test --seed 42 --compile_vae --lightvae_pruning_rate 0.5 --vae_type mg_lightvae --output_dir ./output
# "num_iterations" refers to the number of iterations you want to generate. The total number of frames generated is given by:57 + (num_iterations - 1) * 40
```

**Multi-GPU** (`torchrun` + FSDP):
``` sh
torchrun --nproc_per_node=$NUM_GPUS generate.py --size 704*1280 --dit_fsdp --t5_fsdp --ckpt_dir Matrix-Game-3.0 --fa_version 3 --use_int8 --num_iterations 12 --num_inference_steps 3 --image demo_images/000/image.png --prompt "a vintage gas station with a classic car parked under a canopy, set against a desert landscape." --save_name test --seed 42 --compile_vae --lightvae_pruning_rate 0.5 --vae_type mg_lightvae --output_dir ./output
```
Tips:
If you want to use the base model, you can use "--use_base_model --num_inference_steps 50". Otherwise if you want to generating the interactive videos with your own input actions, you can use "--interactive".
With multiple GPUs, you can pass `--use_async_vae --async_vae_warmup_iters 1` to speed up inference.

### WorldCache (Inference Speedup)

WorldCache ([github.com/umair1221/WorldCache](https://github.com/umair1221/WorldCache)) accelerates inference by caching DiT block outputs across denoising timesteps and skipping recomputation when the hidden-state drift is below a threshold. This is a training-free optimization.

Enable with `--worldcache`:
``` sh
python generate.py --size 704*1280 --ckpt_dir Matrix-Game-3.0 --fa_version 3 --use_int8 \
  --num_iterations 12 --num_inference_steps 3 \
  --image demo_images/000/image.png --prompt "..." --save_name test \
  --compile_vae --lightvae_pruning_rate 0.5 --vae_type mg_lightvae --output_dir ./output \
  --worldcache --worldcache_thresh 0.40 --worldcache_warmup 1
```

| Argument | Default | Description |
|---|---|---|
| `--worldcache` | off | Enable WorldCache block-level feature caching |
| `--worldcache_thresh` | `0.40` | Drift threshold for skipping a block. Lower = more accurate, fewer skips; higher = more skips, faster |
| `--worldcache_warmup` | `1` | Number of denoising steps before caching starts per clip |

Note: WorldCache effectiveness scales with `--num_inference_steps`. With the default distilled model (`--num_inference_steps 3`) there are few steps to cache across, so gains are modest. It is more effective with the base model (`--use_base_model --num_inference_steps 50`).

## ⭐ Acknowledgements
- [WorldCache](https://github.com/umair1221/WorldCache) for the training-free denoising-step block caching technique
- [Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their excellent work
- [GameFactory](https://github.com/KwaiVGI/GameFactory) for their idea of action control module
- [LightX2V](https://github.com/ModelTC/lightx2v) for their excellent quantization framework
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) for their strong base model
- [lingbot-world](https://github.com/Robbyant/lingbot-world) for their context parallel framework 

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
