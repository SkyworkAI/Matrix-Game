#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MatrixGame I2V Video Generation

This script generates videos from input images using the MatrixGame model with
specified keyboard and mouse conditions.
"""

import argparse
import os
import glob
from typing import Tuple, Dict, List, Optional
import torch
import numpy as np
from PIL import Image
import imageio
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from safetensors.torch import load_file as safe_load
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from matrixgame.sample.pipeline_matrixgame import MatrixGameVideoPipeline
from matrixgame.model_variants import get_dit
from matrixgame.model_variants.matrixgame_dit_src import MGVideoDiffusionTransformerI2V
from matrixgame.vae_variants import get_vae
from matrixgame.encoder_variants import get_text_enc
from matrixgame.sample.flow_matching_scheduler_matrixgame import FlowMatchDiscreteScheduler
from tools.visualize import process_video
from condtions import Bench_actions_76
from teacache_forward import teacache_forward
from condtions import Bench_actions_76


class VideoGenerator:
    """Main class for video generation using MatrixGame model."""
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the video generator with configuration parameters.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scheduler = FlowMatchDiscreteScheduler(
            shift=15.0,
            reverse=True,
            solver="euler"
        )
        self.video_length = args.video_length
        self.guidance_scale = args.guidance_scale
        
        # Initialize models
        self._init_models()
        
        # Teacache settings
        self._setup_teacache()
    
    def _init_models(self) -> None:
        """Initialize all required models (VAE, text encoder, transformer)."""
        # Initialize VAE
        vae_path = self.args.vae_path if self.args.vae_path else "/mnt/data_genie/models/HunyuanVideo-I2V/hunyuan-video-i2v-720p/vae/"
        self.vae = get_vae("matrixgame", vae_path, torch.float16)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.enable_tiling()
        
        # Initialize DIT (Transformer)
        dit = MGVideoDiffusionTransformerI2V.from_pretrained(self.args.pretrained)
        dit.requires_grad_(False)
        dit.eval()
        
        # Initialize text encoder
        model_path = self.args.model_path if self.args.model_path else "/mnt/data_genie/models/HunyuanVideo-I2V/"
        weight_dtype = torch.bfloat16 if self.args.bfloat16 else torch.float16
        self.text_enc = get_text_enc('matrixgame', model_path, weight_dtype=weight_dtype, i2v_type='refiner')
        
        # Move models to devices
        if self.args.use_cpu_offload:
            self.text_enc.to(weight_dtype).to("cpu")
            self.vae.to(weight_dtype).to("cpu")
            self.compiled_dit = torch.compile(dit, mode="reduce-overhead").to(weight_dtype).to(self.device)
            
            self.pipeline = MatrixGameVideoPipeline(
                vae=self.vae.vae,
                text_encoder=self.text_enc,
                transformer=self.compiled_dit,
                scheduler=self.scheduler,
            )
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline = MatrixGameVideoPipeline(
                vae=self.vae.vae,
                text_encoder=self.text_enc,
                transformer=dit,
                scheduler=self.scheduler,
            ).to(weight_dtype).to(self.device)
    
    def _setup_teacache(self) -> None:
        """Configure teacache for the transformer."""
        self.pipeline.transformer.__class__.enable_teacache = True
        self.pipeline.transformer.__class__.cnt = 0
        self.pipeline.transformer.__class__.num_steps = self.args.num_steps if hasattr(self.args, 'num_steps') else 50  # should be aligned with infer_steps
        self.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
        self.pipeline.transformer.__class__.rel_l1_thresh = self.args.rel_l1_thresh if hasattr(self.args, 'rel_l1_thresh') else 0.075
        self.pipeline.transformer.__class__.previous_modulated_input = None
        self.pipeline.transformer.__class__.previous_residual = None
        self.pipeline.transformer.__class__.forward = teacache_forward
    
    def _resize_and_crop_image(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize and crop image to target dimensions while maintaining aspect ratio.
        
        Args:
            image: Input PIL image
            target_size: Target (width, height) tuple
            
        Returns:
            Resized and cropped PIL image
        """
        w, h = image.size
        tw, th = target_size
        
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        
        return image.crop((left, top, right, bottom))
    
    def _load_images(self, root_dir: str) -> List[str]:
        """
        Load image paths from directory with specified extensions.
        
        Args:
            root_dir: Root directory to search for images
            
        Returns:
            List of image file paths
        """
        image_extensions = ('*.png', '*.jpg', '*.jpeg')
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
            
        return image_paths[:self.args.max_images] if hasattr(self.args, 'max_images') else image_paths
    
    def _process_condition(self, condition: Dict, image_path: str) -> None:
        """
        Process a single condition and generate video.
        
        Args:
            condition: Condition dictionary containing action and conditions
            image_path: Path to input image
        """
        try:
            # Prepare conditions
            keyboard_condition = torch.tensor(condition['keyboard_condition'], dtype=torch.float32).unsqueeze(0)
            mouse_condition = torch.tensor(condition['mouse_condition'], dtype=torch.float32).unsqueeze(0)
            
            # Move to device
            keyboard_condition = keyboard_condition.to(torch.bfloat16 if self.args.bfloat16 else torch.float16).to(self.device)
            mouse_condition = mouse_condition.to(torch.bfloat16 if self.args.bfloat16 else torch.float16).to(self.device)
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            new_width, new_height = self.args.resolution if hasattr(self.args, 'resolution') else (1280, 720)
            initial_image = self._resize_and_crop_image(image, (new_height, new_width))
            semantic_image = initial_image
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)
            initial_image = video_processor.preprocess(initial_image, height=new_height, width=new_width)
            
            if self.args.num_pre_frames > 0:
                past_frames = initial_image.repeat(self.args.num_pre_frames, 1, 1, 1)
                initial_image = torch.cat([initial_image, past_frames], dim=0)
            
            # Generate video
            with torch.no_grad():
                video = self.pipeline(
                    height=new_height,
                    width=new_width,
                    video_length=self.video_length,
                    mouse_condition=mouse_condition,
                    keyboard_condition=keyboard_condition,
                    initial_image=initial_image,
                    num_inference_steps=self.args.inference_steps if hasattr(self.args, 'inference_steps') else 50,
                    guidance_scale=self.guidance_scale,
                    embedded_guidance_scale=None,
                    data_type="video",
                    vae_ver='884-16c-hy',
                    enable_tiling=True,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                    i2v_type='refiner',
                    args=self.args,
                    semantic_images=semantic_image
                ).videos[0]
            
            # Save video
            img_tensors = rearrange(video.permute(1, 0, 2, 3) * 255, 't c h w -> t h w c').contiguous()
            img_tensors = img_tensors.cpu().numpy().astype(np.uint8)
            
            config = (
                keyboard_condition[0].float().cpu().numpy(),
                mouse_condition[0].float().cpu().numpy()
            )
            
            action_name = condition['action_name']
            output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{action_name}.mp4"
            output_path = os.path.join(self.args.output_path, output_filename)
            
            process_video(
                img_tensors,
                output_path,
                config,
                mouse_icon_path=self.args.mouse_icon_path if hasattr(self.args, 'mouse_icon_path') else '/mnt/datasets_genie/chunli.peng/model.pt/mouse.png',
                mouse_scale=self.args.mouse_scale if hasattr(self.args, 'mouse_scale') else 0.1,
                mouse_rotation=self.args.mouse_rotation if hasattr(self.args, 'mouse_rotation') else -20,
                fps=self.args.fps if hasattr(self.args, 'fps') else 16
            )
            
        except Exception as e:
            print(f"Error processing condition {condition['action_name']}: {str(e)}")
    
    def generate_videos(self) -> None:
        """Main method to generate videos for all conditions."""
        # Create output directory
        os.makedirs(self.args.output_path, exist_ok=True)
        
        # Load conditions
        conditions = Bench_actions_76()
        print(f"Found {len(conditions)} conditions to process")
        
        # Load sample images
        root_dir = self.args.image_path if hasattr(self.args, 'image_path') else '/mnt/datasets_genie/.ossutil_checkpoint/ossutil_yf.cp/ossutil_output/initial_image/'
        image_paths = self._load_images(root_dir)
        
        if not image_paths:
            print("No images found in the specified directory")
            return
        
        # Process each condition
        for idx, condition in enumerate(conditions):
            for image_path in image_paths:
                print(f"Processing condition {idx+1}/{len(conditions)} with image {os.path.basename(image_path)}")
                self._process_condition(condition, image_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MatrixGame I2V Video Generation")
    
    # Basic parameters
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image or image directory.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output video.")
    parser.add_argument("--video_length", type=int, default=65,
                        help="Number of frames in the generated video.")
    parser.add_argument("--guidance_scale", type=float, default=6,
                        help="Guidance scale for generation.")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path to the pretrained model directory (with .safetensors).") 
    parser.add_argument("--use-cpu-offload", 
                        action="store_true",
                        help="Use CPU offload for the model load.") 
    
    # Model paths
    parser.add_argument("--model_path", type=str, default="/mnt/data_genie/models/HunyuanVideo-I2V/",
                        help="Path to the text encoder model directory.")
    parser.add_argument("--vae_path", type=str, default="/mnt/data_genie/models/HunyuanVideo-I2V/hunyuan-video-i2v-720p/vae/",
                        help="Path to the VAE model directory.")
    
    # Inference parameters
    parser.add_argument("--inference_steps", type=int, default=50,
                        help="Number of inference steps.")
    parser.add_argument("--num_pre_frames", type=int, default=5,
                        help="Number of pre-generated frames.")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of steps for teacache.")
    parser.add_argument("--rel_l1_thresh", type=float, default=0.075,
                        help="Relative L1 threshold for teacache.")
    
    # Resolution and format
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720],
                        help="Resolution of the output video (width height).")
    parser.add_argument("--bfloat16", action="store_true",
                        help="Use bfloat16 precision instead of float16.")
    parser.add_argument("--max_images", type=int, default=3,
                        help="Maximum number of images to process from the input directory.")
    
    # Mouse icon parameters
    parser.add_argument("--mouse_icon_path", type=str, 
                        default='/mnt/datasets_genie/chunli.peng/model.pt/mouse.png',
                        help="Path to the mouse icon image.")
    parser.add_argument("--mouse_scale", type=float, default=0.1,
                        help="Scale of the mouse icon in the output video.")
    parser.add_argument("--mouse_rotation", type=float, default=-20,
                        help="Rotation of the mouse icon in the output video.")
    parser.add_argument("--fps", type=int, default=16,
                        help="Frames per second for the output video.")
    
    # GPU settings
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="Separated gpu ids (e.g. 0,1,2,3)")
    
    return parser.parse_args()


def main():
    """Main entry point for video generation."""
    args = parse_args()
    
    # Set GPU device if specified
    if args.gpu_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    generator = VideoGenerator(args)
    generator.generate_videos()


if __name__ == "__main__":
    main()