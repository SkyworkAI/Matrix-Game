import os
import cv2
import base64
import threading
import time
import queue
from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import torch
from PIL import Image
import io

# Import Matrix-Game components
from inference_streaming import InteractiveGameInference
import argparse

app = Flask(__name__)

class RealTimeMatrixGame:
    def __init__(self):
        self.args = self.setup_args()
        self.pipeline = InteractiveGameInference(self.args)
        self.current_frame = None
        self.is_generating = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.action_queue = queue.Queue()
        self.mode = 'universal'

    def setup_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml")
        parser.add_argument("--checkpoint_path", type=str, default="Matrix-Game-2.0/universal/model.safetensors")
        parser.add_argument("--output_folder", type=str, default="outputs/")
        parser.add_argument("--max_num_output_frames", type=int, default=360)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
        return parser.parse_args([])

    def start_generation(self, image_path):
        """Start video generation in a separate thread"""
        if self.is_generating:
            return False

        self.is_generating = True
        thread = threading.Thread(target=self.generate_video_loop, args=(image_path,))
        thread.daemon = True
        thread.start()
        return True

    def generate_video_loop(self, image_path):
        """Main generation loop"""
        try:
            # Load and preprocess image
            from diffusers.utils import load_image
            image = load_image(image_path)
            image = self.pipeline._resizecrop(image, 352, 640)
            image = self.pipeline.frame_process(image)[None, :, None, :, :].to(
                dtype=self.pipeline.weight_dtype, device=self.pipeline.device)

            # Encode image
            padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.pipeline.args.max_num_output_frames - 1), 1, 1)
            img_cond = torch.concat([image, padding_video], dim=2)
            tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
            img_cond = self.pipeline.vae.encode(img_cond, device=self.pipeline.device, **tiler_kwargs).to(self.pipeline.device)

            # Setup conditional inputs
            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = self.pipeline.vae.clip.encode_video(image)

            # Generate initial noise
            sampled_noise = torch.randn([1, 16, self.pipeline.args.max_num_output_frames, 44, 80],
                                      device=self.pipeline.device, dtype=self.pipeline.weight_dtype)

            # Setup conditional dictionary
            from utils.conditions import Bench_actions_universal
            num_frames = (self.pipeline.args.max_num_output_frames - 1) * 4 + 1
            cond_data = Bench_actions_universal(num_frames)

            conditional_dict = {
                "cond_concat": cond_concat.to(device=self.pipeline.device, dtype=self.pipeline.weight_dtype),
                "visual_context": visual_context.to(device=self.pipeline.device, dtype=self.pipeline.weight_dtype),
                "mouse_cond": cond_data['mouse_condition'].unsqueeze(0).to(device=self.pipeline.device, dtype=self.pipeline.weight_dtype),
                "keyboard_cond": cond_data['keyboard_condition'].unsqueeze(0).to(device=self.pipeline.device, dtype=self.pipeline.weight_dtype)
            }

            # Start generation
            with torch.no_grad():
                videos = self.pipeline.pipeline.inference(
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    return_latents=False,
                    output_folder=self.pipeline.args.output_folder,
                    name="realtime",
                    mode=self.mode
                )

        except Exception as e:
            print(f"Generation error: {e}")
        finally:
            self.is_generating = False

# Global instance
matrix_game = RealTimeMatrixGame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_generation', methods=['POST'])
def start_generation():
    data = request.json
    image_path = data.get('image_path')

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 400

    success = matrix_game.start_generation(image_path)
    return jsonify({'success': success})

@app.route('/send_action', methods=['POST'])
def send_action():
    data = request.json
    action = data.get('action')
    matrix_game.action_queue.put(action)
    return jsonify({'success': True})

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            if not matrix_game.frame_queue.empty():
                frame = matrix_game.frame_queue.get()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
