import cv2
import numpy as np
import torch
import threading
import time
import queue
from PIL import Image
import argparse
import os
from inference_streaming import InteractiveGameInference

class RealTimeMatrixGameOpenCV:
    def __init__(self):
        self.args = self.setup_args()
        self.pipeline = InteractiveGameInference(self.args)
        self.current_frame = None
        self.is_generating = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.action_queue = queue.Queue()
        self.mode = 'universal'

        # OpenCV window setup
        self.window_name = "Matrix-Game 2.0 Real-Time"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        # Control state
        self.current_mouse_action = 'u'  # no movement
        self.current_keyboard_action = 'q'  # no movement

    def setup_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml")
        parser.add_argument("--checkpoint_path", type=str, default="Matrix-Game-2.0/universal/model.safetensors")
        parser.add_argument("--output_folder", type=str, default="outputs/")
        parser.add_argument("--max_num_output_frames", type=int, default=360)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
        return parser.parse_args([])

    def create_control_overlay(self, frame):
        """Create control overlay on the video frame"""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Control instructions
        instructions = [
            "MATRIX-GAME 2.0 REAL-TIME CONTROLS",
            "",
            "MOVEMENT (WASD):",
            "W - Forward    S - Backward",
            "A - Left       D - Right",
            "Q - Stop",
            "",
            "CAMERA (IJKL):",
            "I - Up         K - Down",
            "J - Left       L - Right",
            "U - Center",
            "",
            "PRESS 'ESC' TO EXIT",
            f"STATUS: {'GENERATING' if self.is_generating else 'READY'}"
        ]

        # Draw semi-transparent background
        cv2.rectangle(overlay, (10, 10), (400, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        y_offset = 30
        for i, line in enumerate(instructions):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            font_scale = 0.6 if i == 0 else 0.4
            thickness = 2 if i == 0 else 1

            cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, color, thickness)
            y_offset += 20

        # Draw current action status
        cv2.putText(frame, f"Mouse: {self.current_mouse_action.upper()}",
                   (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"Keyboard: {self.current_keyboard_action.upper()}",
                   (20, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame

    def handle_keyboard_input(self, key):
        """Handle keyboard input for controls"""
        key_map = {
            ord('w'): ('keyboard', 'w'),
            ord('s'): ('keyboard', 's'),
            ord('a'): ('keyboard', 'a'),
            ord('d'): ('keyboard', 'd'),
            ord('q'): ('keyboard', 'q'),
            ord('i'): ('mouse', 'i'),
            ord('j'): ('mouse', 'j'),
            ord('k'): ('mouse', 'k'),
            ord('l'): ('mouse', 'l'),
            ord('u'): ('mouse', 'u'),
        }

        if key in key_map:
            action_type, action = key_map[key]
            if action_type == 'mouse':
                self.current_mouse_action = action
            else:
                self.current_keyboard_action = action

            # Send action to generation thread
            if self.is_generating:
                self.action_queue.put({
                    'mouse': self.current_mouse_action,
                    'keyboard': self.current_keyboard_action
                })

            print(f"Action: {action_type} = {action}")
            return True
        return False

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
            from utils.conditions import Bench_actions_universal

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
            num_frames = (self.pipeline.args.max_num_output_frames - 1) * 4 + 1
            cond_data = Bench_actions_universal(num_frames)

            conditional_dict = {
                "cond_concat": cond_concat.to(device=self.pipeline.device, dtype=self.pipeline.weight_dtype),
                "visual_context": visual_context.to(device=self.pipeline.device, dtype=self.pipeline.weight_dtype),
                "mouse_cond": cond_data['mouse_condition'].unsqueeze(0).to(device=self.pipeline.device, dtype=self.pipeline.weight_dtype),
                "keyboard_cond": cond_data['keyboard_condition'].unsqueeze(0).to(device=self.pipeline.device, dtype=self.pipeline.weight_dtype)
            }

            # Action generator
            def action_generator():
                CAM_VALUE = 0.1
                CAMERA_VALUE_MAP = {
                    "i": [CAM_VALUE, 0], "k": [-CAM_VALUE, 0], "j": [0, -CAM_VALUE],
                    "l": [0, CAM_VALUE], "u": [0, 0]
                }
                KEYBOARD_IDX = {
                    "w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1],
                    "q": [0, 0, 0, 0]
                }
                while self.is_generating:
                    try:
                        # Non-blocking get from queue
                        actions = self.action_queue.get_nowait()
                        self.current_mouse_action = actions.get('mouse', self.current_mouse_action)
                        self.current_keyboard_action = actions.get('keyboard', self.current_keyboard_action)
                    except queue.Empty:
                        pass # Keep previous action

                    yield {
                        "mouse": torch.tensor(CAMERA_VALUE_MAP[self.current_mouse_action]).cuda(),
                        "keyboard": torch.tensor(KEYBOARD_IDX[self.current_keyboard_action]).cuda()
                    }
                    time.sleep(0.1) # Prevent busy-waiting

            # Frame callback
            def frame_callback(video_chunk):
                from einops import rearrange
                video = rearrange(video_chunk, "B T C H W -> T B C H W")
                for frame_tensor in video:
                    frame = ((frame_tensor[0].float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)

            # Start generation
            with torch.no_grad():
                self.pipeline.pipeline.inference(
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    return_latents=False,
                    output_folder=self.pipeline.args.output_folder,
                    name="realtime_opencv",
                    mode=self.mode,
                    action_generator=action_generator(),
                    frame_callback=frame_callback
                )

        except Exception as e:
            print(f"Generation error: {e}")
        finally:
            self.is_generating = False

    def run(self, image_path):
        """Main run loop"""
        print("Starting Matrix-Game 2.0 Real-Time Interface")
        print("Controls:")
        print("  WASD - Movement")
        print("  IJKL - Camera")
        print("  ESC - Exit")

        # Start generation
        if not self.start_generation(image_path):
            print("Failed to start generation")
            return

        # Main display loop
        while True:
            # Get latest frame
            if not self.frame_queue.empty():
                self.current_frame = self.frame_queue.get()

            # Display frame with controls
            if self.current_frame is not None:
                display_frame = self.create_control_overlay(self.current_frame.copy())
                cv2.imshow(self.window_name, display_frame)
            else:
                # Show waiting screen
                waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting_frame, "Loading...", (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.imshow(self.window_name, waiting_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key != 255:  # Any other key
                self.handle_keyboard_input(key)

        # Cleanup
        self.is_generating = False
        cv2.destroyAllWindows()
        print("Exiting...")

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python realtime_opencv_interface.py <image_path>")
        print("Example: python realtime_opencv_interface.py demo_images/universal/0000.png")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Create and run the interface
    interface = RealTimeMatrixGameOpenCV()
    interface.run(image_path)

if __name__ == "__main__":
    main()
