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
        """Main generation loop - simplified version"""
        try:
            print(f"Starting generation with image: {image_path}")

            # Load and preprocess image
            from diffusers.utils import load_image
            image = load_image(image_path)
            image = self.pipeline._resizecrop(image, 352, 640)
            image = self.pipeline.frame_process(image)[None, :, None, :, :].to(
                dtype=self.pipeline.weight_dtype, device=self.pipeline.device)

            # For demo purposes, create a simple video loop
            # In real implementation, this would call the actual generation pipeline
            frame_count = 0
            while self.is_generating:
                # Simulate frame generation
                # In real implementation, this would be the actual generated frame
                dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # Add some visual effects to show it's working
                cv2.putText(dummy_frame, f"Frame: {frame_count}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(dummy_frame, f"Mouse: {self.current_mouse_action}", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(dummy_frame, f"Keyboard: {self.current_keyboard_action}", (50, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Put frame in queue
                if not self.frame_queue.full():
                    self.frame_queue.put(dummy_frame)

                frame_count += 1
                time.sleep(0.033)  # ~30 FPS

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
