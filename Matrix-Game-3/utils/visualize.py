import cv2
import numpy as np
from diffusers.utils import export_to_video

def parse_config(config, default_frame_res=(704, 1280)):
    key_data = {}
    mouse_data = {}

    space_frames = set()
    key, mouse = config
    if key.shape[-1] == 4:
        key = np.concatenate([key, np.zeros((key.shape[0], 2))], axis=-1)


    default_frame_height, default_frame_width = default_frame_res

    for i in range(len(mouse)):
        
        w, s, a, d, space, attack  = key[i]
        mouse_y, mouse_x = mouse[i]
        mouse_y = -1 * mouse_y

        key_data[i] = {
            "W": bool(w),
            "A": bool(a),
            "S": bool(s),
            "D": bool(d),
            "Space": bool(space),
            "Attack": bool(attack),
        }

        if i == 0:
            mouse_data[i] = (default_frame_width//2, default_frame_height//2)
        else:
            global_scale_factor = 0.5
            mouse_scale_x = 15 * global_scale_factor
            mouse_scale_y = 15 * 4 * global_scale_factor

            new_x = mouse_data[i-1][0] + mouse_x * mouse_scale_x
            new_y = mouse_data[i-1][1] + mouse_y * mouse_scale_y
            
            new_x = max(0, min(default_frame_width, new_x))
            new_y = max(0, min(default_frame_height, new_y))
            mouse_data[i] = (new_x, new_y)

    return key_data, mouse_data


def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, alpha=0.5):
    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def draw_keys_on_frame(frame, keys, key_size=(80, 50), spacing=20, bottom_margin=30):
    h, w, _ = frame.shape
    horison_shift = 90
    vertical_shift = -20
    horizon_shift_all = 50

    key_positions = {
        "W": (w // 2 - key_size[0] // 2 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] * 2 + vertical_shift - 20),
        "A": (w // 2 - key_size[0] * 2 + 5 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
        "S": (w // 2 - key_size[0] // 2 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
        "D": (w // 2 + key_size[0] - 5 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
        "Space": (w // 2 + key_size[0] * 2 + spacing * 2 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
        "Attack": (w // 2 + key_size[0] * 3 + spacing * 7 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
    }

    for key, (x, y) in key_positions.items():
        is_pressed = keys.get(key, False)
        top_left = (x, y)
        if key in ["Space", "Attack", "Use"]:
            bottom_right = (x + key_size[0]+40, y + key_size[1])
        else:
            bottom_right = (x + key_size[0], y + key_size[1])

        color = (0, 255, 0) if is_pressed else (200, 200, 200)
        alpha = 0.8 if is_pressed else 0.5

        draw_rounded_rectangle(frame, top_left, bottom_right, color, radius=10, alpha=alpha)

        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        if key in ["Space", "Attack", "Use"]:
            text_x = x + (key_size[0]+40 - text_size[0]) // 2
        else:
            text_x = x + (key_size[0] - text_size[0]) // 2
        text_y = y + (key_size[1] + text_size[1]) // 2
        cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def overlay_icon(frame, icon, position, scale=1.0, rotation=0):
    x, y = position
    h, w, _ = icon.shape

    scaled_width = int(w * scale)
    scaled_height = int(h * scale)
    icon_resized = cv2.resize(icon, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    center = (scaled_width // 2, scaled_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    icon_rotated = cv2.warpAffine(icon_resized, rotation_matrix, (scaled_width, scaled_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    h, w, _ = icon_rotated.shape
    frame_h, frame_w, _ = frame.shape

    top_left_x = max(0, int(x - w // 2))
    top_left_y = max(0, int(y - h // 2))
    bottom_right_x = min(frame_w, int(x + w // 2))
    bottom_right_y = min(frame_h, int(y + h // 2))

    icon_x_start = max(0, int(-x + w // 2))
    icon_y_start = max(0, int(-y + h // 2))
    icon_x_end = icon_x_start + (bottom_right_x - top_left_x)
    icon_y_end = icon_y_start + (bottom_right_y - top_left_y)

    icon_region = icon_rotated[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
    alpha = icon_region[:, :, 3] / 255.0
    icon_rgb = icon_region[:, :, :3]

    frame_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    min_h = min(frame_region.shape[0], icon_rgb.shape[0])
    min_w = min(frame_region.shape[1], icon_rgb.shape[1])
    if min_h == 0 or min_w == 0:
        return
    frame_region = frame_region[:min_h, :min_w]
    icon_rgb = icon_rgb[:min_h, :min_w]
    alpha = alpha[:min_h, :min_w]

    for c in range(3):
        frame_region[:, :, c] = (1 - alpha) * frame_region[:, :, c] + alpha * icon_rgb[:, :, c]

    frame[top_left_y:top_left_y+min_h, top_left_x:top_left_x+min_w] = frame_region


def process_video(input_video, output_video, config, mouse_icon_path, mouse_scale=1.0, mouse_rotation=0, default_frame_res=(704, 1280), fps=24):
    key_data, mouse_data = parse_config(config, default_frame_res)

    frame_width = input_video[0].shape[1]
    frame_height = input_video[0].shape[0]
    frame_count = len(input_video)

    mouse_icon = cv2.imread(mouse_icon_path, cv2.IMREAD_UNCHANGED)

    out_video = []
    frame_idx = 0
    for frame in input_video:

        keys = key_data.get(frame_idx, {"W": False, "A": False, "S": False, "D": False, "Sp": False, "Sh": False, "Ct": False})
        mouse_position = mouse_data.get(frame_idx, (frame_width // 2, frame_height // 2))

        draw_keys_on_frame(frame, keys, key_size=(50, 50), spacing=10, bottom_margin=20)
        overlay_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
        out_video.append(frame / 255)
        frame_idx += 1
        # print(f"Processing frame {frame_idx}/{frame_count}", end="\r")
    export_to_video(out_video, output_video, fps=fps)
    # print("\nProcessing complete!")
