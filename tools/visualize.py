from operator import index
import cv2
import numpy as np
import os
import subprocess
from diffusers.utils import export_to_video

def parse_config(config):
    """
    根据配置生成按键数据和鼠标数据
    - config: list_actions[i] 的配置
    - 返回: key_data 和 mouse_data
    """
    key_data = {}
    mouse_data = {}

    # 解析 Space 按键的帧范围
    space_frames = set()
    key, mouse = config

    for i in range(len(mouse)):

        if  len(key[i])==7:
            w, s, a, d, space, attack, _ = key[i]
        else:
            w, s, a, d, space, attack = key[i]
            
        mouse_y, mouse_x = mouse[i]
        mouse_y = -1 * mouse_y
        
        # 按键状态
        key_data[i] = {
            "W": bool(w),
            "A": bool(a),
            "S": bool(s),
            "D": bool(d),
            "Space": bool(space),
            "Attack": bool(attack), 
        }
        # 鼠标位置
        if i == 0:
            mouse_data[i] = (320, 176)  # 默认初始位置
        else:
            global_scale_factor = 0.2
            mouse_scale_x = 15 * global_scale_factor
            mouse_scale_y = 15 * 4 * global_scale_factor
            mouse_data[i] = (
                mouse_data[i-1][0] + mouse_x * mouse_scale_x,  # x 坐标累计
                mouse_data[i-1][1] + mouse_y * mouse_scale_y,  # y 坐标累计
            )

    return key_data, mouse_data


# 绘制圆角矩形
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

# 在帧上绘制按键
def draw_keys_on_frame(frame, keys, key_size=(80, 50), spacing=20, bottom_margin=30):
    h, w, _ = frame.shape
    horison_shift = 90
    vertical_shift = -20
    horizon_shift_all = 50
    key_positions = {
        "W": (w // 2 - key_size[0] // 2 - horison_shift - horizon_shift_all + spacing* 2, h - bottom_margin - key_size[1] * 2 + vertical_shift - 20),
        "A": (w // 2 - key_size[0] * 2 + 5 - horison_shift - horizon_shift_all+ spacing* 2, h - bottom_margin - key_size[1] + vertical_shift),
        "S": (w // 2 - key_size[0] // 2 - horison_shift - horizon_shift_all+ spacing* 2, h - bottom_margin - key_size[1] + vertical_shift),
        "D": (w // 2 + key_size[0] - 5 - horison_shift - horizon_shift_all+ spacing* 2, h - bottom_margin - key_size[1] + vertical_shift),
        "Space": (w // 2 + key_size[0] * 2 + spacing * 4 - horison_shift - horizon_shift_all , h - bottom_margin - key_size[1] + vertical_shift),
        "Attack": (w // 2 + key_size[0] * 3 + spacing * 9 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
    }

    for key, (x, y) in key_positions.items():
        is_pressed = keys.get(key, False)
        top_left = (x, y)
        if key in ["Space", "Attack"]:
            bottom_right = (x + key_size[0]+40, y + key_size[1])
        else:
            bottom_right = (x + key_size[0], y + key_size[1])

        color = (0, 255, 0) if is_pressed else (200, 200, 200)
        alpha = 0.8 if is_pressed else 0.5

        draw_rounded_rectangle(frame, top_left, bottom_right, color, radius=10, alpha=alpha)

        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        if key in ["Space", "Attack"]:
            text_x = x + (key_size[0]+40 - text_size[0]) // 2
        else:
            text_x = x + (key_size[0] - text_size[0]) // 2
        text_y = y + (key_size[1] + text_size[1]) // 2
        cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# 在帧上叠加鼠标图案
def overlay_icon(frame, icon, position, scale=1.0, rotation=0):
    x, y = position
    h, w, _ = icon.shape

    # 缩放图标
    scaled_width = int(w * scale)
    scaled_height = int(h * scale)
    icon_resized = cv2.resize(icon, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    # 旋转图标
    center = (scaled_width // 2, scaled_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    icon_rotated = cv2.warpAffine(icon_resized, rotation_matrix, (scaled_width, scaled_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    h, w, _ = icon_rotated.shape
    frame_h, frame_w, _ = frame.shape

    # 计算绘制区域
    top_left_x = max(0, int(x - w // 2))
    top_left_y = max(0, int(y - h // 2))
    bottom_right_x = min(frame_w, int(x + w // 2))
    bottom_right_y = min(frame_h, int(y + h // 2))

    icon_x_start = max(0, int(-x + w // 2))
    icon_y_start = max(0, int(-y + h // 2))
    icon_x_end = icon_x_start + (bottom_right_x - top_left_x)
    icon_y_end = icon_y_start + (bottom_right_y - top_left_y)

    # 提取图标区域
    icon_region = icon_rotated[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
    alpha = icon_region[:, :, 3] / 255.0
    icon_rgb = icon_region[:, :, :3]

    # 提取帧对应区域
    frame_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # 叠加图标
    print(frame_region.shape, icon_rgb.shape, alpha.shape)
    # import ipdb; ipdb.set_trace()
    for c in range(3):
        frame_region[:, :, c] = (1 - alpha) * frame_region[:, :, c] + alpha * icon_rgb[:, :, c]

    # 替换帧对应区域
    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame_region


# 处理视频
def process_video(input_video, output_video, config, mouse_icon_path, mouse_scale=2.0, mouse_rotation=0,fps=16):
    key_data, mouse_data = parse_config(config)
    fps = fps
    frame_width = input_video[0].shape[1]
    frame_height = input_video[0].shape[0]
    frame_count = len(input_video)

    mouse_icon = cv2.imread(mouse_icon_path, cv2.IMREAD_UNCHANGED) 
    out_video = []
    frame_idx = 0
    for frame in input_video:
        keys = key_data.get(frame_idx, {"W": False, "A": False, "S": False, "D": False, "Space": False, "Attack": False})
        raw_mouse_pos = mouse_data.get(frame_idx, (frame_width // 2 // 2, frame_height // 2 // 2))  # fallback 也用小分辨率中心
        mouse_position = (int(raw_mouse_pos[0] * 2), int(raw_mouse_pos[1] * 2))
        draw_keys_on_frame(frame, keys, key_size=(75, 75), spacing=10, bottom_margin=20)
        overlay_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
        out_video.append(frame / 255)
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{frame_count}", end="\r")
    export_to_video(out_video, output_video, fps=fps)
    print("\nProcessing complete!")

# 处理视频
def save_video(input_video, output_video, fps=16):
    fps = fps
    frame_width = input_video[0].shape[1]
    frame_height = input_video[0].shape[0]
    frame_count = len(input_video)
    out_video = []
    frame_idx = 0
    for frame in input_video:
        out_video.append(frame / 255)
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{frame_count}", end="\r")
    export_to_video(out_video, output_video, fps=fps)
    print("\nProcessing complete!")