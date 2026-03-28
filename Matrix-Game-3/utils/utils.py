import numpy as np
import torch
import torch.nn.functional as torch_F
from einops import rearrange
from .cam_utils import compute_relative_poses, get_plucker_embeddings, _interpolate_camera_poses_handedness, get_extrinsics
from .conditions import Bench_actions_universal
from .transform import get_video_transform
WSAD_OFFSET = 12.35  # units per frame for single direction
DIAGONAL_OFFSET = 8.73  # units per frame for diagonal (12.35 / sqrt(2))
MOUSE_PITCH_SENSITIVITY = 15.0  # degrees per unit of mouse_x
MOUSE_YAW_SENSITIVITY = 15.0    # degrees per unit of mouse_y
MOUSE_THRESHOLD = 0.02  # Small mouse values below this are treated as noise (deadzone)

def compute_all_poses_from_actions(keyboard_conditions, mouse_conditions, first_pose=None, return_last_pose=False):
    """
    Compute all camera poses from a sequence of actions.
    First frame pose is all zeros.
    """
    T = len(keyboard_conditions)
    all_poses = np.zeros((T, 5), dtype=np.float32)
    if first_pose is not None:
        all_poses[0] = first_pose
    for i in range(T - 1):
        all_poses[i + 1] = compute_next_pose_from_action(
            all_poses[i],
            keyboard_conditions[i],
            mouse_conditions[i]
        )
    if return_last_pose:
        last_pose = compute_next_pose_from_action(
                all_poses[-1],
                keyboard_conditions[-1],
                mouse_conditions[-1]
            )
        return all_poses, last_pose
    return all_poses

def compute_next_pose_from_action(current_pose, keyboard_action, mouse_action):
    """
    Compute the next camera pose based on the current pose and action inputs.
    Uses average yaw for position transformation to ensure symmetric paths.
    Small mouse values below MOUSE_THRESHOLD are ignored (deadzone).
    """
    x, y, z, pitch, yaw = current_pose
    w, s, a, d = keyboard_action[:4]
    mouse_x, mouse_y = mouse_action[:2]
    
    # 1. Compute rotation changes from mouse (with deadzone for small values)
    delta_pitch = MOUSE_PITCH_SENSITIVITY * mouse_x if abs(mouse_x) >= MOUSE_THRESHOLD else 0.0
    delta_yaw = MOUSE_YAW_SENSITIVITY * mouse_y if abs(mouse_y) >= MOUSE_THRESHOLD else 0.0
    
    new_pitch = pitch + delta_pitch
    new_yaw = yaw + delta_yaw
    
    # Normalize yaw to [-180, 180]
    while new_yaw > 180:
        new_yaw -= 360
    while new_yaw < -180:
        new_yaw += 360
    
    # 2. Compute position offset in camera-local coordinates
    local_forward = 0.0
    if w > 0.5 and s < 0.5:
        local_forward = WSAD_OFFSET
    elif s > 0.5 and w < 0.5:
        local_forward = -WSAD_OFFSET
    
    local_right = 0.0
    if d > 0.5 and a < 0.5:
        local_right = WSAD_OFFSET
    elif a > 0.5 and d < 0.5:
        local_right = -WSAD_OFFSET
    
    # Handle diagonal movement
    if abs(local_forward) > 0.1 and abs(local_right) > 0.1:
        local_forward = np.sign(local_forward) * DIAGONAL_OFFSET
        local_right = np.sign(local_right) * DIAGONAL_OFFSET
    
    # 3. Transform local offset to world coordinates using AVERAGE yaw
    avg_yaw = float((yaw + new_yaw) / 2.0)
    yaw_rad = float(np.deg2rad(avg_yaw))
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    delta_x = cos_yaw * local_forward - sin_yaw * local_right
    delta_y = sin_yaw * local_forward + cos_yaw * local_right
    delta_z = 0.0
    
    new_x = x + delta_x
    new_y = y + delta_y
    new_z = z + delta_z
    
    return np.array([new_x, new_y, new_z, new_pitch, new_yaw])

def get_data(num_frames, height, width, pil_image, device=None, dtype=None, key_strength=1.0):
    input_image = torch.from_numpy(np.array(pil_image)).unsqueeze(0)
    input_image = input_image.permute(0, 3, 1, 2)
    def normalize_to_neg_one_to_one(x):
        return 2. * x - 1.
    transform = get_video_transform(height, width, normalize_to_neg_one_to_one)
    input_image = transform(input_image)
    input_image = input_image.transpose(0, 1).unsqueeze(0) # b c t h w

    actions = Bench_actions_universal(num_frames)
    keyboard_condition_all = actions['keyboard_condition'] * key_strength
    mouse_condition_all = actions['mouse_condition']

    first_pose = np.concatenate([np.zeros(3), np.zeros(2)], axis=0)
    # Compute poses from actions (first frame pose is all zeros)
    all_poses = compute_all_poses_from_actions(keyboard_condition_all, mouse_condition_all, first_pose=first_pose)
    positions = all_poses[:, :3].tolist()  # (T, 3)
    rotations = np.concatenate([
        np.zeros((all_poses.shape[0], 1)), # roll = 0
        all_poses[:, 3:5]  # pitch, yaw
    ], axis=1).tolist()  # (T, 3) - [roll, pitch, yaw]
    extrinsics_all = get_extrinsics(rotations, positions)
    return (input_image.to(device, dtype), extrinsics_all, keyboard_condition_all.to(device, dtype).unsqueeze(0), mouse_condition_all.to(device, dtype).unsqueeze(0))

def build_plucker_from_c2ws(c2ws_seq, src_indices, tgt_indices, framewise, base_K, target_h, target_w, lat_h, lat_w):
    device = c2ws_seq.device
    c2ws_np = c2ws_seq.cpu().numpy()
    c2ws_infer = _interpolate_camera_poses_handedness(
        src_indices=src_indices,
        src_rot_mat=c2ws_np[:, :3, :3],
        src_trans_vec=c2ws_np[:, :3, 3],
        tgt_indices=tgt_indices,
    )
    c2ws_infer = c2ws_infer.to(device=device)
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=framewise)
    Ks = base_K.to(c2ws_infer.device, dtype=c2ws_infer.dtype).repeat(c2ws_infer.shape[0], 1)
    plucker = get_plucker_embeddings(c2ws_infer, Ks, target_h, target_w)
    c1 = target_h // lat_h
    c2 = target_w // lat_w
    plucker = rearrange(
        plucker,
        'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
        c1=c1,
        c2=c2,
    )
    plucker = plucker[None, ...]
    plucker = rearrange(
        plucker,
        'b (f h w) c -> b c f h w',
        f=len(tgt_indices),
        h=lat_h,
        w=lat_w,
    )
    return plucker

def build_plucker_from_pose(c2ws_pose, base_K, target_h, target_w, lat_h, lat_w):
    Ks = base_K.to(c2ws_pose.device, dtype=c2ws_pose.dtype).repeat(c2ws_pose.shape[0], 1)
    plucker = get_plucker_embeddings(c2ws_pose, Ks, target_h, target_w)
    c1 = target_h // lat_h
    c2 = target_w // lat_w
    plucker = rearrange(
        plucker,
        'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
        c1=c1,
        c2=c2,
    )
    plucker = plucker[None, ...]
    plucker = rearrange(
        plucker,
        'b (f h w) c -> b c f h w',
        f=c2ws_pose.shape[0],
        h=lat_h,
        w=lat_w,
    )
    return plucker
