import json
import os
from einops import rearrange
import torch

def Bench_actions_76():
    actions_single_action = [
        "forward",
        "back",
        "left",
        "right",
        "jump",
        "attack"
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
        "back_left",
        "back_right",
        "forward_jump",
        "back_jump",
        "left_jump",
        "right_jump",
        "forward_attack",
        "back_attack",
        "left_attack",
        "right_attack",
        "jump_attack"
    ]

    actions_single_camera = [    
        "camera_up",
        "camera_down",
        "camera_l",
        "camera_r",
        "camera_ur",
        "camera_ul",
        "camera_dl",
        "camera_dr"
    ]
    actions_to_test = actions_single_action + actions_single_camera + ['empty'] + actions_double_action
    for action in actions_single_action:
        for camera in actions_single_camera:
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3,
        "jump": 4,  "attack": 5
    }

    CAM_VALUE = 0.05
    CAMERA_VALUE_MAP = {
        "camera_up":  [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
        "camera_ur":  [CAM_VALUE, CAM_VALUE],
        "camera_ul":  [CAM_VALUE, -CAM_VALUE],
        "camera_dr":  [-CAM_VALUE, CAM_VALUE],
        "camera_dl":  [-CAM_VALUE, -CAM_VALUE],
    }

    # output_dir = "wpy/Bench76"
    # os.makedirs(output_dir, exist_ok=True)

    num_samples_per_action = 65

    data = []

    for action_name in actions_to_test:
        # 前，后，左，右，跳跃，攻击
        keyboard_condition = [[0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP: # camera_dr
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act == "attack":
                # to do 只有帧数 (idx % 16 >= 8) & (idx % 16 < 16)才为1
                for idx in range(num_samples_per_action):
                    if 8 <= (idx % 16) < 16:
                        keyboard_condition[idx][KEYBOARD_IDX["attack"]] = 1

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "action_name": action_name,
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition
        })

    return data

# 5 + 8 + 40 + 8 + 1 = 62
def Bench_actions_unreal():
    actions_single_action = [
        "forward",
        "back",
        "left",
        "right",
        "jump",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
        "back_left",
        "back_right",
        "forward_jump",
        "back_jump",
        "left_jump",
        "right_jump",
    ]

    actions_single_camera = [    
        "camera_up",
        "camera_down",
        "camera_l",
        "camera_r",
        "camera_ur",
        "camera_ul",
        "camera_dl",
        "camera_dr"
    ]
    actions_to_test = actions_single_action + actions_single_camera + ['empty'] + actions_double_action
    for action in actions_single_action:
        for camera in actions_single_camera:
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3,
        "jump": 4,  "attack": 5
    }

    CAM_VALUE = 0.35
    CAMERA_VALUE_MAP = {
        "camera_up":  [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
        "camera_ur":  [CAM_VALUE, CAM_VALUE],
        "camera_ul":  [CAM_VALUE, -CAM_VALUE],
        "camera_dr":  [-CAM_VALUE, CAM_VALUE],
        "camera_dl":  [-CAM_VALUE, -CAM_VALUE],
    }

    # output_dir = "wpy/Bench76"
    # os.makedirs(output_dir, exist_ok=True)

    num_samples_per_action = 65

    data = []

    for action_name in actions_to_test:
        # 前，后，左，右，跳跃，攻击
        keyboard_condition = [[0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}") # test
            if sub_act in CAMERA_VALUE_MAP: # camera_dr
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            # elif sub_act == "attack":
            #     # to do 只有帧数 (idx % 16 >= 8) & (idx % 16 < 16)才为1
            #     for idx in range(num_samples_per_action):
            #         if 8 <= (idx % 16) < 16:
            #             keyboard_condition[idx][KEYBOARD_IDX["attack"]] = 1

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        if 'camera_down' in action_name:
            print(f"action name: {action_name}")
            print('keyboard_condition: ', keyboard_condition)
            print('mouse_condition: ', mouse_condition)
        data.append({
            "action_name": action_name,
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition
        })

    return data

def Bench_camera_unreal():
    actions_single_action = [
        "forward",
        "back",
        "left",
        "right",
        "jump",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
        "back_left",
        "back_right",
        "forward_jump",
        "back_jump",
        "left_jump",
        "right_jump",
    ]

    actions_single_camera = [    
        "camera_up",
        "camera_down",
        "camera_l",
        "camera_r",
        "camera_ur",
        "camera_ul",
        "camera_dl",
        "camera_dr"
    ]
    actions_to_test = actions_single_camera
    for action in actions_single_action:
        for camera in actions_single_camera:
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3,
        "jump": 4,  "attack": 5
    }

    CAM_VALUE = 0.35
    CAMERA_VALUE_MAP = {
        "camera_up":  [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
        "camera_ur":  [CAM_VALUE, CAM_VALUE],
        "camera_ul":  [CAM_VALUE, -CAM_VALUE],
        "camera_dr":  [-CAM_VALUE, CAM_VALUE],
        "camera_dl":  [-CAM_VALUE, -CAM_VALUE],
    }

    # output_dir = "wpy/Bench76"
    # os.makedirs(output_dir, exist_ok=True)

    num_samples_per_action = 65

    data = []

    for action_name in actions_to_test:
        # 前，后，左，右，跳跃，攻击
        keyboard_condition = [[0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}") # test
            if sub_act in CAMERA_VALUE_MAP: # camera_dr
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            # elif sub_act == "attack":
            #     # to do 只有帧数 (idx % 16 >= 8) & (idx % 16 < 16)才为1
            #     for idx in range(num_samples_per_action):
            #         if 8 <= (idx % 16) < 16:
            #             keyboard_condition[idx][KEYBOARD_IDX["attack"]] = 1

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        if 'camera_down' in action_name:
            print(f"action name: {action_name}")
            print('keyboard_condition: ', keyboard_condition)
            print('mouse_condition: ', mouse_condition)
        data.append({
            "action_name": action_name,
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition
        })

    return data

if __name__ == "__main__":
    # conditions = Bench_actions_76()
    conditions = Bench_actions_unreal()
    keyboard_condition = torch.tensor(conditions[0]['keyboard_condition'], dtype=torch.float32).unsqueeze(0)
    mouse_condition = torch.tensor(conditions[0]['mouse_condition'], dtype=torch.float32).unsqueeze(0)
    print(keyboard_condition.shape, mouse_condition.shape)