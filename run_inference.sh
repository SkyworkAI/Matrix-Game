#!/usr/bin/env bash
# SCRIPT=inference_bench_gpus.py

# # 公共参数（除 gpu 相关之外都放在这里）
# COMMON_ARGS="
#   --gpus 8
#   --output_path /mnt/datasets_genie/puyi.wang/videosft-i2v-textfree-pf-flashattn3-65f-puyi/wpy/bench76_cuiqi
#   --video_length 65
#   --guidance_scale 6
#   --pretrained /mnt/data_genie/yifan.zhang/genie/videosft-i2v-textfree-pf-flashattn3-33f/outputs/250415_hunyuan_i2v_action_pastframe5_robust_foundation3epoch_all2epoch_sub3e8e2k_65f/checkpoint-8000-1647/model
#   --num_pre_frames 5
#   --image_path /mnt/workspace/yifan.zhang/genie/videosft-i2v-textfree-pf/test_image/city1.png
#   --use-cpu-offload
# "

# # 需要使用的 GPU 列表
# for GPU_ID in $(seq 0 7); do
#   CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} ${COMMON_ARGS} --gpu_id ${GPU_ID} &
# done

# wait

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python inference_bench.py --pretrained /mnt/data_genie/yifan.zhang/genie/videosft-i2v-textfree-pf-flashattn3-33f/outputs/250424_hunyuan_i2v_action_pastframe5_robust_foundation3epoch_all2epoch_sub3e8e2k_65f5e_new/checkpoint-1000-1000/model/  --image_path /mnt/datasets_genie/.ossutil_checkpoint/ossutil_yf.cp/ossutil_output/initial_image/  --output_path ./test  --num_pre_frames 5 --bfloat16