

# Set environment variable for CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_ROOT="models/matrixgame" # Replace with the actual path to your model directory
export DIT_PATH="$MODEL_ROOT/dit/"
export TEXTENC_PATH="$MODEL_ROOT"
export VAE_PATH="$MODEL_ROOT/vae/"
export MOUSE_ICON_PATH="$MODEL_ROOT/assets/mouse.png"
export IMAGE_PATH="initial_image/" # Replace with the actual path to your initial image
export OUTPUT_PATH="./test"
export INFERENCE_STEPS=50
# Execute inference script with parameters
python inference_bench.py \
    --dit_path $DIT_PATH \
    --textenc_path $TEXTENC_PATH \
    --vae_path $VAE_PATH \
    --mouse_icon_path $MOUSE_ICON_PATH \
    --image_path $IMAGE_PATH \
    --output_path $OUTPUT_PATH \
    --inference_steps $INFERENCE_STEPS \
    --bfloat16
