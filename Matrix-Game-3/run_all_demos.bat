@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

set SIZE=704*1280
set CKPT_DIR=Matrix-Game-3.0
set FA_VERSION=3
set NUM_ITERATIONS=12
set NUM_INFERENCE_STEPS=3
set OUTPUT_DIR=./output

echo Output directory: %~dp0output

for /d %%D in (demo_images\*) do (
    set FOLDER=%%~nxD
    set IMAGE=%%D\image.png
    set PROMPT_FILE=%%D\prompt.txt

    if exist "!PROMPT_FILE!" (
        set /p PROMPT=<"!PROMPT_FILE!"
        echo.
        echo === Running !FOLDER! ===
        echo Prompt: !PROMPT!
        python generate.py ^
            --size %SIZE% ^
            --ckpt_dir %CKPT_DIR% ^
            --fa_version %FA_VERSION% ^
            --use_int8 ^
            --num_iterations %NUM_ITERATIONS% ^
            --num_inference_steps %NUM_INFERENCE_STEPS% ^
            --image "!IMAGE!" ^
            --prompt "!PROMPT!" ^
            --save_name !FOLDER! ^
            --seed 42 ^
            --compile_vae ^
            --lightvae_pruning_rate 0.5 ^
            --vae_type mg_lightvae ^
            --output_dir %OUTPUT_DIR%

        if !errorlevel! neq 0 (
            echo ERROR: !FOLDER! failed with exit code !errorlevel!
        ) else (
            echo Done: !FOLDER!
        )
    ) else (
        echo SKIP: !FOLDER! has no prompt.txt
    )
)

echo.
echo All demos complete.
endlocal
