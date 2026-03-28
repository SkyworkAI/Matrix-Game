@echo off
:: Matrix-Game-3 - VBench Batch Inference
:: Generates num_samples videos per prompt for scenery+indoor image types.
:: Usage: run_vbench.bat [output_base] [num_samples] [image_types]

setlocal enabledelayedexpansion

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h"     goto :help
if /i "%~1"=="/?"     goto :help
goto :run

:help
echo.
echo Matrix-Game-3 - VBench Batch Inference
echo.
echo Usage:
echo   run_vbench.bat [output_base] [num_samples] [image_types]
echo.
echo Arguments (positional, all optional):
echo   1  output_base    Base output directory            (default: out\vbench)
echo   2  num_samples    Videos to generate per prompt   (default: 5)
echo   3  image_types    Comma-separated type filter      (default: scenery,indoor)
echo.
echo Notes:
echo   - VBench requires 5 samples per prompt
echo   - num_iterations=12 -^> 57 + 11*40 = 497 frames per video
echo   - Already-generated videos are skipped automatically
echo   - Outputs: {output_base}\videos\{caption}-{0..N-1}.mp4
echo   - Stats:   {output_base}\vbench_stats.csv
echo.
exit /b 0

:run
set PYTHONIOENCODING=utf-8
:: ── configurable defaults ──────────────────────────────────────────────────
set CKPT=Matrix-Game-3.0
set NUM_ITERATIONS=4
set NUM_INFERENCE_STEPS=5
set VAE_TYPE=mg_lightvae_v2
set LIGHTVAE_PRUNING_RATE=0.75
set FA_VERSION=2
set USE_INT8=1
set WORLDCACHE=0
set WORLDCACHE_THRESH=0.40
set WORLDCACHE_WARMUP=1
:: ──────────────────────────────────────────────────────────────────────────

set OUTPUT_BASE=%~1
if "%OUTPUT_BASE%"=="" set OUTPUT_BASE=out\vbench
set NUM_SAMPLES=%~2
if "%NUM_SAMPLES%"=="" set NUM_SAMPLES=5
set IMAGE_TYPES=%~3
if "%IMAGE_TYPES%"=="" set IMAGE_TYPES=scenery,indoor

set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

:: Prepend ROOT only for relative OUTPUT_BASE
if "%OUTPUT_BASE:~1,1%"==":" (
    rem absolute path — keep as-is
) else (
    set OUTPUT_BASE=%ROOT%\%OUTPUT_BASE%
)
set VBENCH_OUTPUT_DIR=%OUTPUT_BASE%\videos

if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"

for /f "tokens=2 delims==." %%a in ('wmic os get localdatetime /value 2^>nul') do set _DT=%%a
set LOG_FILE=%OUTPUT_BASE%\vbench_run_%_DT:~0,8%_%_DT:~8,6%.log

if not exist "%ROOT%\%CKPT%" (
    echo ERROR: Checkpoint not found: %ROOT%\%CKPT%
    echo Run download_model.bat first.
    exit /b 1
)

set /a FRAME_COUNT=57 + (%NUM_ITERATIONS% - 1) * 40

echo ============================================================
echo Matrix-Game-3  ^|  VBench batch  ^|  Windows
echo ============================================================
echo   ckpt      : %ROOT%\%CKPT%
echo   output    : %VBENCH_OUTPUT_DIR%
echo   samples   : %NUM_SAMPLES%
echo   types     : %IMAGE_TYPES%
echo   iters     : %NUM_ITERATIONS%  ^(=%FRAME_COUNT% frames/video^)
echo   steps     : %NUM_INFERENCE_STEPS%
echo   vae       : %VAE_TYPE%  pruning=%LIGHTVAE_PRUNING_RATE%
echo   int8      : %USE_INT8%
echo   fa        : %FA_VERSION%
if "%WORLDCACHE%"=="1" (
    echo   worldcache: ON  thresh=%WORLDCACHE_THRESH%  warmup=%WORLDCACHE_WARMUP%
) else (
    echo   worldcache: OFF
)
echo ============================================================

set START_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a START_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)

set PY_ARGS=--ckpt_dir "%ROOT%\%CKPT%"
set PY_ARGS=%PY_ARGS% --vbench_output_dir "%VBENCH_OUTPUT_DIR%"
set PY_ARGS=%PY_ARGS% --num_samples %NUM_SAMPLES%
set PY_ARGS=%PY_ARGS% --image_types "%IMAGE_TYPES%"
set PY_ARGS=%PY_ARGS% --num_iterations %NUM_ITERATIONS%
set PY_ARGS=%PY_ARGS% --num_inference_steps %NUM_INFERENCE_STEPS%
set PY_ARGS=%PY_ARGS% --vae_type %VAE_TYPE%
set PY_ARGS=%PY_ARGS% --lightvae_pruning_rate %LIGHTVAE_PRUNING_RATE%
set PY_ARGS=%PY_ARGS% --fa_version %FA_VERSION%
if "%USE_INT8%"=="1" set PY_ARGS=%PY_ARGS% --use_int8
if "%WORLDCACHE%"=="1" set PY_ARGS=%PY_ARGS% --worldcache --worldcache_thresh %WORLDCACHE_THRESH% --worldcache_warmup %WORLDCACHE_WARMUP%

echo.
echo [MG3-VBench] Generating %NUM_SAMPLES% samples per prompt...
chcp 65001 >nul
python "%ROOT%\generate_vbench.py" %PY_ARGS% 2>&1 | powershell -Command "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $input | Tee-Object -FilePath '%LOG_FILE%'; exit $LASTEXITCODE"
set EXIT_CODE=%ERRORLEVEL%
echo [MG3-VBench] Done. Exit: %EXIT_CODE%

for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a END_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)
set /a ELAPSED=END_S-START_S
if %ELAPSED% lss 0 set /a ELAPSED+=86400
set /a ELAPSED_H=ELAPSED/3600
set /a ELAPSED_M=(ELAPSED%%3600)/60
set /a ELAPSED_SS=ELAPSED%%60

echo ============================================================
echo Done. Elapsed: %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s  Exit: %EXIT_CODE%
echo Stats: %OUTPUT_BASE%\vbench_stats.csv
echo Log:   %LOG_FILE%
echo ============================================================

exit /b %EXIT_CODE%
