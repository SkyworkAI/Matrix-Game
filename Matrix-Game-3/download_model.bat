@echo off
:: Matrix-Game-3 - Model Download
:: Usage: download_model.bat [--repo <hf_repo>] [--dir <local_dir>]

setlocal enabledelayedexpansion

set LOCAL_DIR=Matrix-Game-3.0
set HF_REPO=Skywork/Matrix-Game-3.0

:parse
if "%~1"=="" goto :run
if /i "%~1"=="--repo" ( set HF_REPO=%~2 & shift & shift & goto :parse )
if /i "%~1"=="--dir"  ( set LOCAL_DIR=%~2 & shift & shift & goto :parse )
shift & goto :parse

:run
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

if "%LOCAL_DIR:~1,1%"==":" (
    rem absolute — keep as-is
) else (
    set LOCAL_DIR=%ROOT%\%LOCAL_DIR%
)

echo ============================================================
echo Matrix-Game-3  ^|  Model Download
echo ============================================================
echo   repo      : %HF_REPO%
echo   local dir : %LOCAL_DIR%
echo ============================================================

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='%HF_REPO%', local_dir=r'%LOCAL_DIR%')"
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==0 (
    echo ============================================================
    echo Done. Model saved to: %LOCAL_DIR%
    echo ============================================================
) else (
    echo ============================================================
    echo ERROR: Download failed with exit code %EXIT_CODE%
    echo Make sure huggingface_hub is installed:
    echo   pip install huggingface_hub
    echo ============================================================
)

exit /b %EXIT_CODE%
