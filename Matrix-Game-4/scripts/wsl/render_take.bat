@echo off
rem Copy a recorded take to the DGX, render it with the model and copy the result back to outputs\<name>\.
rem Usage: scripts\wsl\render_take.bat [take-NNN] [output_name] [-- extra matrix-game render arguments]
rem   take-NNN     a folder in captures\ (default: the newest one)
rem   output_name  default racing_<take>_hq; it must not exist yet on the DGX
rem Defaults: quality preset, Wan VAE, 4 denoising steps (--steps 1000,750,500,250). Extra arguments after -- are added to those, e.g.
rem   scripts\wsl\render_take.bat take-003 racing_take003_yellow -- --prompt "Sunny day, yellow sedan" --seed 1
rem Set SINK_IMAGE=<full path of a 16:9 picture> to use it as the reference image (default: the scene's built-in one).
rem Set MATRIX_GAME_RENDER_SIZE (for example 1728x960) before running to change the render resolution (unvalidated, slower).
setlocal enabledelayedexpansion
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "DGX=kschmid@10.74.11.118"
set "DGX_REPO=/localhome/kschmid/Matrix-Game/Matrix-Game-4"

set "TAKE="
set "NAME="
set "EXTRA="
set "SEEN="
for %%A in (%*) do (
  if defined SEEN (
    set "EXTRA=!EXTRA! %%A"
  ) else if "%%A"=="--" (
    set "SEEN=1"
  ) else if not defined TAKE (
    set "TAKE=%%A"
  ) else if not defined NAME (
    set "NAME=%%A"
  )
)
if defined EXTRA set "EXTRA=!EXTRA:"=\"!"

if not defined TAKE (
  for /f "delims=" %%D in ('dir /b /ad /o-n "%REPO%\captures\take-*" 2^>nul') do if not defined TAKE set "TAKE=%%D"
)
if not defined TAKE (
  echo ERROR: no take found in %REPO%\captures. Record one with scripts\wsl\play_racing.bat ^(F9 starts and stops recording^).
  exit /b 1
)
if not exist "%REPO%\captures\%TAKE%\session.json" (
  echo ERROR: %REPO%\captures\%TAKE% is not a finished take ^(no session.json^).
  exit /b 1
)
if not defined NAME set "NAME=racing_%TAKE:-=%_hq"

echo Take: %REPO%\captures\%TAKE%
echo Output name: %NAME%
set "SIZE_ENV="
if defined MATRIX_GAME_RENDER_SIZE set "SIZE_ENV=MATRIX_GAME_RENDER_SIZE=%MATRIX_GAME_RENDER_SIZE% "

rem the render refuses an existing output folder, so use the first free name: <name>, <name>_2, <name>_3 ...
set "FREE="
for /f "delims=" %%N in ('ssh -o BatchMode^=yes %DGX% "cd %DGX_REPO% && n=%NAME%; i=2; while [ -e outputs/$n ]; do n=%NAME%_$i; i=`expr $i + 1`; done; echo $n"') do set "FREE=%%N"
if not defined FREE (
  echo ERROR: could not reach the DGX.
  exit /b 1
)
if not "%FREE%"=="%NAME%" echo Output folder %NAME% already exists on the DGX; using %FREE% instead.
set "NAME=%FREE%"
rem optional reference image: set SINK_IMAGE=<full path to a 16:9 picture> before running
if defined SINK_IMAGE (
  if not exist "!SINK_IMAGE!" (
    echo ERROR: SINK_IMAGE !SINK_IMAGE! does not exist.
    exit /b 1
  )
  scp -o BatchMode=yes "!SINK_IMAGE!" %DGX%:%DGX_REPO%/captures/sink_%NAME%.jpg
  if errorlevel 1 exit /b 1
  set "EXTRA=!EXTRA! --sink-image captures/sink_%NAME%.jpg"
)
scp -r -o BatchMode=yes "%REPO%\captures\%TAKE%" %DGX%:%DGX_REPO%/captures/
if errorlevel 1 exit /b 1
ssh -o BatchMode=yes %DGX% "cd %DGX_REPO% && nvidia-smi --query-gpu=memory.free --format=csv,noheader && %SIZE_ENV%bash scripts/dgx/run_racing.sh --take captures/%TAKE% --preset quality --output outputs/%NAME% -- --steps 1000,750,500,250 --first-steps 1000,750,500,250%EXTRA%"
if errorlevel 1 exit /b 1
if not exist "%REPO%\outputs\%NAME%" mkdir "%REPO%\outputs\%NAME%"
scp -o BatchMode=yes %DGX%:%DGX_REPO%/outputs/%NAME%/rendered.mp4 %DGX%:%DGX_REPO%/outputs/%NAME%/render.json "%REPO%\outputs\%NAME%"
if errorlevel 1 exit /b 1
echo Done: %REPO%\outputs\%NAME%\rendered.mp4
exit /b 0
