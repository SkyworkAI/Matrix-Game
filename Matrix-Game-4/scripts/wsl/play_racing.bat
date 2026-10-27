@echo off
rem Play City Racing in WSL (WSLg window) and record takes into captures\ of this repository.
rem Usage: scripts\wsl\play_racing.bat [game arguments]   (no arguments = downtown time trial, HUD, 48 FPS offline recording)
rem The first run installs the native packages with sudo and builds the game; WSL asks for your sudo password in this window.
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
for /f "delims=" %%P in ('wsl wslpath -a "%REPO%"') do set "WSL_REPO=%%P"
if not defined WSL_REPO (
  echo ERROR: WSL is not available. Install it with: wsl --install -d Ubuntu-24.04
  exit /b 1
)
wsl bash "%WSL_REPO%/scripts/wsl/play_racing.sh" %*
exit /b %ERRORLEVEL%
