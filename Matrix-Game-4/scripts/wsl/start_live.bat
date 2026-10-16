@echo off
rem Start, stop or inspect live City Racing: starts MediaMTX and the server on the DGX (if not running), opens the SSH tunnel from this PC and
rem opens the page in the browser. Defaults: racing, realtime preset, 24 fps, fixed prompt.
rem Usage: scripts\wsl\start_live.bat [start|stop|status] [--preset realtime|quality|rtx5090] [--fps 24] [--prompt-mode fixed|dynamic] [--no-browser]
rem   stop     stops the DGX processes this script started and closes the tunnel on this PC; a stack started by hand is left alone
setlocal enabledelayedexpansion
set "DGX=kschmid@10.74.11.118"
set "DGX_REPO=/localhome/kschmid/Matrix-Game/Matrix-Game-4"
set "URL=http://127.0.0.1:8025/"

set "ACTION=start"
set "ARGS="
set "BROWSER=1"
for %%A in (%*) do (
  if "%%A"=="--no-browser" (
    set "BROWSER=0"
  ) else if "%%A"=="start" (
    set "ACTION=start"
  ) else if "%%A"=="stop" (
    set "ACTION=stop"
  ) else if "%%A"=="status" (
    set "ACTION=status"
  ) else (
    set "ARGS=!ARGS! %%A"
  )
)

if "%ACTION%"=="status" (
  ssh -o BatchMode=yes %DGX% "cd %DGX_REPO% && bash scripts/dgx/start_live.sh status"
  exit /b %ERRORLEVEL%
)

if "%ACTION%"=="stop" (
  ssh -o BatchMode=yes %DGX% "cd %DGX_REPO% && bash scripts/dgx/start_live.sh stop"
  call :close_tunnel
  exit /b 0
)

ssh -o BatchMode=yes %DGX% "cd %DGX_REPO% && bash scripts/dgx/start_live.sh start%ARGS%"
if errorlevel 1 exit /b 1

call :tunnel_pid
if defined TUNNEL_PID (
  echo tunnel already open ^(pid !TUNNEL_PID!^)
) else (
  start "live tunnel" /min ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 8025:127.0.0.1:8025 -L 8890:127.0.0.1:8890 -L 8190:127.0.0.1:8190 %DGX%
  timeout /t 3 /nobreak >nul
  echo tunnel opened
)
if "%BROWSER%"=="1" start "" "%URL%"
echo Live page: %URL%
exit /b 0

:tunnel_pid
set "TUNNEL_PID="
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /R /C:"127.0.0.1:8025 .*LISTENING"') do (
  for /f "tokens=1" %%N in ('tasklist /FI "PID eq %%P" /NH ^| findstr /I "ssh.exe"') do set "TUNNEL_PID=%%P"
)
exit /b 0

:close_tunnel
call :tunnel_pid
if defined TUNNEL_PID (
  taskkill /PID !TUNNEL_PID! >nul
  echo closed the tunnel ^(pid !TUNNEL_PID!^)
) else (
  echo no tunnel to close
)
exit /b 0
