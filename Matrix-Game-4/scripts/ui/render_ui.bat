@echo off
rem Start the local racing render UI and open it in the browser. Needs ssh, scp and ffmpeg on PATH and key-based ssh access to the DGX.
rem Usage: scripts\ui\render_ui.bat [--port 8765] [--no-browser]
python "%~dp0render_ui.py" %*
