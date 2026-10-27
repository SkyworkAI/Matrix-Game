#!/usr/bin/env bash
# Start, stop or inspect the live City Racing stack on the DGX: MediaMTX (video relay) and `matrix-game serve` (headless game + model + browser page).
# Defaults: racing scene, realtime preset (TAEHV, 1 step), 24 fps, fixed prompt. Start is idempotent: a stack that is already listening is left alone.
#
# Usage:   bash scripts/dgx/start_live.sh [start|stop|status] [--preset realtime|quality|rtx5090] [--fps 24] [--prompt-mode fixed|dynamic]
#   start   launch whatever is not running yet and wait until the page answers on port 8025 (model load takes about a minute)
#   stop    stop only the processes this script started (its pid files in runtime/live/); a stack started by hand is never touched
#   status  show ports and pids
# Needs MediaMTX (~/bin/mediamtx or on PATH), GStreamer with the x264 and RTSP plugins, and the racing resources (see docs/dgx_racing_runs.md).
# Logs: runtime/live/mediamtx.log and runtime/live/serve.log. The GPU is checked first: it refuses to start with less than MIN_FREE_MIB (default 30000) free.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

action="start"
if [[ $# -gt 0 && "$1" != --* ]]; then
  action="$1"
  shift
fi
preset="realtime"
fps=24
prompt_mode="fixed"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:?--preset needs a name}"; shift 2 ;;
    --fps) fps="${2:?--fps needs a number}"; shift 2 ;;
    --prompt-mode) prompt_mode="${2:?--prompt-mode needs fixed or dynamic}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

state="runtime/live"
mkdir -p "${state}"
http_port=8025

listening() { ss -ltn | awk '{print $4}' | grep -q ":$1\$"; }
alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

case "${action}" in
  status)
    for port in 8555 8890 8190 "${http_port}"; do
      if listening "${port}"; then echo "port ${port}: listening"; else echo "port ${port}: closed"; fi
    done
    for name in mediamtx serve; do
      if alive "${state}/${name}.pid"; then echo "${name}: started by this script, pid $(cat "${state}/${name}.pid")"; else echo "${name}: no live pid file"; fi
    done
    ;;
  stop)
    for name in serve mediamtx; do
      if alive "${state}/${name}.pid"; then
        pid="$(cat "${state}/${name}.pid")"
        echo "stopping ${name} (pid ${pid})"
        kill "${pid}"
        for _ in $(seq 1 30); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
      else
        echo "${name}: not started by this script, left alone"
      fi
      rm -f "${state}/${name}.pid"
    done
    ;;
  start)
    mediamtx_bin="$(command -v mediamtx || true)"
    [[ -n "${mediamtx_bin}" ]] || mediamtx_bin="${HOME}/bin/mediamtx"
    [[ -x "${mediamtx_bin}" ]] || { echo "ERROR: mediamtx not found. Install it first (see docs/dgx_racing_runs.md)." >&2; exit 1; }
    [[ -x .venv/bin/python ]] || { echo "ERROR: .venv is missing. Run scripts/dgx/setup_dgx.sh first." >&2; exit 1; }

    if listening 8555 || listening 8890; then
      echo "MediaMTX ports are already listening; not starting another"
    else
      nohup "${mediamtx_bin}" configs/mediamtx.yml > "${state}/mediamtx.log" 2>&1 < /dev/null &
      echo $! > "${state}/mediamtx.pid"
      echo "started MediaMTX (pid $!)"
      sleep 2
    fi

    if listening "${http_port}"; then
      echo "the page already answers on port ${http_port}; not starting another server"
    else
      min_free="${MIN_FREE_MIB:-30000}"
      free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
      echo "GPU free: ${free_mib} MiB (needs at least ${min_free})"
      if [[ "${free_mib}" -lt "${min_free}" ]]; then
        echo "ERROR: not enough free GPU memory; the GPU is shared. Not starting." >&2
        exit 1
      fi
      CUDA_VISIBLE_DEVICES=0 PATH="${root}/.venv/bin:${PATH}" PYTHON="${root}/.venv/bin/python" nohup bash scripts/serve.sh \
        --scene racing --resources resources --preset "${preset}" --fps "${fps}" --prompt-mode "${prompt_mode}" \
        > "${state}/serve.log" 2>&1 < /dev/null &
      echo $! > "${state}/serve.pid"
      echo "started the server (pid $!): racing, ${preset}, ${fps} fps, ${prompt_mode} prompt; loading the model"
      for _ in $(seq 1 180); do
        listening "${http_port}" && break
        kill -0 "$(cat "${state}/serve.pid")" 2>/dev/null || { echo "ERROR: the server exited; see ${state}/serve.log" >&2; tail -5 "${state}/serve.log" >&2; exit 1; }
        sleep 1
      done
      listening "${http_port}" || { echo "ERROR: the page did not come up within 180 s; see ${state}/serve.log" >&2; exit 1; }
    fi
    echo "ready: forward ports 8025, 8890 and 8190 and open http://127.0.0.1:${http_port}/"
    ;;
  *)
    echo "Unknown action: ${action} (start, stop or status)" >&2
    exit 2
    ;;
esac
