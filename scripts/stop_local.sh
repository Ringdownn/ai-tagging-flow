#!/usr/bin/env bash
# 停止本地三个服务
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT/temp/pids"

stop_one() {
  local name="$1"
  local pid_file="$PID_DIR/${name}.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "[STOP] $name (PID $pid)"
    fi
    rm -f "$pid_file"
  else
    echo "[SKIP] $name 未运行"
  fi
}

stop_one "orchestration"
stop_one "detect-crop"
stop_one "remove-bg"

echo "完成"
