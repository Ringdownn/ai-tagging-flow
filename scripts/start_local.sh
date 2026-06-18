#!/usr/bin/env bash
# 本地启动三个服务（方案 A：文件直传 + 工具服务 + 编排 API）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/temp/logs"
PID_DIR="$ROOT/temp/pids"
mkdir -p "$LOG_DIR" "$PID_DIR" "$ROOT/temp/orchestration"

# 加载 .env（若存在）
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export LOCAL_VLM_PATH="${LOCAL_VLM_PATH:-$ROOT/models/qwen2.5-vl-7b-merge}"
export REMOVE_BG_URL="${REMOVE_BG_URL:-http://127.0.0.1:8001}"
export DETECT_CROP_URL="${DETECT_CROP_URL:-http://127.0.0.1:8002}"
export LOCAL_VLM_DEVICE="${LOCAL_VLM_DEVICE:-auto}"
export RMBG_ONNX_MODEL="${RMBG_ONNX_MODEL:-model_int8.onnx}"
export YOLO_DEVICE="${YOLO_DEVICE:-cpu}"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

start_service() {
  local name="$1"
  local module="$2"
  local port="$3"
  local pid_file="$PID_DIR/${name}.pid"
  local log_file="$LOG_DIR/${name}.log"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "[SKIP] $name 已在运行 (PID $(cat "$pid_file"))"
    return
  fi

  echo "[START] $name → :$port (日志: $log_file)"
  nohup "$PYTHON" -m uvicorn "$module" --host 0.0.0.0 --port "$port" \
    >"$log_file" 2>&1 &
  echo $! >"$pid_file"
}

echo "=== AI Tagging Flow 本地部署 ==="
echo "项目目录: $ROOT"
echo "本地模型: $LOCAL_VLM_PATH"
echo ""

start_service "remove-bg" "services.remove_bg_service:app" 8001
start_service "detect-crop" "services.detect_crop_service:app" 8002
start_service "orchestration" "services.orchestration_service:app" 8000

echo ""
echo "等待服务就绪..."
sleep 3

for port in 8001 8002 8000; do
  if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "  ✓ :$port 健康"
  else
    echo "  ✗ :$port 未就绪，请查看 temp/logs/ 下日志"
  fi
done

echo ""
echo "前端页面: http://127.0.0.1:8000/"
echo "打标接口: POST http://127.0.0.1:8000/tag/upload"
echo ""
echo "停止服务: bash scripts/stop_local.sh"
