#!/usr/bin/env bash
# 本地启动三个服务（方案 A：文件直传 + 工具服务 + 编排 API）
set -uo pipefail

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

export LOCAL_VLM_PATH="${LOCAL_VLM_PATH:-$ROOT/models/qwen2.5-vl-3b-merge}"
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

  # 释放端口（避免上次异常退出占用）
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti:"$port" | xargs kill -9 2>/dev/null || true
  fi

  echo "[START] $name → :$port (日志: $log_file)"
  nohup "$PYTHON" -m uvicorn "$module" --host 0.0.0.0 --port "$port" \
    >"$log_file" 2>&1 &
  echo $! >"$pid_file"
  disown
}

wait_health() {
  local port="$1"
  local name="$2"
  local retries=36
  while (( retries > 0 )); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "  ✓ :$port $name 就绪"
      return 0
    fi
    sleep 2
    retries=$((retries - 1))
  done
  echo "  ✗ :$port $name 未就绪，请查看 temp/logs/ 下日志"
  return 1
}

echo "=== AI Tagging Flow 本地部署 ==="
echo "项目目录: $ROOT"
echo "本地模型: $LOCAL_VLM_PATH"
echo ""

start_service "orchestration" "services.orchestration_service:app" 8000

echo "  等待编排服务就绪..."
wait_health 8000 "orchestration" || true

echo ""
echo "  启动工具服务（懒加载模型，按需占用内存）..."
start_service "remove-bg" "services.remove_bg_service:app" 8001
start_service "detect-crop" "services.detect_crop_service:app" 8002

echo ""
echo "等待工具服务就绪..."

wait_health 8001 "remove-bg" || true
wait_health 8002 "detect-crop" || true

if [[ ! -d "$ROOT/.venv" ]]; then
  echo ""
  echo "提示: 首次使用请先安装依赖:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r services/requirements.txt"
fi

echo ""
echo "前端页面: http://127.0.0.1:8000/"
echo "打标接口: POST http://127.0.0.1:8000/tag/upload"
echo ""
echo "停止服务: bash scripts/stop_local.sh"
