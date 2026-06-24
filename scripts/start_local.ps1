# AI Tagging Flow - Local Start Script (Windows PowerShell)
# Starts orchestration / remove-bg / detect-crop services

$ROOT = Split-Path -Parent $PSScriptRoot
$LOG_DIR = "$ROOT\temp\logs"
$PID_DIR = "$ROOT\temp\pids"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $PID_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$ROOT\temp\orchestration" | Out-Null

# Load .env if exists
$ENV_FILE = "$ROOT\.env"
if (Test-Path $ENV_FILE) {
    Get-Content $ENV_FILE | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)\s*=\s*(.*)\s*$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
}

# Defaults
if (-not $env:LOCAL_VLM_PATH) { $env:LOCAL_VLM_PATH = "$ROOT\models\qwen2.5-vl-7b-merge" }
if (-not $env:REMOVE_BG_URL) { $env:REMOVE_BG_URL = 'http://127.0.0.1:8001' }
if (-not $env:DETECT_CROP_URL) { $env:DETECT_CROP_URL = 'http://127.0.0.1:8002' }
if (-not $env:LOCAL_VLM_DEVICE) { $env:LOCAL_VLM_DEVICE = 'auto' }
if (-not $env:RMBG_ONNX_MODEL) { $env:RMBG_ONNX_MODEL = 'model_int8.onnx' }
if (-not $env:YOLO_DEVICE) { $env:YOLO_DEVICE = 'cpu' }

# Fix Ultralytics config dir permission issue
$env:ULTRALYTICS_CONFIG_DIR = "$ROOT\temp\ultralytics"
New-Item -ItemType Directory -Force -Path $env:ULTRALYTICS_CONFIG_DIR | Out-Null

# Fix HF home to avoid permission issues
$env:HF_HOME = "$ROOT\temp\hf_home"
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

# Find Python: conda env > .venv > system
$CONDA_PYTHON = "C:\conda\envs\ai-tagging-flow\python.exe"
$VENV_PYTHON = "$ROOT\.venv\Scripts\python.exe"

if (Test-Path $CONDA_PYTHON) {
    $PYTHON = $CONDA_PYTHON
} elseif (Test-Path $VENV_PYTHON) {
    $PYTHON = $VENV_PYTHON
} else {
    $PYTHON = 'python'
}

function Start-ServiceIfNotRunning($name, $module, $port) {
    $pidFile = "$PID_DIR\$name.pid"
    if (Test-Path $pidFile) {
        $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
            Write-Host "[SKIP] $name already running (PID $oldPid)"
            return
        }
    }

    # Kill any process on the port
    $listener = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        $proc = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "[KILL] Freeing port $port (PID $($proc.Id))"
            $proc.Kill()
            Start-Sleep -Seconds 1
        }
    }

    $logFile = "$LOG_DIR\$name.log"
    $errFile = "$LOG_DIR\$name.err.log"
    Write-Host "[START] $name -> :$port (log: $logFile)"

    $process = Start-Process -FilePath $PYTHON `
        -ArgumentList "-m uvicorn $module --host 0.0.0.0 --port $port" `
        -WorkingDirectory $ROOT `
        -WindowStyle Hidden `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $errFile `
        -PassThru

    $process.Id | Out-File $pidFile
    Write-Host "  PID: $($process.Id)"
}

function Wait-Health($port, $name) {
    $retries = 36
    $tmpPy = "$PID_DIR\_healthcheck_$port.py"
    @"
import sys, requests
try:
    r = requests.get('http://127.0.0.1:$port/health', timeout=2)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception:
    sys.exit(1)
"@ | Out-File -FilePath $tmpPy -Encoding utf8

    while ($retries -gt 0) {
        $proc = Start-Process -FilePath $PYTHON -ArgumentList $tmpPy -Wait -WindowStyle Hidden -PassThru
        if ($proc.ExitCode -eq 0) {
            Write-Host "  OK :$port $name ready"
            Remove-Item $tmpPy -ErrorAction SilentlyContinue
            return
        }
        Start-Sleep -Seconds 2
        $retries--
    }
    Remove-Item $tmpPy -ErrorAction SilentlyContinue
    Write-Host "  FAIL :$port $name not ready, check logs in $LOG_DIR"
}

Write-Host "=== AI Tagging Flow Local Deploy ==="
Write-Host "Project: $ROOT"
Write-Host "Model:  $env:LOCAL_VLM_PATH"
Write-Host "Python: $PYTHON"
Write-Host ""

Start-ServiceIfNotRunning 'orchestration' 'services.orchestration_service:app' 8000
Write-Host "Waiting for orchestration service..."
Wait-Health 8000 'orchestration'

Write-Host ""
Write-Host "Starting tool services..."
Start-ServiceIfNotRunning 'remove-bg' 'services.remove_bg_service:app' 8001
Start-ServiceIfNotRunning 'detect-crop' 'services.detect_crop_service:app' 8002

Write-Host ""
Write-Host "Waiting for tool services..."
Wait-Health 8001 'remove-bg'
Wait-Health 8002 'detect-crop'

Write-Host ""
Write-Host "Frontend:  http://127.0.0.1:8000/"
Write-Host "Tag API:   POST http://127.0.0.1:8000/tag/upload"
Write-Host ""
Write-Host "Stop:      .\scripts\stop_local.ps1"
