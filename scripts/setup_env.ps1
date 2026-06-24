# AI Tagging Flow 环境初始化脚本（Windows PowerShell）
# 使用清华镜像安装 Python 依赖

$ROOT = Split-Path -Parent $PSScriptRoot
$VENV_DIR = "$ROOT\.venv"
$PIP_TIMEOUT = 300

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Err($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

Write-Info "项目目录: $ROOT"

# 1. 创建虚拟环境
if (-not (Test-Path $VENV_DIR)) {
    Write-Info "创建虚拟环境 .venv ..."
    python -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Err "创建虚拟环境失败，请检查 python 是否可用"
        exit 1
    }
}

$PYTHON = "$VENV_DIR\Scripts\python.exe"
$PIP = "$VENV_DIR\Scripts\python.exe -m pip"

# 2. 升级 pip（使用清华镜像）
Write-Info "升级 pip ..."
& $PYTHON -m pip install --upgrade pip --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 3. 配置 pip 使用清华镜像（可选：写入 venv 配置，后续安装不用每次带参数）
$PIP_CONFIG_DIR = "$VENV_DIR\pip"
$PIP_CONFIG_FILE = "$PIP_CONFIG_DIR\pip.ini"
New-Item -ItemType Directory -Force -Path $PIP_CONFIG_DIR | Out-Null
@"
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
mirrors.tuna.tsinghua.edu.cn
timeout = $PIP_TIMEOUT
"@ | Out-File -Encoding utf8 $PIP_CONFIG_FILE
Write-Info "已写入 pip 清华镜像配置: $PIP_CONFIG_FILE"

# 4. 安装基础依赖
Write-Info "安装项目依赖（清华镜像）..."
& $PYTHON -m pip install -r "$ROOT\services\requirements.txt" --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
if ($LASTEXITCODE -ne 0) {
    Write-Err "依赖安装失败"
    exit 1
}

# 5. 安装 PyTorch CUDA 版本（清华镜像）
# 如果你的 CUDA 版本不是 12.6，请修改下面的 cu126
$CUDA_VERSION = "cu126"
Write-Info "安装 PyTorch + torchvision (CUDA $CUDA_VERSION，清华镜像)..."
& $PYTHON -m pip install torch torchvision --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --trusted-host mirrors.tuna.tsinghua.edu.cn
if ($LASTEXITCODE -ne 0) {
    Write-Err "PyTorch 安装失败，请检查 CUDA 版本或网络"
    exit 1
}

# 6. 验证
Write-Info "验证安装 ..."
& $PYTHON -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"

Write-Ok "环境初始化完成"
Write-Host ""
Write-Host "启动服务: .\scripts\start_local.ps1" -ForegroundColor Yellow
Write-Host "前端页面: http://127.0.0.1:8000/" -ForegroundColor Yellow
