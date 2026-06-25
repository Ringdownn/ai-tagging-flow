# AI Tagging Flow

基于多模态大模型的二手电商商品自动打标系统。

## 主要功能

- **智能商品打标**：上传商品图片，自动输出结构化标签（类目、颜色、材质/面料、款式特征、成色等）。
- **商品/非商品判定**：自动识别非商品图片（风景、人物、文字截图等）并拒绝打标。
- **本地 VLM 推理**：默认使用 `Qwen2.5-VL` 多模态模型，支持 3B/7B 版本，可在 Mac (MPS)、NVIDIA GPU (CUDA) 或 CPU 上运行。
- **抠图预处理**：调用 RMBG-2.0 模型去除背景，突出商品主体。
- **检测裁切**：使用 YOLOv8-Nano 检测多商品场景并裁切主体。
- **复杂度与置信度**：输出 `confidence`（置信度）和 `complexity`（复杂度），用于路由简单图/疑难图分支。
- **Web 界面**：访问 `http://127.0.0.1:8000/` 即可上传图片并查看结果。

## 环境配置方法

### 1. 克隆仓库并进入目录

```bash
git clone git@github.com:Ringdownn/ai-tagging-flow.git
cd ai-tagging-flow
```

### 2. 创建并激活虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r services/requirements.txt
```

### 4. 准备模型文件

将以下模型放入 `models/` 目录：

| 模型 | 路径 |
|------|------|
| Qwen2.5-VL（推荐 3B） | `models/qwen2.5-vl-3b-merge/` |
| RMBG-2.0 ONNX | `models/RMBG-2.0/onnx/model_int8.onnx` |
| YOLOv8-Nano | `models/yolo/yolov8n.pt`（首次运行会自动下载） |

### 5. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env`，至少配置以下项：

```bash
# 智谱 GLM-4V API Key（疑难图分支使用，可选）
ZHIPU_API_KEY=your_key_here

# 本地 VLM 路径
LOCAL_VLM_PATH=models/qwen2.5-vl-3b-merge

# 设备与精度（Mac 上 auto 会自动检测 MPS）
LOCAL_VLM_DEVICE=auto
LOCAL_VLM_DTYPE=auto
```

> **注意**：`.env` 已加入 `.gitignore`，请勿提交到仓库。

## 如何运行

### 一键启动（推荐）

```bash
bash scripts/start_local.sh
```

脚本会依次启动三个服务：

| 服务 | 端口 | 说明 |
|------|------|------|
| 编排服务 | `8000` | 主入口，VLM 推理与标签生成 |
| 抠图服务 | `8001` | RMBG-2.0 背景移除 |
| 检测裁切服务 | `8002` | YOLOv8 商品检测与裁切 |

启动后访问：

```
http://127.0.0.1:8000/
```

### 手动启动

如果一键脚本不可用，也可分别启动：

```bash
# 1. 抠图服务
python -m uvicorn services.remove_bg_service:app --host 0.0.0.0 --port 8001

# 2. 检测裁切服务
python -m uvicorn services.detect_crop_service:app --host 0.0.0.0 --port 8002

# 3. 编排服务（在新终端中）
python -m uvicorn services.orchestration_service:app --host 0.0.0.0 --port 8000
```

### 停止服务

```bash
bash scripts/stop_local.sh
```

### API 调用示例

```bash
curl -X POST http://127.0.0.1:8000/tag/upload \
  -F "file=@path/to/your/image.jpg"
```

返回示例：

```json
{
  "is_product": true,
  "tags": {
    "类目": ["服饰", "女装", "连衣裙"],
    "颜色": ["黑色"],
    "材质/面料": "聚酯纤维",
    "款式特征": "短袖,收腰,A字裙",
    "成色": "九成新"
  },
  "confidence": 0.85,
  "complexity": 0.32,
  "reason": ""
}
```

## 硬件建议

| 配置 | 推荐 |
|------|------|
| Mac (Apple Silicon, 16GB 内存) | 使用 Qwen2.5-VL-3B，`LOCAL_VLM_DEVICE=auto`，约占用 6GB 内存 |
| NVIDIA GPU | 使用 Qwen2.5-VL-7B，`LOCAL_VLM_DEVICE=cuda`，可开启 4-bit/8-bit 量化 |
| 无 GPU | 使用 Qwen2.5-VL-3B，`LOCAL_VLM_DEVICE=cpu`，速度较慢但可运行 |

## 目录结构

```
ai-tagging-flow/
├── config/              # 配置
├── data/                # 数据集与处理结果
├── docker/              # Docker 构建依赖
├── models/              # 模型权重
├── scripts/             # 启动/停止脚本
├── services/            # 后端服务
│   ├── orchestration/   # 编排 + 本地 VLM
│   ├── detect_crop_service.py
│   └── remove_bg_service.py
├── static/              # 前端页面
├── docker-compose.yml
└── README.md
```
