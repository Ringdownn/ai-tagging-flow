# Docker 部署指南

本文档说明如何使用 Docker Compose 部署 ai-tagging-flow 的三个推理服务。

## 架构

| 服务 | 端口 | 说明 |
|------|------|------|
| `orchestration` | 8000 | 打标编排入口，`POST /tag` |
| `remove-bg` | 8001 | RMBG-2.0 抠图 |
| `detect-crop` | 8002 | YOLOv8 检测裁切 |

模型权重通过宿主机目录 `./models` 只读挂载到容器 `/app/models`，不打进镜像。

## 前置条件

1. 安装 [Docker](https://docs.docker.com/get-docker/) 与 [Docker Compose](https://docs.docker.com/compose/install/)
2. **GPU 部署**：安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. 准备模型文件（宿主机路径）：

```
models/
├── RMBG-2.0/onnx/model_int8.onnx    # 抠图（必需）
├── qwen2.5-vl-7b-merge/             # 本地 VLM 完整权重（必需）
│   ├── config.json
│   └── model-*.safetensors
└── yolo/yolov8n.pt                    # 可选，缺失时首次启动自动下载
```

4. 复制环境变量文件并填写 API Key：

```bash
cp .env.example .env
# 编辑 .env，设置 ZHIPU_API_KEY
```

## 本地快速体验（含前端页面）

无需 Docker，Mac 本机可直接启动：

```bash
# 1. 安装依赖（首次）
python3 -m venv .venv
.venv/bin/pip install -r services/requirements.txt

# 2. 配置 API Key（疑难图分支需要）
cp .env.example .env
# 编辑 .env 填写 ZHIPU_API_KEY

# 3. 启动三个服务
bash scripts/start_local.sh

# 4. 浏览器打开
open http://127.0.0.1:8000/
```

前端支持拖拽上传图片，调用 `POST /tag/upload` 返回标签可视化结果。

停止服务：`bash scripts/stop_local.sh`

## 启动

### GPU 部署（推荐）

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### CPU 部署（开发/演示）

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

首次构建镜像较慢（需下载 PyTorch 等依赖），请耐心等待。

## 健康检查

```bash
curl http://localhost:8000/health   # orchestration
curl http://localhost:8001/health   # remove-bg
curl http://localhost:8002/health   # detect-crop
```

查看日志：

```bash
docker compose logs -f orchestration
docker compose logs -f remove-bg
docker compose logs -f detect-crop
```

## 调用打标接口

```bash
curl -X POST http://localhost:8000/tag \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/product.jpg"}'
```

响应示例：

```json
{
  "tags": {
    "类目": ["服饰", "女装"],
    "颜色": ["黑色"],
    "材质/面料": "棉",
    "款式特征": "休闲",
    "成色": "九成新",
    "置信度": 0.85,
    "复杂度": 0.2
  },
  "tags_list": null,
  "branch": "local",
  "confidence": 0.85,
  "complexity": 0.2,
  "is_product": true,
  "raw_state": { ... }
}
```

`branch` 取值：

- `local`：本地 Qwen2.5-VL 直接返回
- `glm4v`：疑难图，经 GLM-4V + 工具链处理
- `non_product`：非商品图
- `error`：处理失败，返回降级标签

## 测试工具服务

宿主机上运行（需先 `pip install requests pillow`）：

```bash
python scripts/test_tools.py data/open_datasets/images/fashion_0000.jpg
```

确保 `.env` 或环境变量中服务地址为 `localhost`（默认端口 8001/8002）。

## 停止与清理

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
# 或 CPU profile
docker compose -f docker-compose.yml -f docker-compose.cpu.yml down
```

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ZHIPU_API_KEY` | - | 智谱 GLM-4V API Key |
| `LOCAL_VLM_PATH` | `/app/models/qwen2.5-vl-7b-merge` | 本地 VLM 模型目录 |
| `LOCAL_VLM_DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
| `CONFIDENCE_THRESHOLD` | `0.75` | 本地模型置信度阈值 |
| `COMPLEXITY_THRESHOLD` | `0.5` | 复杂度阈值 |
| `RMBG_ONNX_MODEL` | `model_int8.onnx` | 抠图 ONNX 文件名 |
| `YOLO_DEVICE` | `cpu` | YOLO 推理设备 |

## 常见问题

### 1. orchestration 启动失败：模型路径不存在

确认宿主机 `models/qwen2.5-vl-7b-merge/` 包含完整权重，或修改 `.env` 中 `LOCAL_VLM_PATH` 指向正确子目录。

### 2. GLM 分支报错

检查 `ZHIPU_API_KEY` 是否已设置：`docker compose exec orchestration env | grep ZHIPU`

### 3. GPU 不可用

- 确认已安装 nvidia-container-toolkit
- 运行 `nvidia-smi` 验证驱动
- 改用 CPU profile：`-f docker-compose.cpu.yml`

### 4. 显存不足（OOM）

- 使用 `RMBG_ONNX_MODEL=model_int8.onnx`（默认）
- 降低并发请求
- CPU profile 下本地 VLM 推理较慢但无需 GPU 显存

### 5. remove-bg / detect-crop 健康检查超时

模型首次加载较慢，`start_period` 已设为 120s/180s。若仍失败，查看对应容器日志确认 ONNX/YOLO 权重是否挂载正确。
