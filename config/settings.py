"""
全局配置
========
各模块共享的路径和参数配置。
"""
from pathlib import Path

# ==================== 项目根路径 ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==================== 数据路径 ====================
DATA_DIR = PROJECT_ROOT / "data"
OPEN_DATASETS_DIR = DATA_DIR / "open_datasets"
RAW_DIR = DATA_DIR / "raw"               # 原始商品照片
PROCESSED_DIR = DATA_DIR / "processed"    # 清洗后的训练数据

# ==================== 模型路径 ====================
MODELS_DIR = PROJECT_ROOT / "models"
QWEN2VL_DIR = MODELS_DIR / "qwen2.5-vl-3b-merge"        # 本地 VLM 路径
RMBG_DIR = MODELS_DIR / "RMBG-2.0"               # 抠图模型路径
YOLO_DIR = MODELS_DIR / "yolo"                   # 检测模型目录
YOLO_MODEL_PATH = YOLO_DIR / "yolov8n.pt"

# ==================== 服务配置 ====================
# 抠图服务
REMOVE_BG_HOST = "0.0.0.0"
REMOVE_BG_PORT = 8001
RMBG_MODEL_NAME = "briaai/RMBG-2.0"

# 检测裁切服务
DETECT_CROP_HOST = "0.0.0.0"
DETECT_CROP_PORT = 8002

# 业务中台（调度层）
ORCHESTRATION_HOST = "0.0.0.0"
ORCHESTRATION_PORT = 8000

# ==================== 图像处理 ====================
MAX_SIDE = 768        # 最大边长（方案要求）
MAX_CONCURRENCY = 2   # 工具处理并发
LOCAL_CONCURRENCY = 4  # 本地识别并发

# ==================== 大模型 API（智谱 GLM-4V，与打标脚本对齐） ====================
ZHIPU_API_KEY_ENV = "ZHIPU_API_KEY"
ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPU_MODEL = "glm-4v-plus"

# ==================== 训练配置 ====================
TRAINING_DIR = PROJECT_ROOT / "training"
TRAINING_CONFIGS_DIR = TRAINING_DIR / "configs"
TRAINING_SCRIPTS_DIR = TRAINING_DIR / "scripts"
TRAINING_OUTPUTS_DIR = TRAINING_DIR / "outputs"

# ==================== 临时文件 ====================
TEMP_DIR = PROJECT_ROOT / "temp"

# ==================== 商品照片目录（原始） ====================
LOCAL_PRODUCT_DIR = PROJECT_ROOT / "商品照片"
