"""
调度层私有配置
================
业务中台编排服务专用配置，优先从项目级 config/settings.py 读取。
"""
import os
from pathlib import Path

# 加载 .env 文件（项目根目录）
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

# 允许在未安装项目 config 时独立运行
try:
    from config.settings import (
        PROJECT_ROOT,
        QWEN2VL_DIR,
        REMOVE_BG_HOST,
        REMOVE_BG_PORT,
        DETECT_CROP_HOST,
        DETECT_CROP_PORT,
        ORCHESTRATION_HOST,
        ORCHESTRATION_PORT,
        MAX_SIDE,
        ZHIPU_URL,
        ZHIPU_MODEL,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    QWEN2VL_DIR = PROJECT_ROOT / "models" / "qwen2.5-vl-7b-merge"
    REMOVE_BG_HOST = "0.0.0.0"
    REMOVE_BG_PORT = 8001
    DETECT_CROP_HOST = "0.0.0.0"
    DETECT_CROP_PORT = 8002
    ORCHESTRATION_HOST = "0.0.0.0"
    ORCHESTRATION_PORT = 8000
    MAX_SIDE = 768
    ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    ZHIPU_MODEL = "glm-4v-plus"

# ==================== 本地模型 ====================
LOCAL_VLM_PATH = os.getenv("LOCAL_VLM_PATH", str(QWEN2VL_DIR))
LOCAL_VLM_DEVICE = os.getenv("LOCAL_VLM_DEVICE", "auto")   # auto / cuda / cpu
LOCAL_VLM_DTYPE = os.getenv("LOCAL_VLM_DTYPE", "auto")     # auto / float16 / bfloat16
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() in ("true", "1", "yes")   # INT4 量化加载
LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "false").lower() in ("true", "1", "yes")   # INT8 量化加载

# ==================== GLM-4V API ====================
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# ==================== 工具服务地址 ====================
REMOVE_BG_URL = os.getenv("REMOVE_BG_URL", f"http://{REMOVE_BG_HOST}:{REMOVE_BG_PORT}")
DETECT_CROP_URL = os.getenv("DETECT_CROP_URL", f"http://{DETECT_CROP_HOST}:{DETECT_CROP_PORT}")

# ==================== 业务阈值 ====================
# 本地模型置信度 >= 该阈值且复杂度 <= 复杂度阈值时直接返回
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.5"))
TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT", "30"))

# ==================== 多商品模式 ====================
# true:  对 detect_crop 返回的每个子图分别调用 GLM-4V，返回标签列表
# false: 只选质量最高的子图，返回单商品标签（默认，与训练数据对齐）
MULTI_OBJECT_MODE = os.getenv("MULTI_OBJECT_MODE", "false").lower() in ("true", "1", "yes")
