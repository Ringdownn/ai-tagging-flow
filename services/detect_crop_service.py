"""
商品检测裁切服务 - YOLOv8-Nano
===============================
检测图片中的商品，返回检测框坐标和裁切后的子图。

依赖:
  pip install ultralytics pillow fastapi uvicorn

模型下载:
  1. 初始权重（通用 COCO）: 首次运行时 ultralytics 自动下载 yolov8n.pt
  2. 自有数据微调后: 替换 model_path 为你的 .pt 权重路径

启动:
  python detect_crop_service.py
  # 或: uvicorn services.detect_crop_service:app --host 0.0.0.0 --port 8002
"""

import base64
import io
import os
import sys
import time
from pathlib import Path

# Fix Ultralytics config dir permission issue on Windows
_proj_root = Path(__file__).resolve().parent.parent
_ultra_dir = _proj_root / "temp" / "ultralytics"
_ultra_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultra_dir))
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(_ultra_dir))

import requests
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    DETECT_CROP_HOST,
    DETECT_CROP_PORT,
    MAX_SIDE,
    YOLO_MODEL_PATH,
)

# ==================== 配置 ====================
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.4"))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.5"))
MODEL_PATH = str(YOLO_MODEL_PATH) if YOLO_MODEL_PATH.exists() else "yolov8n.pt"
DEVICE = os.getenv("YOLO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


# ==================== 模型加载 ====================

class ObjectDetector:
    def __init__(self, model_path: str):
        print(f"[INFO] 加载 YOLOv8 模型 ({model_path}) 到 {DEVICE} ...")
        self.model = YOLO(model_path)
        print("[INFO] YOLOv8 模型加载完成")

    def detect(self, pil_image: Image.Image) -> list[dict]:
        """检测图片中的商品，返回检测结果列表。"""
        results = self.model(
            pil_image,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        detections = []
        if len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        w, h = pil_image.size
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = results[0].names[cls_id]

            detections.append({
                "bbox": [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                "bbox_norm": [xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "class_name": cls_name,
            })

        return detections

    def crop_sub_images(
        self,
        pil_image: Image.Image,
        detections: list[dict],
        padding: int = 10,
    ) -> list[Image.Image]:
        """根据检测结果裁切子图。"""
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(pil_image.width, int(x2) + padding)
            y2 = min(pil_image.height, int(y2) + padding)
            crops.append(pil_image.crop((x1, y1, x2, y2)))
        return crops


# ==================== FastAPI ====================

app = FastAPI(title="商品检测裁切服务 - YOLOv8-Nano")
_detector: ObjectDetector | None = None


def get_detector() -> ObjectDetector:
    """懒加载模型，避免三服务同时启动时内存峰值过高。"""
    global _detector
    if _detector is None:
        _detector = ObjectDetector(MODEL_PATH)
    return _detector


def limit_max_side(pil_image: Image.Image, max_side: int) -> Image.Image:
    """等比缩放，限制最大边长。"""
    w, h = pil_image.size
    if max(w, h) <= max_side:
        return pil_image
    if w > h:
        new_w = max_side
        new_h = int(h * max_side / w)
    else:
        new_h = max_side
        new_w = int(w * max_side / h)
    return pil_image.resize((new_w, new_h), Image.LANCZOS)


def image_to_base64(pil_image: Image.Image) -> str:
    """PIL Image 转 base64 字符串。"""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_image_bytes(image_data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"图片解析失败: {e}") from e


async def _load_image_from_request(request: Request) -> tuple[bytes, bool, str]:
    """从请求中解析图片字节及裁切参数。返回 (image_data, return_crops, crop_format)。"""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        image_b64 = body.get("image")
        if not image_b64:
            raise HTTPException(400, "请提供 image(base64)")
        return (
            base64.b64decode(image_b64),
            body.get("return_crops", True),
            body.get("crop_format", "base64"),
        )

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        file = form.get("file")
        url = form.get("url")
        return_crops = str(form.get("return_crops", "true")).lower() not in ("false", "0", "no")
        crop_format = str(form.get("crop_format", "base64"))

        if file is not None and hasattr(file, "read"):
            return await file.read(), return_crops, crop_format
        if url:
            resp = requests.get(str(url), timeout=10)
            resp.raise_for_status()
            return resp.content, return_crops, crop_format
        raise HTTPException(400, "请提供 file 或 url")

    raise HTTPException(400, "不支持的 Content-Type，请使用 application/json 或 multipart/form-data")


async def _run_detect_crop(
    image_data: bytes,
    return_crops: bool = True,
    crop_format: str = "base64",
) -> dict | Response:
    """执行检测裁切核心逻辑。"""
    start_time = time.time()
    pil_image = _decode_image_bytes(image_data)
    original_size = pil_image.size
    pil_image = limit_max_side(pil_image, MAX_SIDE)

    try:
        detections = get_detector().detect(pil_image)
    except Exception as e:
        raise HTTPException(500, f"检测推理失败: {e}") from e

    crops_base64: list[str] = []
    crops_pil: list[Image.Image] = []
    if return_crops and crop_format != "none" and detections:
        crops_pil = get_detector().crop_sub_images(pil_image, detections)
        if crop_format == "base64":
            crops_base64 = [image_to_base64(c) for c in crops_pil]

    elapsed_ms = int((time.time() - start_time) * 1000)
    result = {
        "original_size": f"{original_size[0]}x{original_size[1]}",
        "processed_size": f"{pil_image.width}x{pil_image.height}",
        "detections": detections,
        "num_detections": len(detections),
        "process_time_ms": elapsed_ms,
    }

    if crops_base64:
        result["crops_base64"] = crops_base64

    if crop_format == "image" and crops_pil:
        if len(crops_pil) == 1:
            buf = io.BytesIO()
            crops_pil[0].save(buf, format="PNG")
            buf.seek(0)
            return Response(
                content=buf.getvalue(),
                media_type="image/png",
                headers={
                    "X-Detections": str(len(detections)),
                    "X-Process-Time-ms": str(elapsed_ms),
                },
            )
        result["crops_base64"] = [image_to_base64(c) for c in crops_pil]

    return result


@app.post("/detect_crop")
async def detect_crop(request: Request):
    """
    检测图片中的商品，返回检测框坐标和裁切子图。

    请求方式（三选一）：
    - application/json: {"image": "base64_string", "return_crops": true, "crop_format": "base64"}
    - multipart/form-data: 上传 file
    - application/x-www-form-urlencoded: 传入 url
    """
    image_data, return_crops, crop_format = await _load_image_from_request(request)
    return await _run_detect_crop(image_data, return_crops, crop_format)


@app.post("/detect_only")
async def detect_only(request: Request):
    """仅检测，不返回裁切子图。"""
    image_data, _, _ = await _load_image_from_request(request)
    return await _run_detect_crop(image_data, return_crops=False, crop_format="none")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_PATH,
        "conf_threshold": CONFIDENCE_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=DETECT_CROP_HOST, port=DETECT_CROP_PORT)
