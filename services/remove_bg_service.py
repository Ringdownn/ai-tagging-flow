"""
商品抠图服务 - RMBG-2.0 (ONNX)
==============================
基于 briaai/RMBG-2.0 ONNX 模型，返回透明背景商品图。

依赖:
  pip install onnxruntime numpy pillow fastapi uvicorn requests

启动:
  python remove_bg_service.py
  # 或: uvicorn remove_bg_service:app --host 0.0.0.0 --port 8001
"""

import io
import os
import sys
import time
import base64
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import Response
from PIL import Image

# 将项目根目录加入 Python 路径，确保能导入 config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    RMBG_DIR,
    REMOVE_BG_HOST,
    REMOVE_BG_PORT,
    MAX_SIDE,
)

# ==================== 配置 ====================
RMBG_ONNX_MODEL = os.getenv("RMBG_ONNX_MODEL", "model_int8.onnx")
ONNX_MODEL_PATH = str(RMBG_DIR / "onnx" / RMBG_ONNX_MODEL)
IMAGE_SIZE = (1024, 1024)

# ==================== 模型加载 ====================

class BackgroundRemover:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        print(f"[INFO] 加载 RMBG-2.0 ONNX 模型 ({model_path}) ...")
        available_providers = ort.get_available_providers()
        print(f"[INFO] 可用 providers: {available_providers}")

        # macOS 上 CoreML 首次编译极慢，优先使用 CPU；CUDA 环境可改为 GPU
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available_providers]
        if not providers:
            providers = ["CPUExecutionProvider"]
        print(f"[INFO] 使用 providers: {providers}")

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[INFO] 模型输入: {self.session.get_inputs()[0].shape}")
        print("[INFO] RMBG-2.0 ONNX 模型加载完成")

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        """预处理为模型输入。"""
        image = pil_image.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
        image_np = np.array(image).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std

        # HWC -> CHW
        image_np = np.transpose(image_np, (2, 0, 1))
        # 添加 batch 维度
        return np.expand_dims(image_np, axis=0).astype(np.float32)

    def remove(self, pil_image: Image.Image) -> Image.Image:
        """移除背景，返回带透明通道的 RGBA 图片。"""
        original_size = pil_image.size

        input_tensor = self._preprocess(pil_image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        pred = outputs[0][0, 0]  # (1024, 1024)

        # sigmoid + resize回原图
        mask = (1 / (1 + np.exp(-pred))).astype(np.float32)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(original_size, Image.LANCZOS)

        output = pil_image.convert("RGBA")
        output.putalpha(mask_pil)
        return output


# ==================== FastAPI ====================

app = FastAPI(title="商品抠图服务 - RMBG-2.0 ONNX")
_remover: BackgroundRemover | None = None


def get_remover() -> BackgroundRemover:
    """懒加载模型，避免三服务同时启动时内存峰值过高。"""
    global _remover
    if _remover is None:
        _remover = BackgroundRemover(ONNX_MODEL_PATH)
    return _remover


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


def image_to_base64(pil_image: Image.Image, fmt: str = "PNG") -> str:
    """PIL Image 转 base64 字符串。"""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def base64_to_image(image_b64: str) -> Image.Image:
    """base64 字符串转 PIL Image。"""
    return Image.open(io.BytesIO(base64.b64decode(image_b64)))


@app.post("/remove_bg")
async def remove_bg(request: Request):
    """
    移除图片背景，返回透明背景商品图。

    请求方式（三选一）：
    - application/json: {"image": "base64_string"}
    - multipart/form-data: 上传 file
    - application/x-www-form-urlencoded: 传入 url 或 image(base64)

    返回:
    - return_format=base64: JSON {"image": "base64_string"}
    - return_format=image/png: 直接返回 PNG 图片
    """
    start_time = time.time()
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        image_b64 = body.get("image")
        if not image_b64:
            raise HTTPException(400, "请提供 image(base64)")
        pil_image = base64_to_image(image_b64)
        return_format = body.get("return_format", "base64")
    else:
        form = await request.form()
        file = form.get("file")
        url = form.get("url")
        image = form.get("image")
        return_format = str(form.get("return_format", "base64"))

        if file is not None and hasattr(file, "read"):
            image_data = await file.read()
            pil_image = Image.open(io.BytesIO(image_data))
        elif url:
            resp = requests.get(str(url), timeout=10)
            resp.raise_for_status()
            pil_image = Image.open(io.BytesIO(resp.content))
        elif image:
            pil_image = base64_to_image(str(image))
        else:
            raise HTTPException(400, "请提供 file、url 或 image(base64)")

    original_size = pil_image.size
    pil_image = limit_max_side(pil_image, MAX_SIDE)

    # 3. 抠图
    try:
        output = get_remover().remove(pil_image)
    except Exception as e:
        raise HTTPException(500, f"抠图推理失败: {e}")

    elapsed_ms = int((time.time() - start_time) * 1000)

    # 4. 返回
    if return_format in ("image", "png"):
        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"X-Process-Time-ms": str(elapsed_ms)},
        )

    return {
        "original_size": f"{original_size[0]}x{original_size[1]}",
        "processed_size": f"{pil_image.width}x{pil_image.height}",
        "image": image_to_base64(output, fmt="PNG"),
        "process_time_ms": elapsed_ms,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": ONNX_MODEL_PATH,
        "max_side": MAX_SIDE,
    }


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=REMOVE_BG_HOST, port=REMOVE_BG_PORT)
