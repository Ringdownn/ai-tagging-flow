"""
本地工具调用封装
================
通过 HTTP 调用本地 FastAPI 工具服务：
- 抠图服务 (RMBG-2.0): /remove_bg
- 检测裁切服务 (YOLOv8): /detect_crop
"""
import base64
import io
from typing import Any

import httpx
from PIL import Image

from .config import REMOVE_BG_URL, DETECT_CROP_URL, TOOL_TIMEOUT


def _b64_to_pil(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _pil_to_b64(pil_image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


async def remove_background(image_b64: str) -> str:
    """
    调用 RMBG-2.0 抠图服务，返回抠图后的 base64 图片。

    接口约定（需与 remove_bg_service.py 保持一致）:
        POST /remove_bg
        Body: {"image": "base64_string"}
        Resp: {"image": "base64_string"}
    """
    async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
        resp = await client.post(
            f"{REMOVE_BG_URL}/remove_bg",
            json={"image": image_b64},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["image"]


async def detect_and_crop(image_b64: str) -> list[str]:
    """
    调用 YOLOv8 检测裁切服务，返回裁切子图 base64 列表。

    接口约定（detect_crop_service.py 已实现）:
        POST /detect_crop
        Body: {"image": "base64_string"} 或 multipart file/url
        Resp: {"detections": [...], "crops_base64": [...]}
    """
    async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
        resp = await client.post(
            f"{DETECT_CROP_URL}/detect_crop",
            json={"image": image_b64},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("crops_base64", [])


def _preprocess_pil_image(pil_image: Image.Image, max_side: int) -> dict[str, Any]:
    """将 PIL 图片压缩并转为 base64。"""
    original_size = pil_image.size
    w, h = pil_image.size
    if max(w, h) > max_side:
        if w > h:
            new_w = max_side
            new_h = int(h * max_side / w)
        else:
            new_h = max_side
            new_w = int(w * max_side / h)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

    return {
        "image_b64": _pil_to_b64(pil_image),
        "original_size": original_size,
        "processed_size": pil_image.size,
    }


async def preprocess_image_bytes(image_data: bytes, max_side: int = 768) -> dict[str, Any]:
    """从本地字节流预处理图片。"""
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return _preprocess_pil_image(pil_image, max_side)


async def preprocess_image(image_url: str, max_side: int = 768) -> dict[str, Any]:
    """
    下载图片并做统一预处理：
    - 限制最大边长
    - 转 RGB
    - 输出 base64

    返回:
        {
            "image_b64": base64 字符串,
            "original_size": (w, h),
            "processed_size": (w, h),
        }
    """
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(image_url, timeout=30) as resp:
            resp.raise_for_status()
            image_data = await resp.read()

    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return _preprocess_pil_image(pil_image, max_side)

