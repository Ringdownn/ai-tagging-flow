"""
业务中台商品打标入口
====================
对外暴露统一的 `tag_image` / `tag_image_bytes` 异步接口。
"""
import base64
import io
import uuid
from pathlib import Path

from .config import MAX_SIDE
from .graph import tagging_graph
from .tools import preprocess_image_bytes


def _format_result(result: dict) -> dict:
    # 提取工具调用链，兼容 None / 空列表 / 多商品模式
    tool_outputs = result.get("tool_outputs") or {}
    tool_chain = tool_outputs.get("tool_chain") or []
    return {
        "tags": result.get("final_tags", {}),
        "tags_list": result.get("final_tags_list"),
        "branch": result.get("branch", "unknown"),
        "confidence": result.get("local_confidence", 0.0),
        "complexity": result.get("local_complexity", 0.0),
        "is_product": result.get("is_product", True),
        "original_size": result.get("original_size"),
        "processed_size": result.get("processed_size"),
        "tools": tool_chain,
        "raw_state": result,
    }


async def tag_image(image_url: str) -> dict:
    """
    为商品图片 URL 生成标准化标签。
    """
    result = await tagging_graph.ainvoke({"image_url": image_url})
    return _format_result(result)


async def tag_image_bytes(image_data: bytes) -> dict:
    """
    为上传的图片字节流生成标准化标签。
    """
    from PIL import Image

    preprocessed = await preprocess_image_bytes(image_data, max_side=MAX_SIDE)

    tmp_dir = Path(__file__).resolve().parent.parent.parent / "temp" / "orchestration"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{uuid.uuid4().hex}.jpg"

    pil_image = Image.open(io.BytesIO(base64.b64decode(preprocessed["image_b64"])))
    pil_image.save(tmp_path, quality=95)

    result = await tagging_graph.ainvoke({
        "image_url": "upload://local",
        "image_b64": preprocessed["image_b64"],
        "image_path": str(tmp_path),
        "original_size": preprocessed["original_size"],
        "processed_size": preprocessed["processed_size"],
    })
    return _format_result(result)


__all__ = ["tag_image", "tag_image_bytes", "tagging_graph"]
