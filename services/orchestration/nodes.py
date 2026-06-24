"""
LangGraph 节点实现（多 Agent 协作版）
======================================
- preprocess_node: 图片预处理
- supervisor_node: 调度 LocalAgent / HardCaseAgent
- error_node: 错误降级
"""
import base64
import io
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from .state import TaggingState
from .config import MAX_SIDE, MULTI_OBJECT_MODE
from .tools import preprocess_image
from .agents import Supervisor


async def preprocess_node(state: TaggingState) -> dict[str, Any]:
    """下载、压缩图片，生成 base64，并保存临时文件供本地模型使用。"""
    try:
        if state.get("image_b64") and state.get("image_path"):
            return {**state}

        preprocessed = await preprocess_image(state["image_url"], max_side=MAX_SIDE)

        # 将 base64 保存到临时文件，适配 Qwen2.5-VL processor
        tmp_dir = Path(__file__).resolve().parent.parent.parent / "temp" / "orchestration"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "local_input.jpg"

        pil_image = Image.open(io.BytesIO(base64.b64decode(preprocessed["image_b64"])))
        pil_image.save(tmp_path, quality=95)

        return {
            **state,
            "image_b64": preprocessed["image_b64"],
            "image_path": str(tmp_path),
            "original_size": preprocessed["original_size"],
            "processed_size": preprocessed["processed_size"],
        }
    except Exception as e:
        traceback.print_exc()
        return {
            **state,
            "error": f"预处理失败: {e}",
            "branch": "error",
        }


async def supervisor_node(state: TaggingState) -> dict[str, Any]:
    """
    Supervisor 节点：调度 LocalAgent 和 HardCaseAgent。

    路由逻辑:
    - 非商品图 → 返回 "unknown" 标签，不调用 HardCaseAgent
    - 商品图且置信度足够 → 返回本地模型标签
    - 商品图但置信度不足 → 交给 HardCaseAgent (GLM-4V + 工具)
    """
    if state.get("error"):
        return state

    try:
        supervisor = Supervisor()
        result = await supervisor.tag(
            image_b64=state["image_b64"],
            image_path=state["image_path"],
        )

        output = {
            **state,
            "final_tags": result["tags"],
            "branch": result["branch"],
            "local_confidence": result["confidence"],
            "local_complexity": result["complexity"],
            "is_product": result.get("is_product", True),
            "tool_outputs": {"tool_chain": result.get("tool_chain")},
        }

        # 多商品模式：如果 HardCaseAgent 使用了 detect_crop 且返回多个子图
        if (
            MULTI_OBJECT_MODE
            and result["branch"] == "glm4v"
            and result.get("tool_chain")
        ):
            # 需要重新调用 HardCaseAgent 获取 candidates 并对每个子图打标
            # 这里通过 glm_agent 直接执行多商品分支
            from .glm_agent import glm_tagging_agent, generate_tags

            agent_result = await glm_tagging_agent(
                state["image_b64"],
                existing=state.get("local_tags") or result["tags"],
            )
            crop_candidates = [
                c for c in agent_result["candidates"]
                if c["name"].startswith("crop")
            ]
            if len(crop_candidates) > 1:
                print(f"[INFO] 多商品模式：对 {len(crop_candidates)} 个裁切子图并发打标")
                tasks = [
                    generate_tags(c["image_b64"], existing=result["tags"])
                    for c in crop_candidates
                ]
                import asyncio
                tags_list = await asyncio.gather(*tasks)
                output["final_tags_list"] = tags_list
                output["tool_outputs"] = {
                    "tool_chain": agent_result["tool_chain"],
                    "candidates": [c["name"] for c in agent_result["candidates"]],
                }

        return output
    except Exception as e:
        traceback.print_exc()
        return {
            **state,
            "error": f"Supervisor 调度失败: {e}",
            "branch": "error",
        }


async def error_node(state: TaggingState) -> dict[str, Any]:
    """错误节点，返回降级结果，对齐数据集格式。"""
    return {
        **state,
        "final_tags": {
            "类目": ["未知"],
            "颜色": ["未知"],
            "材质/面料": "未知",
            "款式特征": "未知",
            "成色": "未知",
            "置信度": 0.0,
            "复杂度": 0.0,
        },
        "branch": "error",
    }
