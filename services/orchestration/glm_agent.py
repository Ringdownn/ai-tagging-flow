"""
GLM-4V 工具决策 Agent
=====================
让 GLM-4V 自主判断疑难图需要哪些预处理工具，再执行并打标。

调用链:
  1. decide_tools(image_b64) -> 工具列表
  2. run_tools(image_b64, tools) -> 候选图片列表
  3. generate_tags(image_b64, existing) -> 标签

支持的工具:
  - detect_crop: 检测并裁切商品主体
  - remove_bg: 移除背景
  - []: 不需要工具，直接打标
"""

import json
from typing import Any

from .config import ZHIPU_API_KEY, ZHIPU_MODEL, ZHIPU_URL
from .glm_client import image_b64_to_data_url, parse_vlm_json
from .tools import detect_and_crop, remove_background


def _call_glm_for_decision(image_b64: str, prompt: str) -> str:
    """调用 GLM-4V 获取决策/标签文本。"""
    import requests

    payload = {
        "model": ZHIPU_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64_to_data_url(image_b64)},
                    },
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(ZHIPU_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _parse_tool_list(content: str) -> list[str]:
    """从 GLM 输出中解析工具列表。"""
    content = content.strip()

    # 尝试 JSON 解析
    try:
        tools = json.loads(content)
        if isinstance(tools, list):
            return [t for t in tools if t in ("detect_crop", "remove_bg")]
    except json.JSONDecodeError:
        pass

    # 兜底：关键词匹配
    content_lower = content.lower()
    tools = []
    if "detect_crop" in content_lower or "detect" in content_lower or "裁切" in content:
        tools.append("detect_crop")
    if "remove_bg" in content_lower or "remove" in content_lower or "抠图" in content or "背景" in content:
        tools.append("remove_bg")
    return tools


async def decide_tools(image_b64: str) -> list[str]:
    """
    让 GLM-4V 决定需要调用哪些工具。

    返回:
        工具名列表，例如 ["detect_crop", "remove_bg"] 或 []
    """
    prompt = (
        "你是一位专业的二手商品图片分析助手。请观察图片，判断是否需要调用预处理工具来辅助打标。\n\n"
        "可用工具：\n"
        "1. detect_crop：检测图片中的商品主体并裁切，适用于主体不明确、多主体、背景杂乱的图片\n"
        "2. remove_bg：移除背景，适用于背景干扰严重的图片\n\n"
        "规则：\n"
        "- 如果图片主体清晰、背景简单，直接输出 []\n"
        "- 如果背景杂乱或主体边缘不清晰，可以输出 [\"remove_bg\"]\n"
        "- 如果主体不明确、多主体或背景杂物很多，可以输出 [\"detect_crop\"]\n"
        "- 如果同时需要定位和去背景，可以输出 [\"detect_crop\", \"remove_bg\"]\n\n"
        "请直接输出一个 JSON 数组，不要有任何额外解释。例如：\n"
        "[]\n"
        "[\"remove_bg\"]\n"
        "[\"detect_crop\"]\n"
        "[\"detect_crop\", \"remove_bg\"]"
    )

    content = _call_glm_for_decision(image_b64, prompt)
    tools = _parse_tool_list(content)
    print(f"[INFO] GLM Agent 决策工具: {tools}")
    return tools


async def run_tools(image_b64: str, tools: list[str]) -> list[dict[str, Any]]:
    """
    执行 GLM-4V 决定的工具链。

    返回:
        候选图片列表，每项为 {"name": str, "image_b64": str}
    """
    candidates = [{"name": "original", "image_b64": image_b64}]

    if "detect_crop" in tools:
        try:
            crops = await detect_and_crop(image_b64)
            if crops:
                candidates.append({"name": "crop_best", "image_b64": crops[0]})

                if "remove_bg" in tools:
                    try:
                        removed = await remove_background(crops[0])
                        candidates.append({"name": "crop_removed_bg", "image_b64": removed})
                        print("[INFO] Agent 执行: detect_crop + remove_bg(最佳子图)")
                    except Exception as e:
                        print(f"[WARN] 最佳子图抠图失败: {e}")
                else:
                    print("[INFO] Agent 执行: detect_crop")
            else:
                print("[WARN] detect_crop 未返回子图")
        except Exception as e:
            print(f"[WARN] detect_crop 执行失败: {e}")

        # 如果只需要 remove_bg 且没有 detect_crop，对原图抠图
        if "remove_bg" in tools and len(candidates) == 1:
            try:
                removed = await remove_background(image_b64)
                candidates.append({"name": "removed_bg", "image_b64": removed})
                print("[INFO] Agent 执行: remove_bg(原图)")
            except Exception as e:
                print(f"[WARN] 原图抠图失败: {e}")
    elif "remove_bg" in tools:
        try:
            removed = await remove_background(image_b64)
            candidates.append({"name": "removed_bg", "image_b64": removed})
            print("[INFO] Agent 执行: remove_bg(原图)")
        except Exception as e:
            print(f"[WARN] 原图抠图失败: {e}")
    else:
        print("[INFO] Agent 决策: 无需工具，直接打标")

    return candidates


async def generate_tags(
    image_b64: str,
    existing: dict | None = None,
) -> dict:
    """
    对最终选定的图片生成标准化标签。
    复用 glm_client.call_glm4v，与打标脚本对齐。
    """
    from .glm_client import call_glm4v

    return call_glm4v(
        api_key=ZHIPU_API_KEY,
        model=ZHIPU_MODEL,
        api_url=ZHIPU_URL,
        image_b64=image_b64,
        existing=existing,
    )


async def glm_tagging_agent(
    image_b64: str,
    existing: dict | None = None,
) -> dict[str, Any]:
    """
    完整的 GLM 打标 Agent 入口。

    返回:
        {
            "tags": {"类目": ..., ...},
            "tool_chain": ["detect_crop", "remove_bg"],
            "candidates": [{"name": ..., "image_b64": ...}],
        }
    """
    tools = await decide_tools(image_b64)
    candidates = await run_tools(image_b64, tools)

    # 选择最后一张候选图（质量最高）
    selected = candidates[-1]["image_b64"]
    tags = await generate_tags(selected, existing=existing)

    return {
        "tags": tags,
        "tool_chain": tools,
        "candidates": candidates,
    }
