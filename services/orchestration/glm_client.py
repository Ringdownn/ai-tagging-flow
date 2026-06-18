"""
GLM-4V 客户端公共函数
====================
与 scripts/prepare_hf_datasets.py 保持对齐：
- 提示词模板
- safe_str 字段处理
- JSON 解析清洗

注意：此处只保留与推理相关的函数，避免与打标脚本产生实现分歧。
"""
import json
import base64
import time
from typing import Any

import requests

TARGET_LABEL_FIELDS = ["类目", "颜色", "材质/面料", "款式特征", "成色", "置信度", "复杂度"]


def safe_str(value: Any) -> str:
    """将字段值安全转为字符串。"""
    if value is None:
        return "未知"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    return str(value)


def encode_image_to_base64(image_path: str) -> str:
    """将图片转为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_b64_to_data_url(image_b64: str, mime: str = "image/jpeg") -> str:
    """base64 字符串转 GLM-4V 需要的 data URL。"""
    return f"data:{mime};base64,{image_b64}"


def build_label_prompt(existing: dict | None = None) -> tuple[str, str]:
    """
    构造 VLM 打标提示词。
    与 prepare_hf_datasets.py 中的 build_prompt_from_existing 对齐。
    """
    existing = existing or {}

    system_prompt = (
        "你是一位专业的二手商品图片标注员。请根据图片及参考信息，"
        "输出简洁、标准化的二手商品特征标签。所有标签必须使用中文。"
        "无法判断的字段填\"未知\"。"
    )

    reference = f"""
参考信息（可能不完整或格式不统一，请以图片为准并修正）：
- 类目：{safe_str(existing.get('category', existing.get('type', '未知')))}
- 颜色：{safe_str(existing.get('colors', existing.get('color', '未知')))}
- 材质/面料：{safe_str(existing.get('material', '未知'))}
- 款式特征：{safe_str(existing.get('type', existing.get('cut', existing.get('pattern', '未知'))))}
- 成色：{safe_str(existing.get('condition', existing.get('成色', '未知')))}
""".strip()

    user_prompt = (
        "请仔细观察图片，输出以下 7 个字段，直接以 JSON 格式返回，不要多余解释。\n"
        "注意：所有标签值必须是中文。\n\n"
        "类目标签规则（非常重要）：\n"
        "1. 类目需要按层级从粗到细输出，最多 5 层，最少 1 层。\n"
        "2. 每一层都必须是你在图片中高度确信、置信度高的标签，不要为了凑数硬加。\n"
        "3. 格式：类目字段用英文逗号分隔多个层级，例如：服饰,女装,连衣裙。\n"
        "4. 如果只能判断到大类，就只输出一层，例如：服饰。\n"
        "5. 如果图片里同时有多个独立商品，选择最主体的一个商品打标。\n\n"
        "颜色标签规则：\n"
        "如果图片只能判断出一个明确颜色，就只填一个；如果能判断出多个不同颜色（1到5个），用英文逗号分隔。\n\n"
        "置信度规则：\n"
        "输出 0.0 ~ 1.0 之间的数值，表示你对整组标签的整体确信程度。\n\n"
        "复杂度规则：\n"
        "输出 0.0 ~ 1.0 之间的数值，表示图片场景的复杂程度。\n"
        "- 接近 0.0：单主体、背景干净（白底/纯色）、光线清晰、主体完整。\n"
        "- 接近 0.5：单主体但背景较复杂，或多主体但主体仍较清晰。\n"
        "- 接近 1.0：多主体、背景杂乱、遮挡严重、光线差、主体不明确。\n\n"
        "其他字段（材质/面料、款式特征、成色）各填一个。\n"
        f"字段列表：{json.dumps(TARGET_LABEL_FIELDS, ensure_ascii=False)}\n\n"
        f"{reference}\n\n"
        "示例返回格式：\n"
        "{\n"
        '  "类目": "服饰,女装,连衣裙",\n'
        '  "颜色": "黑色",\n'
        '  "材质/面料": "聚酯纤维",\n'
        '  "款式特征": "简约通勤",\n'
        '  "成色": "九成新",\n'
        '  "置信度": 0.95,\n'
        '  "复杂度": 0.1\n'
        "}"
    )
    return system_prompt, user_prompt


def parse_vlm_json(content: str) -> dict:
    """
    解析 GLM-4V 返回的 JSON。
    与 prepare_hf_datasets.py 中的清洗逻辑对齐。
    """
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    return json.loads(content)


def call_glm4v(
    api_key: str,
    model: str,
    api_url: str,
    image_b64: str,
    existing: dict | None = None,
    max_retries: int = 3,
) -> dict:
    """
    调用 GLM-4V API 生成标准化标签。
    与 prepare_hf_datasets.py 中的 call_vlm 对齐。
    """
    system_prompt, user_prompt = build_label_prompt(existing)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64_to_data_url(image_b64)},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.post(api_url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return parse_vlm_json(content)
        except Exception as e:
            last_error = e
            print(f"[WARN] GLM-4V 调用失败（尝试 {attempt + 1}/{max_retries}）: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"GLM-4V 多次调用失败: {last_error}")
