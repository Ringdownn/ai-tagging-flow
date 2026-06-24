"""
下载并预处理开源二手商品数据集，生成 GLM-4V 打标输入 / LLaMA Factory 训练样本。

数据集:
- Nilanjan-2002/fashion-second-hand-front-only-rgb (31.6k 张，带结构化标签，本脚本取 30 张)
- NingLab/MMECInstruct (多任务指令数据，含 category_classification 等子集，本脚本取 70 张)
- ./商品照片 (本地商品照片，自动复制并重命名编号)

输出:
- data/open_datasets/images/      下载/复制的图片
- data/open_datasets/labels/      单条样本、汇总、训练集格式及 image_label_mapping.json

使用方式:
1. 设置环境变量 ZHIPU_API_KEY
2. 安装依赖: pip install datasets requests Pillow tqdm
3. 运行: python scripts/prepare_hf_datasets.py
"""

import os
import json
import base64
import io
import shutil
import time
import requests
from pathlib import Path
from urllib.parse import urlparse

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ==================== 配置 ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "open_datasets"
IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"

# GLM-4V API（智谱）
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPU_MODEL = "glm-4v-plus"

# 采样数量
FASHION_SAMPLE_SIZE = 30
MMEC_CATEGORY_SAMPLE_SIZE = 70  # 仅取 category_classification 任务

# 本地商品照片目录
LOCAL_PRODUCT_DIR = PROJECT_ROOT / "商品照片"

# 负例图目录（非电商场景：风景、人物、模糊、文字等）
NEGATIVE_DIR = PROJECT_ROOT / "负例图"

# 目标标签字段
TARGET_LABEL_FIELDS = ["类目", "颜色", "材质/面料", "款式特征", "成色", "置信度", "复杂度"]

# 统一的训练 instruction
INSTRUCTION = (
    "你是一位专业的二手电商平台商品标注专家。请仔细观察用户上传的图片，输出标准化、结构化的商品标签。\n\n"
    "## 输出字段（共 7 个）\n"
    "1. 类目：层级类目数组，从粗到细，最多 5 层，最少 1 层。每一层必须是你高确信、高置信度的标签，不要为了凑数硬加。\n"
    "2. 颜色：颜色数组。如果只有一个主颜色，数组长度为 1；如果有多个明显颜色，最多列出 5 个。\n"
    "3. 材质/面料：单一字符串。\n"
    "4. 款式特征：单一字符串，描述商品的款式、功能或风格特点。\n"
    "5. 成色：单一字符串，描述商品新旧程度。\n"
    "6. 置信度：0.0 ~ 1.0 之间的数值，表示你对整组标签的整体确信程度。\n"
    "7. 复杂度：0.0 ~ 1.0 之间的数值，表示图片场景的复杂程度。\n"
    "   - 接近 0.0：单主体、背景干净（白底或纯色）、光线清晰、主体完整。\n"
    "   - 接近 0.5：单主体但背景较复杂，或多主体但主体仍较清晰。\n"
    "   - 接近 1.0：多主体、背景杂乱、遮挡严重、光线差、主体不明确。\n\n"
    "## 输出格式要求\n"
    "- 仅输出合法的 JSON 对象，不要添加 markdown 代码块、不要添加解释说明。\n"
    "- 所有标签值必须是中文。\n"
    "- 类目和颜色字段为字符串数组，其他字段为字符串或数值。\n"
    "- 无法判断的字段填写\"未知\"。\n"
    "- 如果图片中有多个商品，只对最主体、最清晰的商品进行打标。\n"
    "- 如果图片明显不是二手商品（如风景、人物、动物、纯文字、表情包、模糊图、建筑等），"
    '  必须输出：{"类目": ["未知"], "颜色": ["未知"], "材质/面料": "未知", "款式特征": "未知", "成色": "未知", "置信度": 0.0, "复杂度": 0.0}\n\n'
    "## 示例 1（女装连衣裙，简单背景）\n"
    '{\n  "类目": ["服饰", "女装", "连衣裙"],\n  "颜色": ["黑色"],\n  "材质/面料": "聚酯纤维",\n  "款式特征": "简约通勤",\n  "成色": "九成新",\n  "置信度": 0.95,\n  "复杂度": 0.1\n}\n\n'
    "## 示例 2（电子产品游戏手柄，复杂背景）\n"
    '{\n  "类目": ["电子产品", "游戏设备", "游戏手柄", "电脑配件", "游戏配件"],\n  "颜色": ["黑色"],\n  "材质/面料": "塑料",\n  "款式特征": "有线手柄",\n  "成色": "九成新",\n  "置信度": 0.92,\n  "复杂度": 0.85\n}\n\n'
    "## 示例 3（食品，中等背景）\n"
    '{\n  "类目": ["食品", "休闲零食", "坚果炒货"],\n  "颜色": ["棕色"],\n  "材质/面料": "未知",\n  "款式特征": "袋装混合坚果",\n  "成色": "全新",\n  "置信度": 0.88,\n  "复杂度": 0.5\n}\n\n'
    "## 示例 4（非商品图）\n"
    '{\n  "类目": ["未知"],\n  "颜色": ["未知"],\n  "材质/面料": "未知",\n  "款式特征": "未知",\n  "成色": "未知",\n  "置信度": 0.0,\n  "复杂度": 0.0\n}\n\n'
    "请为这张图片生成标准化标签。"
)


def ensure_dirs():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)


def encode_image_to_base64(image_path: Path) -> str:
    """将图片转为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_image(image, filename: str) -> Path:
    """保存 PIL Image 到本地，如果已存在则跳过。"""
    path = IMAGE_DIR / filename
    if path.exists():
        return path
    if isinstance(image, dict):  # datasets 中 image 字段有时为 dict
        image = image["bytes"]
        image = Image.open(io.BytesIO(image))
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    image.save(path)
    return path


def download_image(url: str, filename: str, timeout: int = 10) -> Path | None:
    """从 URL 下载图片，如果已存在则跳过。"""
    path = IMAGE_DIR / filename
    if path.exists():
        return path
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        path = IMAGE_DIR / filename
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception as e:
        print(f"[WARN] 下载失败 {url}: {e}")
        return None


def safe_str(value):
    """将字段值安全转为字符串。"""
    if value is None:
        return "未知"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    return str(value)


def build_prompt_from_existing(existing: dict, is_negative: bool = False) -> tuple[str, str]:
    """
    构造 VLM 打标提示词。
    已有标签作为上下文传入，让模型做校验 / 标准化 / 补全。

    is_negative: 是否为已知负例图（非电商场景）。
                 若为 True，强制要求模型输出全未知 + 置信度 0.0，用于训练稳定学习拒绝能力。
    """
    system_prompt = (
        "你是一位专业的二手商品图片标注员。请根据图片及参考信息，"
        "输出简洁、标准化的二手商品特征标签。所有标签必须使用中文。"
        "无法判断的字段填\"未知\"。"
    )

    # 将已有字段映射到目标字段
    reference = f"""
参考信息（可能不完整或格式不统一，请以图片为准并修正）：
- 类目：{safe_str(existing.get('category', existing.get('type', '未知')))}
- 颜色：{safe_str(existing.get('colors', existing.get('color', '未知')))}
- 材质/面料：{safe_str(existing.get('material', '未知'))}
- 款式特征：{safe_str(existing.get('type', existing.get('cut', existing.get('pattern', '未知'))))}
- 成色：{safe_str(existing.get('condition', existing.get('成色', '未知')))}
""".strip()

    if is_negative:
        user_prompt = (
            "这是一张负例图，图片内容不属于二手电商平台可交易的商品场景（可能是风景、人物、动物、"
            "纯文字、表情包、模糊图、建筑等，但食品属于可交易商品）。\n\n"
            "请直接输出以下 JSON，表示无法识别商品：\n"
            "{\n"
            '  "类目": "未知",\n'
            '  "颜色": "未知",\n'
            '  "材质/面料": "未知",\n'
        '  "款式特征": "未知",\n'
        '  "成色": "未知",\n'
        '  "置信度": 0.0,\n'
        '  "复杂度": 0.0\n'
        "}\n\n"
        "不要输出任何其他解释。"
        )
        return system_prompt, user_prompt

    user_prompt = (
        "请仔细观察图片，输出以下 7 个字段，直接以 JSON 格式返回，不要多余解释。\n"
        "注意：所有标签值必须是中文。\n\n"
        "类目标签规则（非常重要）：\n"
        "1. 类目需要按层级从粗到细输出，最多 5 层，最少 1 层。\n"
        "2. 每一层都必须是你在图片中高度确信、置信度高的标签，不要为了凑数硬加。\n"
        "3. 格式：类目字段用英文逗号分隔多个层级，例如：服饰,女装,连衣裙 或 电子产品,游戏设备,游戏手柄,电脑配件,游戏配件。\n"
        "4. 如果只能判断到大类，就只输出一层，例如：服饰。\n"
        "5. 如果图片里同时有多个独立商品，选择最主体的一个商品打标。\n"
        "6. 如果图片明显不是二手商品（如风景、人物、动物、纯文字、表情包、模糊图、建筑等），"
        '   请输出：{"类目": "未知", "颜色": "未知", "材质/面料": "未知", "款式特征": "未知", "成色": "未知", "置信度": 0.0, "复杂度": 0.0}\n\n'
        "颜色标签规则：\n"
        "如果图片只能判断出一个明确颜色，就只填一个；如果能判断出多个不同颜色（1到5个），用英文逗号分隔。\n\n"
        "置信度规则：\n"
        "输出 0.0 ~ 1.0 之间的数值，表示你对整组标签的整体确信程度。非商品图填 0.0，非常确定填 1.0。\n\n"
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
        "}\n\n"
        "示例（多层级类目）：\n"
        "{\n"
        '  "类目": "电子产品,游戏设备,游戏手柄,电脑配件,游戏配件",\n'
        '  "颜色": "黑色",\n'
        '  "材质/面料": "塑料",\n'
        '  "款式特征": "有线手柄",\n'
        '  "成色": "九成新",\n'
        '  "置信度": 0.92,\n'
        '  "复杂度": 0.85\n'
        "}\n\n"
        "示例（非商品图）：\n"
        "{\n"
        '  "类目": "未知",\n'
        '  "颜色": "未知",\n'
        '  "材质/面料": "未知",\n'
        '  "款式特征": "未知",\n'
        '  "成色": "未知",\n'
        '  "置信度": 0.0,\n'
        '  "复杂度": 0.0\n'
        "}"
    )
    return system_prompt, user_prompt


def call_vlm(image_path: Path, existing: dict, max_retries: int = 3, is_negative: bool = False) -> dict:
    """调用 GLM-4V API 生成标准化标签。"""
    if not ZHIPU_API_KEY:
        raise EnvironmentError("请先设置环境变量 ZHIPU_API_KEY")

    b64_image = encode_image_to_base64(image_path)
    system_prompt, user_prompt = build_prompt_from_existing(existing, is_negative=is_negative)

    payload = {
        "model": ZHIPU_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        },
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(ZHIPU_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            # 尝试从内容中解析 JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content)
        except Exception as e:
            print(f"[WARN] VLM 调用失败（尝试 {attempt + 1}/{max_retries}）: {e}")
            time.sleep(2 ** attempt)

    # 全部失败则回退到已有标签
    print(f"[ERROR] VLM 多次失败，回退到原始标签: {image_path.name}")
    return {
        "类目": safe_str(existing.get("category", existing.get("type", "未知"))),
        "颜色": safe_str(existing.get("colors", existing.get("color", "未知"))),
        "材质/面料": safe_str(existing.get("material", "未知")),
        "款式特征": safe_str(existing.get("type", existing.get("cut", existing.get("pattern", "未知")))),
        "成色": safe_str(existing.get("condition", "未知")),
        "置信度": 0.5,
        "复杂度": 0.5,
    }


def labels_to_structured_text(labels: dict) -> str:
    """把标签字典转成带数组的 JSON 字符串输出，便于模型学习结构化格式。"""
    structured = {}
    for k, v in labels.items():
        if k == "类目":
            # 层级类目用英文逗号分隔，转成字符串数组
            structured[k] = [x.strip() for x in str(v).split(",") if x.strip()]
        elif k == "颜色":
            # 颜色可能是逗号分隔多个
            structured[k] = [x.strip() for x in str(v).split(",") if x.strip()]
        elif k in ("置信度", "复杂度"):
            # 置信度和复杂度保持数值
            try:
                structured[k] = float(v)
            except (ValueError, TypeError):
                structured[k] = 0.0
        else:
            # 其他字段保留单个字符串
            structured[k] = str(v)
    return json.dumps(structured, ensure_ascii=False, indent=2)


def to_llama_factory_format(image_path: Path, labels: dict, instruction: str) -> dict:
    """转换为 LLaMA Factory ShareGPT 训练样本格式，assistant 输出为结构化 JSON。"""
    label_text = labels_to_structured_text(labels)
    return {
        "images": [str(image_path)],
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": label_text},
        ],
    }


def to_alpaca_format(image_path: Path, labels: dict, instruction: str) -> dict:
    """转换为 LLaMA Factory Alpaca 训练样本格式。"""
    label_text = labels_to_structured_text(labels)
    return {
        "instruction": instruction,
        "input": "",
        "output": label_text,
        "images": [str(image_path)],
    }


# ==================== 数据集 1: fashion-second-hand-front-only-rgb ====================

def process_fashion_dataset():
    """处理服装二手数据集：已有标签可直接使用，VLM 负责标准化。"""
    print("\n[1/3] 加载 Nilanjan-2002/fashion-second-hand-front-only-rgb (流式加载) ...")
    try:
        ds = load_dataset("Nilanjan-2002/fashion-second-hand-front-only-rgb", split="train", streaming=True)
        ds = ds.take(FASHION_SAMPLE_SIZE)

        records = []

        for idx, item in enumerate(tqdm(ds, desc="处理 fashion 数据集")):
            filename = f"fashion_{idx:04d}.jpg"
            image_path = save_image(item["image"], filename)

            # 数据集中已有的结构化字段，可直接作为初始标签
            existing = {
                "category": item.get("category"),
                "type": item.get("type"),
                "colors": item.get("colors"),
                "material": item.get("material"),
                "condition": item.get("condition"),
                "cut": item.get("cut"),
                "pattern": item.get("pattern"),
                "brand": item.get("brand"),
            }

            # 调用 VLM 做标准化 / 补全
            labels = call_vlm(image_path, existing)

            record = {
                "source": "fashion-second-hand-front-only-rgb",
                "image": str(image_path),
                "raw_labels": existing,
                "vlm_labels": labels,
            }
            records.append(record)

            # 同时保存 LLaMA Factory 格式
            sample = to_llama_factory_format(image_path, labels, INSTRUCTION)
            with open(LABEL_DIR / f"fashion_{idx:04d}.json", "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)

        # 汇总
        with open(LABEL_DIR / "fashion_summary.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[DONE] fashion 数据集: {len(records)} 条样本已保存至 {LABEL_DIR}")
        return records

    except Exception as e:
        print(f"[WARN] 加载或处理 fashion 数据集失败，跳过: {e}")
        return []


# ==================== 数据集 1.5: 本地商品照片 ====================

def process_local_products():
    """处理用户本地商品照片：复制到目标目录并重命名编号，调用 VLM 打标。"""
    if not LOCAL_PRODUCT_DIR.exists():
        print(f"\n[WARN] 本地商品照片目录不存在: {LOCAL_PRODUCT_DIR}，跳过")
        return []

    image_files = sorted([
        p for p in LOCAL_PRODUCT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    if not image_files:
        print(f"\n[WARN] 本地商品照片目录为空: {LOCAL_PRODUCT_DIR}")
        return []

    print(f"\n[2/3] 处理本地商品照片: {len(image_files)} 张 ...")

    records = []

    for idx, src_path in enumerate(tqdm(image_files, desc="处理本地商品照片")):
        ext = src_path.suffix.lower()
        filename = f"local_{idx:04d}{ext}"
        dst_path = IMAGE_DIR / filename

        # 复制到目标目录，已存在则跳过
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        existing = {}  # 本地照片没有已有标签
        labels = call_vlm(dst_path, existing)

        record = {
            "source": "local_product_photos",
            "image": str(dst_path),
            "original_path": str(src_path),
            "vlm_labels": labels,
        }
        records.append(record)

        sample = to_llama_factory_format(dst_path, labels, INSTRUCTION)
        with open(LABEL_DIR / f"local_{idx:04d}.json", "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

    with open(LABEL_DIR / "local_summary.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] 本地商品照片: {len(records)} 条样本已保存至 {LABEL_DIR}")
    return records


# ==================== 数据集 1.6: 负例图 ====================

def process_negative_examples():
    """处理负例图：非电商场景（不含食品），标签全部为未知，置信度 0.0。"""
    if not NEGATIVE_DIR.exists():
        print(f"\n[WARN] 负例图目录不存在: {NEGATIVE_DIR}，跳过")
        return []

    image_files = sorted([
        p for p in NEGATIVE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    if not image_files:
        print(f"\n[WARN] 负例图目录为空: {NEGATIVE_DIR}")
        return []

    print(f"\n[1.6/3] 处理负例图: {len(image_files)} 张 ...")

    records = []

    for idx, src_path in enumerate(tqdm(image_files, desc="处理负例图")):
        ext = src_path.suffix.lower()
        filename = f"negative_{idx:04d}{ext}"
        dst_path = IMAGE_DIR / filename

        # 复制到目标目录，已存在则跳过
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        # 调用 VLM 并明确告知这是负例图（食品除外），让模型学习拒绝输出
        existing = {}
        labels = call_vlm(dst_path, existing, is_negative=True)

        # 兜底：如果 API 未按预期返回全未知，强制覆盖
        if labels.get("类目", "") != "未知":
            labels = {
                "类目": "未知",
                "颜色": "未知",
                "材质/面料": "未知",
                "款式特征": "未知",
                "成色": "未知",
                "置信度": 0.0,
                "复杂度": 0.0,
            }

        record = {
            "source": "negative_examples",
            "image": str(dst_path),
            "original_path": str(src_path),
            "vlm_labels": labels,
        }
        records.append(record)

        sample = to_llama_factory_format(dst_path, labels, INSTRUCTION)
        with open(LABEL_DIR / f"negative_{idx:04d}.json", "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

    with open(LABEL_DIR / "negative_summary.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] 负例图: {len(records)} 条样本已保存至 {LABEL_DIR}")
    return records


# ==================== 数据集 2: MMECInstruct ====================

def process_mmec_dataset():
    """
    处理 MMECInstruct 数据集。
    该数据集为指令格式，包含 category_classification / sentiment_analysis 等任务。
    我们只取 category_classification 子集，利用其图片 + 标题做商品打标。
    若加载失败（网络/SSL 问题），则自动跳过，不影响整体流程。
    """
    print("\n[3/3] 加载 NingLab/MMECInstruct (流式加载) ...")
    try:
        ds = load_dataset("NingLab/MMECInstruct", split="train", streaming=True)
    except Exception as e:
        print(f"[WARN] 加载 MMECInstruct 失败，跳过: {e}")
        return []

    # 仅保留与商品识别相关的任务；不同版本字段名可能不同，这里做兼容处理
    task_field = None
    # 先拿一条样本探测字段名
    try:
        sample = next(iter(ds))
    except Exception as e:
        print(f"[WARN] 读取 MMECInstruct 样本失败，跳过: {e}")
        return []

    for candidate in ["task", "instruction_type", "category"]:
        if candidate in sample:
            task_field = candidate
            break

    if task_field:
        ds = ds.filter(lambda x: str(x.get(task_field, "")).lower() in [
            "category_classification", "product_categorization"
        ])

    ds = ds.take(MMEC_CATEGORY_SAMPLE_SIZE)

    records = []

    for idx, item in enumerate(tqdm(ds, desc="处理 MMEC 数据集")):
        # 获取图片 URL（字段名可能为 images / image / image_url）
        image_urls = item.get("images") or item.get("image") or []
        if isinstance(image_urls, str):
            # 可能是 JSON 数组字符串，如 '["https://..."]'
            try:
                parsed = json.loads(image_urls)
                image_urls = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                image_urls = [image_urls]
        if not image_urls:
            continue

        url = image_urls[0]
        ext = Path(urlparse(url).path).suffix or ".jpg"
        filename = f"mmec_{idx:04d}{ext}"
        image_path = download_image(url, filename)
        if image_path is None:
            continue

        # 尝试提取标题/类别作为参考
        title = item.get("title", "")
        existing = {
            "category": title,  # 用标题辅助类目判断
            "type": title,
        }

        labels = call_vlm(image_path, existing)

        record = {
            "source": "NingLab/MMECInstruct",
            "image": str(image_path),
            "url": url,
            "raw_title": title,
            "vlm_labels": labels,
        }
        records.append(record)

        sample = to_llama_factory_format(image_path, labels, INSTRUCTION)
        with open(LABEL_DIR / f"mmec_{idx:04d}.json", "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

    with open(LABEL_DIR / "mmec_summary.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] MMEC 数据集: {len(records)} 条样本已保存至 {LABEL_DIR}")
    return records


# ==================== 合并为训练集 ====================

def merge_to_dataset_json(records: list[dict]):
    """将所有样本合并为 LLaMA Factory 可用的 ShareGPT 和 Alpaca 格式训练集。"""
    sharegpt_samples = []
    alpaca_samples = []

    for r in records:
        labels = r.get("vlm_labels") or r.get("deepseek_labels", {})
        image_path = Path(r["image"])
        sharegpt_samples.append(
            to_llama_factory_format(image_path, labels, INSTRUCTION)
        )
        alpaca_samples.append(
            to_alpaca_format(image_path, labels, INSTRUCTION)
        )

    # ShareGPT 格式
    sharegpt_path = LABEL_DIR / "train_open_datasets_sharegpt.json"
    with open(sharegpt_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_samples, f, ensure_ascii=False, indent=2)
    print(f"[DONE] ShareGPT 训练集: {sharegpt_path} ({len(sharegpt_samples)} 条)")

    # Alpaca 格式
    alpaca_path = LABEL_DIR / "train_open_datasets_alpaca.json"
    with open(alpaca_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_samples, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Alpaca 训练集: {alpaca_path} ({len(alpaca_samples)} 条)")


def save_image_label_mapping(records: list[dict], output_name: str = "image_label_mapping.json"):
    """生成简洁的图片-标签映射表，便于人工审核和查看。"""
    mapping = []
    for r in records:
        mapping.append({
            "image": r["image"],
            "source": r.get("source", ""),
            "labels": r.get("vlm_labels") or r.get("deepseek_labels", {}),
        })

    output_path = LABEL_DIR / output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 图片-标签映射表: {output_path} ({len(mapping)} 条)")


# ==================== 主入口 ====================

def _load_existing_records(summary_name: str) -> list[dict]:
    """从已有的 summary.jsonl 加载记录，用于增量更新时合并。"""
    path = LABEL_DIR / summary_name
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"[WARN] 读取已有记录失败 {path}: {e}")
        return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description="生成商品打标训练数据集")
    parser.add_argument(
        "--only",
        choices=["fashion", "local", "negative", "mmec", "all"],
        default="all",
        help="只重新处理指定数据源，其余从已有 summary 合并（默认处理全部）",
    )
    args = parser.parse_args()

    ensure_dirs()

    # 默认重新处理；如果被跳过，尝试从已有 summary 加载
    if args.only in ("all", "fashion"):
        fashion_records = process_fashion_dataset()
    else:
        fashion_records = _load_existing_records("fashion_summary.jsonl")

    if args.only in ("all", "local"):
        local_records = process_local_products()
    else:
        local_records = _load_existing_records("local_summary.jsonl")

    if args.only in ("all", "negative"):
        negative_records = process_negative_examples()
    else:
        negative_records = _load_existing_records("negative_summary.jsonl")

    if args.only in ("all", "mmec"):
        mmec_records = process_mmec_dataset()
    else:
        mmec_records = _load_existing_records("mmec_summary.jsonl")

    all_records = fashion_records + local_records + negative_records + mmec_records
    merge_to_dataset_json(all_records)
    save_image_label_mapping(all_records)

    print(f"\n全部完成，共生成 {len(all_records)} 条训练样本。")
    print(f"  - fashion 数据集: {len(fashion_records)}")
    print(f"  - 本地商品照片: {len(local_records)}")
    print(f"  - 负例图: {len(negative_records)}")
    print(f"  - MMEC 数据集: {len(mmec_records)}")
    print(f"图片目录: {IMAGE_DIR}")
    print(f"标签目录: {LABEL_DIR}")


if __name__ == "__main__":
    main()
