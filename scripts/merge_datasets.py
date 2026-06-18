"""
合并新旧训练数据到 train_sharegpt.json（ShareGPT 格式）。
同时提供 dataset_info.json 配置参考。
"""
import json
from pathlib import Path

LABELS_DIR = Path("data/open_datasets/labels")

# 标准 instruction（与训练脚本保持一致）
INSTRUCTION = (
    "你是一位专业的二手电商平台商品标注专家。请仔细观察用户上传的商品图片，输出标准化、结构化的商品标签，用于平台的商品分类、搜索和推荐。\n\n"
    "## 输出字段（共 7 个）\n"
    "1. 类目：层级类目数组，从粗到细，最多 5 层，最少 1 层。每一层必须是你高确信、高置信度的标签，不要为了凑数硬加。\n"
    "2. 颜色：颜色数组。如果只有一个主颜色，数组长度为 1；如果有多个明显颜色，最多列出 5 个。\n"
    "3. 材质/面料：单一字符串。\n"
    "4. 款式特征：单一字符串，描述商品的款式、功能或风格特点。\n"
    "5. 成色：单一字符串，描述商品新旧程度。\n"
    "6. 置信度：0.0 ~ 1.0 之间的数值，表示对整组标签的整体确信程度。\n"
    "7. 复杂度：0.0 ~ 1.0 之间的数值，表示图片场景的复杂程度（0.0 简单 ~ 1.0 复杂）。\n\n"
    "## 输出格式要求\n"
    "- 仅输出合法的 JSON 对象，不要添加 markdown 代码块、不要添加解释说明。\n"
    "- 所有标签值必须是中文。\n"
    "- 类目和颜色字段为字符串数组，其他字段为字符串或数值。\n"
    "- 无法判断的字段填写\"未知\"。\n"
    "- 如果图片中有多个商品，只对最主体、最清晰的商品进行打标。\n"
    "- 如果图片明显不是二手商品（如风景、人物、动物、纯文字、表情包、模糊图、建筑等），必须输出全未知 + 置信度 0.0 + 复杂度 0.0。\n\n"
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


def normalize_image_path(img_path: str) -> str:
    """统一图片路径格式为 images/xxx.jpg。"""
    p = Path(img_path)
    return f"images/{p.name}"


def convert_new_to_old(sample: dict) -> dict:
    """将 messages 格式转为 conversations 格式。"""
    messages = sample.get("messages", [])
    user_msg = ""
    assistant_output = ""
    for m in messages:
        if m.get("role") == "user":
            user_msg = m.get("content", "")
        elif m.get("role") == "assistant":
            assistant_output = m.get("content", "")

    return {
        "conversations": [
            {"from": "human", "value": f"<image>\n{user_msg}"},
            {"from": "gpt", "value": assistant_output},
        ],
        "images": [normalize_image_path(img) for img in sample.get("images", [])],
    }


def convert_old_to_new(sample: dict) -> dict:
    """将 conversations 格式转为 messages 格式。"""
    conversations = sample.get("conversations", [])
    user_msg = ""
    assistant_output = ""
    for c in conversations:
        if c.get("from") == "human":
            # 去掉 <image>\n 前缀
            raw = c.get("value", "")
            if raw.startswith("<image>\n"):
                raw = raw[len("<image>\n"):]
            user_msg = raw
        elif c.get("from") == "gpt":
            assistant_output = c.get("value", "")

    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_output},
        ],
        "images": [normalize_image_path(img) for img in sample.get("images", [])],
    }


def merge():
    # 读取旧数据（conversations 格式）
    old_path = LABELS_DIR / "train_sharegpt.json"
    old_data = []
    if old_path.exists():
        with open(old_path) as f:
            old_data = json.load(f)
        print(f"旧数据: {len(old_data)} 条 ({old_path})")

    # 读取新数据（messages 格式）
    new_path = LABELS_DIR / "train_open_datasets_sharegpt.json"
    new_data = []
    if new_path.exists():
        with open(new_path) as f:
            new_data = json.load(f)
        print(f"新数据: {len(new_data)} 条 ({new_path})")

    # 转换新数据为旧格式（conversations）
    old_format_data = []
    for s in new_data:
        if "messages" in s:
            old_format_data.append(convert_new_to_old(s))
        else:
            old_format_data.append(s)

    # 按图片路径去重，新数据覆盖旧数据
    seen = set()
    merged = []

    # 先加新数据
    for s in old_format_data:
        for img in s.get("images", []):
            seen.add(img)
        merged.append(s)

    # 再加旧数据中不重复的
    added_from_old = 0
    for s in old_data:
        imgs = s.get("images", [])
        if not imgs or imgs[0] not in seen:
            for img in imgs:
                seen.add(img)
            merged.append(s)
            added_from_old += 1

    # 写回 train_sharegpt.json
    output = LABELS_DIR / "train_sharegpt.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n合并完成: {len(merged)} 条（新增 {len(old_format_data)} 条，"
          f"保留旧数据中不重复的 {added_from_old} 条）")
    print(f"输出: {output}")

    # ---- 也生成 messages 格式的最新版 ----
    msgs_format = []
    for s in merged:
        if "conversations" in s:
            msgs_format.append(convert_old_to_new(s))
        else:
            msgs_format.append(s)

    msgs_output = LABELS_DIR / "train_sharegpt_messages.json"
    with open(msgs_output, "w", encoding="utf-8") as f:
        json.dump(msgs_format, f, ensure_ascii=False, indent=2)
    print(f"messages 格式副本: {msgs_output} ({len(msgs_format)} 条)")


def print_dataset_info():
    """打印 dataset_info.json 配置参考。"""
    print("\n" + "=" * 60)
    print("dataset_info.json 配置参考")
    print("=" * 60)
    info = {
        "train_sharegpt": {
            "file_name": "data/open_datasets/labels/train_sharegpt.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        },
        "train_sharegpt_messages": {
            "file_name": "data/open_datasets/labels/train_sharegpt_messages.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("\n将此配置放入 LLaMA Factory 的 data/dataset_info.json 中即可使用。")
    print("\nLLaMA Factory 训练时指定 --dataset train_sharegpt")


if __name__ == "__main__":
    merge()
    print_dataset_info()
