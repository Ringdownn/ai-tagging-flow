"""
将已生成的标签文件转换为 LLaMA Factory 支持的 Alpaca / ShareGPT 多模态格式。

输入: data/open_datasets/labels/image_label_mapping.json
输出:
  data/open_datasets/labels/train_alpaca.json
  data/open_datasets/labels/train_sharegpt.json

Alpaca 多模态格式:
{
  "instruction": "...",
  "input": "",
  "output": "{\"类目\": [...], ...}",
  "images": ["data/open_datasets/images/fashion_0000.jpg"]
}

ShareGPT 多模态格式:
{
  "conversations": [
    { "from": "human", "value": "<image>\\ninstruction" },
    { "from": "gpt", "value": "{\"类目\": [...], ...}" }
  ],
  "images": ["images/fashion_0000.jpg"]
}
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "open_datasets" / "labels" / "image_label_mapping.json"
ALPACA_OUTPUT = PROJECT_ROOT / "data" / "open_datasets" / "labels" / "train_alpaca.json"
SHAREGPT_OUTPUT = PROJECT_ROOT / "data" / "open_datasets" / "labels" / "train_sharegpt.json"

INSTRUCTION = (
    "你是一位专业的二手电商平台商品标注专家。请仔细观察用户上传的商品图片，"
    "输出标准化、结构化的商品标签，用于平台的商品分类、搜索和推荐。\n\n"
    "## 输出字段（共 5 个）\n"
    "1. 类目：层级类目数组，从粗到细，最多 5 层，最少 1 层。"
    "每一层必须是你高确信、高置信度的标签，不要为了凑数硬加。\n"
    "2. 颜色：颜色数组。如果只有一个主颜色，数组长度为 1；"
    "如果有多个明显颜色，最多列出 5 个，用英文逗号分隔。\n"
    "3. 材质/面料：单一字符串。\n"
    "4. 款式特征：单一字符串，描述商品的款式、功能或风格特点。\n"
    "5. 成色：单一字符串，描述商品新旧程度。\n\n"
    "## 输出格式要求\n"
    "- 仅输出合法的 JSON 对象，不要添加 markdown 代码块、不要添加解释说明。\n"
    "- 所有标签值必须是中文。\n"
    "- 类目和颜色字段为字符串数组，其他字段为字符串。\n"
    "- 无法判断的字段填写\"未知\"。\n"
    "- 如果图片中有多个商品，只对最主体、最清晰的商品进行打标。\n\n"
    "## 示例 1（女装连衣裙）\n"
    '{\n  "类目": ["服饰", "女装", "连衣裙"],\n  "颜色": ["黑色"],\n  "材质/面料": "聚酯纤维",\n  "款式特征": "简约通勤",\n  "成色": "九成新"\n}\n\n'
    "## 示例 2（电子产品游戏手柄）\n"
    '{\n  "类目": ["电子产品", "游戏设备", "游戏手柄", "电脑配件", "游戏配件"],\n  "颜色": ["黑色"],\n  "材质/面料": "塑料",\n  "款式特征": "有线手柄",\n  "成色": "九成新"\n}\n\n'
    "## 示例 3（多颜色上衣）\n"
    '{\n  "类目": ["服饰", "女装", "上衣"],\n  "颜色": ["蓝色", "白色"],\n  "材质/面料": "棉",\n  "款式特征": "印花T恤",\n  "成色": "八成新"\n}\n\n'
    "请为这张二手商品图片生成标准化标签。"
)


def labels_to_output(labels: dict) -> str:
    """将标签字典转换为结构化 JSON 字符串。"""
    structured = {}
    for k, v in labels.items():
        if k == "类目":
            structured[k] = [x.strip() for x in str(v).split(",") if x.strip()]
        elif k == "颜色":
            structured[k] = [x.strip() for x in str(v).split(",") if x.strip()]
        else:
            structured[k] = str(v)
    return json.dumps(structured, ensure_ascii=False, indent=2)


def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] 输入文件不存在: {INPUT_FILE}")
        print("请先运行 prepare_hf_datasets.py 生成标签数据。")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    alpaca_samples = []
    sharegpt_samples = []
    for item in mapping:
        labels = item.get("labels", {})
        output_text = labels_to_output(labels)

        # Alpaca 格式（图片路径保持原样）
        alpaca_samples.append({
            "instruction": INSTRUCTION,
            "input": "",
            "output": output_text,
            "images": [item["image"]],
        })

        # ShareGPT 格式（图片路径改为相对路径 + <image> 占位符）
        image_name = Path(item["image"]).name
        sharegpt_samples.append({
            "conversations": [
                {"from": "human", "value": f"<image>\n{INSTRUCTION}"},
                {"from": "gpt", "value": output_text},
            ],
            "images": [f"images/{image_name}"],
        })

    # 保存 Alpaca
    with open(ALPACA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(alpaca_samples, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Alpaca 格式训练集已保存: {ALPACA_OUTPUT} ({len(alpaca_samples)} 条)")

    # 保存 ShareGPT
    with open(SHAREGPT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(sharegpt_samples, f, ensure_ascii=False, indent=2)
    print(f"[DONE] ShareGPT 格式训练集已保存: {SHAREGPT_OUTPUT} ({len(sharegpt_samples)} 条)")


if __name__ == "__main__":
    main()
