import json
from pathlib import Path

label_dir = Path("data/open_datasets/labels")
low_confidence = []

skip_files = {
    "train_open_datasets_sharegpt.json",
    "train_open_datasets_alpaca.json",
    "image_label_mapping.json",
    "fashion_summary.jsonl",
    "local_summary.jsonl",
    "negative_summary.jsonl",
    "mmec_summary.jsonl",
}

for path in sorted(label_dir.glob("*.json")):
    if path.name in skip_files:
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "messages" in data:
            assistant_text = data["messages"][1]["content"]
        elif "output" in data:
            assistant_text = data["output"]
        else:
            continue
        labels = json.loads(assistant_text)
        conf = float(labels.get("置信度", 1.0))
        if conf < 0.8:
            low_confidence.append((path.name, conf, labels.get("类目", "unknown")))
    except Exception as e:
        print(f"[WARN] 解析失败 {path.name}: {e}")

print(f"置信度低于 0.8 的样本共 {len(low_confidence)} 个：\n")
for name, conf, cat in low_confidence:
    print(f"{name}: 置信度={conf}, 类目={cat}")
