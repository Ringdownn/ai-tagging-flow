"""查看当前数据集的复杂度分布。"""
import json
from pathlib import Path

labels_dir = Path("data/open_datasets/labels")

# 检查合并训练集
train_path = labels_dir / "train_open_datasets_sharegpt.json"
if train_path.exists():
    print(f"读取 {train_path} ...")
    with open(train_path) as f:
        samples = json.load(f)
    print(f"共 {len(samples)} 条样本\n")

    complexities = []
    for s in samples:
        output = s.get("messages", [{}])[-1].get("content", "{}")
        try:
            parsed = json.loads(output)
            c = parsed.get("复杂度", None)
            if c is not None:
                complexities.append(float(c))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    if complexities:
        low = sum(1 for c in complexities if c <= 0.33)
        mid = sum(1 for c in complexities if 0.33 < c <= 0.66)
        high = sum(1 for c in complexities if c > 0.66)
        over_05 = sum(1 for c in complexities if c > 0.5)

        print(f"=== 复杂度分布（{len(complexities)} 条有数据） ===")
        print(f"  平均值: {sum(complexities)/len(complexities):.3f}")
        print(f"  最小值: {min(complexities):.3f}")
        print(f"  最大值: {max(complexities):.3f}")
        print(f"  低(≤0.33): {low} ({low/len(complexities)*100:.1f}%)")
        print(f"  中(≤0.66): {mid} ({mid/len(complexities)*100:.1f}%)")
        print(f"  高(>0.66): {high} ({high/len(complexities)*100:.1f}%)")
        print(f"  >0.5: {over_05} ({over_05/len(complexities)*100:.1f}%)")
        print(f"\n  详细值: {sorted([round(c,3) for c in complexities])}")
    else:
        print("没有找到复杂度数据")
else:
    print(f"{train_path} 不存在，逐个检查 summary 文件...")

    summary_files = {
        "fashion": "fashion_summary.jsonl",
        "local": "local_summary.jsonl",
        "negative": "negative_summary.jsonl",
        "mmec": "mmec_summary.jsonl",
    }

    for name, fname in summary_files.items():
        path = labels_dir / fname
        if not path.exists():
            print(f"  {name}: 不存在")
            continue
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]

        complexities = []
        for r in records:
            labels = r.get("vlm_labels", {})
            c = labels.get("复杂度", None)
            if c is not None:
                try:
                    complexities.append(float(c))
                except (ValueError, TypeError):
                    pass

        if complexities:
            low = sum(1 for c in complexities if c <= 0.33)
            mid = sum(1 for c in complexities if 0.33 < c <= 0.66)
            high = sum(1 for c in complexities if c > 0.66)
            print(f"\n  === {name} ({len(complexities)} 条) ===")
            print(f"    平均值: {sum(complexities)/len(complexities):.3f}")
            print(f"    分布: 低={low}, 中={mid}, 高={high}")
            print(f"    值: {sorted([round(c,3) for c in complexities])}")
        else:
            print(f"  {name}: 无复杂度数据")
