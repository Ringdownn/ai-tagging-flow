"""
把需要上传到云端算力训练的文件打包到一个目录。

输出: upload_to_cloud/
├── data/
│   ├── train_alpaca.json     # 图片路径已改为 images/xxx
│   └── images/               # 所有商品图片
├── dataset_info.json         # LLaMA Factory 数据集注册配置
├── qwen2vl_second_hand.yaml  # 训练配置
└── README.md                 # 云端操作说明
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_LABELS = PROJECT_ROOT / "data" / "open_datasets" / "labels"
SRC_IMAGES = PROJECT_ROOT / "data" / "open_datasets" / "images"
UPLOAD_DIR = PROJECT_ROOT / "temp" / "upload_to_cloud"
UPLOAD_DATA = UPLOAD_DIR / "data"
UPLOAD_IMAGES = UPLOAD_DATA / "images"


def main():
    # 清理并创建目录
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_IMAGES.mkdir(parents=True, exist_ok=True)

    # 读取 train_alpaca.json 并修正图片路径
    with open(SRC_LABELS / "train_alpaca.json", "r", encoding="utf-8") as f:
        samples = json.load(f)

    for s in samples:
        old_path = s["images"][0]
        new_path = "images/" + Path(old_path).name
        s["images"] = [new_path]

    with open(UPLOAD_DATA / "train_alpaca.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 复制 ShareGPT 格式（路径已是相对路径，直接复制）
    src_sharegpt = SRC_LABELS / "train_sharegpt.json"
    if src_sharegpt.exists():
        shutil.copy2(src_sharegpt, UPLOAD_DATA / "train_sharegpt.json")

    # 复制图片
    image_files = sorted(SRC_IMAGES.glob("*"))
    for src in image_files:
        shutil.copy2(src, UPLOAD_IMAGES / src.name)

    # dataset_info.json
    dataset_info = {
        "second_hand_tagging": {
            "file_name": "train_alpaca.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "images": "images"
            }
        },
        "second_hand_tagging_sharegpt": {
            "file_name": "train_sharegpt.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            }
        }
    }
    with open(UPLOAD_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    # 训练配置
    yaml_content = """### model
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
template: qwen2_vl

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16

### dataset
dataset: second_hand_tagging
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: saves/qwen2vl-7b/lora/second_hand_tagging
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
"""
    with open(UPLOAD_DIR / "qwen2vl_second_hand.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # README
    readme = """# 云端训练上传包

## 文件说明

- `data/train_alpaca.json`：训练数据集（Alpaca 格式）
- `data/train_sharegpt.json`：训练数据集（ShareGPT 格式，带 `<image>` 占位符）
- `data/images/`：所有训练图片
- `dataset_info.json`：LLaMA Factory 数据集注册配置
- `qwen2vl_second_hand.yaml`：训练配置文件

## 云端操作步骤

1. 把 `data/` 目录下的内容放到 LLaMA Factory 项目的 `data/` 目录中
2. 把 `dataset_info.json` 中的内容合并到 LLaMA Factory 的 `data/dataset_info.json`
3. 把 `qwen2vl_second_hand.yaml` 放到 LLaMA Factory 的 `examples/train_lora/` 目录
4. 安装依赖并启动训练：

```bash
pip install -e .
llamafactory-cli train examples/train_lora/qwen2vl_second_hand.yaml
```

5. 训练完成后验证：

```bash
llamafactory-cli webchat \
  --model_name_or_path saves/qwen2vl-7b/lora/second_hand_tagging \
  --template qwen2_vl \
  --finetuning_type lora
```

6. 导出 AWQ INT4 模型（用于本地 RTX 4060 部署）：

```bash
llamafactory-cli export \
  --model_name_or_path saves/qwen2vl-7b/lora/second_hand_tagging \
  --template qwen2_vl \
  --finetuning_type lora \
  --export_dir models/qwen2vl-7b-second-hand-awq \
  --export_quantization_bit 4 \
  --export_size 2 \
  --export_device cpu \
  --export_legacy_format false
```

7. 把导出的 `models/qwen2vl-7b-second-hand-awq/` 目录压缩后传回本地。
"""
    with open(UPLOAD_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"[DONE] 上传包已生成: {UPLOAD_DIR.resolve()}")
    print(f"[INFO] 共 {len(samples)} 条训练样本，{len(image_files)} 张图片")
    print("\n目录结构:")
    for p in sorted(UPLOAD_DIR.rglob("*")):
        depth = len(p.relative_to(UPLOAD_DIR).parts) - 1
        indent = "  " * depth
        marker = "📁 " if p.is_dir() else "📄 "
        print(f"{indent}{marker}{p.name}")


if __name__ == "__main__":
    main()
