# LLaMA Factory 训练配置说明

## 1. 数据集准备

运行转换脚本，生成 Alpaca 格式训练集：

```bash
python scripts/convert_to_alpaca.py
```

输出文件：
- `data/open_datasets/labels/train_alpaca.json`

每条样本格式示例：

```json
{
  "instruction": "请为这张二手商品图片生成标准化标签，包含类目、颜色、材质/面料、款式特征、成色。",
  "input": "",
  "output": "{\n  \"类目\": [\"服饰\", \"女装\", \"连衣裙\"],\n  \"颜色\": [\"黑色\"],\n  \"材质/面料\": \"聚酯纤维\",\n  \"款式特征\": \"简约通勤\",\n  \"成色\": \"九成新\"\n}",
  "images": ["data/open_datasets/images/fashion_0002.jpg"]
}
```

## 2. dataset_info.json 配置

在 LLaMA Factory 项目目录下找到 `data/dataset_info.json`，添加以下内容：

```json
{
  "second_hand_tagging": {
    "file_name": "train_alpaca.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "images": "images"
    }
  }
}
```

> 注意：如果你把 `train_alpaca.json` 放在 LLaMA Factory 的 `data/` 目录下，就填 `file_name`；如果放在其他位置，填相对 LLaMA Factory 根目录的路径。

## 3. 模型训练配置 (Qwen2-VL-7B-Instruct)

在 LLaMA Factory 中新建训练配置，例如 `examples/train_lora/qwen2vl_second_hand.yaml`：

```yaml
### model
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
```

## 4. 启动训练

```bash
llamafactory-cli train examples/train_lora/qwen2vl_second_hand.yaml
```

## 5. 推理部署

训练完成后导出为 AWQ 量化模型（RTX 4060 8G 推荐 INT4）：

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

之后可用 vLLM 启动服务：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/qwen2vl-7b-second-hand-awq \
  --quantization awq \
  --gpu-memory-utilization 0.8 \
  --max-model-len 2048 \
  --port 8000
```

## 6. 训练数据量建议

当前数据集：
- fashion 数据集：约 30 张
- 本地商品照片：33 张
- MMECInstruct：根据下载情况

总计 60+ 张样本用于 SFT 微调。对于 LoRA 而言样本偏少，建议：

1. 增加本地商品照片到 100+ 张
2. 用更强的多模态模型（GLM-4V / Qwen2.5-VL）标注，保证标签质量
3. 训练 5-10 个 epoch
4. 如效果不佳，可先用完整 Qwen2-VL 模型做 few-shot 验证标签输出格式是否正确
