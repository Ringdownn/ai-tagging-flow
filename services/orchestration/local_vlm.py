"""
本地 Qwen2.5-VL-7B 模型封装
==========================
提供与 LangGraph 节点适配的 predict 接口。
"""
import json
import re
from pathlib import Path
from typing import Any

from .config import LOCAL_VLM_PATH, LOCAL_VLM_DEVICE, LOCAL_VLM_DTYPE, LOAD_IN_4BIT, LOAD_IN_8BIT


def _parse_model_output(text: str) -> dict:
    """从模型输出文本中提取 JSON。支持商品标签和非商品识别格式。"""
    text = text.strip()

    # 尝试直接 JSON 解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    if "```json" in text:
        text = text.split("```json")[-1]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 兜底：正则提取 key-value（商品标签格式）
    result = {"is_product": True}
    for field in ["类目", "颜色", "材质/面料", "款式特征", "成色"]:
        pattern = r'["\']?' + re.escape(field) + r'["\']?\s*[:：]\s*["\']?([^"\'\n,}]*)["\']?'
        match = re.search(pattern, text)
        result[field] = match.group(1).strip() if match else "未知"
    return result


class LocalVLM:
    """本地 Qwen2.5-VL-7B-Instruct 封装。"""

    def __init__(
        self,
        model_path: str = LOCAL_VLM_PATH,
        device: str = LOCAL_VLM_DEVICE,
        dtype: str = LOCAL_VLM_DTYPE,
        load_in_4bit: bool = LOAD_IN_4BIT,
        load_in_8bit: bool = LOAD_IN_8BIT,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.processor = None

    def _load(self):
        """懒加载模型。"""
        if self.model is not None:
            return

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "运行本地 Qwen2.5-VL 需要安装 transformers 和 qwen-vl-utils: "
                "pip install transformers qwen-vl-utils accelerate"
            ) from e

        print(f"[INFO] 加载本地模型: {self.model_path}")
        import torch as _torch

        # 设备处理：量化时用 auto（允许 CPU offload），非量化时也用 auto
        load_kwargs = {
            "device_map": self.device,  # "auto" 由 accelerate 决定
            "trust_remote_code": True,
        }

        # dtype 处理：auto 时用 float16（GPU）或 float32（CPU）
        if self.dtype and self.dtype != "auto":
            load_kwargs["torch_dtype"] = self.dtype
        else:
            if _torch.cuda.is_available():
                load_kwargs["torch_dtype"] = _torch.float16
            else:
                load_kwargs["torch_dtype"] = _torch.float32

        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import torch
            except ImportError as e:
                raise ImportError(
                    "INT4 量化需要安装 bitsandbytes: pip install bitsandbytes"
                ) from e

            print("[INFO] 启用 INT4 (bitsandbytes 4-bit) 量化加载（允许 CPU offload）")
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            # 限制 GPU 显存使用，自动将装不下的层放到 CPU
            load_kwargs["max_memory"] = {0: "6.5GiB", "cpu": "30GiB"}
            # 4-bit 量化时由 device_map 自动决定 torch_dtype，避免冲突
            load_kwargs.pop("torch_dtype", None)
        elif self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                import torch
            except ImportError as e:
                raise ImportError(
                    "INT8 量化需要安装 transformers/torch: pip install transformers torch"
                ) from e

            print("[INFO] 启用 INT8 (bitsandbytes 8-bit) 量化加载（允许 CPU offload）")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=["visual"],
            )
            # 限制 GPU 显存使用，自动将装不下的层放到 CPU
            import torch
            load_kwargs["max_memory"] = {0: "6.5GiB", "cpu": "30GiB"}
            # 8-bit 量化时由 device_map 自动决定 torch_dtype
            load_kwargs.pop("torch_dtype", None)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        print("[INFO] 本地模型加载完成")

    def predict(self, image_input: str | Path, max_new_tokens: int = 256) -> dict:
        """
        预测单张图片的标签。

        参数:
            image_input: 图片本地路径

        返回:
            {
                "tags": {"类目": ..., "颜色": ..., ...},
                "confidence": 0.0 ~ 1.0,
                "raw": "模型原始输出"
            }
        """
        self._load()

        prompt = (
            "你是一位二手商品图片识别助手。请按以下步骤处理图片：\n"
            "1. 首先判断图片是否包含可以在二手平台交易的商品（如服饰、3C、家居、美妆、鞋包、食品等）。\n"
            "2. 如果是风景、人物、动物、纯文字、表情包、截图、建筑等非商品图片，\n"
            '   请输出：{"is_product": false, "reason": "非商品，原因是..."}\n'
            "3. 如果是商品图片，请输出：\n"
            '   {"is_product": true, "tags": {"类目": "...", "颜色": "...", "材质/面料": "...", "款式特征": "...", "成色": "..."}, "confidence": 0.0-1.0, "complexity": 0.0-1.0}\n'
            "注意：\n"
            "- confidence 表示你对标签的整体置信度，越高越确定\n"
            "- complexity 表示图片场景复杂程度：0.0 为单主体白底简单图，1.0 为多主体杂乱背景疑难图\n"
            "- 类目标签按层级从粗到细，用英文逗号分隔\n"
            "- 无法判断的字段填\"未知\"\n"
            "请直接输出 JSON，不要有任何多余解释。"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_input)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError("请安装 qwen-vl-utils: pip install qwen-vl-utils") from e

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 将输入移到模型主设备（量化模型在 cuda:0）
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        raw_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"[DEBUG] 本地模型原始输出: {raw_output[:500]}")
        parsed = _parse_model_output(raw_output)
        print(f"[DEBUG] 解析结果: is_product={parsed.get('is_product')}, tags={parsed.get('tags')}, confidence={parsed.get('confidence')}")

        is_product = parsed.get("is_product", True)
        tags = parsed.get("tags", {}) if is_product else {}
        reason = parsed.get("reason", "")

        # 如果模型自己输出了 confidence/complexity 就使用，否则估算
        confidence = parsed.get("confidence")
        if confidence is None:
            confidence = self._estimate_confidence(tags) if is_product else 0.0

        complexity = parsed.get("complexity")
        if complexity is None:
            # 未训练前，用背景简单度启发式估算
            complexity = self._estimate_complexity(image_input) if is_product else 0.0

        return {
            "is_product": is_product,
            "tags": tags,
            "confidence": float(confidence),
            "complexity": float(complexity),
            "reason": reason,
            "raw": raw_output,
        }

    def _estimate_confidence(self, tags: dict) -> float:
        """简易置信度估计。TODO: 替换为 token 级 logprob 均值。"""
        valid_fields = ["类目", "颜色", "材质/面料", "款式特征", "成色"]
        valid_count = sum(1 for f in valid_fields if tags.get(f) and tags.get(f) != "未知")
        return round(valid_count / len(valid_fields), 4)

    def _estimate_complexity(self, image_input: str | Path) -> float:
        """
        简易复杂度估计。训练前作为兜底，训练后由模型直接输出。
        基于颜色丰富度和边缘密度估算，0.0 简单 ~ 1.0 复杂。
        """
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(str(image_input)).convert("RGB")
            img.thumbnail((256, 256))
            arr = np.array(img).astype(np.float32) / 255.0

            # 颜色丰富度：标准差越大越复杂
            color_std = float(np.std(arr))

            # 边缘密度：拉普拉斯算子
            gray = np.mean(arr, axis=2)
            lap = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
            edge_density = float(np.mean(lap))

            # 归一化到 0~1（简单启发式阈值）
            score = min(1.0, (color_std * 2.0 + edge_density * 0.3) / 2.0)
            return round(score, 4)
        except Exception:
            return 0.5


# 全局单例，避免重复加载
_local_vlm_instance: LocalVLM | None = None


def get_local_vlm() -> LocalVLM:
    """获取本地 VLM 单例。"""
    global _local_vlm_instance
    if _local_vlm_instance is None:
        _local_vlm_instance = LocalVLM()
    return _local_vlm_instance
