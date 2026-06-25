"""
本地 Qwen2.5-VL 模型封装
========================
提供与 LangGraph 节点适配的 predict 接口。
支持 CUDA / MPS (Apple Silicon) / CPU 后端。
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
    """本地 Qwen2.5-VL 模型封装。支持 3B/7B，CUDA/MPS/CPU 后端。"""

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

    def _resolve_device(self) -> tuple[str, str]:
        """
        解析运行设备和数据类型。
        返回 (device, dtype_str)，其中 device 为 "cuda" / "mps" / "cpu"。
        """
        import torch

        device = self.device
        dtype_str = self.dtype

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if dtype_str == "auto":
            if device in ("cuda", "mps"):
                dtype_str = "float16"
            else:
                dtype_str = "float32"

        return device, dtype_str

    def _load(self):
        """懒加载模型。支持 CUDA / MPS (Apple Silicon) / CPU。"""
        if self.model is not None:
            return

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "运行本地 Qwen2.5-VL 需要安装 transformers 和 qwen-vl-utils: "
                "pip install transformers qwen-vl-utils accelerate"
            ) from e

        import torch

        device, dtype_str = self._resolve_device()
        print(f"[INFO] 加载本地模型: {self.model_path}")
        print(f"[INFO] 设备: {device}, 精度: {dtype_str}")

        torch_dtype = getattr(torch, dtype_str)

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        # CUDA: 使用 device_map 多 GPU 分配 + bitsandbytes 量化
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            # auto 时优先 bfloat16（Qwen2.5-VL 原生精度），不支持则回退 float16
            if self.dtype == "auto":
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    load_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    load_kwargs["torch_dtype"] = torch.float16

            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                print("[INFO] 启用 INT4 (bitsandbytes 4-bit) 量化")
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs.pop("torch_dtype", None)
            elif self.load_in_8bit:
                from transformers import BitsAndBytesConfig
                print("[INFO] 启用 INT8 (bitsandbytes 8-bit) 量化")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_skip_modules=["visual"],
                )
                load_kwargs.pop("torch_dtype", None)

        # MPS / CPU: 不使用 device_map（这是 CUDA 多 GPU 特性），
        # 也不使用 bitsandbytes（CUDA only）
        if device in ("mps", "cpu"):
            if self.load_in_4bit or self.load_in_8bit:
                print(
                    "[WARN] bitsandbytes 量化仅支持 CUDA，"
                    f"在 {device} 上将使用 {dtype_str} 全精度加载"
                )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_kwargs,
        )

        # MPS: 加载到 CPU 后再移到 MPS（避免 device_map 兼容问题）
        if device == "mps":
            print("[INFO] 将模型移至 MPS (Apple Silicon GPU)...")
            self.model = self.model.to("mps")

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
            "你是一位二手电商平台商品标注专家。请仔细观察图片，输出标准化、结构化的商品标签。\n\n"
            "## 判断规则\n"
            "1. 首先判断图片是否包含可以在二手平台交易的商品（如服饰、3C、家居、美妆、鞋包、食品等）。\n"
            "2. 如果是风景、人物、动物、纯文字、表情包、截图、建筑等明显非商品图片，\n"
            '   请输出：{"is_product": false, "reason": "非商品，原因是..."}\n'
            "3. 如果是商品图片，请输出如下 JSON：\n"
            '   {"is_product": true, "tags": {"类目": ["..."], "颜色": ["..."], "材质/面料": "...", "款式特征": "...", "成色": "..."}, "confidence": 0.0-1.0, "complexity": 0.0-1.0}\n\n'
            "## 字段说明\n"
            "- 类目：层级类目数组，从粗到细，最少 1 层，最多 5 层。每一层必须是你高确信、高置信度的标签，不要为了凑数硬加。\n"
            "- 颜色：颜色数组。如果只有一个主颜色，数组长度为 1；如果有多个明显颜色，最多列出 5 个。\n"
            "- 材质/面料：字符串。单一材质直接写一种；如果商品明显由多种材质组成（如\"金属,塑料\"），可用英文逗号分隔，最多 3 种。\n"
            "- 款式特征：字符串。描述款式、功能或风格特点。如有多个显著特征（如\"花卉图案,短袖,收腰\"），可用英文逗号分隔，最多 3 个。\n"
            "- 成色：单一字符串，描述商品新旧程度（如全新、九成新、八成新、七成新等）。\n"
            "- confidence：0.0~1.0，表示你对整组标签的整体确信程度。\n"
            "- complexity：0.0~1.0，表示图片场景复杂程度。0.0 为单主体白底简单图，1.0 为多主体杂乱背景疑难图。\n\n"
            "## 输出要求\n"
            "- 仅输出合法 JSON，不要 markdown 代码块、不要解释说明。\n"
            "- 所有标签值必须是中文。\n"
            "- 类目和颜色字段为字符串数组，其他字段为字符串或数值。\n"
            "- 无法判断的字段填\"未知\"。\n"
            "- 如果图片中有多个商品，只对最主体、最清晰的商品进行打标。"
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
