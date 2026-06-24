"""
多 Agent 协作封装
==================
- LocalAgent:   本地 Qwen2.5-VL-7B，负责初筛 + 非商品过滤 + 置信度评估
- HardCaseAgent: GLM-4V + 工具调用，负责疑难图打标
- Supervisor:   调度器，决定走本地还是疑难分支
"""

from typing import Any

from .config import CONFIDENCE_THRESHOLD, COMPLEXITY_THRESHOLD
from .local_vlm import get_local_vlm
from .glm_agent import glm_tagging_agent


UNKNOWN_TAGS = {
    "类目": ["未知"],
    "颜色": ["未知"],
    "材质/面料": "未知",
    "款式特征": "未知",
    "成色": "未知",
    "置信度": 0.0,
    "复杂度": 0.0,
}


def normalize_tags(tags: dict) -> dict:
    """
    把模型输出的字符串标签对齐到训练数据格式：
    - 类目 -> 字符串数组
    - 颜色 -> 字符串数组
    - 置信度/复杂度 -> float
    """
    normalized = {}
    for k, v in tags.items():
        if k == "类目" or k == "颜色":
            if isinstance(v, list):
                normalized[k] = [str(x).strip() for x in v if str(x).strip()]
            elif isinstance(v, str):
                normalized[k] = [x.strip() for x in v.split(",") if x.strip()] or ["未知"]
            else:
                normalized[k] = ["未知"]
        elif k in ("置信度", "复杂度"):
            try:
                normalized[k] = float(v)
            except (ValueError, TypeError):
                normalized[k] = 0.0
        else:
            normalized[k] = str(v) if v is not None else "未知"

    # 补齐缺失字段
    for field in ["类目", "颜色", "材质/面料", "款式特征", "成色", "置信度", "复杂度"]:
        if field not in normalized:
            normalized[field] = ["未知"] if field in ("类目", "颜色") else (0.0 if field in ("置信度", "复杂度") else "未知")
    return normalized


class LocalAgent:
    """本地模型 Agent：初筛 + 非商品识别 + 置信度评估。"""

    def __init__(self):
        self.vlm = get_local_vlm()

    def invoke(self, image_path: str) -> dict[str, Any]:
        """
        调用本地模型。

        返回:
            {
                "is_product": bool,
                "tags": dict,
                "confidence": float,
                "complexity": float,
                "reason": str,  # 非商品原因
            }
        """
        result = self.vlm.predict(image_path)
        return {
            "is_product": result.get("is_product", True),
            "tags": result.get("tags", {}),
            "confidence": result.get("confidence", 0.0),
            "complexity": result.get("complexity", 0.0),
            "reason": result.get("reason", ""),
        }


class HardCaseAgent:
    """疑难图 Agent：GLM-4V 自主工具决策 + 打标。"""

    async def invoke(
        self,
        image_b64: str,
        existing: dict | None = None,
    ) -> dict[str, Any]:
        """
        调用 GLM-4V Agent。

        返回:
            {
                "tags": dict,
                "tool_chain": list,
                "candidates": list,
            }
        """
        return await glm_tagging_agent(image_b64, existing=existing)


class Supervisor:
    """调度器：根据 LocalAgent 结果决定路由。"""

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        complexity_threshold: float = COMPLEXITY_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        self.complexity_threshold = complexity_threshold
        self.local_agent = LocalAgent()
        self.hard_case_agent = HardCaseAgent()

    async def tag(
        self,
        image_b64: str,
        image_path: str,
    ) -> dict[str, Any]:
        """
        完整调度逻辑。

        返回:
            {
                "tags": dict,
                "branch": "local" | "glm4v" | "non_product" | "error",
                "confidence": float,
                "complexity": float,
                "reason": str,
                "tool_chain": list | None,
            }
        """
        # 1. 本地模型初筛
        local_result = self.local_agent.invoke(image_path)
        is_product = local_result["is_product"]
        confidence = local_result["confidence"]
        complexity = local_result["complexity"]
        reason = local_result["reason"]

        # 2. 非商品图：直接返回未知，不调用 HardCaseAgent
        if not is_product:
            print(f"[INFO] Supervisor: 非商品图，原因: {reason}")
            return {
                "tags": UNKNOWN_TAGS,
                "branch": "non_product",
                "is_product": False,
                "confidence": 0.0,
                "complexity": 0.0,
                "reason": reason,
                "tool_chain": None,
            }

        # 3. 置信度足够且复杂度较低：直接返回本地标签
        if confidence >= self.confidence_threshold and complexity <= self.complexity_threshold:
            print(f"[INFO] Supervisor: 本地模型置信度 {confidence}，复杂度 {complexity}，直接返回")
            tags = normalize_tags(local_result["tags"])
            tags["置信度"] = confidence
            tags["复杂度"] = complexity
            return {
                "tags": tags,
                "branch": "local",
                "is_product": True,
                "confidence": confidence,
                "complexity": complexity,
                "reason": "",
                "tool_chain": None,
            }

        # 4. 疑难图：交给 HardCaseAgent
        print(f"[INFO] Supervisor: 置信度 {confidence} / 复杂度 {complexity}，转 HardCaseAgent")
        try:
            hard_result = await self.hard_case_agent.invoke(
                image_b64,
                existing=local_result["tags"],
            )
            tags = normalize_tags(hard_result["tags"])
        except Exception as e:
            # GLM API 失败时降级：返回本地模型结果
            print(f"[WARN] HardCaseAgent 调用失败: {e}，降级使用本地模型结果")
            tags = normalize_tags(local_result["tags"])

        tags["置信度"] = confidence  # 保留本地模型的置信度用于参考
        tags["复杂度"] = complexity

        return {
            "tags": tags,
            "branch": "glm4v",
            "is_product": True,
            "confidence": confidence,
            "complexity": complexity,
            "reason": "",
            "tool_chain": hard_result.get("tool_chain", []) if "hard_result" in locals() else [],
        }
