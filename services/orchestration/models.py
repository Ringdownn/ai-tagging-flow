"""
GLM-4V 模型封装
===============
LangGraph 节点可直接调用的 GLM-4V-plus 包装类。
"""
from .config import ZHIPU_API_KEY, ZHIPU_MODEL, ZHIPU_URL
from .glm_client import call_glm4v


class GLMVLM:
    """智谱 GLM-4V-plus 封装，与打标脚本调用方式对齐。"""

    def __init__(self):
        if not ZHIPU_API_KEY:
            raise EnvironmentError("请先设置环境变量 ZHIPU_API_KEY")
        self.api_key = ZHIPU_API_KEY
        self.model = ZHIPU_MODEL
        self.api_url = ZHIPU_URL

    def invoke(self, image_b64: str, existing: dict | None = None) -> dict:
        """
        调用 GLM-4V 生成标准化标签。

        参数:
            image_b64: 图片 base64 字符串（不含 data URL 前缀）
            existing:  已有标签/参考信息

        返回:
            {"类目": ..., "颜色": ..., ...}
        """
        return call_glm4v(
            api_key=self.api_key,
            model=self.model,
            api_url=self.api_url,
            image_b64=image_b64,
            existing=existing,
        )


# 全局单例
glm_vlm_instance: GLMVLM | None = None


def get_glm_vlm() -> GLMVLM:
    """获取 GLM-4V 单例。"""
    global glm_vlm_instance
    if glm_vlm_instance is None:
        glm_vlm_instance = GLMVLM()
    return glm_vlm_instance
