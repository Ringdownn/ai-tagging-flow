"""
LangGraph 状态定义
==================
"""
from typing import Optional, Dict, Any, List
from langgraph.graph import MessagesState


class TaggingState(MessagesState):
    """商品打标流程状态。"""

    image_url: Optional[str] = None         # 原始图片 URL（上传模式可为空）
    image_path: Optional[str] = None        # 本地缓存路径
    image_b64: Optional[str] = None         # 预处理后 base64
    original_size: Optional[tuple] = None   # 原始尺寸 (w, h)
    processed_size: Optional[tuple] = None  # 处理后尺寸

    local_tags: Optional[Dict[str, Any]] = None
    local_confidence: float = 0.0           # 本地模型置信度
    local_complexity: float = 0.0           # 本地模型输出复杂度
    local_raw_output: Optional[str] = None  # 本地模型原始输出
    is_product: Optional[bool] = None       # 本地模型判断是否商品图

    glm_tags: Optional[Dict[str, Any]] = None       # GLM-4V 输出标签
    glm_description: Optional[str] = None           # GLM-4V 图文描述

    tool_outputs: Optional[Dict[str, Any]] = None     # 工具预处理结果
    final_tags: Optional[Dict[str, Any]] = None       # 最终标准化标签（单商品模式）
    final_tags_list: Optional[List[Dict[str, Any]]] = None  # 多商品模式下的所有标签

    branch: Optional[str] = None            # local / glm4v / non_product / error
    error: Optional[str] = None             # 错误信息
