"""
LangGraph 流程构建（多 Agent 协作版）
======================================
"""
from langgraph.graph import StateGraph, START, END

from .state import TaggingState
from .nodes import (
    preprocess_node,
    supervisor_node,
    error_node,
)


def build_tagging_graph() -> StateGraph:
    """构建商品打标 LangGraph。"""
    builder = StateGraph(TaggingState)

    builder.add_node("preprocess", preprocess_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("error", error_node)

    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: "error" if state.get("error") else "end",
        {
            "end": END,
            "error": "error",
        },
    )
    builder.add_edge("error", END)

    return builder.compile()


# 全局编译后的图
tagging_graph = build_tagging_graph()
