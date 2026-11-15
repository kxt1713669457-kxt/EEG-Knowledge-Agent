"""查询图的编排定义；节点实现由入口注入，便于隔离外部服务检查真实拓扑。"""
from collections.abc import Callable, Mapping

from langgraph.graph import END, StateGraph

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.task_updates import input_checkpoint


RETRIEVAL_NODE_NAMES = (
    "node_search_embedding",
    "node_search_embedding_hyde",
    "node_web_search_mcp",
)
QUERY_NODE_NAMES = (
    "node_item_name_confirm",
    *RETRIEVAL_NODE_NAMES,
    "node_rrf",
    "node_rerank",
    "node_answer_output",
)


def after_node_item_name_confirm(state: QueryGraphState) -> str | tuple[str, ...]:
    """已有澄清回答则直达输出；开放或主题过滤检索都启动三路召回。"""
    if state.get("answer"):
        return "node_answer_output"
    return RETRIEVAL_NODE_NAMES


def after_node_rrf(state: QueryGraphState) -> str:
    """全空时已有证据不足说明，跳过重排与生成模型。"""
    return "node_answer_output" if state.get("answer") else "node_rerank"


def build_query_graph_builder(nodes: Mapping[str, Callable]) -> StateGraph:
    """线上节点和教学替身共用此处的节点、条件边及并行汇合定义。"""
    builder = StateGraph(QueryGraphState)
    for name in QUERY_NODE_NAMES:
        # 三路召回全部汇合后再检查，避免只中止一条并行支路。
        builder.add_node(name, nodes[name] if name in RETRIEVAL_NODE_NAMES else input_checkpoint(nodes[name]))
    builder.set_entry_point("node_item_name_confirm")
    builder.add_conditional_edges(
        "node_item_name_confirm",
        after_node_item_name_confirm,
        {name: name for name in (*RETRIEVAL_NODE_NAMES, "node_answer_output")},
    )
    # 等三个召回节点全部完成后，RRF 才执行一次。
    builder.add_edge(list(RETRIEVAL_NODE_NAMES), "node_rrf")
    builder.add_conditional_edges("node_rrf", after_node_rrf,
                                  {name: name for name in ("node_rerank", "node_answer_output")})
    builder.add_edge("node_rerank", "node_answer_output")
    builder.add_edge("node_answer_output", END)
    return builder
