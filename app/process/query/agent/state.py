from typing_extensions import TypedDict
from typing import List
import copy

class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如 state["session_id"]、state.get("answer")）。
    """
    session_id: str  # 会话唯一标识
    task_id: str  # 一次请求的进度/SSE身份，不能用会话ID代替
    run_id: str | None  # Agent执行身份；固定RAG查询可以为空
    managed_input: bool
    input_revision: int
    defer_history: bool
    defer_delivery: bool
    requested_answer_format: str
    source_requirement: str
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档
    web_search_status: dict  # not_run / ok / empty / failed，联网完成不等于结果可信

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档
    rerank_status: dict  # not_run / disabled / model / no_evidence

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案
    evidence_sources: list  # 本轮编号与实际原文的快照
    citations: list  # 回答实际使用的来源，E编号在本轮内有效
    citation_status: dict  # ID校验状态，不等于事实核验结果
    answer_audit: dict  # 原始模型输出及校验结果，供排查
    context_budget: dict
    generation_usage: dict

    # 辅助信息
    item_names: List[str]  # 已对齐知识库名称的主题过滤；空列表表示开放检索
    entities: List[str]  # 模型从问题中提取的实体，不直接作为数据库过滤条件
    retrieval_scope: str  # open / filtered / clarify
    clarification_question: str
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记
    image_urls: List[str]  # 答案中引用的图片链接


# ========================
# 默认状态（全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "run_id": None,
    "managed_input": False,
    "input_revision": 1,
    "defer_history": False,
    "defer_delivery": False,
    "requested_answer_format": "auto",
    "source_requirement": "auto",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "web_search_status": {"status": "not_run"},
    "rrf_chunks": [],
    "reranked_docs": [],
    "rerank_status": {"mode": "not_run"},
    "prompt": "",
    "answer": "",
    "evidence_sources": [],
    "citations": [],
    "citation_status": {"status": "not_run", "support_check": "not_assessed"},
    "answer_audit": {},
    "context_budget": {},
    "generation_usage": {},
    "item_names": [],
    "entities": [],
    "retrieval_scope": "open",
    "clarification_question": "",
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "image_urls": []
}


# ========================
# 创建默认状态（可覆盖）
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# ========================
# 获取干净状态
# ========================
def get_query_default_state() -> QueryGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(query_graph_default_state)


def execution_id(state) -> str:
    """HTTP必须传task_id；旧的独立节点样例/隔离测试保留session回退。"""
    return state.get("task_id") or state["session_id"]




if __name__ == "__main__":
    # 测试
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
        is_stream=False
    )
    print("初始化状态：", state)
