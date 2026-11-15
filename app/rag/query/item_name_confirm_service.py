"""把原始问题与会话历史转成独立查询；主体标签仅用于可选过滤。"""
import json
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_providers
from app.infra.persistence.history_repository import history_repository
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.query_understanding import (
    HISTORY_MESSAGE_LIMIT, comparison_pronoun_clarification, format_history_messages,
    topic_key, topic_variants, validate_understanding,
    preserve_named_document_scope,
)
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def get_data_and_validates(state: QueryGraphState) -> tuple[str, str]:
    session_id, original_query = state.get("session_id"), state.get("original_query")
    if not session_id or not isinstance(original_query, str) or not original_query.strip():
        raise ValueError("session_id或original_query为空，不能进行问题理解")
    return session_id, original_query


def get_history_messages_and_context(session_id: str, messages: list[dict] | None = None) -> str:
    if messages is None:
        messages = history_repository.list_recent(session_id=session_id, limit=HISTORY_MESSAGE_LIMIT)
    return format_history_messages(messages)


def call_item_name_and_rewritten(history_text: str, original_query: str) -> dict:
    prompt = load_prompt("rewritten_query_and_itemnames", query=original_query, history_text=history_text)
    chain = llm_providers.chat(json_mode=True) | JsonOutputParser()
    result = chain.invoke([HumanMessage(content=prompt)])
    # JSON语法可解析不等于字段类型、澄清条件及业务意义正确。
    return validate_understanding(result, original_query)


def search_by_item_names(item_names: list[str]) -> dict[str, list[dict]]:
    """按登记名称/别名核对主题，正文语义检索留给后续召回节点。"""
    if not item_names:
        return {}
    client = milvus_gateway.milvus_client
    collection = milvus_gateway.item_name_collection_name
    if not client.has_collection(collection_name=collection):
        return {name: [] for name in item_names}
    matches = {}
    for name in item_names:
        rows = client.query(collection_name=collection,
                            filter=f"item_name in {json.dumps(topic_variants(name), ensure_ascii=False)}",
                            output_fields=["item_name"], limit=100)
        matches[name] = [{"item_name": value, "match_type": "registered_name"}
                         for value in dict.fromkeys(row["item_name"] for row in rows)]
    return matches


def select_item_names(milvus_result: dict[str, list[dict]]) -> dict:
    """名字或登记别名一致才作为强过滤条件，相关性不能代替身份。"""
    confirmed, options, unmatched = [], [], []
    for entity, candidates in milvus_result.items():
        aligned = [item for item in candidates if topic_key(item.get("item_name", "")) == topic_key(entity)]
        if aligned:
            confirmed.append(aligned[0])
        else:
            unmatched.append(entity)
            options.extend(candidates[:2])
    return {"confirmed_list": confirmed, "option_list": options, "unmatched_entities": unmatched}


def apply_item_name_result(state: dict, list_dict: dict, rewritten_query: str):
    # 比较问题只对齐一半实体时不能硬过滤，否则会提前丢掉另一半证据。
    names = [] if list_dict.get("unmatched_entities") else [
        item["item_name"] for item in list_dict.get("confirmed_list", [])
    ]
    state["item_names"] = list(dict.fromkeys(names))
    state["rewritten_query"] = rewritten_query
    state["answer"] = ""
    state["retrieval_scope"] = "filtered" if names else "open"


def save_history_message(state: QueryGraphState):
    history_repository.save_message(
        session_id=state["session_id"], role="user", text=state["original_query"],
        task_id=state.get("task_id") or "", run_id=state.get("run_id"),
        rewritten_query=state["rewritten_query"], item_names=state.get("item_names", []), image_urls=[],
    )


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    session_id, original_query = get_data_and_validates(state)
    # 固定本轮历史快照，后面的回答生成也复用，避免再读到本次刚写的提问。
    messages = history_repository.list_recent(session_id=session_id, limit=HISTORY_MESSAGE_LIMIT)
    state["history"] = messages or []
    history_text = get_history_messages_and_context(session_id, state["history"])
    result = call_item_name_and_rewritten(history_text, original_query)
    result['item_names']=preserve_named_document_scope(original_query,result['item_names'])
    clarification = comparison_pronoun_clarification(original_query, state["history"])
    if clarification:
        result = {"item_names": [], "rewritten_query": original_query,
                  "needs_clarification": True, "clarification_question": clarification}
    state["entities"] = result["item_names"]
    state["clarification_question"] = result["clarification_question"]
    if result["needs_clarification"]:
        state["item_names"] = []
        state["rewritten_query"] = result["rewritten_query"]
        state["answer"] = result["clarification_question"]
        state["retrieval_scope"] = "clarify"
    else:
        matches = search_by_item_names(result["item_names"])
        apply_item_name_result(state, select_item_names(matches), result["rewritten_query"])
    if not state.get("defer_history"):
        save_history_message(state)
    logger.info("问题理解完成: entities={}, filters={}, scope={}",
                state["entities"], state["item_names"], state["retrieval_scope"])
    return state
