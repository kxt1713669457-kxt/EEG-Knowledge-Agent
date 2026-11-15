"""应用输入预算按UTF-8字节精确计量；它不是远端模型tokenizer或窗口规格。"""
from app.shared.config.common import env_str
from app.rag.query.evidence_service import build_evidence_sources, format_evidence_context
from app.rag.query.query_understanding import format_history_messages
from app.shared.runtime.load_prompt import load_prompt
from app.workspace.connections import completion_limit
from app.rag.query.answer_quality import output_budget

def _positive_int(name, default):
    value = int(env_str(name, str(default)))
    if value < 1:
        raise ValueError(f"{name}必须为正整数")
    return value


ANSWER_MAX_PROMPT_BYTES = _positive_int("ANSWER_MAX_PROMPT_BYTES", 32000)
REVIEW_MAX_PROMPT_BYTES = _positive_int("REVIEW_MAX_PROMPT_BYTES", 48000)
ANSWER_MAX_OUTPUT_TOKENS = _positive_int("ANSWER_MAX_OUTPUT_TOKENS", 1024)
REVIEW_MAX_OUTPUT_TOKENS = _positive_int("REVIEW_MAX_OUTPUT_TOKENS", 4096)


class ContextBudgetError(ValueError):
    pass


def check_prompt_budget(prompt: str, limit: int):
    size = len(prompt.encode("utf-8"))
    if size > limit:
        raise ContextBudgetError("完整问题或复核提示超出应用输入预算，请缩小问题范围。")
    return size


def pack_answer_prompt(state: dict, history: list, *, limit=ANSWER_MAX_PROMPT_BYTES):
    question = state.get("rewritten_query") or state.get("original_query") or ""
    messages = list(history or [])
    candidates = [doc for doc in state.get("reranked_docs") or [] if isinstance(doc.get("text"), str) and doc["text"].strip()]
    def render(docs, past):
        sources = build_evidence_sources(docs)
        # 从本次实际装入的来源生成清单，不用固定E示例诱导产物引用改号。
        allowed = "、".join(f"[{source['evidence_id']}]" for source in sources) or "无可用编号"
        context = f"【本轮允许引用】{allowed}\n只复制实际来源的引用标识；不改变前缀，不自行新增编号。\n" + format_evidence_context(sources)
        prompt = load_prompt("answer_out", question=question, context=context,
                             item_names=", ".join(state.get("item_names") or []), history=format_history_messages(past))
        return prompt, sources
    while messages:
        prompt, _ = render(candidates, messages)
        if len(prompt.encode("utf-8")) <= limit:
            break
        messages.pop(0)
    base, _ = render([], messages)
    check_prompt_budget(base, limit)  # 不截断问题来腾地方。
    selected, omitted = [], []
    for index, doc in enumerate(candidates):
        trial, _ = render([*selected, doc], messages)
        if len(trial.encode("utf-8")) <= limit:
            selected.append(doc)
        else:
            omitted.append({"candidate_index": index, "doc_id": doc.get("doc_id"), "chunk_uid": doc.get("chunk_uid"),
                            "reason": "whole_source_does_not_fit"})
    prompt, sources = render(selected, messages)
    state["evidence_sources"] = sources
    state["context_budget"] = {"unit": "utf8_bytes", "limit": limit, "used": len(prompt.encode("utf-8")),
                               "candidate_count": len(candidates), "selected_count": len(sources),
                               "omitted_sources": omitted, "dropped_history_messages": len(history or []) - len(messages),
                               "output_max_tokens": completion_limit(output_budget(question, ANSWER_MAX_OUTPUT_TOKENS))["max_tokens"],
                               "requested_answer_space_tokens": output_budget(question, ANSWER_MAX_OUTPUT_TOKENS)}
    return prompt
