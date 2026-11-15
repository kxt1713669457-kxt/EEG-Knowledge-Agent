import re

from langchain_core.messages import HumanMessage

from app.infra.llm.providers import llm_providers
from app.workspace.connections import completion_limit
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState, execution_id
from app.rag.query.config import SUPPORTED_IMAGE_EXTENSIONS
from app.rag.query.evidence_service import (
    INSUFFICIENT_ANSWER,
    validate_answer_citations,
)
from app.rag.query.answer_support_service import review_answer_support
from app.rag.query.context_budget import pack_answer_prompt, ContextBudgetError, ANSWER_MAX_OUTPUT_TOKENS
from app.rag.query.answer_quality import answer_issues, output_budget, IncompleteAnswerError
from app.shared.utils.task_utils import add_done_task,add_running_task,push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
from app.shared.runtime.telemetry import trace_span, trace_mark
# 1. 判断state是否有answer,有我们直接返回(state) -> bool 有没有answer
#     没有 -> return false
#     有   -> 流式  push_to_session(session_id,delta,{delta:answer })
#            非流式 return True state => answer
#     2. 上一次返回的值是false
#         3. answer_prompt_text = 声明一个提示词拼接的方法(state) 拼接
#         4. 调用模型处理字符串answer回答的问题(state,answer_prompt_text)
#             流式
#                 llm - stream -> push ... -> state answer
#             非流式
#                 llm - invoke -> state -> answer
#             state[answer] =
#         5. 提取图片存储到state(reranked [text ![]()] mcp url -> 是不是图片, state)
#             url -> 后缀名
#             text -> 网络搜索 md ![xxxx](http://xxx) -> r"\!\[.*?\]\((.*?)\)" findall
#             iamge_urls -> state
# 6. 保存聊天记录 [回答]
# 7. 返回state
# return state

def state_exists_answer(state) -> bool:
    # 1. 判断是否存在
    answer = state.get("answer")
    if not answer:
        # 没有
        # item_name 确定,正常流程
        logger.info(f"answer内容为空,业务正常进行查询的!跳入回答环节!")
        return False
    # 2. 存在 判定是不是流式
    is_stream = state.get("is_stream",False)
    if is_stream and not state.get("defer_delivery"):
        # 是流式 现在就要结果给推送队列中
        push_to_session(execution_id(state), SSEEvent.DELTA, data={"delta":answer, "task_id": execution_id(state)})
    logger.info("使用前置回答，跳过回答模型")
    return True
def load_answer_prompt(state : QueryGraphState) -> str:
    """在应用预算内组装完整问题、近期历史与整条证据。"""
    # 理解节点读到的历史快照不包含刚保存的本次提问，避免重复拼入当前问题。
    message_list = state.get("history") if "history" in state else history_repository.list_recent(state.get("session_id"), limit=6)
    return pack_answer_prompt(state, message_list or [])

def call_llm_deal_answer(state, answer_prompt_text):
    """
    处理answer
    :param state:
    :param answer_prompt_text:
    :return:
    """
    # 流式和非流式
    is_stream = state.get("is_stream", False)
    answer = ""
    finish_reason = None
    # 准备模型
    question = state.get("rewritten_query") or state.get("original_query") or ""
    llm_client = llm_providers.chat().bind(**completion_limit(output_budget(question, ANSWER_MAX_OUTPUT_TOKENS)))
    messages= [HumanMessage(
            content=answer_prompt_text
        )]
    # 流式和非流式
    # 流式和非流式
    if is_stream:
        # 流式
        stream = llm_client.stream(messages)
        for chunk in stream:
            # 增量数据 delta -> 队列中
            # 草稿先在服务端收集；来源复核后才向用户交付正文。
            if chunk.content:trace_mark(state.get('task_id') or state.get('session_id'),'draft_first_token')
            answer += chunk.content
            finish_reason = (getattr(chunk, "response_metadata", None) or {}).get("finish_reason") or finish_reason
            if getattr(chunk,"usage_metadata",None):state["generation_usage"] = chunk.usage_metadata
    else:
        # 非流
        response = llm_client.invoke(messages)  # AIMessage(content = 结果)
        answer = response.content
        state["generation_usage"] = getattr(response, "usage_metadata", None) or {}
        finish_reason = (getattr(response, "response_metadata", None) or {}).get("finish_reason")
    if finish_reason == "length":
        state["answer_audit"] = {"raw_answer": answer, "finish_reason": "length"}
        raise ContextBudgetError("回答达到输出上限，不能将截断草稿作为完整答案。")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("回答模型未返回有效文本")
    attempts = [{"phase":"initial_answer","raw_answer":answer,"finish_reason":finish_reason,
                 "usage":state.get("generation_usage",{})}]
    issues = answer_issues(question,answer,finish_reason)
    if issues:
        response = llm_client.invoke([*messages, HumanMessage(content="上一次草稿未通过完整性检查："+", ".join(issues)+
            "。请重新完整回答当前问题并遵守原文、引用和篇幅要求；不能只输出标题或开头，不用无依据内容凑字数。")])
        answer = response.content
        finish_reason = (getattr(response,"response_metadata",None) or {}).get("finish_reason")
        usage = getattr(response,"usage_metadata",None) or {}
        attempts.append({"phase":"bounded_answer_repair","raw_answer":answer,"finish_reason":finish_reason,"usage":usage,"reasons":issues})
        issues = answer_issues(question,answer,finish_reason)
        if issues:
            raise IncompleteAnswerError(issues,{"generation_attempts":attempts})
        state["generation_usage"] = {key:sum((row.get("usage") or {}).get(key,0) or 0 for row in attempts)
                                     for key in ("input_tokens","output_tokens","total_tokens")}
    state.setdefault("answer_audit",{})["generation_attempts"] = attempts
    state["answer"] = answer

def extract_text_image_url(state):
    """
    处理图片
        text
        url
    :param state:
    :return:
    """
    # 获取reranked_docs  text url
    reranked_docs = (state.get("citations") or []) if state.get("citation_status", {}).get("status") in {
        "valid", "invalid", "insufficient"
    } else (state.get("reranked_docs") or [])
    # 定义正则规则
    image_re = re.compile(r"\!\[.*?\]\((.*?)\)")  # findall
    image_urls = []
    # 循环出来每个doc url | text
    for doc in reranked_docs:
        url: str = doc.get("url")
        text: str = doc.get("text") or ""
        # str.endswith 只接受 str 或 tuple，SUPPORTED_IMAGE_EXTENSIONS 是 set。
        # 同时忽略 URL 查询参数，避免合法图片链接因 ?token=... 判断失败。
        url_path = url.lower().split("?", 1)[0].split("#", 1)[0] if url else ""
        if url_path.endswith(tuple(SUPPORTED_IMAGE_EXTENSIONS)):
            image_urls.append(url)
        url_list = image_re.findall(text)
        if url_list and len(url_list) > 0:
            image_urls.extend(url_list)
    # image_urls
    image_urls = list(dict.fromkeys(image_urls))
    state['image_urls'] = image_urls
    # 图片地址由证据资产确定，避免模型逐字生成哈希地址时写错字符。
    answer = state.get("answer")
    if isinstance(answer, str) and "【图片】" in answer:
        body = answer.split("【图片】", 1)[0].rstrip()
        state["answer"] = body + ("\n\n【图片】\n" + "\n".join(f"<{url}>" for url in image_urls) if image_urls else "")

def save_answer_message_history(state):
    state.setdefault("answer_audit", {})["context_budget"] = state.get("context_budget", {})
    history_repository.save_message(
        session_id=state.get("session_id"),
        task_id=state.get("task_id") or "",
        run_id=state.get("run_id"),
        input_revision=state.get("input_revision", 1),
        message_kind="answer",
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", []),
        citations=state.get("citations", []),
        citation_status=state.get("citation_status", {}),
        evidence_sources=state.get("evidence_sources", []),
        answer_audit=state.get("answer_audit", {}),
    )

def generate_answer(state: QueryGraphState, *, persist_history: bool = True) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    # 1. state中是否存在answer -> 可以返回字符串了
    persist_history = persist_history and not state.get("defer_history")
    has_answer: bool = state_exists_answer(state)
    # 2. 判断不存在...
    if not has_answer:
        # 没有answer 模型润色answer 提取image_urls
        # answer_prompt_text =  声明一个提示词拼接的方法(state) 拼接
        try:
            answer_prompt_text: str = load_answer_prompt(state)
        except ContextBudgetError:
            state.update(answer="完整问题与提示超出本次回答预算，请缩小问题范围。", citations=[],
                         citation_status={"status": "invalid", "reason": "context_budget_exceeded", "support_check": "not_assessed"})
            if persist_history:
                save_answer_message_history(state)
            return state
        if not state["evidence_sources"]:
            state["answer"] = ("当前资料片段无法完整装入本次回答预算，请缩小问题范围或先细分资料。"
                               if state.get("context_budget", {}).get("candidate_count") else INSUFFICIENT_ANSWER)
            state["citations"] = []
            state["citation_status"] = {"status": "insufficient", "reason": "empty_evidence",
                                        "support_check": "not_assessed"}
            if state.get("is_stream") and not state.get("defer_delivery"):
                push_to_session(execution_id(state), SSEEvent.DELTA, data={"delta": state["answer"], "task_id": execution_id(state)})
        else:
            try:
                with trace_span(state.get('task_id') or state.get('session_id'),'answer_generate'):
                    call_llm_deal_answer(state, answer_prompt_text)
                validate_answer_citations(state)
                state["answer_audit"]["generation_usage"] = state.get("generation_usage", {})
                with trace_span(state.get('task_id') or state.get('session_id'),'source_review'):
                    review_answer_support(state)
                if state.get("citation_status",{}).get("status") == "valid":
                    issues = answer_issues(state.get("rewritten_query") or state.get("original_query") or "",state["answer"])
                    if issues:raise IncompleteAnswerError(issues,state.get("answer_audit"))
            except ContextBudgetError:
                state.update(answer="回答达到本次输出预算，请缩小问题范围后重试。", citations=[],
                             citation_status={"status": "invalid", "reason": "output_budget_exceeded", "support_check": "not_assessed"})
        # 使用正则或者图片url匹配获取image_urls
        extract_text_image_url(state)
        if state.get("is_stream") and not state.get("defer_delivery") and state["evidence_sources"]:
            push_to_session(execution_id(state), SSEEvent.DELTA, data={"delta": state["answer"], "task_id": execution_id(state)})

    else:
        state["citations"] = []
        state["citation_status"] = {"status": "not_required", "support_check": "not_assessed"}
    if persist_history:
        save_answer_message_history(state)
    return state
