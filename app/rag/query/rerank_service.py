import math
import re
from numbers import Real
from pathlib import Path

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_WINDOW_OVERLAP_TOKENS, \
    RERANK_MAX_TOPK, RERANK_MIN_TOPK, RERANK_GAP_ABS, RERANK_GAP_RATIO, \
    FINAL_LOCAL_EVIDENCE_TOPK, FINAL_WEB_EVIDENCE_TOPK
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_providers
from app.shared.config.reranker_config import reranker_config
from app.rag.query.config import FALLBACK_LOCAL_EVIDENCE_TOPK, FALLBACK_WEB_EVIDENCE_TOPK
from app.rag.import_.chunk_metadata import source_fields
from app.rag.query.coverage_selection import select_with_subject_coverage
def get_data_and_validates(state):
    # 1. 获取参数
    rewritten_query = state.get("rewritten_query")
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    # 2. 非空判断
    # 网页搜索是补充来源；网页服务暂时不可用时，仍应使用知识库结果回答。
    if not rewritten_query or (not rrf_chunks and not web_search_docs):
        logger.error("rewritten_query 为空，或知识库与网页检索结果均为空，无法生成答案")
        raise ValueError("rewritten_query 为空，或知识库与网页检索结果均为空，无法生成答案")
    return rewritten_query, rrf_chunks, web_search_docs

def deal_rrf_and_web_result(rrf_chunks, web_search_docs):
    # 1. 定义一个列表
    reranker_docs = []
    # 2. 先循环rrf
    for chunk in rrf_chunks:
        reranker_docs.append({
            **source_fields(chunk),
            **({key:chunk.get(key) for key in ("file_id", "workspace_id", "source_url", "content_scope")}
               if chunk.get("type") == "workspace_file" else {}),
            "chunk_id":chunk.get("chunk_id"),
            "text":chunk.get("content") or "",
            "title":chunk.get("title"),
            "file_title": chunk.get("file_title"),
            "item_name": chunk.get("item_name"),
            "score":chunk.get("score", 0.0), # 未启用模型时保留 RRF 分数
            "rrf_score": chunk.get("rrf_score", chunk.get("score", 0.0)),
            "rrf_contributions": chunk.get("rrf_contributions", []),
            "type":chunk.get("type") or "milvus",
            "url":None
        })
    # 3. 再循环web_search
    # 3. 再循环web_search
    for doc in web_search_docs:
        reranker_docs.append({
            "chunk_id": None,
            "text": doc.get("snippet") or "",
            "title": doc.get("title"),
            "score": 0, # rrf分 -> reranker打的分
            "type": "web",
            "retrieved_at": doc.get("retrieved_at"),
            "url": doc.get("url")
        })
    return reranker_docs


def create_question_answer_list(rewritten_query, reranker_docs):
    """按实际token预算构造问题/原文窗口；完整text保留供最终回答。

    长片段使用重叠窗口，避免LLM压缩改写数值，也不让SDK静默截掉尾部。
    rerank_windows记录窗口对应哪一个输入对，后续将分数聚合回原文档。
    """
    tokenizer = llm_providers.reranker_model().tokenizer
    query_tokens = len(tokenizer.encode(rewritten_query, add_special_tokens=False, verbose=False))
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    passage_budget = RERANK_MAX_INPUT_TOKENS - query_tokens - special_tokens
    if passage_budget < 1:
        raise ValueError("重排问题query过长，请先缩短或拆分问题，不能静默截断")
    pairs = []
    for doc in reranker_docs:
        text = doc.get("text") or ""
        token_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        windows = []
        start = 0
        while True:
            end = min(start + passage_budget, len(token_ids))
            passage = text if start == 0 and end == len(token_ids) else tokenizer.decode(
                token_ids[start:end], skip_special_tokens=True)
            # decode再encode可能改变边界分词，复核实际送入模型的长度。
            while True:
                pair_tokens = query_tokens + len(tokenizer.encode(passage, add_special_tokens=False, verbose=False)) + special_tokens
                if pair_tokens <= RERANK_MAX_INPUT_TOKENS:
                    break
                end -= max(1, pair_tokens - RERANK_MAX_INPUT_TOKENS)
                if end <= start:
                    raise ValueError("重排窗口无法在token预算内表示，需要先细分原文")
                passage = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
            windows.append({"pair_index": len(pairs), "token_start": start,
                            "token_end": end, "pair_tokens": pair_tokens})
            pairs.append([rewritten_query, passage])
            if end >= len(token_ids):
                break
            overlap = min(RERANK_WINDOW_OVERLAP_TOKENS, (end - start) // 2)
            start = end - overlap
        doc["rerank_windows"] = windows
    return pairs


def use_reranker_deal_score(question_answer_pair_list, reranker_docs):
    """完整校验所有分数后，以最佳窗口相关性聚合到原文档，再排序。"""
    reranker_model = llm_providers.reranker_model()
    raw_scores = reranker_model.compute_score(
        question_answer_pair_list, normalize=True,
        batch_size=reranker_config.batch_size, max_length=RERANK_MAX_INPUT_TOKENS,
        query_max_length=RERANK_MAX_INPUT_TOKENS,
    )
    scores = [raw_scores] if isinstance(raw_scores, Real) else list(raw_scores)
    if len(scores) != len(question_answer_pair_list):
        raise RuntimeError("重排score分数数量与输入对不一致，不能保留部分旧分数")
    scores = [float(score) for score in scores]
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise RuntimeError("重排score分数包含非有限值或超出归一化范围")
    updates = []
    for index, doc in enumerate(reranker_docs):
        windows = doc.get("rerank_windows") or [{"pair_index": index}]
        window_scores = [scores[window["pair_index"]] for window in windows]
        updates.append((max(window_scores), window_scores))
    for doc, (score, window_scores) in zip(reranker_docs, updates):
        doc.update(score=score, rerank_score=score, rerank_window_count=len(window_scores),
                   rerank_window_scores=window_scores)
    reranker_docs.sort(key=lambda doc: doc["rerank_score"], reverse=True)


def dyn_limit_reranker_docs(reranker_docs):
    """旧分数断崖策略，保留作对照；生产路径使用select_evidence。

    少于最少条数时全部返回，不填充候选；最少条数不代表事实证据质量保证。
    """
    top_max = min(RERANK_MAX_TOPK, len(reranker_docs))
    topk = top_max
    for index in range(RERANK_MIN_TOPK - 1, top_max - 1):
        previous = reranker_docs[index]["score"]
        following = reranker_docs[index + 1]["score"]
        difference = previous - following
        ratio = difference / previous if previous > 0 else 0.0
        if difference > RERANK_GAP_ABS or ratio > RERANK_GAP_RATIO:
            topk = index + 1
            break
    return reranker_docs[:topk]


def filter_conflicting_web_documents(query, docs):
    """只排除明确的2a/2b身份冲突；没有标签或比较问题不作此硬过滤。"""
    def labels(text):
        return set(re.findall(r"(?<![A-Za-z0-9])2[ab](?![A-Za-z0-9])", text, flags=re.I))
    expected = {label.lower() for label in labels(query)}
    kept = []
    for doc in docs:
        found = {label.lower() for label in labels(str(doc.get("title") or "") + " " + doc["text"])}
        if doc["type"] == "web" and len(expected) == 1 and found and expected.isdisjoint(found):
            continue
        kept.append(doc)
    return kept


def select_evidence(reranker_docs, rrf_chunks, item_names=()):
    """按真实语义分数调整各源内部顺序，沿用较宽本地额度及网页独立额度。"""
    local_docs = [doc for doc in reranker_docs if doc["type"] in {"milvus", "workspace_file"}]
    web_docs = [doc for doc in reranker_docs if doc["type"] == "web"]
    prior_ranks = {doc["chunk_id"]: index for index, doc in enumerate(rrf_chunks, 1)}
    for semantic_rank, doc in enumerate(local_docs, 1):
        prior_rank = prior_ranks[doc["chunk_id"]]
        doc.update(rrf_rank=None if doc["type"] == "workspace_file" else prior_rank, semantic_rank=semantic_rank,
            selection_score=doc["rerank_score"], selection_basis="explicit_file_cross_encoder" if doc["type"] == "workspace_file" else "cross_encoder_with_local_quota")
    for doc in web_docs:
        doc.update(selection_score=doc["rerank_score"], selection_basis="cross_encoder_with_web_quota")
    return select_with_subject_coverage(local_docs,item_names,FINAL_LOCAL_EVIDENCE_TOPK) + web_docs[:FINAL_WEB_EVIDENCE_TOPK]


def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 各源内部按语义分数排序，本地与网页独立限额
    4. 回写 reranked_docs
    """
    #1.获取并校验参数参数（state）rewritten_query  rrf_chunks  web_search_docs
    rewritten_query, rrf_chunks, web_search_docs = get_data_and_validates(state)
    # 2. 数据格式化处理
    reranker_docs = deal_rrf_and_web_result(rrf_chunks, web_search_docs)
    original_document_count = len(reranker_docs)
    reranker_docs = filter_conflicting_web_documents(rewritten_query, reranker_docs)
    input_document_count = len(reranker_docs)
    if not reranker_docs:
        state["reranked_docs"] = []
        state["answer"] = "本次候选资料与所问数据集存在明确冲突，未找到可用证据。请补充资料或明确查询范围。"
        state["rerank_status"] = {"mode": "no_evidence", "input_documents": 0,
            "filtered_web_count": original_document_count, "selected_count": 0}
        return state
    # bge-reranker-large 在内存不足的 Windows 进程中可能直接触发原生崩溃，
    # 这类错误无法被 try/except 捕获。关闭时沿用 RRF 排序，保证回答链路可用。
    if not reranker_config.enabled:
        logger.warning("本地重排模型未启用，使用 RRF 分数排序继续生成答案")
        local_docs = sorted((doc for doc in reranker_docs if doc["type"] in {"milvus", "workspace_file"}),
                            key=lambda item: item.get("score", 0.0), reverse=True)
        web_docs = [doc for doc in reranker_docs if doc["type"] == "web"]
        state["reranked_docs"] = local_docs[:FALLBACK_LOCAL_EVIDENCE_TOPK] + web_docs[:FALLBACK_WEB_EVIDENCE_TOPK]
        state["rerank_status"] = {"mode": "disabled", "input_documents": len(reranker_docs),
                                  "selected_count": len(state["reranked_docs"])}
        return state

    # 3. 组装问题和答案的列表(rewritten_query,reranker_list) -> question_answer_pair_list
    question_answer_pair_list: list[list[str]] = create_question_answer_list(rewritten_query, reranker_docs)
    #4.打分+排序
    logger.info("BGE精排开始：候选{}条，输入窗口{}个", len(reranker_docs), len(question_answer_pair_list))
    use_reranker_deal_score(question_answer_pair_list,reranker_docs)
    logger.info("BGE精排打分完成，开始选择证据")
    reranker_docs = select_evidence(reranker_docs, rrf_chunks, state.get('item_names') or [])
    #更新state
    state["reranked_docs"] = reranker_docs
    state["rerank_status"] = {"mode": "model", "model": Path(reranker_config.bge_reranker_large).name,
        "input_documents": input_document_count,
        "pair_count": len(question_answer_pair_list), "selected_count": len(reranker_docs),
        "batch_size": reranker_config.batch_size, "max_input_tokens": RERANK_MAX_INPUT_TOKENS,
        "filtered_web_count": original_document_count - input_document_count,
        "selection": "source_balanced_semantic_sort", "local_topk": FINAL_LOCAL_EVIDENCE_TOPK,
        "web_topk": FINAL_WEB_EVIDENCE_TOPK,
        "selected_local_chunk_uids": [doc.get("chunk_uid") for doc in reranker_docs if doc["type"] == "milvus"],
        "selected_web_urls": [doc.get("url") for doc in reranker_docs if doc["type"] == "web"]}
    return state
