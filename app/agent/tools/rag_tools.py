"""复用现有RAG能力，不递归调用完整聊天图，不让模型提交所谓的论文原文。"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

from app.agent.tools.schemas import (
    CompareEvidenceInput, DocumentContextInput, KnowledgeSearchInput, ToolError,
    ToolPayload, WebSearchInput, SourceReference,
)
from pydantic import ValidationError
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.query.agent.state import create_query_default_state
from app.rag.query import answer_service, embedding_search_service, hyde_search_service
from app.rag.query import rerank_service, rrf_service, web_search_service
from app.rag.query.evidence_service import original_source_url, safe_web_url
from app.rag.query.official_source_lookup import lookup_official_sources
from app.rag.import_.chunk_metadata import SOURCE_FIELDS


def _source_links(documents):
    docs = deepcopy(documents)
    for doc in docs:
        doc["source_url"] = safe_web_url(doc.get("url")) if doc.get("type") == "web" else original_source_url(doc)
    return docs


def search_knowledge(args: KnowledgeSearchInput) -> ToolPayload:
    state = create_query_default_state(rewritten_query=args.query, original_query=args.query)
    with ThreadPoolExecutor(max_workers=2) as pool:
        ordinary = pool.submit(embedding_search_service.search_by_embedding, deepcopy(state))
        hyde = pool.submit(hyde_search_service.search_by_hyde, deepcopy(state))
        state["embedding_chunks"], state["hyde_embedding_chunks"] = ordinary.result(), hyde.result()
    if not state["embedding_chunks"] and not state["hyde_embedding_chunks"]:
        return ToolPayload(status="empty", data={"documents": []})
    rrf_service.fuse_by_rrf(state)
    rerank_service.rerank_documents(state)
    traceable = []
    for doc in state["reranked_docs"]:
        try:
            SourceReference.model_validate({key: doc.get(key) for key in ("doc_id", "chunk_uid")})
        except ValidationError:
            continue
        traceable.append(doc)
    documents = _source_links(traceable[:args.top_k])
    return ToolPayload(status="ok" if documents else "empty", data={
        "documents": documents, "rerank_status": state["rerank_status"],
        "branch_counts": {"ordinary": len(state["embedding_chunks"]), "hyde": len(state["hyde_embedding_chunks"])},
        "legacy_without_identity_excluded": len(state["reranked_docs"]) - len(traceable),
    })


def search_web(args: WebSearchInput) -> ToolPayload:
    official,lookup_errors=lookup_official_sources(args.query)
    if official:
        documents=rerank_service.deal_rrf_and_web_result([],official[:args.top_k])
        for doc,page in zip(documents,official):
            doc['content_scope']=page['content_scope'];doc['retrieval_backend']=page['retrieval_backend']
        return ToolPayload(status='ok',data={'documents':_source_links(documents),
            'web_search_status':{'status':'ok','backend':'registered_official_source','returned_count':len(documents),'mcp_called':False,'lookup_errors':lookup_errors},
            'content_scope':'actual_official_source_partial_extract_not_full_text'})
    state = create_query_default_state(rewritten_query=args.query)
    web_search_service.search_by_web(state)
    status = state["web_search_status"]
    if status["status"] == "failed":
        return ToolPayload(status="failed", data={"web_search_status": status}, error=ToolError(
            code="web_search_failed", message="联网检索未完成，本次没有可用联网证据。", retryable=True))
    candidates = rerank_service.deal_rrf_and_web_result([], state["web_search_docs"])
    filtered = rerank_service.filter_conflicting_web_documents(args.query, candidates)
    docs = filtered[:args.top_k]
    return ToolPayload(status="ok" if docs else "empty", data={
        "documents": _source_links(docs), "web_search_status": status,
        "content_scope": "search_summary_not_full_page",
        "explicit_conflicts_excluded": len(candidates) - len(filtered),
        "official_lookup_errors":lookup_errors,
    })


CONTEXT_FIELDS = list(dict.fromkeys([
    "chunk_id", "content", "title", "parent_title", "file_title", "item_name", "part",
    *SOURCE_FIELDS, "index_revision",
]))


def read_document_context(args: DocumentContextInput) -> list[dict]:
    client = milvus_gateway.milvus_client
    if client is None:
        raise ConnectionError("Milvus不可用")
    doc_filter = f"doc_id == {json.dumps(args.doc_id)}"
    anchor_filter = f"{doc_filter} and chunk_uid == {json.dumps(args.chunk_uid)}"
    anchors = client.query(collection_name=milvus_gateway.chunk_collection_name, filter=anchor_filter,
                           output_fields=CONTEXT_FIELDS, limit=2, consistency_level="Strong", timeout=30)
    if not isinstance(anchors, list):
        raise RuntimeError("原文读取返回格式错误，不能当作正常零命中")
    if not anchors:
        return []
    if len(anchors) != 1:
        raise ValueError("切片身份同时对应多个发布批次，不能任选或混合")
    anchor = anchors[0]
    if anchor.get("doc_id") != args.doc_id or anchor.get("chunk_uid") != args.chunk_uid:
        raise RuntimeError("读取结果与请求的原文身份不对应")
    rows = anchors
    revision, center = anchor.get("index_revision"), anchor.get("chunk_index")
    if args.neighbors and revision and type(center) is int:
        low, high = max(0, center - args.neighbors), center + args.neighbors
        expression = (f"{doc_filter} and index_revision == {json.dumps(revision)} "
                      f"and chunk_index >= {low} and chunk_index <= {high}")
        rows = client.query(collection_name=milvus_gateway.chunk_collection_name, filter=expression,
                            output_fields=CONTEXT_FIELDS, limit=2 * args.neighbors + 1,
                            consistency_level="Strong", timeout=30)
        if (not rows or not any(row.get("chunk_uid") == args.chunk_uid for row in rows)
                or any(row.get("doc_id") != args.doc_id or row.get("index_revision") != revision for row in rows)):
            raise RuntimeError("扩展上下文时来源已变化，请重新检索定位")
        rows.sort(key=lambda row: row["chunk_index"])
    chunks = embedding_search_service.deal_milvus_list([{"entity": row, "distance": 0.0} for row in rows])
    docs = _source_links(rerank_service.deal_rrf_and_web_result(chunks, []))
    for doc, row in zip(docs, rows):
        if row.get("index_revision"):
            doc["index_revision"] = row["index_revision"]
    return docs


def get_document_context(args: DocumentContextInput) -> ToolPayload:
    documents = read_document_context(args)
    return ToolPayload(status="ok" if documents else "empty", data={
        "documents": documents, "anchor_chunk_uid": args.chunk_uid,
        "requested_neighbors": args.neighbors, "returned_count": len(documents),
    })


def compare_evidence(args: CompareEvidenceInput) -> ToolPayload:
    docs = []
    for source in args.sources:
        found = read_document_context(DocumentContextInput(**source.model_dump(), neighbors=0))
        if not found:
            return ToolPayload(status="empty", data={"reason": "source_not_found", "documents": []})
        docs.extend(found)
    state = create_query_default_state(original_query=args.question, rewritten_query=args.question,
                                       reranked_docs=docs, history=[])
    # 工具只是比较所选真实片段；最终会话消息由外层Agent保存，避免内部每次调用都污染历史。
    answer_service.generate_answer(state, persist_history=False)
    status = state["citation_status"]["status"]
    if status == "invalid":
        return ToolPayload(status="failed", error=ToolError(
            code="comparison_review_failed", message="证据比较未通过来源检查，暂不提供事实结论。"))
    return ToolPayload(status="empty" if status == "insufficient" else "ok", data={
        "answer": state["answer"], "citations": state["citations"],
        "citation_status": state["citation_status"], "documents": docs,
        "scope": "selected_source_fragments_only",
    })
