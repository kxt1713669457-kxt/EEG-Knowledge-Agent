"""本轮证据编号与引用校验；编号有效不代表主张已经通过事实核验。"""
from copy import deepcopy
import json
import re
from urllib.parse import urlsplit


INSUFFICIENT_ANSWER = "当前检索资料不足以回答这一问题，请补充相关原文或缩小问题范围。"
INVALID_REFERENCE_ANSWER = "本次生成的引用未能对应到检索原文，暂不提供该回答。请重新提问或补充资料。"
NO_EVIDENCE_MARKER = "[NO_EVIDENCE]"
REFERENCE_RE = re.compile(r"\[([EA]\d+)\]")
ORIGINAL_TYPES = {"pdf", "docx", "md", "html", "htm"}


def original_source_url(doc: dict) -> str:
    """只为本项目实际归档规则对应的对象生成原件入口。"""
    image_id = doc.get("image_id") or ""
    if doc.get("type") == "image_observation" and re.fullmatch(r"[0-9a-f]{32}",image_id) and doc.get("source_url") == f"/workspace/images/{image_id}":
        return doc["source_url"]
    file_id = doc.get("file_id") or ""
    if (doc.get("type") == "workspace_file" and re.fullmatch(r"[0-9a-f]{32}", file_id)
            and doc.get("source_url") == f"/workspace/files/{file_id}/download"):
        return doc["source_url"]
    doc_id = doc.get("doc_id") or ""
    source_type = str(doc.get("source_type") or "").lower().lstrip(".")
    if (re.fullmatch(r"[0-9a-f]{64}", doc_id) and source_type in ORIGINAL_TYPES
            and doc.get("object_key") == f"originals/{doc_id}/source.{source_type}"):
        return f"/sources/original/{doc_id}/{source_type}"
    return ""


def safe_web_url(value) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme in {"http", "https"} and parsed.netloc else ""
    except ValueError:
        return ""


def build_evidence_sources(documents: list[dict]) -> list[dict]:
    """E编号只在本轮回答内使用；真实文档身份另由doc_id/chunk_uid定位。"""
    sources = []
    counts={"E":0,"A":0}
    for doc in documents:
        text = doc.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        source = deepcopy(doc)
        is_artifact=doc.get("type")=="analysis_result"
        prefix="A" if is_artifact else "E"
        if is_artifact and not re.fullmatch(r"[0-9a-f]{32}",doc.get("artifact_id") or ""):
            continue
        counts[prefix]+=1
        source.update(evidence_id=f"{prefix}{counts[prefix]}", text=text,
                      source_url=safe_web_url(doc.get("url")) if doc.get("type") == "web"
                      else f"/agent/eeg/artifacts/{doc['artifact_id']}" if is_artifact else original_source_url(doc))
        # 没有页码就保留缺失；不把章节、MD行号或chunk序号伪装成PDF页码。
        for field in ("page_start", "page_end"):
            value = source.get(field)
            source[field] = value if type(value) is int and value > 0 else None
        sources.append(source)
    return sources


def format_evidence_context(sources: list[dict]) -> str:
    records = []
    for source in sources:
        records.append({
            "引用标识": f"[{source['evidence_id']}]",
            "来源性质": ("真实模型分类执行产物（预测不是人工真值；分数未校准）" if source.get("analysis_output", {}).get("classification_available") else "真实模型执行产物（不是论文；表征不是分类预测）") if source.get("type")=="analysis_result" else
                        "登记官方来源的实际提取片段（未读取全文）" if source.get('retrieval_backend')=='registered_official_source' else
                        "公开原件实际页面摘录（仅所列页面）" if source.get("retrieval_backend")=="public_document_reader" else "联网摘要（未抓取全文）" if source.get("type") == "web" else "用户图片的模型观察（不是论文原文或数值分类）" if source.get("type") == "image_observation" else "本地资料原文",
            "artifact_id":source.get("artifact_id"),
            "内容范围":source.get('content_scope'),
            "检索获取时间":source.get("retrieved_at"),
            "标题": source.get("title"), "文件": source.get("file_title"),
            "关联主体": source.get("item_name"), "doc_id": source.get("doc_id"),
            "chunk_uid": source.get("chunk_uid"), "内容快照": source.get("version"),
            "章节": source.get("section") or source.get("title"),
            "页码起": source.get("page_start"), "页码止": source.get("page_end"),
            "来源链接": source.get("source_url"), "获取时间": source.get("retrieved_at"),
            "排序分数（非事实正确概率）": source.get("score"), "原文": source["text"],
        })
    return json.dumps(records, ensure_ascii=False, indent=2)


def validate_answer_citations(state: dict) -> None:
    """核对引用是否指向本轮证据；不将ID检查冒充语义支持判断。"""
    raw_answer = state.get("answer") or ""
    # 中文模型偶尔仅改变引用括号；只归一明确的编号，不补造来源或改前缀。
    normalized_answer = re.sub(r"【([EA]\d+)】|［([EA]\d+)］",
                               lambda match: f"[{match.group(1) or match.group(2)}]", raw_answer)
    sources = state.get("evidence_sources") or []
    by_id = {source["evidence_id"]: source for source in sources}
    used_ids = list(dict.fromkeys(REFERENCE_RE.findall(normalized_answer)))
    invalid_ids = [evidence_id for evidence_id in used_ids if evidence_id not in by_id]
    status = {"status": "valid", "used_ids": used_ids, "invalid_ids": invalid_ids,
              "support_check": "not_assessed"}
    state.setdefault("answer_audit", {})["raw_answer"] = raw_answer
    if normalized_answer != raw_answer:
        state["answer_audit"].update(citation_format_normalized=True, normalized_answer=normalized_answer)
    if NO_EVIDENCE_MARKER in raw_answer or "【NO_EVIDENCE】" in raw_answer:
        status.update(status="insufficient", used_ids=[])
        state["answer"] = INSUFFICIENT_ANSWER
        state["citations"] = []
    elif invalid_ids or not used_ids:
        status.update(status="invalid", reason="unknown_reference" if invalid_ids else "missing_reference")
        state["answer"] = INVALID_REFERENCE_ANSWER
        state["citations"] = []
    else:
        state["answer"] = normalized_answer
        state["citations"] = [deepcopy(by_id[evidence_id]) for evidence_id in used_ids]
    state["citation_status"] = status
    state["answer_audit"]["citation_status"] = deepcopy(status)
