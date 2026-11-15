"""原件身份随切片传递；数据库主键与内容定位标识各有用途。"""
import hashlib
import json
import re

from app.infra.config.providers import infra_config
from app.rag.import_ import config


SOURCE_FIELDS = (
    "doc_id", "file_hash", "version", "original_filename", "source_type",
    "object_key", "bucket", "section", "chunk_uid", "chunk_index", "index_profile",
)


def processing_profile() -> str:
    # 解析/增强/切分代码或模型语义改变时也要提升revision，不把旧结果当新配置结果。
    spec = {"revision": "day03-tolerant600-overlap100-source-spans-v3", "parser": config.MINERU_MODEL_VERSION,
            "chunk_size": config.CHUNK_SIZE, "overlap": config.CHUNK_OVERLAP,
            "max_size": config.CHUNK_MAX_SIZE, "min_size": config.CHUNK_MIN,
            "embedding_max_tokens": infra_config.embedding_config.max_input_tokens,
            "embedding": infra_config.embedding_config.bge_m3 or "bge-m3"}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    return f"d3-{digest}"


def require_doc_id(metadata: dict) -> str:
    doc_id = metadata.get("doc_id", "")
    if not isinstance(doc_id, str) or not re.fullmatch(r"[0-9a-f]{64}", doc_id):
        raise ValueError("缺少有效的原件doc_id，不能按文件名代替文档身份")
    return doc_id


def attach_chunk_metadata(chunks: list[dict], metadata: dict) -> list[dict]:
    doc_id = require_doc_id(metadata)
    profile = processing_profile()
    for index, chunk in enumerate(chunks):
        section = chunk.get("parent_title") or chunk.get("title") or ""
        identity = json.dumps([doc_id, profile, index, chunk["content"]], ensure_ascii=False)
        chunk.update({key: metadata.get(key, "") for key in SOURCE_FIELDS[:8]})
        chunk.update(doc_id=doc_id, section=section, chunk_index=index,
                     chunk_uid=hashlib.sha256(identity.encode("utf-8")).hexdigest(), index_profile=profile)
    # 原文没有可靠页码时不添加页码，不能把Markdown行号伪装为PDF页码。
    return chunks


def source_fields(row: dict) -> dict:
    """旧记录没有新增字段时保持缺失，兼容尚未迁移的知识。"""
    return {key: row[key] for key in SOURCE_FIELDS if key in row}
