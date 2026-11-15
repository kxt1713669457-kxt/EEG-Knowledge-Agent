"""成功回执用于重复导入复用；不代替全局任务持久化或图断点恢复。"""
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.rag.import_.chunk_metadata import processing_profile, require_doc_id
from app.rag.import_.index_publication import count_rows
from app.shared.runtime.logger import PROJECT_ROOT

_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def document_import_lock(doc_id: str):
    """当前单进程中，同内容的检查、处理与回执提交串行，其他内容可并行。"""
    require_doc_id({"doc_id": doc_id})
    with _guard:
        return _locks.setdefault(doc_id, threading.Lock())


def receipt_path(doc_id: str) -> Path:
    require_doc_id({"doc_id": doc_id})
    return PROJECT_ROOT / "output/import_receipts" / f"{doc_id}-{processing_profile()}.json"


def find_completed_receipt(doc_id: str, source_type: str | None = None) -> dict | None:
    path = receipt_path(doc_id)
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return None
    index = receipt.get("index_metadata") or {}
    if (receipt.get("doc_id") != doc_id or receipt.get("profile") != processing_profile()
            or index.get("status") != "completed" or not index.get("chunk_count") or not index.get("index_revision")):
        return None
    client = milvus_gateway.milvus_client
    if client is None:
        raise ConnectionError("无法核验Milvus成功回执")
    if not (client.has_collection(collection_name=milvus_gateway.chunk_collection_name)
            and client.has_collection(collection_name=milvus_gateway.item_name_collection_name)):
        return None
    revision_filter = f'doc_id == {json.dumps(doc_id)} and index_revision == {json.dumps(index["index_revision"])}'
    if count_rows(client, milvus_gateway.chunk_collection_name, revision_filter) != index["chunk_count"]:
        return None
    doc_filter = f'doc_id == {json.dumps(doc_id)}'
    if count_rows(client, milvus_gateway.chunk_collection_name, doc_filter) != index["chunk_count"]:
        return None
    if source_type and count_rows(client, milvus_gateway.chunk_collection_name,
                                  f'{doc_filter} and source_type == {json.dumps(source_type)}') != index["chunk_count"]:
        return None
    name_filter = f'doc_id == {json.dumps(doc_id)} and item_name == {json.dumps(receipt.get("item_name", ""))}'
    if count_rows(client, milvus_gateway.item_name_collection_name, name_filter) != 1:
        return None
    if count_rows(client, milvus_gateway.item_name_collection_name, doc_filter) != 1:
        return None
    return receipt


def save_completed_receipt(state: dict) -> dict:
    metadata = state.get("document_metadata") or {}
    doc_id = require_doc_id(metadata)
    index = state.get("index_metadata") or {}
    if index.get("status") != "completed" or not index.get("chunk_count"):
        raise ValueError("缺少已核验的入库结果，不能写成功回执")
    receipt = {"schema_version": 1, "doc_id": doc_id, "profile": processing_profile(),
               "completed_at": datetime.now(timezone.utc).isoformat(), "task_id": state["task_id"],
               "item_name": state["item_name"], "index_metadata": index,
               "parse_metadata": state.get("parse_metadata", {})}
    target = receipt_path(doc_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return receipt
