"""新批次先写入并核验，再按doc_id替换；不按文件名删除其他资料。"""
import json
import uuid

from app.rag.import_.chunk_metadata import require_doc_id
from app.shared.runtime.logger import logger


def count_rows(client, collection: str, expression: str) -> int:
    rows = client.query(collection_name=collection, filter=expression,
                        output_fields=["count(*)"], consistency_level="Strong", timeout=30)
    return int(rows[0]["count(*)"]) if rows else 0


def publish_document_rows(client, collection: str, rows: list[dict], doc_id: str) -> dict:
    require_doc_id({"doc_id": doc_id})
    if not rows or any(row.get("doc_id") != doc_id for row in rows):
        raise ValueError("待写入记录的文档身份不一致")
    revision = uuid.uuid4().hex
    doc_filter = f"doc_id == {json.dumps(doc_id)}"
    current_filter = f"{doc_filter} and index_revision == {json.dumps(revision)}"
    payload = [{**row, "index_revision": revision} for row in rows]
    try:
        result = client.insert(collection_name=collection, data=payload, timeout=30)
        if result.get("insert_count") != len(payload) or count_rows(client, collection, current_filter) != len(payload):
            raise RuntimeError("新索引批次写入数量与预期不一致")
    except Exception:
        # 只清理本次可能部分写入的批次，先前可用数据不受影响。
        try:
            client.delete(collection_name=collection, filter=current_filter, timeout=30)
        except Exception as cleanup_error:
            logger.warning(f"失败批次清理未完成（{type(cleanup_error).__name__}），保留原始写入异常")
        raise
    # 这不是跨集合数据库事务；清理失败则上抛，不生成成功回执，重试再处理。
    client.delete(collection_name=collection,
                  filter=f"{doc_filter} and index_revision != {json.dumps(revision)}", timeout=30)
    if count_rows(client, collection, doc_filter) != len(payload):
        raise RuntimeError("文档索引清理后数量异常，不能标记导入完成")
    return {"status": "completed", "index_revision": revision,
            "chunk_count": len(payload), "ids": list(result.get("ids", []))}
