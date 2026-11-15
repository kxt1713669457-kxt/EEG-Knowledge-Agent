"""原件身份、对象归档及本地来源记录；不承担解析或向量入库。"""
import hashlib
import json
from pathlib import Path

from app.infra.object_storage.minio_gateway import minio_gateway
from app.process.import_.agent.state import ImportGraphState


def describe_document(local_file_path: str | Path) -> dict:
    source = Path(local_file_path)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    if not size:
        raise ValueError("文件内容为空")
    file_hash = digest.hexdigest()
    return {
        "doc_id": file_hash,
        "file_hash": file_hash,
        # 内容快照标识，不自动推断论文的投稿/出版版本关系。
        "version": file_hash[:12],
        "original_filename": source.name,
        "title": source.stem,
        "source_type": source.suffix.lower().lstrip("."),
        "size_bytes": size,
        "archive_status": "pending",
        "bucket": "",
        "object_key": "",
    }


def write_source_manifest(local_dir: str | Path, task_id: str, metadata: dict) -> Path:
    directory = Path(local_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "source.json"
    temporary = directory / "source.json.tmp"
    temporary.write_text(json.dumps({"schema_version": 1, "task_id": task_id,
                                     "document": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def archive_original_document(state: ImportGraphState) -> dict:
    metadata = describe_document(state["local_file_path"])
    previous = state.get("document_metadata") or {}
    if previous.get("file_hash") and previous["file_hash"] != metadata["file_hash"]:
        raise ValueError("文件内容在任务登记后发生变化")
    object_key = minio_gateway.archive_original(state["local_file_path"], metadata["file_hash"])
    metadata.update(archive_status="completed", bucket=minio_gateway.bucket_name, object_key=object_key)
    state["document_metadata"] = metadata
    write_source_manifest(state["local_dir"], state["task_id"], metadata)
    return metadata
