"""显式选择的文件夹快照：只读资料，不执行目录里的程序。"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import uuid
import zipfile

from app.shared.runtime.logger import PROJECT_ROOT
from app.workspace import store

FILE_ROOT = PROJECT_ROOT / "output/runtime/workspace-files"
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".html", ".htm", ".py", ".r", ".m", ".toml"}
DOCUMENT_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}
MAX_TOTAL_BYTES, MAX_FILE_BYTES = 24 * 1024 * 1024, 8 * 1024 * 1024
_selected = ContextVar("workspace_file_documents", default=())
_selection_options = ContextVar("workspace_file_options", default=None)


def safe_relative_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or len(path.parts) > 16 or any(part in {".", ".."} or ":" in part for part in path.parts):
        raise ValueError("目录相对路径不合法。")
    if any(part.startswith(".") or part.lower() in {"node_modules", "venv", "__pycache__", "models", "dist", "build"} for part in path.parts):
        raise ValueError("跳过隐藏文件、环境与依赖目录。")
    if re.search(r"(?:^|[-_.])(?:secret|credentials|private-key)(?:[-_.]|$)", path.name, re.I) or path.suffix.lower() in {".pem", ".key", ".pfx"}:
        raise ValueError("凭据文件不作为分析资料。")
    if path.suffix.lower() not in DOCUMENT_EXTENSIONS:
        raise ValueError("该格式不属于文档资料；EEG波形请使用样本登记。")
    return str(path)


def extract_text(blob, suffix):
    if suffix in TEXT_EXTENSIONS:
        text = blob.decode("utf-8-sig")
        if "\x00" in text:
            raise ValueError("该文件不是有效UTF-8文本。")
        return text
    if suffix == ".docx":
        from xml.etree import ElementTree as ET
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError("Word正文解压后过大。")
            root = ET.fromstring(archive.read(info))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        return "\n".join("".join(p.itertext()) for p in root.findall(".//w:p", ns))
    if suffix == ".pdf":
        import pymupdf
        with pymupdf.open(stream=blob, filetype="pdf") as document:
            if document.is_encrypted:
                raise ValueError("PDF已加密，请先提供可读取版本。")
            if document.page_count > 100:
                raise ValueError("文件夹预览最多读取100页；长文档请到知识库完整解析。")
            pages = []
            for index, page in enumerate(document):
                pages.append(f"[PDF原件第{index+1}页]\n" + page.get_text())
                if sum(map(len, pages)) > 400000:
                    raise ValueError("PDF正文过大，请到知识库完整解析。")
        text = "\n\n".join(pages)
        if len(re.sub(r"\[PDF原件第\d+页\]|\s", "", text)) < 20:
            raise ValueError("未读取到PDF文字，扫描件请使用知识库MinerU解析。")
        return text
    raise ValueError("不支持的资料类型。")


def workspace(identity):
    row = store.read("folder", identity)
    if not row or row.get("deleted_at"):
        raise ValueError("工作文件夹不存在。")
    return row


def file_record(identity):
    row = store.read("file", identity)
    if not row or row.get("deleted_at"):
        raise ValueError("文件不存在。")
    workspace(row["workspace_id"])
    return row


def stored_path(row):
    path = (FILE_ROOT / row["storage_name"]).resolve()
    if not path.is_relative_to(FILE_ROOT.resolve()) or not path.is_file():
        raise ValueError("文件快照不可读取。")
    return path


def import_folder(files, relative_paths, *, project_id=None):
    if not files or len(files) != len(relative_paths) or len(files) > 200:
        raise ValueError("一次选择1至200份文件，文件与路径必须对应。")
    # 接收的是用户通过文件夹选择器授权的文件快照，不读取任意服务端绝对路径。
    name = PurePosixPath(relative_paths[0].replace("\\", "/")).parts[0][:100]
    prepared, skipped, seen, total = [], [], set(), 0
    for upload, value in zip(files, relative_paths):
        try:
            relative = safe_relative_path(value)
        except ValueError as exc:
            skipped.append({"path": str(value)[:200], "reason": str(exc)})
            continue
        if relative.casefold() in seen:
            raise ValueError("文件夹中存在重复路径。")
        seen.add(relative.casefold())
        blob = upload.file.read(MAX_FILE_BYTES + 1)
        total += len(blob)
        if len(blob) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("资料超出限制：单份8MB、每批24MB。")
        try:
            text = extract_text(blob, PurePosixPath(relative).suffix.lower())
            if not text.strip():
                raise ValueError("文件没有可读取正文。")
            if len(text.encode("utf-8")) > 1200000:
                raise ValueError("文字内容超过预览限制，请到知识库入库。")
        except Exception as exc:
            message = str(exc) if isinstance(exc, (ValueError, UnicodeError)) else "文档解析未完成，请检查格式或通过知识库导入。"
            skipped.append({"path": relative, "reason": message[:200]})
            continue
        prepared.append((relative, blob, text))
    if not prepared:
        raise ValueError("没有可读取的文档。支持文本、Markdown、PDF、Word；EEG数组需到样本登记。")
    folder = store.new_record("folder", name=name, project_id=project_id, file_count=len(prepared), deleted_at=None)
    FILE_ROOT.mkdir(parents=True, exist_ok=True)
    for relative, blob, text in prepared:
        identity = uuid.uuid4().hex
        original = FILE_ROOT / (identity + PurePosixPath(relative).suffix.lower())
        original.write_bytes(blob)
        store.write("file", identity, {"id": identity, "workspace_id": folder["id"], "path": relative,
            "name": PurePosixPath(relative).name, "storage_name": original.name,
            "doc_id": hashlib.sha256(blob).hexdigest(), "text": text, "size": len(blob),
            "source_type": original.suffix.lstrip("."), "created_at": store.now(), "deleted_at": None})
    return {"workspace": folder, "files": list_files(folder["id"]), "skipped": skipped}


def list_files(identity):
    workspace(identity)
    return [{k:v for k,v in row.items() if k not in {"text", "storage_name"}} for row in store.all_records("file")
            if row["workspace_id"] == identity and not row.get("deleted_at")]


def documents_for_selection(workspace_id, identities, query=""):
    if not identities:
        return []
    if not workspace_id or len(identities) > 8 or len(set(identities)) != len(identities):
        raise ValueError("每轮可选择同一文件夹内1至8份资料。")
    workspace(workspace_id)
    documents = []
    terms = set(re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]", query.lower()))
    for identity in identities:
        row = file_record(identity)
        if row["workspace_id"] != workspace_id:
            raise ValueError("所选文件不属于当前工作文件夹。")
        text = row["text"]
        chunks = [(start, text[start:start+800]) for start in range(0, len(text), 700)]
        ranked = sorted(chunks, key=lambda pair: -sum(term in pair[1].lower() for term in terms))
        # 每份资料最多两个候选窗口；完整文件可在文件树读取，不能声称所有文件全文都装入模型。
        for start, chunk in ranked[:2 if len(identities) <= 4 else 1]:
            uid = hashlib.sha256(f"{identity}:{start}:{chunk}".encode()).hexdigest()
            documents.append({"type": "workspace_file", "file_id": identity, "workspace_id": workspace_id,
                "doc_id": row["doc_id"], "chunk_uid": uid, "chunk_id": "file-"+uid,
                "text": chunk, "content": chunk, "title": row["path"], "file_title": row["name"],
                "source_type": row["source_type"], "section": f"文件正文字符{start+1}–{start+len(chunk)}",
                "source_url": f"/workspace/files/{identity}/download", "version": row["doc_id"][:12],
                "content_scope": "explicitly_selected_file_excerpt", "workspace_snapshot": True})
    return documents


@contextmanager
def selection_scope(documents, options=None):
    token = _selected.set(tuple(documents))
    option_token = _selection_options.set(options)
    try:
        yield
    finally:
        _selected.reset(token)
        _selection_options.reset(option_token)


def selected_documents(query=""):
    options = _selection_options.get()
    if query and options and options.get("file_ids"):
        return documents_for_selection(options.get("workspace_id"), options["file_ids"], query)
    return list(_selected.get())
