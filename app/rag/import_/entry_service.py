from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.parse_result_service import record_parse_result
from app.rag.import_.document_asset_service import describe_document
from app.shared.runtime.logger import step_log


@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """检查原件并选择解析器；无效输入显式失败。"""
    if not state.get("local_file_path"):
        raise FileNotFoundError("未提供导入文件")
    source = Path(state["local_file_path"])
    if not source.is_file():
        raise FileNotFoundError("导入文件不存在")
    if not source.stat().st_size:
        raise ValueError("导入文件为空")
    suffix = source.suffix.lower()
    if suffix not in {".md", ".pdf", ".docx"}:
        raise ValueError("当前仅支持PDF、DOCX和Markdown")
    if not state.get("document_metadata"):
        state["document_metadata"] = describe_document(source)
    state.update(is_md_read_enabled=False, is_pdf_read_enabled=False, is_docx_read_enabled=False)
    state["file_title"] = source.stem
    if suffix == ".md":
        try:
            content = source.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("Markdown文件必须为UTF-8文本") from error
        if not content.strip():
            raise ValueError("Markdown正文为空")
        state.update(md_path=str(source), md_content=content, is_md_read_enabled=True)
        record_parse_result(state, "markdown_utf8")
    elif suffix == ".pdf":
        with source.open("rb") as stream:
            if b"%PDF-" not in stream.read(1024):
                raise ValueError("文件扩展名为PDF，但缺少PDF文件头")
        state.update(pdf_path=str(source), is_pdf_read_enabled=True)
    else:
        state.update(docx_path=str(source), is_docx_read_enabled=True)
    return state


