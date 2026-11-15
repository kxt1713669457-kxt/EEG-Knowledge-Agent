"""各解析器共用的结果记录；原件身份保持不变。"""
import json
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.shared.utils.task_utils import set_task_result


def record_parse_result(state: ImportGraphState, parser: str, **details) -> ImportGraphState:
    markdown = Path(state["md_path"]).read_text(encoding="utf-8-sig")
    if not markdown.strip():
        raise ValueError("解析结果为空，不能继续切块入库")
    metadata = {"parser": parser, "status": "completed", "output_format": "markdown", "markdown_chars": len(markdown),
                "warnings": [], **details}
    return _write_parse_metadata(state, metadata)


def record_parse_progress(state: ImportGraphState, **details) -> ImportGraphState:
    """远端排队/下载/失败时也保存解析进度，不需要先有Markdown。"""
    return _write_parse_metadata(state, {**state.get("parse_metadata", {}), **details})


def _write_parse_metadata(state: ImportGraphState, metadata: dict) -> ImportGraphState:
    state["parse_metadata"] = metadata
    directory = Path(state.get("local_dir") or Path(state["md_path"]).parent)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "parse_result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if state.get("task_id"):
        set_task_result(state["task_id"], "parse_metadata", json.dumps(metadata, ensure_ascii=False))
    return state
