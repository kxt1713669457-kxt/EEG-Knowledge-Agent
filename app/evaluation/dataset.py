"""版本化题集及来源检查；冻结后变更必须建立新版本。"""
import hashlib
import json
from pathlib import Path

from app.evaluation.metrics import normalize


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(directory, project_root=None):
    directory = Path(directory)
    lock = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in lock["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"冻结文件已改变：{name}；请另建版本，不能继续使用旧结果")
    sources = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    source_by_id = {row["source_id"]: row for row in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("来源ID重复")
    if len(cases) != 50 or sum(row["split"] == "calibration" for row in cases) != 30 or sum(row["split"] == "frozen" for row in cases) != 20:
        raise ValueError("题集必须为30校准＋20冻结")
    if len({row["case_id"] for row in cases}) != 50 or len({normalize(row["query"]) for row in cases}) != 50:
        raise ValueError("题号或问题重复")
    for case in cases:
        if case["split"] not in {"calibration", "frozen"} or not case.get("acceptance"):
            raise ValueError("缺少分组或任务成功标准")
        for unit in case["evidence_units"]:
            source = source_by_id[unit["source_id"]]
            text = normalize((directory / source["snapshot_path"]).read_text(encoding="utf-8"))
            if not unit.get("anchors") or any(normalize(anchor) not in text for anchor in unit["anchors"]):
                raise ValueError(f"{case['case_id']}的证据锚点在归档原文中不存在")
        if project_root is not None and case.get("fixture"):
            fixture = case["fixture"]
            if sha256(Path(project_root) / fixture["path"]) != fixture["sha256"]:
                raise ValueError("冻结执行产物已改变")
    if project_root is not None:
        for source in sources:
            if sha256(Path(project_root) / source["original_path"]) != source["original_sha256"]:
                raise ValueError(f"原件版本已改变：{source['source_id']}")
    return {"cases": cases, "sources": sources, "freeze": lock}


def select_cases(bundle, split="calibration", case_ids=None):
    if split not in {"calibration", "frozen"}:
        raise ValueError("须明确选择calibration或frozen，不混合统计")
    selected = [row for row in bundle["cases"] if row["split"] == split]
    if case_ids is not None:
        wanted = set(case_ids)
        selected = [row for row in selected if row["case_id"] in wanted]
        if {row["case_id"] for row in selected} != wanted:
            raise ValueError("所选题号不属于指定分组")
    return selected
