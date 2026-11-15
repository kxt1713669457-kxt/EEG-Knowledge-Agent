"""检索、证据覆盖与引用/主张分别计分，不用一项成功冒充整链成功。"""
import re
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)
    text = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def ranking_metrics(ranked_ids, relevant_ids, k):
    if type(k) is not int or k < 1:
        raise ValueError("k必须是正整数")
    ranking = list(dict.fromkeys(ranked_ids))  # 按稳定身份去重，不用重复切片刷命中。
    gold = set(relevant_ids)
    if not gold:
        return {"recall": None, "hit": None, "rr": None, "status": "no_relevance_labels"}
    top = ranking[:k]
    return {"recall": len(set(top) & gold) / len(gold), "hit": int(bool(set(top) & gold)),
            "rr": next((1 / rank for rank, uid in enumerate(top, 1) if uid in gold), 0.0),
            "status": "scored"}


def evidence_metrics(units, hits, k):
    """每个unit包含同一来源内必需的锚点；多个unit允许跨切片/来源组合。"""
    if type(k) is not int or k < 1:
        raise ValueError("k必须是正整数")
    if not units:
        return {"evidence_recall": None, "complete_evidence": None, "status": "not_applicable"}
    unique = {}
    for hit in hits:
        key = (hit["source_id"], hit["chunk_uid"])
        unique.setdefault(key, hit)
    top = list(unique.values())[:k]
    covered = []
    for unit in units:
        texts = [normalize(hit["content"]) for hit in top if hit["source_id"] == unit["source_id"]]
        if not unit.get("anchors"):
            raise ValueError("证据单元不得为空")
        covered.append(all(any(normalize(anchor) in text for text in texts) for anchor in unit["anchors"]))
    return {"evidence_recall": sum(covered) / len(covered), "complete_evidence": int(all(covered)),
            "covered_units": covered, "status": "scored"}


def citation_metrics(answer, available_ids):
    references = list(dict.fromkeys(re.findall(r"\[([EA]\d+)\]", answer)))
    if not references:
        return {"id_validity": None, "status": "no_citations", "used_ids": [], "invalid_ids": []}
    available = set(available_ids)
    invalid = [ref for ref in references if ref not in available]
    return {"id_validity": (len(references) - len(invalid)) / len(references),
            "status": "scored", "used_ids": references, "invalid_ids": invalid}


def claim_support_metrics(verdicts):
    """主张须有独立来源对照标注；缺标注不凭引用ID自动给100%。"""
    if not verdicts:
        return {"support_rate": None, "review_coverage": 0.0, "status": "unreviewed"}
    allowed = {"supported", "unsupported", "unreviewed"}
    if any(verdict not in allowed for verdict in verdicts):
        raise ValueError("未知主张复核标签")
    reviewed = sum(verdict != "unreviewed" for verdict in verdicts)
    return {"support_rate": verdicts.count("supported") / len(verdicts) if reviewed == len(verdicts) else None,
            "review_coverage": reviewed / len(verdicts),
            "status": "scored" if reviewed == len(verdicts) else "partially_reviewed"}


def summarize_cases(rows):
    if not rows:
        raise ValueError("不能汇总空评测集")
    ids = [row["case_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("同一题重复结果不能当多个独立样本")
    completed = [row for row in rows if row["status"] == "completed"]
    judged = [row for row in rows if row.get("task_success") is not None]
    return {"attempted": len(rows), "completed": len(completed),
            "errors": sum(row["status"] == "error" for row in rows),
            "success_rate": sum(bool(row["task_success"]) for row in judged) / len(rows) if len(judged) == len(rows) else None,
            "success_judgment_coverage": len(judged) / len(rows)}


def aggregate_retrieval(rows, k):
    if not rows or len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("检索汇总需要非空、题号唯一的记录")
    metrics = [row["metrics"][str(k)] for row in rows]  # 缺结果直接暴露，不默默丢题。
    names = {"recall": "mean_judged_chunk_recall", "hit": "hit_rate", "rr": "mrr",
             "evidence_recall": "mean_evidence_unit_recall", "complete_evidence": "complete_evidence_rate"}
    result = {"attempted_questions": len(rows), "applicable_questions": sum(m["recall"] is not None for m in metrics)}
    for name, output in names.items():
        values = [m[name] for m in metrics if m[name] is not None]
        result[output] = sum(values) / len(values) if values else None
    return result
