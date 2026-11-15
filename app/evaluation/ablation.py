"""消融的输入身份和候选交接约束；不执行模型，不修改生产配置。"""
import hashlib
from copy import deepcopy

from app.evaluation.metrics import normalize, ranking_metrics, evidence_metrics


def text_key(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def resolve_hit_identity(hit):
    """SDK可按实际主键字段名返回；多个身份字段冲突时拒绝猜测。"""
    values = [hit.get('chunk_id'), hit.get('id'), (hit.get('entity') or {}).get('chunk_id')]
    identities = {str(value) for value in values if value is not None}
    if len(identities) != 1:
        raise ValueError('Milvus命中缺少明确一致的主键')
    return identities.pop()


def annotate_qrels(case, chunks):
    relevant = []
    for row in chunks:
        if any(row['source_id'] == unit['source_id'] and
               any(normalize(a) in normalize(row['content']) for a in unit['anchors'])
               for unit in case['evidence_units']):
            relevant.append(row['chunk_uid'])
    if case['evidence_units'] and not relevant:
        raise ValueError('题目证据未映射到当前切片，不能默记召回零分')
    coverage = evidence_metrics(case['evidence_units'], chunks, len(chunks))
    if coverage.get('complete_evidence') != 1:
        raise ValueError('当前完整语料没有覆盖标注证据，先对齐切片与标注')
    return relevant


def require_same_candidates(before, after):
    a = [row['chunk_uid'] for row in before]
    b = [row['chunk_uid'] for row in after]
    if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) != set(b):
        raise ValueError('精排消融必须对同一候选集合排序，不能偷偷增删候选')


def score_order(case, ranked, qrels, k):
    return {**ranking_metrics([r['chunk_uid'] for r in ranked], qrels, k),
            **evidence_metrics(case['evidence_units'], ranked, k)}


def apply_calibration_judgments(cases, revisions, source_texts):
    """校准标注允许显式修订，保留原题；不能借此更改冻结验收。"""
    result=deepcopy(cases)
    by_id={row['case_id']:row for row in result}
    for qid, units in revisions.items():
        if qid not in by_id or by_id[qid]['split']!='calibration':
            raise ValueError('只允许修订已选校准题，不能修改冻结题')
        if not units:
            raise ValueError('不能清空必需证据以刷通过')
        for unit in units:
            text=source_texts.get(unit['source_id'])
            if text is None or not unit.get('anchors') or any(normalize(a) not in normalize(text) for a in unit['anchors']):
                raise ValueError('修订锚点必须存在于对应归档原文')
        by_id[qid]['evidence_units']=deepcopy(units)
    return result
