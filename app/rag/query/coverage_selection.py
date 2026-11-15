"""有界多主体覆盖选择：只重排真实候选，不新增分数或伪造缺失证据。"""
from app.rag.query.query_understanding import topic_key


def select_with_subject_coverage(ranked, item_names, limit):
    names=list(dict.fromkeys(topic_key(name) for name in item_names if name))
    if not 1<len(names)<=4 or limit<len(names):
        return ranked[:limit]
    # 每个已确认主体预留其排名最高的一条，剩余额度仍按既有排名填充。
    required={}
    for index, doc in enumerate(ranked):
        key=topic_key(doc.get('item_name') or '')
        if key in names:required.setdefault(key,index)
    selected=set(required.values())
    for index in range(len(ranked)):
        if len(selected)>=limit:break
        selected.add(index)
    return [doc for index,doc in enumerate(ranked) if index in selected]
