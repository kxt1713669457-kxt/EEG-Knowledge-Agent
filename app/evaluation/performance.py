"""小样本延迟统计使用明确的nearest-rank口径；不将缺失TTFT填0。"""
import math


def percentiles(values):
    if not values:return {'count':0,'p50':None,'p95':None}
    if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('延迟必须是有限非负数')
    ordered=sorted(values);n=len(ordered)
    return {'count':n,'p50':ordered[math.ceil(.5*n)-1],'p95':ordered[math.ceil(.95*n)-1],
            'method':'nearest_rank','small_sample_note':'n=5时p95就是这5次最大值，不代表稳定尾延迟'}
