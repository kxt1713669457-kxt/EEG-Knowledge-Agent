"""按一次图执行采集单调时钟跨度；草稿首片段与最终交付分别观察。"""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
from threading import RLock
import time
import uuid

TRACE_DIR=Path(__file__).resolve().parents[3]/'output/runtime/timing-traces'
_active={}
_guard=RLock()


class Trace:
    def __init__(self,task_id,mode,revision,clock=time.perf_counter):
        self.clock=clock;self.started=clock();self.lock=RLock()
        self.data={'trace_id':uuid.uuid4().hex,'task_id':task_id,'mode':mode,'input_revision':revision,
            'started_at':datetime.now(timezone.utc).isoformat(),'status':'running','spans':[],'marks':{}}

    def mark(self,name):
        with self.lock:self.data['marks'].setdefault(name,round(self.clock()-self.started,6))

    @contextmanager
    def span(self,name):
        start=self.clock();status='completed';error=None
        try:yield
        except BaseException as exc:
            status='failed';error=type(exc).__name__;raise
        finally:
            row={'name':name,'start_seconds':round(start-self.started,6),
                 'duration_seconds':round(self.clock()-start,6),'status':status}
            if error:row['error_type']=error
            with self.lock:self.data['spans'].append(row)


@contextmanager
def capture_trace(task_id,mode,revision=1,directory=TRACE_DIR):
    trace=Trace(task_id,mode,revision)
    with _guard:
        if task_id in _active:raise RuntimeError('同一task不能同时开启两份图计时')
        _active[task_id]=trace
    try:
        yield trace
        trace.data['status']='completed'
    except BaseException as exc:
        trace.data.update(status='failed',error_type=type(exc).__name__);raise
    finally:
        trace.data['graph_seconds']=round(trace.clock()-trace.started,6)
        with _guard:_active.pop(task_id,None)
        try:
            directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
            target=directory/(trace.data['trace_id']+'.json');temp=target.with_suffix('.tmp')
            temp.write_text(json.dumps(trace.data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(target)
        except OSError:
            # 监测落盘失败不改变业务回答；验收端仍须明确报告缺失，不能伪称测得。
            from app.shared.runtime.logger import logger
            logger.warning('本轮图计时未落盘，不能据此报告阶段耗时')


@contextmanager
def trace_span(task_id,name):
    with _guard:trace=_active.get(task_id)
    if trace is None:yield
    else:
        with trace.span(name):yield


def trace_mark(task_id,name):
    with _guard:trace=_active.get(task_id)
    if trace is not None:trace.mark(name)


def measured_node(name,func):
    @wraps(func)
    def wrapped(state,*args,**kwargs):
        with trace_span(state.get('task_id'),name):return func(state,*args,**kwargs)
    return wrapped
