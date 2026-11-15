"""准入时冻结模型与文件选择；同任务补充沿用，不串到别的会话。"""
from functools import wraps
from contextlib import contextmanager
import inspect
import re
from threading import Lock

from app.workspace.connections import ExecutionOptions, resolve_binding, binding_scope
from app.workspace.files import documents_for_selection, selection_scope
from app.workspace import store
from app.workspace.media import image_record, image_scope
from app.workspace.eeg_attachments import attachment_scope

_bindings, _lock = {}, Lock()


def prepare(options, query):
    if options is None:
        return None
    binding = resolve_binding(options)
    documents = documents_for_selection(options.workspace_id, options.file_ids, query)
    if any(not isinstance(identity,str) or not re.fullmatch(r"[0-9a-f]{32}",identity) for identity in options.image_ids):
        raise ValueError("图片身份格式不合法。")
    for identity in options.image_ids: image_record(identity)
    with attachment_scope(options.eeg_attachment_ids):
        pass  # Validate file identity, total size and immutable hashes at admission.
    values = options.model_dump()
    values["model"] = binding.model  # 省略模型ID时也记录本轮实际解析的模型。
    return binding, documents, values


def attach(task_id, prepared):
    if prepared is None:
        return
    from app.shared.runtime.query_tasks import query_tasks
    with _lock:
        _bindings[task_id] = prepared
    try:
        query_tasks.set_execution_options(task_id, prepared[2])
    except Exception:
        with _lock:
            _bindings.pop(task_id, None)
        raise


@contextmanager
def options_scope(options, query):
    prepared = prepare(options, query)
    if prepared is None:
        yield
    else:
        binding, documents, values = prepared
        with binding_scope(binding), selection_scope(documents, values), image_scope(values.get("image_ids", [])), attachment_scope(values.get("eeg_attachment_ids", [])):
            yield


def scoped_task(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        tid = arguments.get("task_id") or arguments.get("tid")
        with _lock:
            prepared = _bindings.get(tid)
        if prepared is None:
            return function(*args, **kwargs)
        binding, documents, options = prepared
        try:
            with binding_scope(binding), selection_scope(documents, options), image_scope(options.get("image_ids", [])), attachment_scope(options.get("eeg_attachment_ids", [])):
                return function(*args, **kwargs)
        finally:
            with _lock:
                _bindings.pop(tid, None)
    return wrapped
