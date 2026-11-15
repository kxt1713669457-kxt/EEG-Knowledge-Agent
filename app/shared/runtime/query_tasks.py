"""单worker准入；线上配置Mongo快照，独立测试默认使用内存。"""
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
import uuid

from app.shared.utils.task_utils import get_done_task_list, get_running_task_list
from app.infra.persistence.task_repository import TaskStoreUnavailable, TERMINAL_STATUSES

MAX_QUERY_PASSES = 3  # 用户修订触发的有界更新，区别于Agent自动补查。
MAX_TASK_INPUT_BYTES = 12000


class TaskInputError(ValueError):
    def __init__(self, code, message, status_code=409):
        self.code, self.status_code = code, status_code
        super().__init__(message)


class SessionBusy(RuntimeError):
    def __init__(self, task_id):
        self.task_id = task_id
        super().__init__("当前会话仍有任务进行中，请等待完成后再提问。")


class QueryTaskRegistry:
    def __init__(self):
        self._lock = RLock()
        self._active = {}
        self._tasks = {}
        self._repository = None
        self._owner_id = uuid.uuid4().hex
        self._session_heads = {}

    def configure_repository(self, repository):
        with self._lock:
            self._owner_id = uuid.uuid4().hex
            count = repository.interrupt_previous(self._owner_id) if repository is not None else 0
            self._repository = repository
            self._tasks.clear()
            self._active.clear()
            self._session_heads.clear()
            return count

    def _find(self, task_id):
        return self._tasks.get(task_id) or (self._repository.get(task_id) if self._repository is not None else None)

    def _commit(self, record):
        updated = deepcopy(record)
        updated.update(updated_at=self._now(), owner_id=self._owner_id, persisted=self._repository is not None)
        if self._repository is not None:
            updated = self._repository.save(updated)
        self._tasks[updated["task_id"]] = updated
        return updated

    def persist_progress(self, task_id):
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record["status"] in TERMINAL_STATUSES:
                return
            updated = deepcopy(record)
            updated.update(done_list=get_done_task_list(task_id), running_list=get_running_task_list(task_id))
            self._commit(updated)

    def record_artifacts(self, task_id, artifacts):
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record["status"] in TERMINAL_STATUSES:
                return
            updated = deepcopy(record)
            references = {item["artifact_id"]: item for item in updated.get("analysis_artifacts", [])}
            for item in artifacts:
                references[item["artifact_id"]] = {key: item.get(key) for key in
                    ("artifact_id", "model_id", "sample_id", "trial_index")}
            updated["analysis_artifacts"] = list(references.values())
            self._commit(updated)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def reserve(self, session_id, mode, query=None):
        """检查并占用必须在同一临界区；不清空已有队列或会话历史。"""
        with self._lock:
            if session_id in self._active:
                raise SessionBusy(self._active[session_id])
            if query is not None and len(query.encode("utf-8")) > MAX_TASK_INPUT_BYTES:
                raise TaskInputError("task_input_too_long", "本轮问题过长，请缩小范围。", 413)
            task_id = uuid.uuid4().hex
            created = self._now()
            record = {"session_id": session_id, "task_id": task_id,
                      "run_id": uuid.uuid4().hex if mode == "agent" else None,
                      "mode": mode, "status": "pending", "created_at": created,
                      "started_at": None, "finished_at": None, "result": None, "error": None,
                      "input_revision": 1, "accepting_updates": query is not None,
                      "input_messages": ([{"message_id": "initial", "text": query, "revision": 1,
                                            "received_at": created}] if query is not None else []),
                      "attempts": [], "phase": "pending", "done_list": [], "running_list": [],
                      "analysis_artifacts": [], "storage_version": 0}
            record = self._commit(record)
            self._active[session_id] = task_id
            if mode in {"rag", "agent", "chat"}:
                self._session_heads[session_id] = task_id
            return deepcopy(record)

    def begin(self, task_id):
        with self._lock:
            if task_id in self._tasks:
                record = deepcopy(self._tasks[task_id])
                record.update(status="processing", phase="processing", started_at=self._now())
                self._commit(record)

    def append_input(self, task_id, session_id, text, message_id):
        with self._lock:
            record = deepcopy(self._find(task_id))
            if record is None or record["session_id"] != session_id:
                raise TaskInputError("task_not_found", "找不到该会话对应的任务。", 404)
            for old in record["input_messages"]:
                if old["message_id"] == message_id:
                    if old["text"] != text:
                        raise TaskInputError("message_id_conflict", "同一消息标识不能改成不同内容。")
                    return {"task_id": task_id, "input_revision": old["revision"], "duplicate": True}
            if not record["accepting_updates"] or record["status"] not in {"pending", "processing"}:
                raise TaskInputError("task_input_closed", "本轮已进入最终收尾，补充可在回答结束后作为下一轮继续。")
            size = sum(len(row["text"].encode("utf-8")) for row in record["input_messages"]) + len(text.encode("utf-8"))
            if size > MAX_TASK_INPUT_BYTES:
                raise TaskInputError("task_input_too_long", "本轮问题与补充过长，请分成下一轮。", 413)
            record["input_revision"] += 1
            revision = record["input_revision"]
            record["input_messages"].append({"message_id": message_id, "text": text,
                                              "revision": revision, "received_at": self._now()})
            self._commit(record)
            return {"task_id": task_id, "input_revision": revision, "duplicate": False}

    def input_snapshot(self, task_id, *, close_updates=False):
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or not record["input_messages"]:
                return None
            if close_updates:
                record = deepcopy(record)
                record.update(accepting_updates=False, phase="settling")
                self._commit(record)
            return {"input_revision": record["input_revision"], "messages": deepcopy(record["input_messages"]),
                    "mode": record["mode"], "run_id": record["run_id"]}

    def set_run(self, task_id, run_id):
        with self._lock:
            record = deepcopy(self._tasks[task_id])
            record["run_id"] = run_id
            self._commit(record)

    def set_execution_options(self, task_id, options):
        with self._lock:
            record = deepcopy(self._tasks[task_id])
            record["execution_options"] = deepcopy(options)  # 只有连接/模型/文件ID，不包含密钥。
            self._commit(record)

    def add_attempt(self, task_id, attempt):
        with self._lock:
            record = deepcopy(self._tasks[task_id])
            record["attempts"].append(deepcopy(attempt))
            self._commit(record)

    def seal_if_current(self, task_id, revision):
        """检查版本与关闭输入原子完成，防止已接受的补充在最终提交时丢失。"""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return True
            if record["input_revision"] != revision:
                return False
            record = deepcopy(record)
            record.update(accepting_updates=False, phase="finalizing")
            self._commit(record)
            return True

    def finish(self, task_id, *, result=None, error=None):
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return  # 兼容不经HTTP的隔离图测试，不创建虚假会话任务。
            record = deepcopy(record)
            record.update(status="failed" if error else "completed", result=deepcopy(result),
                          error=deepcopy(error), finished_at=self._now(), accepting_updates=False, phase="finished")
            record["elapsed_ms"] = max(0, round((datetime.fromisoformat(record["finished_at"]) - datetime.fromisoformat(record["created_at"])).total_seconds() * 1000))
            record.update(done_list=get_done_task_list(task_id), running_list=[])
            if self._active.get(record["session_id"]) == task_id:
                del self._active[record["session_id"]]
            try:
                self._commit(record)
                if record["mode"] == "history_clear" and not error:
                    self._session_heads[record["session_id"]] = None
            except TaskStoreUnavailable:
                record.update(status="failed", persisted=False, result=None, error={"code": "task_store_unavailable",
                              "message": "任务执行收尾未确认保存，请稍后查询。系统未自动重新执行。"})
                self._tasks[task_id] = record
                raise

    def get(self, task_id):
        with self._lock:
            record = self._find(task_id)
            snapshot = deepcopy(record) if record else None
        if snapshot is not None:
            if self._repository is None:
                snapshot.update(done_list=get_done_task_list(task_id), running_list=get_running_task_list(task_id))
            snapshot["storage"] = ("mongo" if snapshot.get("persisted") else "process_memory_unsaved") if self._repository is not None else "process_memory"
        return snapshot

    def latest(self, session_id):
        with self._lock:
            if self._repository is not None:
                record = self._repository.latest(session_id)
                return self.get(record["task_id"]) if record else None
            tid = self._session_heads.get(session_id)
            return self.get(tid) if tid else None

    def active_task(self, session_id):
        with self._lock:
            return self._active.get(session_id)


query_tasks = QueryTaskRegistry()
