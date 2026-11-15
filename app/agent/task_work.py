"""一次用户任务共享的有界工具预算与已完成EEG产物；不跨任务缓存。"""
from copy import deepcopy
import json
import uuid

from app.agent import planning
from app.agent.tools import registry


class TaskWork:
    def __init__(self, task_id=None):
        self.calls = 0
        self.supplements = 0
        self.analysis = {}
        self.checkpoint = lambda: None
        self.task_id = task_id

    def execute(self, name, arguments):
        self.checkpoint()
        self.calls += 1
        key = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        cache_key = name + ":" + key
        cacheable = name in {"analyze_eeg_sample", "preprocess_eeg_recording", "analyze_preprocessed_eeg", "register_eeg_recording", "register_eeg_sample"}
        if cacheable and cache_key in self.analysis:
            result = deepcopy(self.analysis[cache_key])
            result.update(tool_call_id=uuid.uuid4().hex, elapsed_ms=0)
            result["data"].update(cache_hit=True, inference_performed=False)
            if name == "preprocess_eeg_recording":
                result["data"]["preprocessing_performed"] = False
            return result
        result = registry.execute_tool(name, arguments).model_dump()
        if cacheable and result["status"] == "ok":
            self.analysis[cache_key] = deepcopy(result)
            if self.task_id:
                from app.shared.runtime.query_tasks import query_tasks
                query_tasks.record_artifacts(self.task_id, result["data"].get("artifacts") or [])
        return result

    def summary(self):
        return {"tool_calls": self.calls, "max_tool_calls": planning.MAX_TOOL_CALLS,
                "supplement_rounds": self.supplements,
                "completed_analysis": [{"tool": key.split(":", 1)[0], "arguments": json.loads(key.split(":", 1)[1]),
                    "artifacts": result["data"].get("artifacts", [])} for key, result in self.analysis.items()]}
