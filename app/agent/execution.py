"""按已校验计划顺序执行，依赖失败不会伪造来源或重新无限调用。"""
from copy import deepcopy

from app.agent import planning
from app.agent.tools import registry
from app.agent.tools.schemas import SourceReference


class SourceUnavailable(ValueError):
    pass


def resolve_source(binding, results):
    result = results.get(binding.step_id)
    if result is None or result["status"] != "ok":
        raise SourceUnavailable("前置步骤没有可用结果")
    docs = result["data"].get("documents") or []
    if binding.document_index >= len(docs):
        raise SourceUnavailable("前置步骤的实际结果少于计划引用的下标")
    doc = docs[binding.document_index]
    return SourceReference.model_validate({key: doc.get(key) for key in ("doc_id", "chunk_uid")}).model_dump()


def execute_plan(payload, *, max_tool_calls=planning.MAX_TOOL_CALLS, task_work=None):
    plan = planning.validate_plan(payload)  # 全量验证，后面的非法步骤不能等前面的工具跑完才发现。
    results, trace, documents = {}, [], {}
    calls = task_work.calls if task_work is not None else 0
    for step in plan.steps:
        if calls >= max_tool_calls:
            trace.append({"step_id": step.step_id, "tool": step.tool, "status": "skipped", "reason": "tool_budget_exhausted"})
            continue
        try:
            if isinstance(step, planning.ContextStep):
                arguments = {**resolve_source(step.arguments.source, results), "neighbors": step.arguments.neighbors}
            elif isinstance(step, planning.CompareStep):
                arguments = {"question": step.arguments.question,
                             "sources": [resolve_source(ref, results) for ref in step.arguments.sources]}
            elif isinstance(step, (planning.PreprocessingStep, planning.AnalysisStep)) and step.arguments.registration_step_id:
                producer = results.get(step.arguments.registration_step_id)
                registration = producer["data"].get("registration", {}) if producer and producer["status"] == "ok" else {}
                kind = "recording" if isinstance(step, planning.PreprocessingStep) else "sample"
                key = "recording_id" if kind == "recording" else "sample_id"
                if registration.get("kind") != kind or not registration.get(key):
                    raise SourceUnavailable("前置登记失败或缺少真实ID")
                arguments = step.arguments.model_dump(exclude={"registration_step_id", key})
                arguments[key] = registration[key]
            elif isinstance(step, (planning.PreprocessingStep, planning.AnalysisStep)):
                arguments = step.arguments.model_dump(exclude={"registration_step_id"})
            elif isinstance(step, planning.PreparedAnalysisStep):
                arguments = {"trial_index": step.arguments.trial_index, "model_id": step.arguments.model_id}
                if step.arguments.preprocessing_step_id:
                    producer = results.get(step.arguments.preprocessing_step_id)
                    artifacts = producer["data"].get("artifacts", []) if producer and producer["status"] == "ok" else []
                    if len(artifacts) != 1 or artifacts[0].get("kind") != "mi_preprocessing" or step.arguments.trial_index >= artifacts[0]["output"]["shape"][0]:
                        raise SourceUnavailable("前置预处理失败或实际试次数不足")
                    arguments["artifact_id"] = artifacts[0]["artifact_id"]
                else:
                    arguments["artifact_id"] = step.arguments.artifact_id
            else:
                arguments = step.arguments.model_dump()
        except (SourceUnavailable, ValueError):
            trace.append({"step_id": step.step_id, "tool": step.tool, "status": "skipped", "reason": "source_unavailable"})
            continue
        result = (task_work.execute(step.tool, arguments) if task_work is not None
                  else registry.execute_tool(step.tool, arguments).model_dump())
        results[step.step_id] = result
        calls += 1
        trace.append({"step_id": step.step_id, "tool": step.tool, "status": result["status"],
                      "arguments": arguments,
                      "tool_call_id": result["tool_call_id"], "elapsed_ms": result["elapsed_ms"],
                      "artifact_ids": [item["artifact_id"] for item in result["data"].get("artifacts") or [] if item.get("artifact_id")],
                      "inference_performed": result["data"].get("inference_performed",False),
                      "cache_hit": result["data"].get("cache_hit", False),
                      "error": result["error"], "returned_documents": len(result["data"].get("documents") or [])})
        if result["status"] != "ok":
            continue
        # 只合并原文/网页摘要，不把工具生成的比较answer或HyDE假设当成新来源。
        for doc in result["data"].get("documents") or []:
            key = (("analysis",doc["artifact_id"]) if doc.get("type")=="analysis_result" else
                   ("web", doc.get("url")) if doc.get("type") == "web" else (doc.get("doc_id"), doc.get("chunk_uid")))
            if key not in documents:
                documents[key] = deepcopy(doc)
    complete = all(row["status"] in {"ok", "empty"} for row in trace)
    return {"plan": plan.model_dump(), "trace": trace, "tool_results": results,
            "documents": list(documents.values()), "execution_status": "completed" if complete else "partial",
            "tool_calls": calls, "max_tool_calls": max_tool_calls,
            "prior_supplement_rounds": task_work.supplements if task_work is not None else 0}
