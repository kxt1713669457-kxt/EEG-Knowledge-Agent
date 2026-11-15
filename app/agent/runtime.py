"""有限计划图：规划/验证→执行→观察→至多一次补查→真实来源回答。"""
from typing_extensions import TypedDict
import uuid

from langgraph.graph import END, StateGraph

from app.agent import planning
from app.agent.execution import execute_plan
from app.agent.observation import observe_execution, execute_supplement, requires_original, has_local_original
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import create_query_default_state
from app.rag.query.answer_service import generate_answer, save_answer_message_history
from app.shared.runtime.task_updates import input_checkpoint, ensure_current_input
from app.shared.runtime.telemetry import measured_node


class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    task_id: str
    run_id: str
    history: list
    supplied_plan: dict | None
    plan: dict
    planning_audit: dict
    status: str
    error: dict
    execution: dict
    answer_state: dict
    observation: dict
    supplement_rounds: int
    managed_input: bool
    input_revision: int
    task_work: object
    requested_answer_format: str
    source_requirement: str


def node_plan(state):
    audit = {}
    try:
        plan = planning.validate_plan_subject(planning.validate_plan(state["supplied_plan"]), state["query"]) if state.get("supplied_plan") is not None else planning.generate_plan(state["query"], state.get("history") or [], audit=audit)
        return {"plan": plan.model_dump(), "planning_audit": audit, "status": "validated"}
    except planning.PlanRejected as exc:
        return {"status": "rejected", "planning_audit": audit, "error": {"code": exc.code, "message": str(exc), "details": exc.details}}
    except Exception:
        return {"status": "failed", "planning_audit": audit, "error": {"code": "planning_failed", "message": "计划服务未完成，未执行任何工具。"}}


def node_execute(state):
    work = state.get("task_work")
    if work is not None:
        work.checkpoint = lambda: ensure_current_input(state)
    return {"execution": execute_plan(state["plan"], **({"task_work": work} if work is not None else {}))}


def node_observe(state):
    from app.agent.eeg_receipts import operation_receipt
    if operation_receipt(state["execution"]) is not None:
        return {"observation": {"status": "accepted", "decision": "answer", "model_calls": 0,
                                "reason": "附件检查/登记返回实际元信息，直接交付回执或询问缺失字段。"}}
    controls = {"source_requirement": state["source_requirement"]} if state.get("source_requirement", "auto") != "auto" else {}
    return {"observation": observe_execution(state["query"], state["plan"]["goal"], state["execution"], **controls)}


def after_observe(state):
    result = state["observation"]
    return "supplement" if result["status"] == "accepted" and result["decision"] == "supplement" else "answer"


def node_supplement(state):
    work = state.get("task_work")
    controls = {"source_requirement": state["source_requirement"]} if state.get("source_requirement", "auto") != "auto" else {}
    execution = execute_supplement(state["query"], state["plan"]["goal"], state["execution"], state["observation"],
                                   **({"task_work": work} if work is not None else {}), **controls)
    return {"execution": execution, "supplement_rounds": 1}


def node_answer(state):
    answer_state = create_query_default_state(session_id=state["session_id"], task_id=state.get("task_id") or state["run_id"],
        run_id=state["run_id"], original_query=state["query"],
        rewritten_query=state["query"] + "\n任务补全：" + state["plan"]["goal"],
        history=state.get("history") or [], reranked_docs=state["execution"]["documents"],
        input_revision=state.get("input_revision", 1), defer_delivery=state.get("managed_input", False),
        requested_answer_format=state.get("requested_answer_format", "auto"),
        source_requirement=state.get("source_requirement", "auto"))
    observation = state.get("observation") or {}
    needs_original = (state.get("source_requirement") == "original" or
                      state.get("source_requirement", "auto") == "auto" and requires_original(state["query"]))
    has_recorded_source = has_local_original(state["execution"]) or any(
        doc.get("type") == "analysis_result" for doc in state["execution"]["documents"])
    from app.agent.eeg_receipts import operation_receipt
    receipt = operation_receipt(state["execution"])
    if receipt is not None:
        answer_state.update(answer=receipt, citations=[], citation_status={"status": "not_required", "support_check": "verified_tool_metadata"})
    elif needs_original and not has_recorded_source:
        answer_state.update(answer="本次尚未取得可核对的原件，仅有网页摘要或没有资料，暂不提供要求原文支持的事实结论。", citations=[],
                            citation_status={"status": "insufficient", "reason": "original_evidence_unavailable", "support_check": "not_assessed"})
    elif observation.get("status") == "accepted" and observation.get("decision") == "insufficient":
        answer_state.update(answer="当前已取得资料不足以回答这个问题，请补充相关资料或缩小问题范围。", citations=[],
                            citation_status={"status": "insufficient", "reason": "observed_evidence_gap", "support_check": "not_assessed"})
    else:
        try:
            generate_answer(answer_state, persist_history=False)
        except Exception as exc:
            # 只记异常类别/HTTP状态，不能把带供应商请求或凭据的异常全文交给页面。
            answer_state.setdefault("answer_audit", {})["generation_error"] = {
                "type": type(exc).__name__, "status_code": getattr(exc, "status_code", None)}
            answer_state.update(answer="回答服务暂未完成，已保留工具执行记录，请稍后重试。", citations=[],
                                citation_status={"status": "invalid", "reason": "answer_service_failed"})
    if state["execution"]["execution_status"] == "partial":
        notice = ("部分联网调用未完成；以下仅依据已取得资料回答。" if any(
            row["tool"] == "search_web" and row["status"] == "failed" for row in state["execution"]["trace"])
            else "部分步骤未完成；以下仅依据已取得资料回答。")
        answer_state["answer"] = (notice + "\n\n" + answer_state["answer"]
                                  if state["execution"]["documents"] else "本次工具未取得可用证据，请检查调用记录或稍后重试。")
    if observation.get("status") in {"failed", "rejected"}:
        answer_state["answer"] = "本次资料观察或补查建议未完成；以下仅使用已取得资料。\n\n" + answer_state["answer"]
    answer_state.setdefault("answer_audit", {})["agent_execution"] = {
        "task_id": state.get("task_id") or state["run_id"],
        "run_id": state["run_id"], "plan": state["plan"], "trace": state["execution"]["trace"],
        "planning_audit": state.get("planning_audit", {}),
        "observation": observation, "supplement_rounds": state.get("supplement_rounds", 0),
        "artifact_references": [{key:doc.get(key) for key in ("type","artifact_id","sample_id","model_id","trial_index")}
                                for doc in state["execution"]["documents"] if doc.get("type")=="analysis_result"],
        "tool_calls": state["execution"]["tool_calls"], "max_tool_calls": state["execution"]["max_tool_calls"]}
    status = "partial" if observation.get("status") in {"failed", "rejected"} else state["execution"]["execution_status"]
    return {"answer_state": answer_state, "status": status}


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", input_checkpoint(measured_node('agent_plan',node_plan)))
    graph.add_node("execute", input_checkpoint(measured_node('agent_execute',node_execute)))
    graph.add_node("observe", input_checkpoint(measured_node('agent_observe',node_observe)))
    graph.add_node("supplement", input_checkpoint(measured_node('agent_supplement',node_supplement)))
    graph.add_node("answer", input_checkpoint(measured_node('agent_answer',node_answer)))
    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", lambda state: "execute" if state["status"] == "validated" else END)
    graph.add_edge("execute", "observe")
    graph.add_conditional_edges("observe", after_observe, {"supplement": "supplement", "answer": "answer"})
    graph.add_edge("supplement", "answer")  # 没有回到observe的边，一轮后必定收尾。
    graph.add_edge("answer", END)
    return graph.compile()


agent_graph = build_agent_graph()


def run_agent(query: str, session_id: str | None = None, supplied_plan=None, *, task_id=None, run_id=None,
              defer_history=False, input_revision=1, task_work=None, requested_answer_format="auto", source_requirement="auto"):
    sid = session_id or str(uuid.uuid4())
    task_id, run_id = task_id or uuid.uuid4().hex, run_id or uuid.uuid4().hex
    # 外部提供的非法计划先拒绝，不读写会话，也不执行任何工具。
    if supplied_plan is not None:
        try:
            planning.validate_plan_subject(planning.validate_plan(supplied_plan), query)
        except planning.PlanRejected as exc:
            return {"status": "rejected", "task_id": task_id, "run_id": run_id, "session_id": sid, "trace": [], "tool_calls": 0,
                    "error": {"code": exc.code, "message": str(exc)}}
    history = history_repository.list_recent(sid, limit=6)
    if task_work is not None and task_work.analysis:
        import json
        history = history + [{"role": "assistant", "text": "本任务已完成的可信执行产物；分类/表征用get_eeg_analysis，kind=mi_preprocessing用get_eeg_preprocessing读取，无需重新计算：" +
                              json.dumps(task_work.summary()["completed_analysis"], ensure_ascii=False),
                              "evidence_sources": [doc for result in task_work.analysis.values()
                                                   for doc in result["data"].get("documents", [])]}]
    state = agent_graph.invoke({"query": query, "session_id": sid, "task_id": task_id, "run_id": run_id,
                               "supplied_plan": supplied_plan, "history": history,
                               "managed_input": defer_history, "input_revision": input_revision, "task_work": task_work,
                               "requested_answer_format": requested_answer_format, "source_requirement": source_requirement})
    if "answer_state" not in state:
        return {"status": state["status"], "task_id": task_id, "run_id": run_id, "session_id": sid, "trace": [], "tool_calls": 0,
                "planning_audit": state.get("planning_audit", {}), "error": state["error"]}
    result, execution = state["answer_state"], state["execution"]
    if not defer_history:
        history_repository.save_message(session_id=sid, task_id=task_id, run_id=run_id, role="user", text=query)
        save_answer_message_history(result)
    return {"status": state["status"], "task_id": task_id, "run_id": run_id, "session_id": sid, "plan": state["plan"],
            "planning_audit": state.get("planning_audit", {}),
            "trace": execution["trace"], "tool_calls": execution["tool_calls"], "max_tool_calls": execution["max_tool_calls"],
            "observation": state.get("observation", {}), "supplement_rounds": state.get("supplement_rounds", 0),
            "analysis_artifacts": [artifact for tool in execution["tool_results"].values() for artifact in tool["data"].get("artifacts") or []],
            **({"_answer_state": result} if defer_history else {}),
            **{key: result.get(key) for key in ("answer", "citations", "citation_status", "context_budget", "image_urls", "generation_usage")}}
