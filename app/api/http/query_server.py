import datetime
import re
from mimetypes import guess_type
from pathlib import Path
import sys
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.responses import JSONResponse
from minio.error import S3Error

from app.infra.object_storage.minio_gateway import minio_gateway
from app.rag.query.evidence_service import ORIGINAL_TYPES
from app.agent.tools.registry import execute_tool, tool_declarations
from app.agent.tools.schemas import ToolInvocation, ToolResult
from app.agent.tools.schemas import StrictInput
from app.agent import planning
from app.agent.runtime import run_agent
from app.agent.skills.eeg import service as eeg_service
from pydantic import Field, JsonValue

from app.api.schema.query_schema import HealthResponseSchema, QueryRequestSchema, QueryStreamResponseSchema, \
    HistoryClearResponseSchema, HistoryListResponseSchema, HistoryItemResponseSchema, TaskMessageSchema
from app.infra.persistence.history_repository import history_repository
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.runtime.query_tasks import query_tasks, SessionBusy, TaskInputError
from app.shared.runtime.task_updates import run_revisions
from app.agent.task_work import TaskWork
from app.infra.persistence.task_repository import TaskStoreUnavailable, TERMINAL_STATUSES
from app.shared.runtime.query_lifespan import query_lifespan
from app.api.http.workspace_account import router as workspace_router
from app.api.http.harness_api import router as harness_router
from app.api.http.auth_api import install_auth
from app.api.http.source_images import router as source_images_router
from app.workspace import execution as workspace_execution, store as workspace_store
from app.workspace.connections import ExecutionOptions
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_app as query_graph_app, query_app
from app.process.query.agent.state import create_query_default_state, QueryGraphState
from app.shared.utils.sse_utils import (
    SSEEvent,
    create_sse_queue,
    get_sse_queue,
    push_to_session,
    remove_sse_queue,
    sse_generator,
    claim_sse_queue,
    StreamAlreadyConnected,
    _sse_pack,
)
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_PENDING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)

# 定义fastapi对象
app = FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.5.0",
    lifespan=query_lifespan,
)
app.include_router(workspace_router)
app.include_router(harness_router)
app.include_router(source_images_router)

# 跨域处理{前端和后端可能不在一个服务器，非同源，浏览器认为跨域，拦截了}
install_auth(app)


@app.exception_handler(TaskStoreUnavailable)
async def task_store_error(request, exc):
    return JSONResponse(status_code=503, content={"detail": {"code": "task_store_unavailable", "message": str(exc)}})


def finish_failed_task(task_id, error):
    try:
        query_tasks.finish(task_id, error=error)
    except TaskStoreUnavailable:
        logger.error("任务失败状态未确认持久化，task_id={}；保留本地失败标识，禁止作为成功交付。", task_id)


@app.get("/agent/tools")
def list_agent_tools():
    return {"tools": tool_declarations()}


@app.get("/agent/eeg/catalog")
def eeg_model_catalog():
    return eeg_service.model_catalog()


@app.get("/agent/eeg/artifacts/{artifact_id}")
def eeg_artifact(artifact_id:str):
    try:
        return eeg_service.read_artifact(artifact_id)
    except eeg_service.EEGSkillError as exc:
        raise HTTPException(status_code=404 if exc.code=="artifact_not_found" else 400,detail=str(exc)) from None


@app.get("/agent/eeg/artifacts/{artifact_id}/features")
def eeg_features(artifact_id:str):
    eeg_artifact(artifact_id)  # 先校验身份，路径不接受客户端输入。
    path=eeg_service.ARTIFACT_ROOT/artifact_id/"features.npz"
    if not path.is_file():
        raise HTTPException(status_code=404,detail="表征文件不存在")
    return FileResponse(path,media_type="application/octet-stream",filename=f"{artifact_id}-features.npz")


@app.post("/agent/tools/execute", response_model=ToolResult)
def execute_agent_tool(invocation: ToolInvocation):
    return execute_tool(invocation.name, invocation.arguments)


class AgentPlanRequest(StrictInput):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    execution: ExecutionOptions | None = None


class AgentQueryRequest(AgentPlanRequest):
    plan: dict[str, JsonValue] | None = None


@app.post("/chat/tasks", status_code=202)
def submit_chat_task(request: AgentPlanRequest, backgroundtasks: BackgroundTasks):
    sid = (request.session_id or "").strip() or str(uuid.uuid4())
    ticket = reserve_workspace_task(sid, "chat", request.query, request.execution or ExecutionOptions())
    backgroundtasks.add_task(invoke_chat_task, request, sid, ticket["task_id"])
    return {"session_id":sid,"task_id":ticket["task_id"],"mode":"chat","message":"对话任务已登记。"}


@workspace_execution.scoped_task
def invoke_chat_task(request, sid, tid):
    from app.agent.conversation import run_conversation
    from app.rag.query.answer_quality import IncompleteAnswerError
    try:
        query_tasks.begin(tid)
        update_task_status(tid,TASK_STATUS_PROCESSING)
        def execute(effective,revision,run_id,controls):
            return run_conversation(effective,sid,tid,revision,run_id)
        _, result = run_revisions(tid,execute,mode="chat")
        query_tasks.finish(tid,result=result)
        update_task_status(tid,TASK_STATUS_COMPLETED)
    except IncompleteAnswerError as exc:
        finish_failed_task(tid,{"code":"incomplete_response","message":str(exc),"reasons":exc.reasons})
        update_task_status(tid,TASK_STATUS_FAILED)
    except Exception as exc:
        finish_failed_task(tid,{"code":"conversation_failed","message":"对话未完成，请检查所选模型是否支持工具/图片，或缩小问题后重试。","error_type":type(exc).__name__})
        update_task_status(tid,TASK_STATUS_FAILED)


def reserve_query_task(session_id, mode, query=None):
    try:
        return query_tasks.reserve(session_id, mode, query)
    except SessionBusy as exc:
        raise HTTPException(status_code=409, detail={"code": "session_busy", "message": str(exc),
                                                   "task_id": exc.task_id}) from None
    except TaskInputError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from None


def reserve_workspace_task(session_id, mode, query, options):
    try:
        prepared = workspace_execution.prepare(options, query)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    ticket = reserve_query_task(session_id, mode, query)
    if prepared is not None:
        try:
            workspace_store.ensure_session(session_id, query)
            workspace_execution.attach(ticket["task_id"], prepared)
        except Exception:
            finish_failed_task(ticket["task_id"], {"code": "workspace_admission_failed", "message": "工作区选择未确认，请重试。"})
            raise HTTPException(503, "工作区选择未确认，本轮未开始。") from None
    return ticket


@app.get("/sessions/{session_id}/active")
def active_query_task(session_id: str):
    tid = query_tasks.active_task(session_id)
    return {"task_id": tid}


@app.get("/sessions/{session_id}/latest-task")
def latest_query_task(session_id: str):
    return {"task": query_tasks.latest(session_id)}


@app.post("/tasks/{task_id}/messages", status_code=202)
def update_query_input(task_id: str, message: TaskMessageSchema):
    try:
        result = query_tasks.append_input(task_id, message.session_id, message.text, message.message_id)
    except TaskInputError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from None
    if not result["duplicate"] and get_sse_queue(task_id) is not None:
        push_to_session(task_id, "input_updated", {**result, "message": "已收到补充/更正，将在安全检查点合并。"})
    return result


@app.get("/tasks/{task_id}")
def query_task_status(task_id: str):
    snapshot = query_tasks.get(task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="任务不存在或未保存任务记录。")
    return snapshot


@app.post("/agent/plan/validate")
def validate_agent_plan(payload: dict):
    try:
        plan = planning.validate_plan(payload)
        return {"status": "validated", "plan": plan.model_dump()}
    except planning.PlanRejected as exc:
        return {"status": "rejected", "error": {"code": exc.code, "message": str(exc), "details": exc.details}}


@app.post("/agent/plan")
def create_agent_plan(request: AgentPlanRequest):
    try:
        with workspace_execution.options_scope(request.execution, request.query):
            return create_agent_plan_with_context(request)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


def create_agent_plan_with_context(request: AgentPlanRequest):
    audit = {}
    try:
        plan = planning.generate_plan(request.query, history_repository.list_recent(request.session_id, limit=6) if request.session_id else [], audit=audit)
        return {"status": "validated", "plan": plan.model_dump(), "planning_audit": audit}
    except planning.PlanRejected as exc:
        return {"status": "rejected", "planning_audit": audit, "error": {"code": exc.code, "message": str(exc), "details": exc.details}}
    except Exception:
        return {"status": "failed", "error": {"code": "planning_failed", "message": "计划服务未完成，未执行任何工具。"}}


@app.post("/agent/query")
def query_agent(request: AgentQueryRequest):
    sid = (request.session_id or "").strip() or str(uuid.uuid4())
    ticket = reserve_workspace_task(sid, "agent", request.query, request.execution)
    return invoke_agent_task(request, sid, ticket["task_id"])


@app.post("/agent/tasks", status_code=202)
def submit_agent_task(request: AgentQueryRequest, backgroundtasks: BackgroundTasks):
    """页面先取得持久化任务身份，再通过同一任务查询进度与补充要求。"""
    sid = (request.session_id or "").strip() or str(uuid.uuid4())
    ticket = reserve_workspace_task(sid, "agent", request.query, request.execution)
    tid = ticket["task_id"]
    try:
        update_task_status(tid, TASK_STATUS_PENDING)
        backgroundtasks.add_task(invoke_agent_task, request, sid, tid, raise_errors=False)
    except Exception:
        finish_failed_task(tid, {"code": "agent_admission_failed", "message": "Agent任务准备未完成。"})
        raise
    return {"session_id": sid, "task_id": tid, "run_id": ticket.get("run_id"),
            "mode": "agent", "message": "Agent任务已登记，请按任务身份查询进度。"}


@workspace_execution.scoped_task
def invoke_agent_task(request: AgentQueryRequest, sid: str, tid: str, *, raise_errors=True):
    """同步与后台入口复用执行、修订、审计和最终交付控制。"""
    try:
        query_tasks.begin(tid)
        update_task_status(tid, TASK_STATUS_PROCESSING)
        work = TaskWork(task_id=tid)
        def execute(effective, revision, run_id, controls):
            from app.shared.runtime.telemetry import capture_trace
            with capture_trace(tid,'agent',revision) as timing:
                reply = run_agent(effective, sid, request.plan if revision == 1 else None,
                                  task_id=tid, run_id=run_id, defer_history=True,
                                  input_revision=revision, task_work=work, requested_answer_format=controls.answer_format,
                                  source_requirement=controls.source_requirement)
            reply['timing_trace_id']=timing.data['trace_id']
            return reply.pop("_answer_state", None), reply
        _, result = run_revisions(tid, execute, mode="agent", task_work=work)
        error = result.get("error") if result.get("status") in {"failed", "rejected"} else None
        query_tasks.finish(tid, result=result, error=error)
        update_task_status(tid, TASK_STATUS_FAILED if error else TASK_STATUS_COMPLETED)
        return result
    except Exception:
        error = {"code": "agent_execution_failed", "message": "本次Agent执行未完成，请查看任务记录或稍后重试。"}
        finish_failed_task(tid, error)
        update_task_status(tid, TASK_STATUS_FAILED)
        logger.exception("Agent请求失败，task_id={}", tid)
        if raise_errors:
            raise HTTPException(status_code=503, detail={**error, "task_id": tid}) from None
        return None  # 后台失败由/tasks终态交付，不在已返回202后抛连接异常。

#接口1返回页面
@app.get("/html")
def query_html():
    query_html_obj:Path=PROJECT_ROOT/"app"/"resources"/"html"/"chat.html"
    return  FileResponse(
        path=str(query_html_obj),
        media_type=guess_type(query_html_obj.name)[0]
    )

@app.get("/health")
def health():
    logger.info(f"完成健康检查!{datetime.datetime.now()}")
    return HealthResponseSchema(
        code=200,
        message=f"完成健康检查!{datetime.datetime.now()}"
    )


def query_result_payload(state, session_id, task_id):
    return {"session_id": session_id, "task_id": task_id, "run_id": state.get("run_id"),
            "answer": state.get("answer", ""), "image_urls": state.get("image_urls", []),
            "done_list": get_done_task_list(task_id), "running_list": [],
            "status": TASK_STATUS_COMPLETED, "error": "",
            "input_revision": state.get("input_revision", 1),
            **{key: state.get(key, {}) if key not in {"citations"} else state.get(key, [])
               for key in ("web_search_status", "rerank_status", "citations", "citation_status", "context_budget")}}


@workspace_execution.scoped_task
def invoke_query_graph(session_id:str,original_query:str,is_stream:bool,task_id:str | None=None)-> QueryGraphState:
    tid = task_id or session_id  # 无HTTP的旧节点样例兼容；HTTP总是传独立task_id。
    try:
        query_tasks.begin(tid)
        # 非流式请求在这里初始化；流式请求的队列必须在 /query 返回前创建，
        # 否则浏览器先连 /stream 时会找不到队列并立即断开。
        if not is_stream:
            clear_task(tid)
        update_task_status(tid, TASK_STATUS_PROCESSING, push_queue=is_stream)
        def execute(effective, revision, run_id, controls):
            state = create_query_default_state(session_id=session_id, task_id=tid, original_query=effective,
                    is_stream=is_stream, managed_input=True, input_revision=revision,
                    defer_history=True, defer_delivery=True, requested_answer_format=controls.answer_format,
                    source_requirement=controls.source_requirement)
            from app.shared.runtime.telemetry import capture_trace
            with capture_trace(tid,'rag',revision) as timing:
                result = {**state, **query_app.invoke(state)}
            payload=query_result_payload(result, session_id, tid)
            payload['timing_trace_id']=timing.data['trace_id']
            return result, payload
        if query_tasks.input_snapshot(tid):
            result_state, payload = run_revisions(tid, execute, mode="rag")
        else:  # 独立节点/旧脚本不经HTTP准入时仍可执行。
            result_state = query_app.invoke(create_query_default_state(session_id=session_id, task_id=tid,
                                             original_query=original_query, is_stream=is_stream))
            payload = query_result_payload(result_state, session_id, tid)
        query_tasks.finish(tid, result=payload)
        update_task_status(tid, TASK_STATUS_COMPLETED,push_queue=is_stream)
        #执行完毕以后[final-sse-is_stream=True]
        if is_stream:
            push_to_session(
                tid,
                SSEEvent.FINAL,
                data=payload
            )
            push_to_session(tid, SSEEvent.CLOSE, data={})
        return result_state
    except Exception as e:
        failure = {"code": "query_execution_failed", "message": "本次查询未完成，请查看任务记录或稍后重试。"}
        finish_failed_task(tid, failure)
        update_task_status(tid, TASK_STATUS_FAILED,push_queue=is_stream)
        logger.exception(f"执行查询流程报错,错误信息:{str(e)}")
        if is_stream:
            push_to_session(tid, SSEEvent.ERROR, data={"error": failure["message"], "task_id": tid})
            push_to_session(tid, SSEEvent.CLOSE, data={})
            # BackgroundTask 中不能继续向上抛出，否则服务器会把异常当作连接异常。
            return create_query_default_state(
                session_id=session_id,
                task_id=tid,
                original_query=original_query,
                is_stream=True,
            )
        raise

# 接口3: 查询接口
@app.post("/query")
def query(backgroundtasks:BackgroundTasks, query_params:QueryRequestSchema):
    """
    查询数据
    :param backgroundtasks:
    :param query:
    :return:
    """
    # 1. 获取参数
    query = query_params.query
    session_id = query_params.session_id or str(uuid.uuid4())
    is_stream = query_params.is_stream
    ticket = reserve_workspace_task(session_id, "rag", query, query_params.execution)
    task_id = ticket["task_id"]

    # 2. 封装执行图对象的方法
    # 3. 判断是不是流式
    if is_stream:
        # 准入检查在任何队列操作前完成，不删除同一会话上一轮的队列。
        try:
            clear_task(task_id)
            create_sse_queue(task_id)
            update_task_status(task_id, TASK_STATUS_PENDING, push_queue=True)
            backgroundtasks.add_task(invoke_query_graph, session_id=session_id, task_id=task_id,
                                     original_query=query, is_stream=True)
        except Exception:
            finish_failed_task(task_id, {"code": "query_admission_failed", "message": "任务准备未完成。"})
            remove_sse_queue(task_id)
            raise
        return QueryStreamResponseSchema(
            message=f"已经开始:{query}任务查询!",
            session_id=session_id,
            task_id=task_id,
        )

    # 非流式：直接调用图执行器，等待完成后整理响应。
    # 这里没有使用 BackgroundTasks，也没有 await 协程调用。
    else:
        try:
            state = invoke_query_graph(session_id=session_id, task_id=task_id,
                                       original_query=query, is_stream=False)
            result = query_tasks.get(task_id)["result"]
            if result is None:
                result = query_result_payload(state, session_id, task_id)
                query_tasks.finish(task_id, result=result)
            return result
        except Exception:
            finish_failed_task(task_id, {"code": "query_execution_failed", "message": "本次查询未完成。"})
            raise
# 接口4：流式接口
@app.get("/stream/{task_id}")
def stream(task_id:str,request:Request):
    if get_sse_queue(task_id) is None:
        snapshot = query_tasks.get(task_id)
        if snapshot and snapshot["status"] in TERMINAL_STATUSES:
            async def terminal_snapshot():
                yield _sse_pack(SSEEvent.READY, {"task_id": task_id})
                if snapshot["status"] == "completed":
                    yield _sse_pack(SSEEvent.FINAL, snapshot["result"] or {"task_id": task_id, "answer": ""})
                else:
                    yield _sse_pack(SSEEvent.ERROR, {"task_id": task_id,
                        "error": (snapshot.get("error") or {}).get("message", "本轮未完成"),
                        "code": (snapshot.get("error") or {}).get("code"), "status": snapshot["status"]})
            return StreamingResponse(terminal_snapshot(), media_type="text/event-stream")
    try:
        stream_queue = claim_sse_queue(task_id)
    except StreamAlreadyConnected:
        raise HTTPException(status_code=409, detail="该任务已有一个进度连接，请先关闭旧连接。") from None
    if stream_queue is None:
        raise HTTPException(status_code=404, detail="流式任务不存在或已结束，请先提交流式查询")
    # request.is_disconnected() 通过request对象可以检查前端是否已经断开!
    return StreamingResponse(
        sse_generator(task_id,request,stream_queue=stream_queue),
        media_type="text/event-stream"
    )

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    # 删除历史也占用同一会话准入，不能在另一轮读写上下文时清空它。
    ticket = reserve_query_task(session_id, "history_clear")
    try:
        deleted_count = history_repository.clear_session(session_id=session_id)
        query_tasks.finish(ticket["task_id"], result={"deleted_count": deleted_count})
    except Exception:
        finish_failed_task(ticket["task_id"], {"code": "history_clear_failed", "message": "历史清理未完成。"})
        raise
    return HistoryClearResponseSchema(
        message=f"session_id:{session_id}历史聊天记录已经清空!",
        deleted_count=deleted_count
    )

@app.get("/history/{session_id}")
def get_history(session_id: str, limit: int = Query(default=10, ge=1, le=100)):
    history_list : list[dict] = (
        history_repository.list_recent(session_id=session_id, limit=limit)
    )
    return HistoryListResponseSchema(
        session_id=session_id,
        items=[
            HistoryItemResponseSchema(
                id=str(item.get("_id")),
                session_id=session_id,
                task_id=item.get("task_id") or "",
                run_id=item.get("run_id"),
                input_revision=item.get("input_revision", 1),
                message_kind=item.get("message_kind", ""),
                response_elapsed_ms=(item.get("answer_audit") or {}).get("response_elapsed_ms"),
                role=item.get("role"),
                text=item.get("text"),
                rewritten_query=item.get("rewritten_query") or "",
                item_names=item.get("item_names") or [],
                image_urls=item.get("image_urls") or [],
                eeg_attachments=item.get("eeg_attachments") or [],
                citations=item.get("citations") or [],
                citation_status=item.get("citation_status") or {},
                ts=item.get("ts")
            )
            for item in history_list
        ]
    )
@app.get("/sources/original/{doc_id}/{source_type}")
def get_source_original(doc_id: str, source_type: str):
    """按内容身份取归档原件；不接受客户端对象键或本地文件路径。"""
    if not re.fullmatch(r"[0-9a-f]{64}", doc_id) or source_type not in ORIGINAL_TYPES:
        raise HTTPException(status_code=400, detail="无效的原件身份或格式")
    try:
        response = minio_gateway.read_original(f"originals/{doc_id}/source.{source_type}")
    except S3Error as exc:
        status = 404 if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"} else 502
        raise HTTPException(status_code=status, detail="原件不存在或暂不可读取") from None
    except Exception:
        raise HTTPException(status_code=502, detail="原件读取失败，请检查对象存储") from None

    def content():
        try:
            yield from response.stream(amt=64 * 1024)
        finally:
            response.close()
            response.release_conn()

    filename = f"{doc_id[:12]}.{source_type}"
    html_original=source_type in {'html','htm'}
    return StreamingResponse(content(), media_type="application/octet-stream" if html_original else guess_type(filename)[0] or "application/octet-stream",
                             headers={"Content-Disposition": f'{"attachment" if html_original else "inline"}; filename="{filename}"',
                                      "X-Content-Type-Options":"nosniff","Content-Security-Policy":"sandbox; default-src 'none'"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)


