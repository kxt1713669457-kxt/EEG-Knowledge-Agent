"""
导入服务 HTTP 入口模块，直接承载导入接口与相关接口业务逻辑。
"""
import json
import shutil
import sys
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from urllib.parse import quote

from app.api.schema.import_schema import UploadResponseSchema, StatusResponseSchema
from app.process.import_.agent.main_graph import import_app
from app.infra.object_storage.minio_gateway import minio_gateway
from app.rag.import_.document_asset_service import (
    archive_original_document, describe_document, write_source_manifest,
)
from app.rag.import_.import_receipt_service import (
    document_import_lock, find_completed_receipt, save_completed_receipt,
)
from app.rag.import_.parse_errors import MinerUServiceError
from app.shared.runtime.logger import logger,PROJECT_ROOT

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.state import get_default_state, ImportGraphState, create_default_state
from app.infra.config.providers import infra_config
from app.api.http.auth_api import install_auth
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_running_task,
    get_task_result,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
    set_task_result,
)
app = FastAPI(
    title=infra_config.settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0",
)
install_auth(app)

@app.get("/html")
@app.get("/import/html")
def import_html():
    """
    返回导入演示页面。
    Returns:
        FileResponse: 本地导入演示页面文件响应。
    """
    html_path = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/health")
def health():
    """仅检查 HTTP 服务存活；外部依赖由诊断脚本单独探测。"""
    return {"code": 200, "message": "导入服务正在运行"}


def invoke_import_graph(task_id:str,local_file_path:str,local_dir:str):
    stage = "archive_original"
    state: ImportGraphState = create_default_state(
        task_id=task_id, local_file_path=local_file_path, local_dir=local_dir,
    )
    try:
        update_task_status(task_id,status_name=TASK_STATUS_PROCESSING)
        state["document_metadata"] = json.loads(get_task_result(task_id, "document_metadata", "{}"))
        add_running_task(task_id, stage)
        metadata = archive_original_document(state)
        set_task_result(task_id, "document_metadata", json.dumps(metadata, ensure_ascii=False))
        add_done_task(task_id, stage)
        stage = "check_receipt"
        with document_import_lock(metadata["doc_id"]):
            receipt = find_completed_receipt(metadata["doc_id"], metadata["source_type"])
            if receipt:
                state["index_metadata"] = {**receipt["index_metadata"], "action": "reused",
                                           "reused_from_task_id": receipt["task_id"]}
                state["parse_metadata"] = {**receipt.get("parse_metadata", {}), "reused_from_task_id": receipt["task_id"]}
                add_done_task(task_id, "reuse_index")
            else:
                stage = "import_graph"
                state = import_app.invoke(state)
                stage = "receipt_commit"
                save_completed_receipt(state)
            for field in ("parse_metadata", "index_metadata"):
                set_task_result(task_id, field, json.dumps(state.get(field, {}), ensure_ascii=False))
        update_task_status(task_id,status_name=TASK_STATUS_COMPLETED)
    except Exception as e:
        if stage == "archive_original" and state.get("document_metadata", {}).get("archive_status") != "completed":
            metadata = state.get("document_metadata") or {}
            metadata["archive_status"] = "failed"
            set_task_result(task_id, "document_metadata", json.dumps(metadata, ensure_ascii=False))
        set_task_result(task_id, "failed_stage", "node_pdf_to_md" if isinstance(e, MinerUServiceError) else stage)
        set_task_result(task_id, "error_code", e.code if isinstance(e, MinerUServiceError) else "")
        message = (e.public_message if isinstance(e, MinerUServiceError) else
                   f"{'原件归档' if stage == 'archive_original' else '文档处理'}失败（{type(e).__name__}），请查看服务日志。")
        set_task_result(task_id, "error", message)
        update_task_status(task_id,status_name=TASK_STATUS_FAILED)
        logger.exception(f"导入模块发生异常,异常信息{str(e)}")
#接口2 ：导入文件
#异步接口
#前端 文件 接口 langgraph解析（很久） 前端返回结果

@app.post("/upload")
def uploads(backgroundtasks:BackgroundTasks,files:list[UploadFile]):
    """
    本质: 异步调用图对象,解析本次上传的文件
        backgroundtasks.add_task(函数, 参数)

    1. 先定一个一个执行graph的函数 (task_id, local_file_path, local_dir)
    2. 凑齐参数 task_id, local_file_path, local_dir
    3. 异步调用 执行graph的函数 并传递参数
    4. 返回前端数据

    :param backgroundtasks:
    :param files:
    :return:
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个 PDF、DOCX 或 Markdown 文件")

    # 先校验整个批次，再保存；只使用文件名，避免客户端路径进入任务目录。
    prepared_files = []
    for uploaded_file in files:
        filename = Path((uploaded_file.filename or "").replace("\\", "/")).name
        if not filename or Path(filename).suffix.lower() not in {".pdf", ".md", ".docx"}:
            raise HTTPException(status_code=400, detail="当前仅支持 PDF、DOCX 和 Markdown 文件")
        uploaded_file.file.seek(0)
        if not uploaded_file.file.read(1):
            raise HTTPException(status_code=400, detail=f"文件内容为空：{filename}")
        uploaded_file.file.seek(0)
        prepared_files.append((uploaded_file, filename))

    task_ids = []
    documents = []
    time_now_str = datetime.now().strftime("%Y%m%d")
    for uploaded_file, filename in prepared_files:
        task_id = str(uuid.uuid4())
        local_dir_obj = PROJECT_ROOT / "output" / time_now_str / task_id
        local_dir_obj.mkdir(parents=True, exist_ok=True)
        local_file_path_obj = local_dir_obj / filename
        with local_file_path_obj.open("wb") as file_buffer:
            shutil.copyfileobj(uploaded_file.file, file_buffer)
        metadata = describe_document(local_file_path_obj)
        write_source_manifest(local_dir_obj, task_id, metadata)
        set_task_result(task_id, "document_metadata", json.dumps(metadata, ensure_ascii=False))
        documents.append({"task_id": task_id, **metadata})
        update_task_status(task_id, TASK_STATUS_PENDING)
        backgroundtasks.add_task(
            invoke_import_graph,
            task_id=task_id,
            local_file_path=str(local_file_path_obj),
            local_dir=str(local_dir_obj),
        )
        task_ids.append(task_id)

    return UploadResponseSchema(
        code=200,
        message=f"{len(files)}文件上传成功",
        task_ids=task_ids,
        documents=documents,

    )


#接口3 获取请求状态
@app.get("/status/{task_id}")
def task_status(task_id:str):
    # done_list running_list
    # 每次调用的时候,先存储!
    # task_utils . add_running / done _task (task_id , 节点名)
    # _tasks_running_list : dict [str,list] = task_id -> list -> list.append -> 节点名
    # _tasks_done_list :dict [str,list] = task_id -> list -> list.append -> 节点名
    # _tasks_running_list = {001: [node_entry,node_pdf_to_md]}
    # _tasks_running_list = {002: [node_entry,node_pdf_to_md]}
    # _tasks_running_list = {003: [node_entry,node_pdf_to_md]}
    #task_utils 每次解析任务内部节点的状态
    #task_utils 1.定义了存储数据的字典 2.定义了存储数据的方法 3.定义了获取数据的方法
    status = get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在或服务重启后已失效")
    done_list=get_done_task_list(task_id)
    running_list=get_running_task_list(task_id)

    #status 这个任务的总状态


    return StatusResponseSchema(
        code =200,
        task_id=task_id,
        status=status,
        done_list=done_list,
        running_list=running_list,
        document_metadata=json.loads(get_task_result(task_id, "document_metadata", "{}")),
        error=get_task_result(task_id, "error"),
        error_code=get_task_result(task_id, "error_code"),
        failed_stage=get_task_result(task_id, "failed_stage"),
        parse_metadata=json.loads(get_task_result(task_id, "parse_metadata", "{}")),
        index_metadata=json.loads(get_task_result(task_id, "index_metadata", "{}")),
    )


@app.get("/original/{task_id}")
def download_original(task_id: str):
    """通过已登记任务的归档键取回原件，不接受客户端文件路径。"""
    if not get_task_status(task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    metadata = json.loads(get_task_result(task_id, "document_metadata", "{}"))
    if metadata.get("archive_status") != "completed":
        raise HTTPException(status_code=409, detail="原件尚未完成归档")
    try:
        response = minio_gateway.read_original(metadata["object_key"])
    except Exception:
        raise HTTPException(status_code=502, detail="原件读取失败，请检查对象存储") from None

    def content():
        try:
            yield from response.stream(amt=64 * 1024)
        finally:
            response.close()
            response.release_conn()

    filename = metadata.get("original_filename", "document")
    return StreamingResponse(
        content(), media_type=guess_type(filename)[0] or "application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename, safe='')}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=infra_config.settings.app_host, port=infra_config.settings.import_app_port)

