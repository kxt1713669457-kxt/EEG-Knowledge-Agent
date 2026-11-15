import hashlib
import json
import threading
import shutil
from contextlib import nullcontext
from datetime import datetime, timezone
import time
import uuid
import zipfile
from pathlib import Path

from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import PDF_PARSE_SERVICE_LOCAL_DIR, MINERU_MODEL_VERSION, MINERU_POLL_INTERVAL_SECONDS, \
    MINERU_POLL_TIMEOUT_SECONDS, MINERU_REQUEST_TIMEOUT_SECONDS, MINERU_DOWNLOAD_TIMEOUT_SECONDS
from app.rag.import_.parse_result_service import record_parse_result, record_parse_progress
from app.rag.import_.parse_errors import MinerUServiceError, check_mineru_response
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log
import requests
"""
1. 参数获取和校验（state -> str)-> (pdf_path:Path,local_dir:Path)
# 获取参数 并且校验 非空 | 真实存在性校验
1.1 state获取 pdf_path 和 local_dir : str
1.2 进行pdf_path非空校验
    空 -> 打印日志error -> 异常 ValueError
1.3 进行local_path非空校验
    空 -> 打印日志warning -> 给与默认值 项目根路径 / output
1.4 将pdf_path local_dir 转成Path
1.5 pdf_path_obj:Path 判断是否存在
    不存在,打印日志error -> 异常 FileNotFoundError
1.6 local_dir_obj:Path 是不是目录(is_dir())
    不是(不存在,或者不是目录)
    打印日志warning -> 创建 local_dri_obj对应的文件夹
1.7 返回结果 return pdf_path_obj , local_dir_obj
"""

@step_log("validate_pdf_paths")
def validate_pdf_paths(state):
    """
    进行参数获取和校验并且完成文件存在性和文件夹的创建
    :param state:
    :return:
    """
    #1.1state获取pdf_path 和 local_dir : str
    pdf_path=state.get("pdf_path")
    local_dir=state.get("local_dir")
    #1.2 进行pdf_path非空校验
    #空 -> 打印日志error -> 异常 ValueError
    if not pdf_path:
        logger.error(f"pdf_path为空未找到文件，无法继续业务，提前终止")
        raise ValueError(f"pdf_path为空未找到文件，无法继续业务，提前终止")
    #1.3 进行local_path非空校验
    #空 -> 打印日志warning -> 给与默认值 项目根路径 / output
    if not local_dir:
        logger.warning(f"local_dir未空,我们给与默认值,默认为: 项目根地址/output 文件夹!")
        """
        Path  name stem suffix  read... writer...
        拼接地址 Path  / Path or "xxx"
        local_dir_obj = PROJECT_ROOT / "output" / "xxx"  -> 推荐
        """
        local_dir:Path = PROJECT_ROOT / PDF_PARSE_SERVICE_LOCAL_DIR
        #后续节点使用
        state["local_dir"]=str(local_dir)
    # 1.4 将pdf_path local_dir 转成Path
    pdf_path_obj: Path = Path(pdf_path)
    # Path(str | Path) 兼容语法
    local_dir_obj: Path = Path(local_dir)

    # 1.5 pdf_path_obj:Path 判断是否存在
    # 不存在,打印日志error -> 异常 FileNotFoundError
    if not pdf_path_obj.exists():
        logger.error(f"pdf_path_obj:存在地址，但{pdf_path_obj}未找到文件，无法继续业务，提前终止")
        raise FileNotFoundError(f"pdf_path_obj:存在地址，但{pdf_path_obj}未找到文件，无法继续业务，提前终止")
    # 1.6 判断local_dir_obj是否为目录(is_dir())
    # 条件：目录不存在 或者 路径不是文件夹
    if not local_dir_obj.is_dir():
        logger.warning(f"存在local_dir地址:{str(local_dir_obj)},但是没有对应的文件夹,我们创建对应的文件夹,业务继续!!")
        # parents=True：递归创建多级目录，等价于Linux mkdir -p
        # exist_ok=True：目录已存在时不会抛出异常，避免重复创建报错
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    # 1.7 返回Path对象结果
    return pdf_path_obj, local_dir_obj
def _pdf_hash(source: Path) -> str:
    with source.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mineru_checkpoint_path(source: Path) -> Path:
    """原件内容与解析引擎共同确定远端任务记录；不同文件/引擎不会混用。"""
    if MINERU_MODEL_VERSION not in {"vlm", "pipeline"}:
        raise ValueError("PDF解析引擎仅支持vlm或pipeline")
    return PROJECT_ROOT / "output/mineru_jobs" / f"{_pdf_hash(source)}-{MINERU_MODEL_VERSION}.json"


def _write_job(path: Path | None, job: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


_JOB_LOCKS: dict[str, threading.Lock] = {}
_JOB_LOCKS_GUARD = threading.Lock()


def _job_lock(path: Path | None):
    if path is None:
        return nullcontext()
    with _JOB_LOCKS_GUARD:
        return _JOB_LOCKS.setdefault(str(path.resolve()), threading.Lock())


@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj: Path, *, checkpoint_path: Path | None = None,
                        on_progress=None) -> str:
    """上传一次并保存batch_id；超时后重新调用可继续查询同一个远端任务。"""
    header = {"Content-Type": "application/json",
              "Authorization": f"Bearer {infra_config.mineru_config.api_key}"}
    base_url = infra_config.mineru_config.base_url.rstrip("/")
    fingerprint = _pdf_hash(pdf_path_obj) if checkpoint_path is not None else None
    job = {}

    # 只锁住同内容任务的检查和提交，等待远端解析时不阻塞其他文档。
    with _job_lock(checkpoint_path):
        if checkpoint_path is not None and checkpoint_path.exists():
            try:
                saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                saved = {}
            if (saved.get("file_hash") == fingerprint and saved.get("model_version") == MINERU_MODEL_VERSION
                    and saved.get("uploaded") is True and saved.get("batch_id")
                    and saved.get("remote_state") != "failed"):
                job = saved

        resumed = bool(job)
        if not job:
            file = {"name": pdf_path_obj.name}
            if fingerprint:
                file["data_id"] = fingerprint
            response = requests.post(f"{base_url}/file-urls/batch", headers=header,
                                     json={"files": [file], "model_version": MINERU_MODEL_VERSION},
                                     timeout=MINERU_REQUEST_TIMEOUT_SECONDS)
            result = check_mineru_response(response, operation="create_batch")
            data = result.get("data") or {}
            batch_id, upload_urls = data.get("batch_id"), data.get("file_urls") or []
            if not batch_id:
                raise ValueError("MinerU未返回batch_id")
            if not upload_urls:
                raise ValueError("MinerU未返回上传地址")
            with requests.Session() as session:
                session.trust_env = False
                with pdf_path_obj.open("rb") as stream:
                    uploaded = session.put(upload_urls[0], data=stream, timeout=MINERU_REQUEST_TIMEOUT_SECONDS)
            if uploaded.status_code not in {200, 201}:
                raise RuntimeError(f"PDF上传失败，HTTP状态码:{uploaded.status_code}")
            job = {"batch_id": batch_id, "file_hash": fingerprint,
                   "model_version": MINERU_MODEL_VERSION, "filename": pdf_path_obj.name,
                   "uploaded": True, "remote_state": "waiting-file",
                   "submitted_at": datetime.now(timezone.utc).isoformat()}
            _write_job(checkpoint_path, job)

    def progress(phase, **details):
        if on_progress:
            on_progress({"phase": phase, "remote_state": job.get("remote_state", "unknown"),
                         "batch_id": job["batch_id"], "model_version": MINERU_MODEL_VERSION, **details})

    progress("polling", resumed=resumed)
    start = time.monotonic()
    while True:
        remaining = MINERU_POLL_TIMEOUT_SECONDS - (time.monotonic() - start)
        if remaining <= 0:
            progress("timed_out")
            raise TimeoutError(f"MinerU解析轮询超时，最后状态:{job.get('remote_state', 'unknown')}；远端任务已保存，可恢复查询")
        try:
            response = requests.get(f"{base_url}/extract-results/batch/{job['batch_id']}",
                                    headers=header, timeout=min(MINERU_REQUEST_TIMEOUT_SECONDS, remaining))
        except requests.RequestException as error:
            progress("network_retry", error_type=type(error).__name__)
            time.sleep(min(MINERU_POLL_INTERVAL_SECONDS, max(0, remaining)))
            continue
        if 500 <= response.status_code < 600:
            progress("server_retry", http_status=response.status_code)
            time.sleep(min(MINERU_POLL_INTERVAL_SECONDS, max(0, remaining)))
            continue
        result = check_mineru_response(response, operation="poll_batch")
        items = (result.get("data") or {}).get("extract_result") or []
        if items:
            item = next((item for item in items if fingerprint and item.get("data_id") == fingerprint), items[0])
            remote = item.get("state")
            job["remote_state"] = remote if remote in {"done", "waiting-file", "pending", "running", "failed", "converting"} else "unknown"
            job["last_checked_at"] = datetime.now(timezone.utc).isoformat()
            _write_job(checkpoint_path, job)
            progress("polling")
            if remote == "done":
                zip_url = item.get("full_zip_url")
                if not zip_url:
                    raise ValueError("MinerU完成任务未提供解析产物")
                return zip_url
            if remote == "failed":
                raise MinerUServiceError(
                    "mineru_parse_failed", "MinerU 已接收 PDF，但云端解析失败，尚未取得 Markdown。原件已保留，可重新导入；持续失败时请检查原件或解析服务。",
                    operation="parse_document", http_status=response.status_code, recovery_action="retry_parsing")
        time.sleep(min(MINERU_POLL_INTERVAL_SECONDS, max(0, remaining)))
@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_url: str, local_dir_obj: Path, file_name: str) -> Path:
    """下载并展开MinerU产物；保留每次产物目录，不删除先前来源。"""
    label = Path(file_name.replace("\\", "/")).name
    if label in {"", ".", ".."}:
        raise ValueError("无效文档名称")
    response = requests.get(zip_url, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"下载解析产物失败，HTTP状态码:{response.status_code}")
    local_dir_obj = Path(local_dir_obj)
    local_dir_obj.mkdir(parents=True, exist_ok=True)
    zip_path = local_dir_obj / f"{label}.zip"
    zip_path.write_bytes(response.content)
    extract_dir = local_dir_obj / f"{label}_parsed_{uuid.uuid4().hex[:8]}"
    extract_dir.mkdir()
    root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path) as package:
        for member in package.infolist():
            target = (root / member.filename.replace("\\", "/")).resolve()
            if not target.is_relative_to(root):
                raise ValueError("解析产物包含越界的文件路径")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    markdowns = sorted(extract_dir.rglob("*.md"))
    if not markdowns:
        raise ValueError("解析产物未包含Markdown")
    chosen = next((path for path in markdowns if path.stem == label), None)
    chosen = chosen or next((path for path in markdowns if path.stem == "full"), markdowns[0])
    target = chosen.with_name(f"{label}.md")
    return chosen if chosen == target else chosen.rename(target)


@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    进行pdf转成md业务!
    最后修改state md_path属性
    """
    #1.获取并校验参数 validate_pdf_paths(state)->tuple(pdf_path_obj:Path,local_dir_obj:Path)
    pdf_path_obj, local_dir_obj = validate_pdf_paths(state)
    def progress(details):
        record_parse_progress(state, parser="mineru", status="processing", **details)

    try:
        checkpoint = mineru_checkpoint_path(pdf_path_obj)
        zip_url = upload_pdf_and_poll(pdf_path_obj, checkpoint_path=checkpoint, on_progress=progress)
        record_parse_progress(state, phase="downloading")
        md_path_obj = download_and_extract_markdown(zip_url, local_dir_obj, pdf_path_obj.stem)
        state["md_path"] = str(md_path_obj)
        return record_parse_result(state, "mineru", model_version=MINERU_MODEL_VERSION,
                                   batch_id=state["parse_metadata"].get("batch_id"), remote_state="done",
                                   image_count=sum(path.is_file() for path in (md_path_obj.parent / "images").glob("*")))
    except Exception as error:
        details = error.public_details() if isinstance(error, MinerUServiceError) else {}
        record_parse_progress(state, parser="mineru", status="failed", error_type=type(error).__name__, **details)
        raise
