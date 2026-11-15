"""当前工作区只启动一个查询worker；先获得本机文件锁，再处理中断记录。"""
from contextlib import asynccontextmanager, contextmanager
import os

from app.infra.persistence.task_repository import MongoTaskRepository
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.runtime.query_tasks import query_tasks
from app.shared.utils.task_utils import set_progress_observer


@contextmanager
def single_query_worker(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("当前工作区已有查询服务，本版本仅支持单worker；未处理中断记录。") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@asynccontextmanager
async def query_lifespan(app):
    with single_query_worker(PROJECT_ROOT / "output/runtime/query-worker.lock"):
        repository = MongoTaskRepository.connect()
        try:
            interrupted = query_tasks.configure_repository(repository)
            set_progress_observer(query_tasks.persist_progress)
            logger.info("查询任务存储已就绪，恢复状态查询；标记中断任务{}个，未自动重新执行。", interrupted)
            yield
        finally:
            from app.workspace.quota import shutdown_authorizations
            shutdown_authorizations()
            set_progress_observer(None)
            query_tasks.configure_repository(None)
            repository.client.close()
