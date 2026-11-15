"""供应商原生额度读取；百炼使用官方CLI控制台授权，额度与本地usage分开。"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from threading import Lock, Thread, Timer
import uuid

from app.shared.runtime.logger import PROJECT_ROOT
from app.workspace import store, secrets
from app.workspace.connections import get_connection, connection_key, request_json, MODEL_PATTERN

CLI_PATH = PROJECT_ROOT / "output/runtime/bailian-cli/node_modules/bailian-cli/dist/bailian.mjs"
NODE_PATH = Path("F:/Node/node.exe")
AUTH_ROOT = PROJECT_ROOT / "output/runtime/quota-auth"
_jobs, _children, _guard = {}, {}, Lock()


def _identity(row):
    return hashlib.sha256(connection_key(row).encode()).hexdigest()


@contextmanager
def cli_environment(row):
    if not CLI_PATH.is_file() or not NODE_PATH.is_file():
        raise ValueError("百炼授权组件未安装，请联系本机维护者。")
    folder = AUTH_ROOT / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    # CLI只处理控制台认证，不需要模型密钥。该非凭据占位仅使官方CLI跳过
    # “没有模型Key时自动创建Key”的分支；后续仍必须取得真实控制台授权才能查额度。
    config = {"api_key": "managed-separately-no-model-calls", "base_url": row["base_url"]}
    saved = store.read("quota_auth", row["id"])
    if saved and saved.get("connection_fingerprint") == _identity(row):
        config.update(json.loads(secrets.reveal(saved["encrypted_credentials"])))
    path = folder / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    env = os.environ.copy()
    env.update(BAILIAN_CONFIG_DIR=str(folder), NO_COLOR="1", CI="1", NO_UPDATE_NOTIFIER="1")
    try:
        yield folder, env
    finally:
        # 只清理本次创建的明确临时文件；不碰用户原来的CLI配置。
        for name in ("config.json", "config.json.tmp", "credentials.json"):
            target = (folder / name).resolve()
            if target.parent == folder.resolve():
                target.unlink(missing_ok=True)
        try:
            folder.rmdir()
        except OSError:
            pass


def start_authorization(identity):
    row = get_connection(identity)
    if row["provider"] != "bailian":
        raise ValueError("该连接不使用百炼控制台授权。")
    if not CLI_PATH.is_file():
        raise ValueError("百炼官方CLI组件未安装。")
    with _guard:
        pending = next((j for j in _jobs.values() if j["connection_id"] == identity and j["status"] == "pending"), None)
        if pending:
            return pending.copy()
        job = {"id": uuid.uuid4().hex, "connection_id": identity, "status": "pending", "authorization_url": None,
               "message": "正在准备百炼官方授权，请使用该API Key所属账号登录。"}
        _jobs[job["id"]] = job
    Thread(target=_authorize_worker, args=(row, job), daemon=True).start()
    return job.copy()


def _authorize_worker(row, job):
    try:
        with _guard:
            if job["status"] == "cancelled":
                return
        with cli_environment(row) as (folder, env):
            process = subprocess.Popen([str(NODE_PATH), str(CLI_PATH), "auth", "login", "--console", "--no-browser", "--output", "text"],
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with _guard:
                _children[job["id"]] = process
                if job["status"] == "cancelled":
                    process.kill()
            timer = Timer(300, process.kill);timer.start()
            try:
                buffer = ""
                while True:
                    data = process.stdout.read1(4096)
                    if not data:
                        break
                    buffer = (buffer + data.decode("utf-8", errors="replace"))[-20000:]
                    for url in re.findall(r"https://[^\s\x1b\"'<>`]+", buffer):
                        from urllib.parse import urlsplit
                        host = urlsplit(url).hostname or ""
                        if host == "bailian.aliyun.com" or host.endswith(".aliyun.com"):
                            with _guard:
                                job.update(authorization_url=url.rstrip(").,，。"), message="打开百炼官方页面完成登录授权，完成后回到这里。")
                            break
                code = process.wait(timeout=5)
            finally:
                timer.cancel()
                process.stdout.close()
            if code != 0:
                raise ValueError("授权未完成或已超时，请重新开始。")
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            credentials = {k:config[k] for k in ("access_token", "console_region", "console_site") if k in config}
            if not credentials.get("access_token"):
                raise ValueError("没有取得控制台授权，请重试。")
            store.write("quota_auth", row["id"], {"id": row["id"], "encrypted_credentials": secrets.protect(json.dumps(credentials)),
                "connection_fingerprint": _identity(row), "authorized_at": store.now()})
        with _guard:
            job.update(status="completed", authorization_url=None, message="授权已完成，可以刷新该模型的免费额度。")
    except Exception:
        with _guard:
            if job["status"] != "cancelled":
                job.update(status="failed", authorization_url=None, message="授权未完成或已超时，请重新开始；不会把失败显示成零额度。")
    finally:
        with _guard:
            _children.pop(job["id"], None)


def cancel_authorization(identity):
    with _guard:
        job = _jobs.get(identity)
        if not job:
            raise ValueError("授权流程不存在。")
        process = _children.get(identity)
        if job["status"] == "pending":
            job.update(status="cancelled", authorization_url=None, message="已取消本次授权。")
            if process is not None and process.poll() is None:
                process.kill()
        return job.copy()


def shutdown_authorizations():
    with _guard:
        identities = list(_children)
    for identity in identities:
        cancel_authorization(identity)


def authorization_status(identity):
    with _guard:
        job = _jobs.get(identity)
        if not job:
            raise ValueError("授权流程不存在或服务已重启，请重新开始。")
        return job.copy()


def read_quota(identity, model):
    row = get_connection(identity)
    if not re.fullmatch(MODEL_PATTERN, model or ""):
        raise ValueError("需要有效模型ID。")
    if row["provider"] == "deepseek":
        data = request_json("https://api.deepseek.com/user/balance", connection_key(row))
        rows = [{"currency": r["currency"], "remaining": str(r["total_balance"])} for r in data.get("balance_infos", [])
                if r.get("currency") in {"CNY", "USD"} and "total_balance" in r]
        return {"status": "available", "kind": "account_balance", "rows": rows, "checked_at": store.now(), "message": "供应商账户余额"}
    if row["provider"] != "bailian":
        return {"status": "unsupported", "kind": "unknown", "rows": [], "message": "该供应商未接入统一额度接口；模型调用不受此显示限制。"}
    saved = store.read("quota_auth", identity)
    if not saved or saved.get("connection_fingerprint") != _identity(row):
        return {"status": "authorization_required", "kind": "free_tokens", "rows": [], "message": "请先登录该API Key所属的百炼账号授权。"}
    with cli_environment(row) as (_, env):
        try:
            result = subprocess.run([str(NODE_PATH), str(CLI_PATH), "usage", "free", "--model", model, "--output", "json"],
                env=env, capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            data = json.loads(result.stdout)
            if result.returncode:
                return {"status": "authorization_required" if "AUTH" in str(data.get("error", {}).get("code", "")).upper() else "failed",
                        "kind": "free_tokens", "rows": [], "message": "额度查询未完成，授权可能过期；可刷新或重新授权。"}
            if not isinstance(data, list):
                raise ValueError()
            rows = [{k:r.get(k) for k in ("model", "remaining", "total", "expires")} for r in data if isinstance(r, dict)]
            return {"status": "available", "kind": "free_tokens", "rows": rows, "checked_at": store.now(),
                    "message": "所授权控制台账号的模型免费token额度；无免费额度不代表付费账户余额为零。"}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {"status": "failed", "kind": "free_tokens", "rows": [], "message": "额度查询失败，请检查连接后刷新，不据此判断已用完。"}
