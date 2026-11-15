"""本机研究者资料与已记录用量；模型调用密钥不具备控制台余额查询权限。"""
from datetime import datetime, timedelta, timezone
import json
import os
from threading import RLock
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo import MongoClient

from app.shared.config.lm_config import lm_config
from app.shared.runtime.logger import PROJECT_ROOT

router = APIRouter(prefix="/workspace", tags=["研究空间"])
PROFILE_PATH = PROJECT_ROOT / "output/runtime/workspace-profile.json"
_profile_lock = RLock()
DEFAULT_PROFILE = {"name": "个人研究者", "role": "独立研究者", "focus": "运动想象 · EEG"}


class WorkspaceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=40)
    role: str = Field(max_length=60)
    focus: str = Field(max_length=100)

    @field_validator("name", "role", "focus")
    @classmethod
    def no_control_characters(cls, value):
        if any(ord(c) < 32 for c in value):
            raise ValueError("资料不能包含控制字符")
        return value


def read_profile():
    with _profile_lock:
        try:
            return WorkspaceProfile.model_validate(json.loads(PROFILE_PATH.read_text(encoding="utf-8"))).model_dump()
        except (OSError, ValueError):
            return DEFAULT_PROFILE.copy()


def summarize_usage(rows, *, limit=1000):
    """只累计保存过的生成/复核 usage，不用请求数估算 token 或账户余额。"""
    result = {"status": "available", "answer_count": 0, "calls_with_usage": 0,
              "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
              "truncated": len(rows) > limit, "days": 30, "limit": limit,
              "scope": "最近30天最多1000条已保存回答的生成与来源复核；不含改写、HyDE、规划、视觉解析、未落库调用，不是账单总用量。"}
    for row in rows[:limit]:
        result["answer_count"] += 1
        audit = row.get("answer_audit") or {}
        for stage in ("generation_usage", "support_review_usage"):
            usage = audit.get(stage)
            if not isinstance(usage, dict):
                continue
            values = [usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")]
            if not all(type(v) is int and v >= 0 for v in values):
                continue
            result["calls_with_usage"] += 1
            for key, value in zip(("input_tokens", "output_tokens", "total_tokens"), values):
                result[key] += value
    return result


def recorded_usage():
    client = None
    try:
        client = MongoClient(os.environ["MONGO_URL"], serverSelectionTimeoutMS=1500,
                             connectTimeoutMS=1500, socketTimeoutMS=2500)
        since = (datetime.now(timezone.utc) - timedelta(days=30)).timestamp()
        rows = list(client[os.environ["MONGO_DB_NAME"]].chat_message.find(
            {"role": "assistant", "ts": {"$gte": since}},
            {"_id": 0, "answer_audit.generation_usage": 1, "answer_audit.support_review_usage": 1},
        ).sort("ts", -1).limit(1001).max_time_ms(1500))
        return summarize_usage(rows)
    except Exception:
        # 不回显连接串、密钥、数据库异常或错误地把失败显示成零用量。
        return {"status": "unavailable", "message": "用量记录暂不可读取，请稍后刷新。"}
    finally:
        if client is not None:
            client.close()


def require_local_page(request: Request):
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlsplit(origin)
            valid = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port in {8000, 8001, 18000, 18001}
        except ValueError:
            valid = False
        if not valid:
            raise HTTPException(403, "仅允许本机研究页面访问资料")


@router.get("/account")
def workspace_account(request: Request):
    require_local_page(request)
    try:
        host = urlsplit(lm_config.base_url or "").hostname or ""
    except ValueError:
        host = ""
    is_bailian = host == "dashscope.aliyuncs.com" or host.endswith(".dashscope.aliyuncs.com")
    return {
        "profile": read_profile(),
        "service": {"provider": "阿里云百炼" if is_bailian else "OpenAI兼容服务",
                    "model": lm_config.llm_model, "vision_model": lm_config.lv_model,
                    "configured": bool(lm_config.api_key and lm_config.base_url)},
        "quota": {"status": "console_auth_required" if is_bailian else "unavailable", "remaining": None,
                  "message": "剩余额度需在供应商控制台查看，当前项目未接入账户授权。" if is_bailian
                  else "当前供应商未接入余额查询，无法读取剩余额度。",
                  "console_url": "https://bailian.console.aliyun.com/" if is_bailian else None,
                  "help_url": "https://help.aliyun.com/zh/model-studio/cli/installation" if is_bailian else None},
        "usage": recorded_usage(), "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.put("/profile")
def save_workspace_profile(profile: WorkspaceProfile, request: Request):
    require_local_page(request)
    with _profile_lock:
        try:
            PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            temp = PROFILE_PATH.with_suffix(".tmp")
            temp.write_text(json.dumps(profile.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(PROFILE_PATH)
        except OSError:
            raise HTTPException(503, "资料暂未保存，请稍后重试。") from None
    return {"profile": profile.model_dump()}
