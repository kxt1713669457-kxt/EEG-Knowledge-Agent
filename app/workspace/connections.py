"""用户配置的OpenAI兼容连接；按请求绑定，不修改.env或别的会话。"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
import requests

from app.shared.config.lm_config import lm_config
from app.workspace import store, secrets

PRESETS = {
    "bailian": {"name": "阿里云百炼", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "openai": {"name": "OpenAI", "base_url": "https://api.openai.com/v1"},
    "deepseek": {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1"},
    "siliconflow": {"name": "硅基流动", "base_url": "https://api.siliconflow.cn/v1"},
    "openrouter": {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1"},
    "custom": {"name": "自定义兼容接口", "base_url": ""},
}
_current = ContextVar("workspace_model_connection", default=None)
MODEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,199}$"


class ConnectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=60)
    provider: str = Field(pattern="^(bailian|openai|deepseek|siliconflow|openrouter|custom)$")
    base_url: str = Field(max_length=500)
    api_key: SecretStr | None = None
    default_model: str = Field(default="", max_length=200)
    reasoning_format: str = Field(default="auto", pattern="^(auto|none|openai|qwen|deepseek)$")

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value):
        u = urlsplit(value)
        if (not u.hostname or u.username or u.password or u.query or u.fragment
                or not (u.scheme == "https" or u.scheme == "http" and u.hostname in {"localhost", "127.0.0.1", "::1"})):
            raise ValueError("请输入HTTPS接口地址，本地服务可用HTTP；地址不能含密钥或查询参数。")
        return value.rstrip("/")


class ExecutionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    connection_id: str = Field(default="default", pattern=r"^(default|[0-9a-f]{32})$")
    model: str = Field(default="", max_length=200)
    effort: str = Field(default="auto", pattern=r"^(auto|off|low|medium|high|max|xhigh|minimal|none)$")
    workspace_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    file_ids: list[str] = Field(default_factory=list, max_length=8)
    image_ids: list[str] = Field(default_factory=list, max_length=4)
    eeg_attachment_ids: list[str] = Field(default_factory=list, max_length=5)


@dataclass(frozen=True)
class ModelBinding:
    connection_id: str
    provider: str
    base_url: str
    key: str = field(repr=False)
    model: str = ""
    effort: str = "auto"
    reasoning_format: str = "none"
    levels: tuple = ()

    @property
    def cache_identity(self):
        return (self.connection_id, self.base_url, hashlib.sha256(self.key.encode()).hexdigest(), self.model, self.effort, self.reasoning_format)


def default_connection():
    host = urlsplit(lm_config.base_url or "").hostname or ""
    provider = next((p for p, item in PRESETS.items() if item["base_url"] and urlsplit(item["base_url"]).hostname == host), "custom")
    return {"id": "default", "name": "项目默认连接", "provider": provider, "base_url": lm_config.base_url,
            "default_model": lm_config.llm_model, "reasoning_format": "auto", "models": [], "source": "env"}


def get_connection(identity):
    row = default_connection() if identity == "default" else store.read("connection", identity)
    if row is None or row.get("deleted_at"):
        raise ValueError("模型连接不存在，请重新选择。")
    return row


def public_connection(row):
    return {k: v for k, v in row.items() if k != "encrypted_key"} | {"key_configured": bool(lm_config.api_key) if row["id"] == "default" else bool(row.get("encrypted_key"))}


def connection_key(row):
    return lm_config.api_key if row["id"] == "default" else secrets.reveal(row["encrypted_key"])


def save_connection(payload, identity=None):
    if identity == "default":
        raise ValueError("项目默认配置保留在.env，请另存为自定义连接。")
    previous = get_connection(identity) if identity else {}
    if payload.provider != "custom" and urlsplit(payload.base_url).hostname != urlsplit(PRESETS[payload.provider]["base_url"]).hostname:
        raise ValueError("供应商地址不匹配；网关或代理请选自定义接口。")
    key = payload.api_key.get_secret_value() if payload.api_key else ""
    if any(c.isspace() for c in key):
        raise ValueError("API Key不能含空白字符。")
    if not key and not previous.get("encrypted_key"):
        raise ValueError("请填写API Key。")
    if payload.default_model and not re.fullmatch(MODEL_PATTERN, payload.default_model):
        raise ValueError("模型ID格式不合法。")
    changed = bool(key) or previous.get("base_url") != payload.base_url
    fields = payload.model_dump(exclude={"api_key"}) | {"encrypted_key": secrets.protect(key) if key else previous["encrypted_key"],
        "models": [] if changed else previous.get("models", []), "deleted_at": None, "updated_at": store.now()}
    row = store.write("connection", identity, previous | fields) if identity else store.new_record("connection", **fields)
    return public_connection(row)


def request_json(url, key, *, method="GET", payload=None):
    try:
        with requests.request(method, url, headers={"Authorization": "Bearer " + key}, json=payload,
                              timeout=(5, 15), allow_redirects=False, stream=True) as response:
            if not 200 <= response.status_code < 300:
                status = response.status_code
                raise ValueError("密钥无效或没有权限。" if status in {401, 403} else "额度不足。" if status == 402 else f"供应商接口返回HTTP {status}，请检查地址和模型权限。")
            body = bytearray()
            for part in response.iter_content(65536):
                body.extend(part)
                if len(body) > 2_000_000:
                    raise ValueError("供应商响应过大，未载入。")
            import json
            return json.loads(body)
    except requests.RequestException:
        raise ValueError("连接供应商失败，请检查网络和API地址。") from None


def model_capabilities(row, model, metadata=None):
    metadata = metadata or {}
    format_name = row.get("reasoning_format", "auto")
    if format_name == "auto":
        if row["provider"] == "bailian" and re.match(r"qwen3(?:\.[0-9]+)?-(?:flash|plus|max)(?:$|-)", model) and "preview" not in model:
            format_name = "qwen"
        elif row["provider"] == "deepseek":
            format_name = "deepseek"
        elif row["provider"] == "openai" and re.match(r"(?:o[134](?:-|$)|gpt-[56](?:[.-]|$))", model):
            # 未知模型不虚构档位；用户可在连接的高级设置声明通用reasoning_effort兼容。
            format_name = "openai"
        else:
            format_name = "none"
    levels = {"none": [], "qwen": ["off", "low", "medium", "high"],
              "openai": ["low", "medium", "high"], "deepseek": ["off", "low", "high", "max"]}[format_name]
    declared = metadata.get("effort", {}).get("supported_levels") if isinstance(metadata.get("effort"), dict) else None
    if isinstance(declared, list) and declared and all(v in {"low", "medium", "high", "max", "xhigh", "minimal"} for v in declared):
        levels = (["off"] if format_name in {"deepseek", "qwen"} else []) + declared
    return {"reasoning_format": format_name, "efforts": ["auto", *levels]}


def list_models(identity):
    row = get_connection(identity)
    data = request_json(row["base_url"].rstrip("/") + "/models", connection_key(row))
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("供应商没有返回有效模型列表，可手动填写模型ID。")
    models = []
    for item in data["data"][:2000]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not re.fullmatch(MODEL_PATTERN, item["id"]):
            continue
        if re.search(r"embedding|rerank|tts|asr|image-generation|text-to-image", item["id"], re.I):
            continue
        models.append({"id": item["id"], "name": item.get("name") or item["id"], **model_capabilities(row, item["id"], item)})
    if not models:
        raise ValueError("该连接未返回可选对话模型，请确认权限或手动填写模型ID。")
    if identity != "default":
        row.update(models=models, models_checked_at=store.now());store.write("connection", identity, row)
    else:
        store.write("model_cache", identity, {"models": models, "checked_at": store.now()})
    return models


def resolve_binding(options):
    row = get_connection(options.connection_id)
    model = options.model or row.get("default_model")
    if not model or not re.fullmatch(MODEL_PATTERN, model):
        raise ValueError("请先选择或填写有效的模型ID。")
    models = row.get("models") or ((store.read("model_cache", "default") or {}).get("models", []) if row["id"] == "default" else [])
    selected = next((m for m in models if m["id"] == model), None)
    caps = selected or model_capabilities(row, model)
    if options.effort not in caps["efforts"]:
        raise ValueError("所选模型没有声明支持该推理档位，请选择自动或其他档位。")
    key = connection_key(row)
    if not key:
        raise ValueError("该模型连接没有配置API Key。")
    return ModelBinding(row["id"], row["provider"], row["base_url"], key, model, options.effort, caps["reasoning_format"], tuple(caps["efforts"]))


@contextmanager
def binding_scope(binding):
    token = _current.set(binding)
    try:
        yield
    finally:
        _current.reset(token)


def current_binding():
    return _current.get()


def completion_limit(answer_tokens):
    """LangChain将max_tokens转换为包含思考的总上限，为显式Qwen思考另留预算。

    answer_tokens是预留的回答空间，供应商最终按总量限制；不是两个独立硬上限。
    不改用语义随模型变化的旧max_tokens参数，也不偷偷关闭用户选择的思考。
    """
    binding = current_binding()
    thought = 0
    if binding and binding.reasoning_format == "qwen" and binding.effort in {"low", "medium", "high"}:
        thought = {"low": 1024, "medium": 4096, "high": 8192}[binding.effort]
    return {"max_tokens": answer_tokens + thought}


def inference_kwargs(binding, *, json_mode=False):
    result, extra = {}, {}
    effort, form = binding.effort, binding.reasoning_format
    if effort == "auto":
        if form == "qwen":
            extra["enable_thinking"] = False  # 保留既有默认行为，显式选择才启用深度思考。
    elif form == "qwen":
        extra["enable_thinking"] = effort != "off"
        if effort != "off":
            extra["thinking_budget"] = {"low": 1024, "medium": 4096, "high": 8192}[effort]
    elif form == "deepseek":
        extra["thinking"] = {"type": "disabled" if effort == "off" else "enabled"}
        if effort != "off":
            result["reasoning_effort"] = effort
    elif form == "openai" and effort != "auto":
        result["reasoning_effort"] = effort
    if extra:
        result["extra_body"] = extra
    if json_mode:
        result["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return result
