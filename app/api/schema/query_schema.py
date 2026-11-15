#query模块所有json相关的模型
from typing import Any

from pydantic import BaseModel, Field, field_validator
from app.workspace.connections import ExecutionOptions


#健康检查的响应json
class HealthResponseSchema(BaseModel):
    code:int=200
    message:str=None


# 查询接口的请求参数json
class QueryRequestSchema(BaseModel):
    query: str = Field(min_length=1, description="用户问题，去除首尾空白后不能为空")
    session_id: str | None = None
    is_stream:bool = False
    execution: ExecutionOptions | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空或只有空白字符")
        return value

    @field_validator("session_id")
    @classmethod
    def normalize_session_id(cls, value: str | None) -> str | None:
        # 空会话 ID 与未提供一致，由 HTTP 入口生成新 UUID。
        if value is None:
            return None
        return value.strip() or None


# 查询模块流式响应结果
class QueryStreamResponseSchema(BaseModel):
    message:str
    session_id:str
    task_id:str
    run_id:str | None = None


class TaskMessageSchema(BaseModel):
    model_config = {"extra": "forbid", "str_strip_whitespace": True, "strict": True}
    session_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=6000)
    message_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")

class QueryNotStreamResponseSchema(BaseModel):
    message:str
    session_id:str
    answer:str
    done_list:list[str]
    image_urls:list[str]

class HistoryClearResponseSchema(BaseModel):
    message:str
    deleted_count:int

class HistoryItemResponseSchema(BaseModel):
    id: str
    session_id: str
    task_id: str = ""
    run_id: str | None = None
    input_revision: int = 1
    message_kind: str = ""
    response_elapsed_ms: int | None = None
    role: str
    text: str
    rewritten_query: str = ""
    item_names: list[str] = Field(description="已完成任务列表", default_factory=list)
    image_urls: list[str] = Field(description="已完成任务列表", default_factory=list)
    eeg_attachments: list[dict] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    citation_status: dict[str, Any] = Field(default_factory=dict)
    ts: Any


class HistoryListResponseSchema(BaseModel):
    session_id: str
    items: list[HistoryItemResponseSchema] = Field(description="查询的数据记录列表", default_factory=list)
