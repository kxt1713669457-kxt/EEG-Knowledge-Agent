from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class KnowledgeSearchInput(StrictInput):
    query: str = Field(min_length=1, max_length=2000, description="已补全指代的独立知识问题")
    top_k: int = Field(default=10, ge=1, le=10, description="最终本地结果上限，不改变内部召回K")


class WebSearchInput(StrictInput):
    query: str = Field(min_length=1, max_length=2000, description="需要查询公开网页的问题")
    top_k: int = Field(default=5, ge=1, le=5, description="最多返回的网页摘要数")


class PublicDocumentInput(StrictInput):
    url: str = Field(min_length=8, max_length=2000, description="真实搜索结果或用户提供的公开学术原件HTTPS链接；支持arXiv、OpenReview等站点")
    query: str = Field(min_length=1, max_length=2000, description="需要读取的内容，精确结果需保留数据集和实验条件")


class SourceReference(StrictInput):
    doc_id: str = Field(pattern=r"^[0-9a-f]{64}$", description="检索结果返回的原件身份，不按文件名猜测")
    chunk_uid: str = Field(pattern=r"^[0-9a-f]{64}$", description="检索结果返回的具体切片身份")


class DocumentContextInput(SourceReference):
    neighbors: int = Field(default=1, ge=0, le=2, description="前后各取几个切片；只在同一索引发布批次内扩展")


class CompareEvidenceInput(StrictInput):
    question: str = Field(min_length=1, max_length=2000, description="比较维度与对象明确的问题")
    sources: Annotated[list[SourceReference], Field(min_length=2, max_length=4)]

    @model_validator(mode="after")
    def require_unique_sources(self):
        if len({(source.doc_id, source.chunk_uid) for source in self.sources}) != len(self.sources):
            raise ValueError("比较来源不能重复")
        return self


class ToolInvocation(StrictInput):
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, JsonValue]


class AnalyzeEEGSampleInput(StrictInput):
    sample_id: str = Field(min_length=1,max_length=64,pattern=r"^[a-z0-9-]+$",description="服务端登记的样本文件ID，不接受路径")
    trial_index: int = Field(ge=0,le=10000,description="从0计数，实际边界由数据文件检查")
    model_id: str = Field(default="cbramod-pretrained", pattern=r"^[a-z][a-z0-9-]{2,63}$", description="当前启用的skill ID；分类模型与表征模型能力不同")


class GetEEGAnalysisInput(StrictInput):
    artifact_id: str = Field(pattern=r"^[0-9a-f]{32}$",description="真实执行返回的产物ID；读取旧结果不重新推理")


class PreprocessEEGInput(StrictInput):
    recording_id: str = Field(pattern=r"^[0-9a-f]{32}$", description="原始记录目录中的真实ID，不接受路径")
    profile_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$", description="运动想象预处理目录中就绪的方案ID")
    max_trials: int = Field(default=32, ge=1, le=288)


class InspectEEGAttachmentInput(StrictInput):
    attachment_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class RegisterEEGRecordingInput(StrictInput):
    attachment_ids: Annotated[list[Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]], Field(min_length=1, max_length=5)]
    dataset_id: Literal["BCI_IV_2a", "BCI_IV_2b", "MI_GENERIC", "SHU_MI", "OPENBMI_MI"]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    split_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_sources(self):
        if len(set(self.attachment_ids)) != len(self.attachment_ids):
            raise ValueError("附件不能重复")
        if self.dataset_id == "SHU_MI":
            if len(self.attachment_ids) != 5 or not self.split_id:
                raise ValueError("SHU需要五个session附件及split_id")
        elif len(self.attachment_ids) != 1 or self.split_id is not None:
            raise ValueError("连续记录需要一个附件且不能指定split_id")
        return self


class RegisterEEGSampleInput(StrictInput):
    attachment_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sampling_rate_hz: float = Field(gt=0, le=100000)
    unit: Literal["V", "uV"]
    dataset_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class AnalyzePreparedEEGInput(GetEEGAnalysisInput):
    trial_index: int = Field(ge=0, le=287)
    model_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")


class WorkspaceFilesInput(StrictInput):
    file_ids: Annotated[list[Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]], Field(min_length=1, max_length=8)]
    query: str = Field(default="", max_length=2000)


class InspectImageInput(StrictInput):
    image_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    question: str = Field(min_length=1, max_length=2000)


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ToolPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["ok", "empty", "failed"]
    data: dict[str, JsonValue] = Field(default_factory=dict)
    error: ToolError | None = None

    @model_validator(mode="after")
    def check_error_contract(self):
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("失败必须有错误信息，成功或零命中不能携带错误")
        return self


class ToolResult(ToolPayload):
    tool_name: str
    tool_call_id: str
    elapsed_ms: float
