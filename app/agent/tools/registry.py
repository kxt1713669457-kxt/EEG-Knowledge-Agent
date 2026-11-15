"""工具白名单是执行边界；模型给出调用建议，服务端校验后才执行。"""
from dataclasses import dataclass
import time
from typing import Callable
import uuid

from pydantic import BaseModel, ValidationError

from app.agent.tools import rag_tools, eeg_tools, workspace_tools, image_tools, public_documents, mi_tools
from app.agent.tools.schemas import (
    CompareEvidenceInput, DocumentContextInput, KnowledgeSearchInput, ToolError,
    ToolPayload, ToolResult, WebSearchInput,
    AnalyzeEEGSampleInput, GetEEGAnalysisInput, WorkspaceFilesInput, InspectImageInput, PublicDocumentInput,
    PreprocessEEGInput, AnalyzePreparedEEGInput, RegisterEEGRecordingInput, RegisterEEGSampleInput, InspectEEGAttachmentInput,
)
from app.shared.runtime.logger import logger


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Callable


TOOLS = {
    definition.name: definition for definition in (
        ToolDefinition("inspect_eeg_attachment", "读取本轮脑电附件的实际头信息和待补充元数据，不执行推理。", InspectEEGAttachmentInput, mi_tools.inspect_eeg_attachment),
        ToolDefinition("register_eeg_recording", "登记本轮选择的连续脑电附件或五个SHU session，返回真实recording_id。", RegisterEEGRecordingInput, mi_tools.register_eeg_recording),
        ToolDefinition("register_eeg_sample", "登记本轮选择的已切段数值脑电；采样率及单位须有依据，返回真实sample_id。", RegisterEEGSampleInput, mi_tools.register_eeg_sample),
        ToolDefinition("preprocess_eeg_recording", "对登记的连续MI原始记录执行有版本的预处理方案，保存QC/事件/来源并生成数值样本；不执行分类。", PreprocessEEGInput, mi_tools.preprocess_eeg_recording),
        ToolDefinition("get_eeg_preprocessing", "读取真实MI预处理产物，不重新处理；该产物不是分类结果。", GetEEGAnalysisInput, mi_tools.get_eeg_preprocessing),
        ToolDefinition("analyze_preprocessed_eeg", "对真实预处理产物关联的样本执行已启用EEG模型；模型输入条件仍需核验。", AnalyzePreparedEEGInput, mi_tools.analyze_preprocessed_eeg),
        ToolDefinition("read_public_document", "读取已找到的公开学术原件正文，支持arXiv论文PDF；摘要缺少具体实验结果时优先读取原件，返回带页码的实际摘录。不保证获取所有图表。", PublicDocumentInput, public_documents.read_public_document),
        ToolDefinition("inspect_uploaded_image", "识别用户本轮上传图片的可见内容；只接受本轮真实image_id。结果是视觉观察说明，不是原始数值脑电分类。", InspectImageInput, image_tools.inspect_uploaded_image),
        ToolDefinition("read_workspace_files", "读取用户本轮已选择的工作区文件相关正文；只接受可信目录中的file_id，不接受任意路径，不执行代码。", WorkspaceFilesInput, workspace_tools.read_workspace_files),
        ToolDefinition("search_knowledge", "查询本地EEG知识库；普通和HyDE混合检索、RRF及精排，返回来源身份完备的真实切片，不生成最终答案。",
                       KnowledgeSearchInput, rag_tools.search_knowledge),
        ToolDefinition("search_web", "查询公开网页补充信息，返回URL、获取时间和搜索摘要；摘要不是网页全文。",
                       WebSearchInput, rag_tools.search_web),
        ToolDefinition("get_document_context", "用检索返回的doc_id和chunk_uid读取原文及同批次相邻切片；不要猜文件名或ID。",
                       DocumentContextInput, rag_tools.get_document_context),
        ToolDefinition("compare_evidence", "按明确问题比较2至4个已定位的本地来源切片，必须传真实文档/切片ID；结论经过来源复核，不代表比较了整篇论文。",
                       CompareEvidenceInput, rag_tools.compare_evidence),
        ToolDefinition("analyze_eeg_sample", "对已登记的数值EEG试次运行启用的skill。model_id从可信skills目录选择；CBraMod只输出表征，本地DWDaNet/DWTNet有训练分类头并记录预测及未校准分数。不能把波形图片直接传给数值模型。",
                       AnalyzeEEGSampleInput,eeg_tools.analyze_eeg_sample),
        ToolDefinition("get_eeg_analysis", "读取实际产物ID对应的既有单试次结果，用于追问；不重跑模型，不猜产物ID。",
                       GetEEGAnalysisInput,eeg_tools.get_eeg_analysis),
    )
}


def tool_declarations() -> list[dict]:
    return [{"type": "function", "function": {
        "name": tool.name, "description": tool.description,
        "parameters": tool.args_schema.model_json_schema(),
    }} for tool in TOOLS.values()]


def execute_tool(name: str, arguments, *, definitions=None) -> ToolResult:
    started = time.perf_counter()
    call_id = uuid.uuid4().hex
    tools = TOOLS if definitions is None else definitions
    definition = tools.get(name)
    if definition is None:
        payload = ToolPayload(status="failed", error=ToolError(
            code="unknown_tool", message="工具未登记，不能执行。"))
    else:
        try:
            args = definition.args_schema.model_validate(arguments)
        except ValidationError:
            payload = ToolPayload(status="failed", error=ToolError(
                code="invalid_arguments", message="工具参数不符合输入契约，请检查必填字段、类型、范围和额外字段。"))
        else:
            try:
                # 返回契约也要检查，不能把None或格式错误当成正常零命中。
                raw = definition.handler(args)
                payload = ToolPayload.model_validate(raw.model_dump() if isinstance(raw, ToolPayload) else raw)
                payload.model_dump_json()
            except Exception as exc:
                logger.warning("工具{}执行失败，call_id={}，异常类型={}", name, call_id, type(exc).__name__)
                payload = ToolPayload(status="failed", error=ToolError(
                    code="tool_execution_failed", message="工具执行或结果检查失败，请检查该次调用记录。",
                    retryable=isinstance(exc, (ConnectionError, TimeoutError))))
    return ToolResult(**payload.model_dump(), tool_name=name, tool_call_id=call_id,
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
