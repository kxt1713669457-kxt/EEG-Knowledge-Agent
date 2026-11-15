"""应用内EEG模型skill：计算由固定Python适配器执行，不让LLM猜算波形。"""
from app.agent.skills.eeg import service
from app.agent.tools.schemas import AnalyzeEEGSampleInput, GetEEGAnalysisInput, ToolPayload, ToolError


def _payload(result, inference_performed):
    return ToolPayload(status="ok",data={"documents":[service.artifact_document(result)],"artifacts":[result],
                                        "inference_performed":inference_performed})


def analyze_eeg_sample(args: AnalyzeEEGSampleInput):
    try:
        return _payload(service.analyze_sample(args.sample_id,args.trial_index,args.model_id),True)
    except service.EEGSkillError as exc:
        return ToolPayload(status="failed",error=ToolError(code=exc.code,message=str(exc)))


def get_eeg_analysis(args: GetEEGAnalysisInput):
    try:
        return _payload(service.read_artifact(args.artifact_id),False)
    except service.EEGSkillError as exc:
        return ToolPayload(status="failed",error=ToolError(code=exc.code,message=str(exc)))
