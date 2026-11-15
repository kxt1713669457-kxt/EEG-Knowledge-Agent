"""MI tools use registered recordings and artifacts; never LLM-supplied paths."""
import hashlib
import json

from app.agent.skills.mi_preprocessing import service
from app.agent.skills.eeg import service as eeg_service
from app.agent.tools.schemas import ToolPayload, ToolError


def preprocess_eeg_recording(args):
    try:
        result = service.preprocess_recording(args.recording_id, args.profile_id, args.max_trials)
        return ToolPayload(status="ok", data={"documents": [service.artifact_document(result)],
            "artifacts": [result], "inference_performed": False, "preprocessing_performed": True})
    except ValueError as exc:
        return ToolPayload(status="failed", error=ToolError(code="mi_preprocessing_failed", message=str(exc)))


def get_eeg_preprocessing(args):
    try:
        result = service.read_artifact(args.artifact_id)
        return ToolPayload(status="ok", data={"documents": [service.artifact_document(result)],
            "artifacts": [result], "inference_performed": False, "preprocessing_performed": False})
    except ValueError as exc:
        return ToolPayload(status="failed", error=ToolError(code="mi_artifact_unavailable", message=str(exc)))


def analyze_preprocessed_eeg(args):
    try:
        source = service.read_artifact(args.artifact_id)
        service.data_path(args.artifact_id)  # Verify the derived data before inference.
        result = eeg_service.analyze_sample(source["sample_id"], args.trial_index, args.model_id)
        return ToolPayload(status="ok", data={"documents": [eeg_service.artifact_document(result)],
            "artifacts": [result], "inference_performed": True})
    except ValueError as exc:
        return ToolPayload(status="failed", error=ToolError(code="mi_inference_failed", message=str(exc)))


def _registration_document(row, kind):
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return {"type": "eeg_attachment", "doc_id": digest, "chunk_uid": digest,
            "title": "脑电附件" + kind, "text": payload, "content_scope": "verified_attachment_metadata"}


def inspect_eeg_attachment(args):
    try:
        from app.workspace import eeg_attachments
        row = eeg_attachments.inspect_attachment(args.attachment_id)
        return ToolPayload(status="ok", data={"attachment": row, "documents": [_registration_document(row, "检查")]})
    except ValueError as exc:
        return ToolPayload(status="failed", error=ToolError(code="eeg_attachment_unavailable", message=str(exc)))


def register_eeg_recording(args):
    try:
        from app.workspace import eeg_attachments
        if args.dataset_id == "SHU_MI":
            row = eeg_attachments.register_shu_sessions(args.attachment_ids, args.split_id)
        else:
            row = eeg_attachments.register_recording(args.attachment_ids[0], args.dataset_id, args.metadata)
        return ToolPayload(status="ok", data={"registration": {"kind": "recording", **row}, "documents": [_registration_document(row, "登记")]})
    except ValueError as exc:
        return ToolPayload(status="failed", error=ToolError(code="eeg_registration_failed", message=str(exc)))


def register_eeg_sample(args):
    try:
        from app.workspace import eeg_attachments
        row = eeg_attachments.register_sample(args.attachment_id, args.sampling_rate_hz, args.unit, args.dataset_id)
        return ToolPayload(status="ok", data={"registration": {"kind": "sample", **row}, "documents": [_registration_document(row, "登记")]})
    except ValueError as exc:
        return ToolPayload(status="failed", error=ToolError(code="eeg_registration_failed", message=str(exc)))
