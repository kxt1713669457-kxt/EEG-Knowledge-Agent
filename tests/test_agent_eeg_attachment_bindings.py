from app.agent import planning, execution
from app.agent.task_work import TaskWork


def test_registration_bindings_use_actual_returned_ids(monkeypatch):
    calls = []
    def run(name, args):
        calls.append((name, args))
        data = {"register_eeg_recording": {"registration": {"kind": "recording", "recording_id": "b" * 32}},
                "preprocess_eeg_recording": {"artifacts": [{"kind": "mi_preprocessing", "artifact_id": "c" * 32, "output": {"shape": [2, 22, 1000]}}]},
                "analyze_preprocessed_eeg": {"artifacts": []}}[name]
        return {"status": "ok", "data": data, "error": None, "tool_call_id": "x", "elapsed_ms": 0}
    class Work:
        calls = 0
        supplements = 0
        def execute(self, name, args):
            self.calls += 1
            return run(name, args)
    plan = {"goal": "处理上传记录", "steps": [
        {"step_id": "s1", "tool": "register_eeg_recording", "arguments": {"attachment_ids": ["a" * 32], "dataset_id": "MI_GENERIC"}},
        {"step_id": "s2", "tool": "preprocess_eeg_recording", "arguments": {"registration_step_id": "s1", "profile_id": "mi-native-cue4s-v1"}},
        {"step_id": "s3", "tool": "analyze_preprocessed_eeg", "arguments": {"preprocessing_step_id": "s2", "trial_index": 0, "model_id": "dwdanet"}},
    ]}
    result = execution.execute_plan(plan, task_work=Work())
    assert result["execution_status"] == "completed"
    assert calls[1][1]["recording_id"] == "b" * 32
    assert calls[2][1]["artifact_id"] == "c" * 32


def test_invalid_future_registration_rejected():
    import pytest
    with pytest.raises(planning.PlanRejected):
        planning.validate_plan({"goal": "x", "steps": [{"step_id": "s1", "tool": "analyze_eeg_sample", "arguments": {"registration_step_id": "s2", "trial_index": 0, "model_id": "dwdanet"}}]})


def test_selected_sample_metadata_must_be_explicit(monkeypatch):
    import pytest
    from app.workspace import eeg_attachments
    from app.agent.skills import manager
    monkeypatch.setattr(eeg_attachments, "selected_attachments", lambda: [{"attachment_id": "a" * 32, "header_summary": {"shape": [250, 22, 1000]}}])
    monkeypatch.setattr(manager, "get_skill", lambda _: {})
    plan = planning.validate_plan({"goal": "分析", "steps": [{"step_id": "s1", "tool": "register_eeg_sample", "arguments": {"attachment_id": "a" * 32, "sampling_rate_hz": 250, "unit": "uV", "dataset_id": "MI_GENERIC"}}]})
    with pytest.raises(planning.PlanRejected, match="采样率"):
        planning.validate_plan_subject(plan, "分析附件")
    assert planning.validate_plan_subject(plan, "分析附件，采样率250 Hz，单位uV")


def test_microvolt_evidence_cannot_register_volt_sample(monkeypatch):
    import pytest
    from app.workspace import eeg_attachments
    monkeypatch.setattr(eeg_attachments, "selected_attachments", lambda: [{"attachment_id": "a" * 32, "header_summary": {}}])
    plan = planning.validate_plan({"goal": "登记", "steps": [{"step_id": "s1", "tool": "register_eeg_sample", "arguments": {"attachment_id": "a" * 32, "sampling_rate_hz": 250, "unit": "V", "dataset_id": "MI_GENERIC"}}]})
    for unit in ("µV", "μV", "微伏特"):
        with pytest.raises(planning.PlanRejected, match="采样率"):
            planning.validate_plan_subject(plan, "登记，250 Hz，单位" + unit)


def test_followup_can_use_only_this_attachments_registered_sample(monkeypatch):
    import pytest
    from app.workspace import eeg_attachments
    from app.agent.skills import manager
    monkeypatch.setattr(eeg_attachments, "selected_attachments", lambda: [{"attachment_id": "a" * 32, "header_summary": {}, "registrations": [{"sample_id": "sample-current"}]}])
    monkeypatch.setattr(manager, "get_skill", lambda _: {})
    plan = planning.validate_plan({"goal": "继续识别", "steps": [{"step_id": "s1", "tool": "analyze_eeg_sample", "arguments": {"sample_id": "sample-current", "trial_index": 0, "model_id": "dwdanet"}}]})
    assert planning.validate_plan_subject(plan, "识别刚才登记的样本")
    plan.steps[0].arguments.sample_id = "sample-unrelated"
    with pytest.raises(planning.PlanRejected, match="不能替换"):
        planning.validate_plan_subject(plan, "识别刚才登记的样本")


def test_no_repreprocessing_still_allows_new_model_trial_inference(monkeypatch):
    import json
    from types import SimpleNamespace
    from app.workspace import eeg_attachments
    from app.agent.skills import manager
    from app.agent.skills.eeg import service as eeg_service
    from app.agent.skills.mi_preprocessing import service as mi_service
    monkeypatch.setattr(eeg_attachments, "selected_attachments", lambda: [{"attachment_id": "a" * 32, "header_summary": {}, "registrations": [{"sample_id": "sample-current"}]}])
    monkeypatch.setattr(manager, "get_skill", lambda _: {})
    monkeypatch.setattr(manager, "workflow_instructions", lambda: "")
    monkeypatch.setattr(eeg_service, "model_catalog", lambda: {"models": [], "samples": []})
    monkeypatch.setattr(mi_service, "catalog", lambda: {"recordings": [], "profiles": [], "artifacts": []})
    payload = {"goal": "换模型识别", "steps": [{"step_id": "s1", "tool": "analyze_eeg_sample", "arguments": {"sample_id": "sample-current", "trial_index": 1, "model_id": "dwtnet-2a-local"}}]}
    class Client:
        def bind(self, **kwargs):
            return self
        def invoke(self, messages):
            return SimpleNamespace(content=json.dumps(payload))
    monkeypatch.setattr(planning.llm_providers, "chat", lambda **kwargs: Client())
    history = [{"role": "assistant", "evidence_sources": [{"type": "analysis_result", "artifact_id": "b" * 32}]}]
    audit = {}
    result = planning.generate_plan("用DWTNet识别下标1，不要重新预处理", history, audit=audit)
    assert result.steps[0].tool == "analyze_eeg_sample"
    assert audit["model_calls"] == 1
    assert audit["rejected_attempts"] == []
