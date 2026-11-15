from io import BytesIO
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.workspace import eeg_attachments as attachments, samples, store
from app.agent.skills.mi_preprocessing import service


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments, "ROOT", tmp_path / "attachments")
    monkeypatch.setattr(samples, "SAMPLE_ROOT", tmp_path / "samples")
    monkeypatch.setattr(service, "ROOT", tmp_path / "preprocessing")
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "workspace.sqlite3")
    return tmp_path


def array_file(**arrays):
    stream = BytesIO()
    np.savez(stream, **arrays)
    stream.seek(0)
    return stream


def sample_upload():
    return attachments.upload([(array_file(data=np.zeros((2, 22, 1000), dtype=np.float32)), "sample.npz")])["attachments"][0]


def test_upload_is_inert_and_header_only(isolated):
    row = sample_upload()
    assert row["representation"] == "epoched"
    assert row["header_summary"]["shape"] == [2, 22, 1000]
    assert row["header_summary"]["unit"] is None
    assert store.all_records("sample") == []
    assert store.all_records("mi_recording") == []
    assert "storage_name" not in row


def test_sample_registration_is_scoped_idempotent_and_not_inference(isolated):
    row = sample_upload()
    with pytest.raises(ValueError, match="本轮"):
        attachments.register_sample(row["attachment_id"], 250, "V", "BCI_IV_2a")
    with attachments.attachment_scope([row["attachment_id"]]):
        first = attachments.register_sample(row["attachment_id"], 250, "V", "BCI_IV_2a")
    with attachments.attachment_scope([row["attachment_id"]]):
        second = attachments.register_sample(row["attachment_id"], 250, "V", "BCI_IV_2a")
        known = attachments.selected_attachments()[0]["registrations"]
        assert known[0]["sample_id"] == first["sample_id"]
        assert known[0]["unit"] == "V"
    assert first["sample_id"] == second["sample_id"]
    assert first["shape"] == [2, 22, 1000]
    assert len(store.all_records("sample")) == 1
    assert store.all_records("mi_artifact") == []
    assert attachments.selected_attachments() == []


def test_raw_npz_register_preprocess_actual_binding(isolated):
    source = np.arange(22 * 2500, dtype=np.float64).reshape(22, 2500) * 1e-10
    stream = array_file(signal=source, sfreq=np.array(250.), ch_names=np.array([f"EEG-{i}" for i in range(22)]),
                        events=np.array([[0, 0, 769], [1200, 0, 770]]), unit=np.array("V"))
    row = attachments.upload([(stream, "continuous.npz")])["attachments"][0]
    assert row["representation"] == "continuous"
    with attachments.attachment_scope([row["attachment_id"]]):
        with pytest.raises(ValueError, match="连续"):
            attachments.register_sample(row["attachment_id"], 250, "V", "BCI_IV_2a")
        recording = attachments.register_recording(row["attachment_id"], "BCI_IV_2a")
        result = service.preprocess_recording(recording["recording_id"], "bci-cue4s-legacy-v1", 2)
    with np.load(service.data_path(result["artifact_id"]), allow_pickle=False) as archive:
        assert np.array_equal(archive["data"][0], source[:, :1000])
        assert np.array_equal(archive["data"][1], source[:, 1200:2200])


def test_no_object_or_ambiguous_npz_and_partial_upload_cleanup(isolated):
    for stream in (array_file(data=np.array([object()], dtype=object)), array_file(data=np.ones((2, 3)), signal=np.ones((2, 3)))):
        with pytest.raises(ValueError):
            attachments.upload([(array_file(data=np.ones((2, 22, 1000))), "ok.npz"), (stream, "bad.npz")])
        assert list(attachments.ROOT.iterdir()) == []
        assert store.all_records("eeg_attachment") == []


def test_total_limit_rolls_back(isolated, monkeypatch):
    monkeypatch.setattr(attachments, "MAX_BYTES", 1024)
    with pytest.raises(ValueError, match="128MiB"):
        attachments.upload([(BytesIO(b"a" * 1025), "data.npy")])
    assert list(attachments.ROOT.iterdir()) == []


def test_tamper_and_path_rejection(isolated):
    row = sample_upload()
    record = store.read("eeg_attachment", row["attachment_id"])
    attachments.attachment_path(record).write_bytes(b"modified")
    with pytest.raises(ValueError, match="哈希"):
        with attachments.attachment_scope([row["attachment_id"]]):
            pass
    record["storage_name"] = "../outside.npz"
    store.write("eeg_attachment", row["attachment_id"], record)
    with pytest.raises(ValueError, match="原件"):
        attachments.attachment_record(row["attachment_id"])


def test_context_does_not_leak_between_threads(isolated):
    first, second = sample_upload(), sample_upload()
    def read(identity):
        with attachments.attachment_scope([identity]):
            return [row["attachment_id"] for row in attachments.selected_attachments()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(read, [first["attachment_id"], second["attachment_id"]])) == [[first["attachment_id"]], [second["attachment_id"]]]


@pytest.mark.parametrize("rate,unit", [(float("nan"), "V"), (250, "mV"), (True, "uV")])
def test_sample_metadata_must_be_explicit_valid(isolated, rate, unit):
    row = sample_upload()
    with attachments.attachment_scope([row["attachment_id"]]), pytest.raises(ValueError, match="明确"):
        attachments.register_sample(row["attachment_id"], rate, unit, "BCI_IV_2a")
    assert store.all_records("sample") == []


def test_authenticated_upload_api_and_separate_skills(isolated):
    from app.api.http.harness_api import router
    from app.api.http.workspace_account import require_local_page
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_local_page] = lambda: None
    with TestClient(app) as client:
        response = client.post("/workspace/eeg-attachments", files=[("files", ("sample.npz", array_file(data=np.ones((1, 22, 1000))).getvalue(), "application/octet-stream"))])
        assert response.status_code == 201
        assert response.json()["attachments"][0]["representation"] == "epoched"
        skills = client.get("/workspace/skills").json()
        assert len(skills["preprocessing_items"]) == 12
        assert all(row["category"] == "eeg_preprocessing" for row in skills["preprocessing_items"])
        assert all(row.get("category") != "eeg_preprocessing" for row in skills["items"])
    assert not store.all_records("sample")


@pytest.mark.parametrize("tool", ["inspect_eeg_attachment", "register_eeg_sample"])
def test_chat_inspection_and_registration_receipts_without_extra_model_calls(isolated, monkeypatch, tool):
    from app.agent import runtime, execution
    row = sample_upload()
    arguments = {"attachment_id": row["attachment_id"]}
    if tool == "register_eeg_sample":
        arguments.update(sampling_rate_hz=250, unit="V", dataset_id="BCI_IV_2a")
    plan = {"goal": "核验附件", "steps": [{"step_id": "s1", "tool": tool, "arguments": arguments}]}
    with attachments.attachment_scope([row["attachment_id"]]):
        result = execution.execute_plan(plan)
    def forbidden(*args, **kwargs):
        pytest.fail("metadata receipts must not call an answer/observation model")
    monkeypatch.setattr(runtime, "observe_execution", forbidden)
    monkeypatch.setattr(runtime, "generate_answer", forbidden)
    state = {"execution": result, "query": "检查这个附件", "plan": plan, "session_id": "attachment-receipt", "run_id": "receipt"}
    state.update(runtime.node_observe(state))
    answer = runtime.node_answer(state)["answer_state"]["answer"]
    if tool == "inspect_eeg_attachment":
        assert "250 Hz" in answer and "V或uV" in answer and "尚未登记" in answer
    else:
        assert store.all_records("sample")[0]["sample_id"] in answer
        assert "尚未执行" in answer
