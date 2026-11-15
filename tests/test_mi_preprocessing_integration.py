"""Authenticated HTTP and real Agent-tool integration without external services.

Preprocessing is real; only the classifier's weight-backed network is replaced
with a deterministic small callable. Every database and artifact is temporary.
"""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from app.agent import execution, planning
from app.agent.skills import manager
from app.agent.skills.eeg import classifier
from app.agent.skills.eeg import service as eeg_service
from app.agent.skills.mi_preprocessing import catalog as profile_catalog
from app.agent.skills.mi_preprocessing import service
from app.agent.task_work import TaskWork
from app.agent.tools import registry
from app.api.http.auth_api import install_auth
from app.api.http.harness_api import router
from app.workspace import samples, store
from tests.auth_client import AuthenticatedClient


LEGACY = "bci-cue4s-legacy-v1"
MODEL = "dwdanet-2a-local"
BASE = "/workspace/mi-preprocessing"


class MIPreprocessingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for owner, name, value in (
            (service, "ROOT", self.root / "mi"),
            (samples, "SAMPLE_ROOT", self.root / "samples"),
            (store, "DB_PATH", self.root / "workspace.sqlite3"),
            (eeg_service, "ARTIFACT_ROOT", self.root / "classifications"),
            (classifier, "_models", {}),
        ):
            item = patch.object(owner, name, value)
            item.start()
            self.addCleanup(item.stop)
        memory = patch.object(service, "_memory_guard")
        memory.start()
        self.addCleanup(memory.stop)
        app = FastAPI()
        install_auth(app)
        app.include_router(router)
        self.app = app
        self.client = AuthenticatedClient(app, client=("127.0.0.1", 52000))
        self.addCleanup(self.client.close)

    def upload(self, events=None, **data):
        t = np.arange(4500) / 250
        self.signal = np.array([(1 + i / 22) * 10e-6 * np.sin(2 * np.pi * 10 * t)
                                for i in range(22)])
        continuous = np.vstack((self.signal, 100e-6 * np.sin(2 * np.pi * 2 * t)))
        blob = BytesIO()
        np.savez_compressed(blob, signal=continuous,
                            ch_names=[*(f"EEG-{i}" for i in range(22)), "EOG-left"],
                            events=np.array([[1000, 0, 769], [2200, 0, 772]], dtype=np.int64)
                            if events is None else events,
                            sfreq=250, unit="V")
        response = self.client.post(BASE + "/recordings", data={"dataset_id": "BCI_IV_2a", **data},
                                    files={"file": ("continuous.npz", blob.getvalue(), "application/octet-stream")})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def arguments(self, row, max_trials=2):
        return {"recording_id": row["recording_id"], "profile_id": LEGACY, "max_trials": max_trials}

    def plan(self, row, trial=1, max_trials=2):
        return {"goal": "对登记的运动想象原始记录预处理并识别一个试次", "steps": [
            {"step_id": "s1", "tool": "preprocess_eeg_recording", "arguments": self.arguments(row, max_trials)},
            {"step_id": "s2", "tool": "analyze_preprocessed_eeg", "arguments": {
                "preprocessing_step_id": "s1", "trial_index": trial, "model_id": MODEL}},
        ]}

    def network(self):
        import torch

        class SmallNetwork:
            def __init__(self):
                self.inputs = []

            def __call__(self, value):
                self.inputs.append(value.detach().clone())
                return torch.ones((1, 22, 4)), torch.tensor([[0., 3., 1., -1.]])

        return SmallNetwork()

    def test_authenticated_upload_run_artifact_download_catalog_and_sample(self):
        row = self.upload()
        response = self.client.post(BASE + "/run", json=self.arguments(row))
        self.assertEqual(response.status_code, 200, response.text)
        artifact = response.json()
        result = self.client.get(BASE + "/artifacts/" + artifact["artifact_id"])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), artifact)
        download = self.client.get(BASE + "/artifacts/" + artifact["artifact_id"] + "/data")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.headers["x-content-type-options"], "nosniff")
        self.assertIn("attachment", download.headers["content-disposition"])
        with np.load(BytesIO(download.content), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["data"][0], self.signal[:, 1000:2000])
        catalog = self.client.get(BASE + "/catalog").json()
        self.assertEqual(catalog["recordings"][0]["recording_id"], row["recording_id"])
        self.assertEqual(catalog["artifacts"][0]["artifact_id"], artifact["artifact_id"])
        self.assertNotIn("storage_name", catalog["recordings"][0])
        self.assertNotIn("metadata", catalog["recordings"][0])
        sample = samples.active_entries()[artifact["sample_id"]]
        self.assertTrue(sample["compatible"])
        self.assertEqual(sample["source_recording_sha256"], row["sha256"])
        self.assertEqual(samples.read_array(samples.registered_path(sample)).shape, (2, 22, 1000))

    def test_business_routes_keep_auth_and_local_origin_required(self):
        unauthenticated = TestClient(self.app, client=("127.0.0.1", 52001))
        self.addCleanup(unauthenticated.close)
        # Use the same temporary auth database, but no bearer/cookie.
        from app.workspace import auth
        with patch.object(auth, "DB_PATH", self.client._auth_path):
            response = unauthenticated.get(BASE + "/catalog")
        self.assertEqual(response.status_code, 401)
        response = self.client.get(BASE + "/catalog", headers={"Origin": "https://untrusted.example"})
        self.assertEqual(response.status_code, 403)
        response = self.client.post(BASE + "/run", json={}, headers={"Origin": "https://untrusted.example"})
        self.assertEqual(response.status_code, 403)

    def test_http_run_strict_types_limits_extra_fields_and_unregistered_record(self):
        row = self.upload()
        arguments = self.arguments(row)
        for changes in ({"max_trials": True}, {"max_trials": 0}, {"max_trials": 289},
                        {"max_trials": "2"}, {"arbitrary_path": "C:/private.gdf"}):
            with self.subTest(changes=changes):
                response = self.client.post(BASE + "/run", json={**arguments, **changes})
                self.assertEqual(response.status_code, 422, response.text)
        response = self.client.post(BASE + "/run", json={**arguments, "recording_id": "0" * 32})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(samples.active_entries(), {})
        self.assertEqual(self.client.get(BASE + "/artifacts/" + "0" * 32).status_code, 400)

    def test_catalog_displays_missing_optional_dependencies_without_hiding_profile(self):
        original = profile_catalog.find_spec
        with patch.object(profile_catalog, "find_spec", side_effect=lambda name: None if name == "mne_icalabel" else original(name)):
            response = self.client.get(BASE + "/catalog")
            self.assertEqual(response.status_code, 200)
            rows = {p["id"]: p for p in response.json()["profiles"]}
            self.assertFalse(rows["mi-iclabel-v1"]["ready"])
            self.assertIn("mne_icalabel", rows["mi-iclabel-v1"]["reason"])
            self.assertTrue(rows[LEGACY]["ready"])

    def test_upload_rejects_bad_dataset_and_metadata_without_creating_source(self):
        response = self.client.post(BASE + "/recordings", data={"dataset_id": "arbitrary"},
                                    files={"file": ("data.npz", b"not-npz")})
        self.assertEqual(response.status_code, 422)
        response = self.client.post(BASE + "/recordings", data={"dataset_id": "BCI_IV_2a", "metadata_json": '{"labels":[1]}'},
                                    files={"file": ("data.npz", b"not-npz")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(store.all_records("mi_recording"), [])

    def test_registry_real_preprocess_read_and_missing_source_fail_explicitly(self):
        row = self.upload()
        result = registry.execute_tool("preprocess_eeg_recording", self.arguments(row))
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.data["preprocessing_performed"])
        self.assertFalse(result.data["inference_performed"])
        artifact = result.data["artifacts"][0]
        self.assertEqual(artifact["kind"], "mi_preprocessing")
        read = registry.execute_tool("get_eeg_preprocessing", {"artifact_id": artifact["artifact_id"]})
        self.assertEqual(read.status, "ok")
        self.assertFalse(read.data["preprocessing_performed"])
        self.assertEqual(read.data["artifacts"][0], artifact)
        missing = registry.execute_tool("preprocess_eeg_recording", {**self.arguments(row), "recording_id": "0" * 32})
        self.assertEqual(missing.status, "failed")
        self.assertEqual(missing.error.code, "mi_preprocessing_failed")

    def test_registry_rejects_paths_bool_and_unknown_arguments_before_processing(self):
        row = self.upload()
        for changes in ({"recording_id": "../source"}, {"max_trials": True}, {"max_trials": 289},
                        {"source_path": "private.gdf"}):
            with self.subTest(changes=changes), patch.object(service, "preprocess_recording") as process:
                result = registry.execute_tool("preprocess_eeg_recording", {**self.arguments(row), **changes})
                self.assertEqual(result.error.code, "invalid_arguments")
                process.assert_not_called()

    def test_agent_binds_real_preprocessing_output_to_real_classifier_adapter(self):
        row = self.upload()
        small_network = self.network()
        with patch.object(classifier, "load_network", return_value=small_network) as load:
            result = execution.execute_plan(self.plan(row))
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual([s["status"] for s in result["trace"]], ["ok", "ok"])
        prep = result["tool_results"]["s1"]["data"]["artifacts"][0]
        classified = result["tool_results"]["s2"]["data"]["artifacts"][0]
        self.assertEqual(result["trace"][1]["arguments"]["artifact_id"], prep["artifact_id"])
        self.assertEqual(classified["sample"]["sample_id"], prep["sample_id"])
        self.assertEqual(classified["sample"]["preprocessing_artifact_id"], prep["artifact_id"])
        self.assertEqual(classified["sample"]["source_recording_sha256"], row["sha256"])
        self.assertEqual(classified["preprocessing"]["upstream_profile_id"], LEGACY)
        self.assertEqual(classified["output"]["prediction"], "右手")
        self.assertFalse(classified["output"]["score_calibrated"])
        self.assertEqual(len(result["documents"]), 2)
        self.assertEqual(load.call_count, 1)
        norm = manager.get_skill(MODEL)["normalization"]
        expected = (self.signal[:, 2200:3200].astype(np.float32) - norm["mean"]) / norm["std"]
        np.testing.assert_array_equal(small_network.inputs[0].numpy()[0, 0], expected)

    def test_agent_preprocessing_failure_skips_classification(self):
        row = self.upload()
        missing_row = {**row, "recording_id": "0" * 32}
        with patch.object(classifier, "load_network") as load:
            result = execution.execute_plan(self.plan(missing_row))
        self.assertEqual(result["execution_status"], "partial")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["trace"][0]["status"], "failed")
        self.assertEqual(result["trace"][1]["reason"], "source_unavailable")
        load.assert_not_called()

    def test_agent_actual_retained_trial_count_blocks_invalid_followup(self):
        row = self.upload(events=np.array([[1000, 0, 769], [4000, 0, 770]], dtype=np.int64))
        with patch.object(classifier, "load_network") as load:
            result = execution.execute_plan(self.plan(row, trial=1, max_trials=2))
        self.assertEqual(result["tool_results"]["s1"]["status"], "ok")
        self.assertEqual(result["trace"][1]["reason"], "source_unavailable")
        self.assertEqual(result["tool_calls"], 1)
        load.assert_not_called()

    def test_agent_future_self_wrong_kind_and_excess_trial_reject_whole_plan(self):
        row = self.upload()
        future = self.plan(row)
        future["steps"].reverse()
        self_ref = self.plan(row)
        self_ref["steps"][1]["arguments"]["preprocessing_step_id"] = "s2"
        wrong_kind = self.plan(row)
        wrong_kind["steps"][0] = {"step_id": "s1", "tool": "search_web", "arguments": {"query": "test"}}
        excessive = self.plan(row, trial=2, max_trials=2)
        both = self.plan(row)
        both["steps"][1]["arguments"]["artifact_id"] = "0" * 32
        for payload in (future, self_ref, wrong_kind, excessive, both):
            with self.subTest(payload=payload), patch.object(registry, "execute_tool") as tool:
                with self.assertRaises(planning.PlanRejected):
                    execution.execute_plan(payload)
                tool.assert_not_called()

    def test_agent_read_existing_preprocessing_can_bind_classifier(self):
        row = self.upload()
        prep = service.preprocess_recording(**self.arguments(row))
        payload = self.plan(row, trial=0)
        payload["steps"][0] = {"step_id": "s1", "tool": "get_eeg_preprocessing", "arguments": {"artifact_id": prep["artifact_id"]}}
        with patch.object(service, "preprocess_recording") as preprocess, patch.object(classifier, "load_network", return_value=self.network()):
            result = execution.execute_plan(payload)
        self.assertEqual(result["execution_status"], "completed")
        preprocess.assert_not_called()
        self.assertFalse(result["tool_results"]["s1"]["data"]["preprocessing_performed"])

    def test_direct_preprocessed_inference_keeps_actual_bounds_and_tamper_check(self):
        row = self.upload()
        prep = service.preprocess_recording(**self.arguments(row))
        arguments = {"artifact_id": prep["artifact_id"], "model_id": MODEL, "trial_index": 2}
        with patch.object(classifier, "load_network") as load:
            result = registry.execute_tool("analyze_preprocessed_eeg", arguments)
        self.assertEqual(result.error.code, "mi_inference_failed")
        load.assert_not_called()
        with service.data_path(prep["artifact_id"]).open("ab") as stream:
            stream.write(b"changed")
        with patch.object(classifier, "load_network") as load:
            result = registry.execute_tool("analyze_preprocessed_eeg", {**arguments, "trial_index": 0})
        self.assertEqual(result.error.code, "mi_inference_failed")
        self.assertIn("哈希", result.error.message)
        load.assert_not_called()

    def test_taskwork_cache_reuses_preprocessing_only_within_same_task(self):
        row = self.upload()
        arguments = self.arguments(row)
        work = TaskWork()
        with patch.object(service, "preprocess_recording", wraps=service.preprocess_recording) as process:
            first = work.execute("preprocess_eeg_recording", arguments)
            repeated = work.execute("preprocess_eeg_recording", deepcopy(arguments))
        self.assertEqual(process.call_count, 1)
        self.assertTrue(repeated["data"]["cache_hit"])
        self.assertFalse(repeated["data"]["preprocessing_performed"])
        self.assertNotEqual(first["tool_call_id"], repeated["tool_call_id"])
        self.assertEqual(first["data"]["artifacts"], repeated["data"]["artifacts"])
        self.assertEqual(len(service.catalog()["artifacts"]), 1)
        fresh = TaskWork().execute("preprocess_eeg_recording", arguments)
        self.assertNotEqual(fresh["data"]["artifacts"][0]["artifact_id"], first["data"]["artifacts"][0]["artifact_id"])

    def test_modified_registered_sample_copy_cannot_reuse_original_provenance(self):
        row = self.upload()
        prep = service.preprocess_recording(**self.arguments(row))
        entry = samples.active_entries()[prep["sample_id"]]
        # Canonical preprocessing artifact remains unchanged; alter only the
        # separate sample copy that the downstream classifier actually reads.
        np.savez_compressed(samples.registered_path(entry), data=np.zeros((2, 22, 1000)))
        args = {"artifact_id": prep["artifact_id"], "model_id": MODEL, "trial_index": 0}
        with patch.object(classifier, "load_network", return_value=self.network()) as load:
            result = registry.execute_tool("analyze_preprocessed_eeg", args)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.code, "mi_inference_failed")
        load.assert_not_called()

    def test_taskwork_prepared_inference_cache_avoids_second_network_execution(self):
        row = self.upload()
        prep = service.preprocess_recording(**self.arguments(row))
        args = {"artifact_id": prep["artifact_id"], "model_id": MODEL, "trial_index": 0}
        work, small_network = TaskWork(), self.network()
        with patch.object(classifier, "load_network", return_value=small_network):
            first = work.execute("analyze_preprocessed_eeg", args)
            second = work.execute("analyze_preprocessed_eeg", args)
        self.assertEqual(first["status"], "ok")
        self.assertTrue(second["data"]["cache_hit"])
        self.assertFalse(second["data"]["inference_performed"])
        self.assertEqual(len(small_network.inputs), 1)
        self.assertEqual(len(work.summary()["completed_analysis"]), 1)

    def test_failed_taskwork_preprocessing_is_not_cached(self):
        row = {"recording_id": "0" * 32}
        work = TaskWork()
        with patch.object(service, "preprocess_recording", wraps=service.preprocess_recording) as process:
            first = work.execute("preprocess_eeg_recording", self.arguments(row))
            second = work.execute("preprocess_eeg_recording", self.arguments(row))
        self.assertEqual(first["status"], "failed")
        self.assertEqual(second["status"], "failed")
        self.assertEqual(process.call_count, 2)
        self.assertEqual(work.summary()["completed_analysis"], [])


if __name__ == "__main__":
    unittest.main()
