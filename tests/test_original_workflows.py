"""Original DWDaNet/DWTNet contracts and authenticated SHU split integration."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
import numpy as np
from scipy.io import loadmat, savemat

from app.agent.skills.mi_preprocessing import service, catalog
from app.api.http.auth_api import install_auth
from app.api.http.harness_api import router
from app.workspace import samples, store
from tests.auth_client import AuthenticatedClient


class OriginalWorkflowsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for owner, name, value in ((service, "ROOT", root / "mi"), (store, "DB_PATH", root / "store.sqlite3"),
                                  (samples, "SAMPLE_ROOT", root / "samples")):
            item = patch.object(owner, name, value)
            item.start()
            self.addCleanup(item.stop)
        guard = patch.object(service, "_memory_guard")
        guard.start()
        self.addCleanup(guard.stop)
        app = FastAPI()
        install_auth(app)
        app.include_router(router)
        self.client = AuthenticatedClient(app, client=("127.0.0.1", 52000))
        self.addCleanup(self.client.close)

    def shu_files(self):
        files = []
        for session in range(1, 6):
            data = np.repeat((session * 10 + np.arange(4, dtype=np.float32))[:, None, None], 32, axis=1)
            data = np.repeat(data, 1000, axis=2)
            blob = BytesIO()
            savemat(blob, {"data": data, "labels": np.array([[1, 2, 1, 2]])})
            files.append(("files", (f"sub-001_ses-{session:02d}_task_motorimagery_eeg.mat", blob.getvalue(), "application/octet-stream")))
        return files

    def test_two_distinct_contracts_support_three_datasets(self):
        for family in ("dwdanet", "dwtnet"):
            profile = catalog.get_profile(family + "-original-v1")
            self.assertEqual(profile["family"], family)
            self.assertEqual(set(profile["supported_datasets"]), {"BCI_IV_2a", "BCI_IV_2b", "SHU_MI"})
            self.assertEqual(len(profile["workflow_contract_sha256"]), 64)
            self.assertTrue(profile["workflow_contract"])

    def test_both_original_workflows_preserve_same_real_preparation_math(self):
        signal = np.arange(22 * 4000, dtype=np.float64).reshape(22, 4000) * 1e-10
        blob = BytesIO()
        np.savez_compressed(blob, signal=signal, sfreq=250, ch_names=[f"EEG-{i}" for i in range(22)],
                            events=np.array([[1000, 0, 769]], dtype=np.int64), unit="V")
        blob.seek(0)
        row = service.register_recording(blob, "a.npz", "BCI_IV_2a")
        arrays = []
        for family in ("dwdanet", "dwtnet"):
            result = service.preprocess_recording(row["recording_id"], family + "-original-v1", 1)
            with np.load(service.data_path(result["artifact_id"]), allow_pickle=False) as data:
                arrays.append(data["data"])
            self.assertEqual(result["protocol"]["workflow_family"], family)
            self.assertIsNone(result["protocol"]["normalization"])
            self.assertEqual(result["protocol"]["normalization_stage"], "classifier_adapter_only_not_applied_here")
        np.testing.assert_array_equal(arrays[0], arrays[1])
        np.testing.assert_array_equal(arrays[0][0], signal[:, 1000:2000])

    def test_authenticated_shu_registration_split_download_and_sample(self):
        row = self.client.post("/workspace/mi-preprocessing/shu-sessions", files=self.shu_files())
        self.assertEqual(row.status_code, 201, row.text)
        record = row.json()
        result = self.client.post("/workspace/mi-preprocessing/run", json={"recording_id": record["recording_id"],
                    "profile_id": "dwdanet-original-v1", "max_trials": 2})
        self.assertEqual(result.status_code, 200, result.text)
        result = result.json()
        self.assertEqual(result["output"]["shape"], [2, 32, 1000])
        self.assertFalse(samples.active_entries()[result["sample_id"]]["compatible"])
        for partition, count in (("train", 16), ("test", 4)):
            response = self.client.get(f"/workspace/mi-preprocessing/artifacts/{result['artifact_id']}/split/{partition}")
            self.assertEqual(response.status_code, 200, response.text[:200])
            data = loadmat(BytesIO(response.content))
            self.assertEqual(data["data"].shape, (count, 32, 1000))
            self.assertEqual(data["data"].dtype, np.float32)
            self.assertEqual(float(data["data"][0, 0, 0]), 10 if partition == "train" else 50)
        with np.load(service.data_path(result["artifact_id"]), allow_pickle=False) as output:
            self.assertAlmostEqual(float(output["data"][0, 0, 0]), 50e-6, places=10)
        item = next(a for a in service.catalog()["artifacts"] if a["artifact_id"] == result["artifact_id"])
        self.assertEqual(item["profile"]["id"], "dwdanet-original-v1")
        self.assertTrue(item["source_file"])
        document = service.artifact_document(result)
        self.assertIn("已分段", document["text"])
        self.assertNotIn("原始提示事件", document["text"])

    def test_shu_cannot_fall_through_to_continuous_reader_or_unknown_split(self):
        response = self.client.post("/workspace/mi-preprocessing/shu-sessions", files=self.shu_files()[:4])
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/workspace/mi-preprocessing/shu-sessions", data={"split_id": "guess"}, files=self.shu_files())
        self.assertEqual(response.status_code, 422)
        record = self.client.post("/workspace/mi-preprocessing/shu-sessions", files=self.shu_files()).json()
        response = self.client.post("/workspace/mi-preprocessing/run", json={"recording_id": record["recording_id"],
                    "profile_id": "mi-native-cue4s-v1", "max_trials": 2})
        self.assertEqual(response.status_code, 400)
        self.assertIn("SHU", response.text)

    def test_split_integrity_and_absent_exports_are_rejected(self):
        row = self.client.post("/workspace/mi-preprocessing/shu-sessions", files=self.shu_files()).json()
        result = service.preprocess_recording(row["recording_id"], "dwtnet-original-v1", 2)
        with service.split_path(result["artifact_id"], "train").open("ab") as stream:
            stream.write(b"changed")
        response = self.client.get(f"/workspace/mi-preprocessing/artifacts/{result['artifact_id']}/split/train")
        self.assertEqual(response.status_code, 400)
        with self.assertRaises(ValueError):
            service.split_path(result["artifact_id"], "../test")


if __name__ == "__main__":
    unittest.main()
