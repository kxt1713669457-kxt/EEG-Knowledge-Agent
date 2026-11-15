"""Real MAT session preparation, split provenance and failure isolation."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat

from app.agent.skills.mi_preprocessing import service, shu
from app.workspace import samples, store


DIRECT = "shu-session1to4-vs5-v1"
MIXED = "shu-mixed-session-stratified70-v1"


class SHUPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for owner, field, value in [(service, "ROOT", self.root / "mi"),
                                     (samples, "SAMPLE_ROOT", self.root / "samples"),
                                     (store, "DB_PATH", self.root / "workspace.sqlite3")]:
            mock = patch.object(owner, field, value)
            mock.start()
            self.addCleanup(mock.stop)
        guard = patch.object(service, "_memory_guard")
        guard.start()
        self.addCleanup(guard.stop)

    def sessions(self, subject=1, n=4, changes=None):
        items, data = [], []
        t = np.arange(1000, dtype=np.float32) / 250
        for session in range(1, 6):
            x = np.tile(np.sin(2 * np.pi * 10 * t), (n, 32, 1)).astype(np.float32) * np.float32(10)
            x += np.arange(n, dtype=np.float32)[:, None, None] + session * 100
            y = np.tile(np.array([1, 2], dtype=np.int64), (n + 1) // 2)[:n][None]
            contents = {"data": x, "labels": y}
            if changes:
                changes(session, contents)
            blob = BytesIO()
            savemat(blob, contents)
            blob.seek(0)
            items.append((blob, f"sub-{subject:03d}_ses-{session:02d}_task_motorimagery_eeg.mat"))
            data.append(contents)
        return items, data

    def profile(self, family="dwdanet"):
        return {"id": f"{family}-original-v1", "name": family + "原流程", "family": family,
                "method": "identity", "band": None, "model_compatibility": "legacy_protocol"}

    def run_preparation(self, split=DIRECT, family="dwdanet", max_trials=3):
        items, contents = self.sessions()
        row = shu.register_sessions(items, split)
        with service._exclusive():
            result = shu.preprocess(row, self.profile(family), max_trials)
        return row, result, contents

    def test_direct_exports_preserve_microvolts_and_source_order(self):
        row, result, contents = self.run_preparation()
        folder = service.ROOT / "artifacts" / result["artifact_id"]
        train, test = loadmat(folder / "train.mat"), loadmat(folder / "test.mat")
        np.testing.assert_array_equal(train["data"], np.concatenate([m["data"] for m in contents[:4]]))
        np.testing.assert_array_equal(test["data"], contents[4]["data"])
        np.testing.assert_array_equal(train["label"].ravel(), np.concatenate([m["labels"].ravel() for m in contents[:4]]))
        self.assertEqual(train["data"].dtype, np.float32)
        self.assertEqual(row["input_kind"], "shu_epoched_sessions")
        self.assertEqual(row["unit"], "uV")
        self.assertEqual(row["duration_seconds"], 80)
        self.assertEqual(row["event_count"], 20)
        self.assertEqual(result["exports"]["train"]["unit"], "uV")
        self.assertEqual([m["session"] for m in result["events"]], [5, 5, 5])
        self.assertIsNone(result["protocol"]["split"]["seed"])

    def test_inference_converts_unit_once_without_normalization_or_labels(self):
        _, result, contents = self.run_preparation()
        sample = samples.active_entries()[result["sample_id"]]
        np.testing.assert_array_equal(samples.read_array(samples.registered_path(sample)),
                                      contents[4]["data"][:3] * np.float32(1e-6))
        self.assertEqual(sample["unit"], "V")
        self.assertFalse(sample["compatible"])
        self.assertTrue(result["protocol"]["normalization_not_applied"])
        self.assertFalse(result["protocol"]["labels_used"])
        self.assertFalse(result["protocol"]["split"]["labels_used_as_model_input"])
        self.assertNotIn("label", result["events"][0])
        self.assertEqual(result["output"]["shape"], [3, 32, 1000])
        self.assertEqual(result["output"]["channel_names"], shu.CHANNELS)

    def test_mixed_exact_original_randomstate_split_and_replay(self):
        items, contents = self.sessions(subject=2)
        row = shu.register_sessions(items, MIXED)
        result = shu.preprocess(row, self.profile(), 32)
        y = np.concatenate([m["labels"].ravel() for m in contents])
        rng = np.random.RandomState(2028)
        train, test = [], []
        for cls in sorted(np.unique(y)):
            ind = np.where(y == cls)[0]
            rng.shuffle(ind)
            count = max(1, min(int(round(len(ind) * .3)), len(ind) - 1))
            test.extend(ind[:count].tolist())
            train.extend(ind[count:].tolist())
        train, test = np.asarray(train), np.asarray(test)
        rng.shuffle(train)
        rng.shuffle(test)
        split = result["protocol"]["split"]
        self.assertEqual(split["train_indices"], train.tolist())
        self.assertEqual(split["test_indices"], test.tolist())
        self.assertEqual(split["seed"], 2028)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(sorted([*train, *test]), list(range(20)))
        again = shu.preprocess(row, self.profile("dwtnet"), 32)
        self.assertNotEqual(result["artifact_id"], again["artifact_id"])
        self.assertEqual(result["protocol"]["split"], again["protocol"]["split"])
        with np.load(service.data_path(result["artifact_id"])) as a, np.load(service.data_path(again["artifact_id"])) as b:
            np.testing.assert_array_equal(a["data"], b["data"])

    def test_record_and_artifact_survive_generic_service_readers(self):
        row, result, _ = self.run_preparation()
        self.assertEqual(service.recording(row["id"]), row)
        self.assertEqual(service.read_artifact(result["artifact_id"]), result)
        self.assertEqual(store.read("mi_artifact", result["artifact_id"])["profile"], result["profile"])
        self.assertEqual(service._hash(service.data_path(result["artifact_id"])), result["output"]["sha256"])
        self.assertTrue(result["qc"]["before"]["finite"])
        self.assertEqual(result["qc"]["before"], result["qc"]["after"])

    def test_upload_order_is_sorted_and_source_files_immutable(self):
        items, _ = self.sessions()
        originals = {filename: blob.getvalue() for blob, filename in items}
        row = shu.register_sessions(list(reversed(items)))
        self.assertEqual([s["session"] for s in row["source_files"]], [1, 2, 3, 4, 5])
        shu.preprocess(row, self.profile(), 1)
        folder = service.ROOT / "shu-recordings" / row["id"]
        for source in row["source_files"]:
            self.assertEqual((folder / source["storage_name"]).read_bytes(), originals[source["file"]])

    def test_names_subject_and_complete_sessions_are_strict(self):
        for case in ["missing", "duplicate", "mixed_subject", "path", "derived"]:
            with self.subTest(case=case):
                items, _ = self.sessions()
                if case == "missing": items.pop()
                if case == "duplicate": items[-1] = (items[-1][0], items[0][1])
                if case == "mixed_subject": items[-1] = (items[-1][0], items[-1][1].replace("001", "002"))
                if case == "path": items[0] = (items[0][0], "../" + items[0][1])
                if case == "derived": items[0] = (items[0][0], "C01T.mat")
                with self.assertRaises(ValueError): shu.register_sessions(items)
        self.assertEqual(store.all_records("mi_recording"), [])

    def test_invalid_axes_dtype_labels_and_nonfinite_are_rejected(self):
        def alteration(kind):
            def change(session, m):
                if session != 2: return
                if kind == "axes": m["data"] = m["data"].transpose(0, 2, 1)
                if kind == "integer": m["data"] = m["data"].astype(np.int32)
                if kind == "complex": m["data"] = m["data"].astype(np.complex64)
                if kind == "nan": m["data"][0, 0, 0] = np.nan
                if kind == "labels": m["labels"][0, 0] = 0
                if kind == "fraction": m["labels"] = m["labels"].astype(float) + .1
                if kind == "label_count": m["labels"] = m["labels"][:, :-1]
                if kind == "wrong_key": m["label"] = m.pop("labels")
            return change
        for kind in ["axes", "integer", "complex", "nan", "labels", "fraction", "label_count", "wrong_key"]:
            with self.subTest(kind=kind):
                items, _ = self.sessions(changes=alteration(kind))
                with self.assertRaises(ValueError): shu.register_sessions(items)
                self.assertEqual(store.all_records("mi_recording"), [])
                self.assertEqual(list((service.ROOT / "shu-recordings").glob("*")), [])

    def test_split_and_trial_limits_reject_unknown_or_boolean(self):
        items, _ = self.sessions()
        with self.assertRaises(ValueError): shu.register_sessions(items, "unknown")
        row = shu.register_sessions(items)
        for value in [0, 289, True, 1.5, "3"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                shu.preprocess(row, self.profile(), value)
        with self.assertRaises(ValueError): shu.preprocess(row, self.profile("other"), 1)
        with self.assertRaises(ValueError): shu.preprocess(row, self.profile() | {"method": "bandpass"}, 1)

    def test_hash_corruption_fails_without_new_samples(self):
        items, _ = self.sessions()
        row = shu.register_sessions(items)
        path = service.ROOT / "shu-recordings" / row["id"] / row["source_files"][0]["storage_name"]
        with path.open("ab") as f: f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "哈希"):
            shu.preprocess(row, self.profile(), 1)
        self.assertEqual(samples.active_entries(), {})
        self.assertEqual(store.all_records("mi_artifact"), [])

    def test_upload_total_and_decoded_limits_rollback_only_new_uploads(self):
        items, _ = self.sessions()
        existing = shu.register_sessions(items)
        for limit, maximum in [("MAX_FILE_BYTES", 1024), ("MAX_TOTAL_BYTES", 800000), ("MAX_DECODED_BYTES", 2000000)]:
            with self.subTest(limit=limit), patch.object(shu, limit, maximum):
                items, _ = self.sessions()
                with self.assertRaises(ValueError): shu.register_sessions(items)
        self.assertEqual([r["id"] for r in store.all_records("mi_recording")], [existing["id"]])
        self.assertEqual([p.name for p in (service.ROOT / "shu-recordings").iterdir()], [existing["id"]])

    def test_low_memory_guard_is_honored_before_upload(self):
        items, _ = self.sessions()
        with patch.object(service, "_memory_guard", side_effect=ValueError("memory")):
            with self.assertRaisesRegex(ValueError, "memory"): shu.register_sessions(items)
        self.assertFalse((service.ROOT / "shu-recordings").exists())

    def test_failed_artifact_rolls_back_only_new_files_sample_and_index(self):
        row, existing, _ = self.run_preparation()
        write = store.write
        def fail(kind, *args, **kwargs):
            if kind == "mi_artifact": raise RuntimeError("index failed")
            return write(kind, *args, **kwargs)
        with patch.object(store, "write", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "index failed"):
                shu.preprocess(row, self.profile(), 1)
        self.assertEqual(list(samples.active_entries()), [existing["sample_id"]])
        self.assertEqual([p.name for p in (service.ROOT / "artifacts").iterdir()], [existing["artifact_id"]])
        self.assertEqual([r["artifact_id"] for r in store.all_records("mi_artifact")], [existing["artifact_id"]])
        self.assertEqual(service.read_artifact(existing["artifact_id"]), existing)


if __name__ == "__main__":
    unittest.main()
