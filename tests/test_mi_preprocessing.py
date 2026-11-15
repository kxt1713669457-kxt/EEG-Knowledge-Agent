"""Offline MI preprocessing acceptance: signal preservation, cleaning and provenance.

All uploads, artifacts and database records live in a temporary workspace. No
external datasets, weights, databases or model services are required.
"""
from io import BytesIO
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from app.agent.skills.mi_preprocessing import catalog as profile_catalog
from app.agent.skills.mi_preprocessing import service
from app.workspace import samples, store


LEGACY = "bci-cue4s-legacy-v1"


class MIPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.memory_guard = service._memory_guard
        for owner, name, value in (
            (service, "ROOT", self.root / "mi"),
            (samples, "SAMPLE_ROOT", self.root / "samples"),
            (store, "DB_PATH", self.root / "workspace.sqlite3"),
        ):
            item = patch.object(owner, name, value)
            item.start()
            self.addCleanup(item.stop)
        # Resource admission is tested separately; tests must run on busy hosts.
        memory = patch.object(service, "_memory_guard")
        memory.start()
        self.addCleanup(memory.stop)

    def signal(self, channels=22, n_times=4500, eog=True):
        t = np.arange(n_times) / 250.0
        eeg = np.array([(1 + i / 30) * 10e-6 * np.sin(2 * np.pi * 10 * t)
                        for i in range(channels)])
        names = [f"EEG-{i}" for i in range(channels)]
        if eog:
            eeg = np.vstack((eeg, 100e-6 * np.sin(2 * np.pi * 2 * t)))
            names.append("EOG-left")
        return eeg, names

    def blob(self, signal=None, names=None, events=None, unit="V", sfreq=250):
        if signal is None:
            signal, names = self.signal()
        if events is None:
            events = np.array([[1000, 0, 769], [2200, 0, 772]], dtype=np.int64)
        stream = BytesIO()
        np.savez_compressed(stream, signal=signal, ch_names=np.asarray(names),
                            events=events, sfreq=sfreq, unit=unit)
        stream.seek(0)
        return stream

    def register(self, **kwargs):
        metadata = kwargs.pop("metadata", {})
        dataset = kwargs.pop("dataset", "BCI_IV_2a")
        return service.register_recording(self.blob(**kwargs), "record.npz", dataset,
                                          json.dumps(metadata))

    def run_profile(self, profile=LEGACY, max_trials=32, **kwargs):
        row = self.register(**kwargs)
        return row, service.preprocess_recording(row["recording_id"], profile, max_trials)

    def data(self, result):
        with np.load(service.data_path(result["artifact_id"]), allow_pickle=False) as arrays:
            return arrays["data"]

    def test_legacy_exact_cue_windows_original_codes_eog_and_provenance(self):
        signal, names = self.signal()
        original = self.blob(signal=signal, names=names).getvalue()
        row = service.register_recording(BytesIO(original), "record.npz", "BCI_IV_2a")
        result = service.preprocess_recording(row["recording_id"], LEGACY, 32)
        output = self.data(result)
        np.testing.assert_array_equal(output[0], signal[:22, 1000:2000])
        np.testing.assert_array_equal(output[1], signal[:22, 2200:3200])
        self.assertEqual([e["cue_code"] for e in result["events"]], [769, 772])
        self.assertEqual([e["source_sample"] for e in result["events"]], [1000, 2200])
        self.assertEqual(result["output"]["channel_names"], names[:22])
        self.assertEqual(row["eog_channels"], ["EOG-left"])
        self.assertEqual(result["output"]["shape"], [2, 22, 1000])
        self.assertEqual(result["output"]["unit"], "V")
        self.assertFalse(result["protocol"]["labels_used"])
        self.assertIsNone(result["protocol"]["normalization"])
        self.assertIsNone(result["protocol"]["resample"])
        self.assertIsNone(result["protocol"]["band_hz"])
        self.assertEqual(service._path(service.recording(row["recording_id"])).read_bytes(), original)
        self.assertEqual(service.read_artifact(result["artifact_id"]), result)
        document = service.artifact_document(result)
        self.assertEqual(document["sample_id"], result["sample_id"])
        self.assertIn(row["sha256"], document["text"])

    def test_microvolts_are_converted_once_to_volts(self):
        signal, names = self.signal()
        _, result = self.run_profile(signal=signal * 1e6, names=names, unit="uV")
        np.testing.assert_allclose(self.data(result)[0], signal[:22, 1000:2000], rtol=1e-14, atol=1e-20)

    def test_output_registered_for_classifier_without_changing_channel_order(self):
        _, result = self.run_profile()
        entry = samples.active_entries()[result["sample_id"]]
        self.assertTrue(entry["compatible"])
        self.assertEqual(entry["preprocessing_artifact_id"], result["artifact_id"])
        self.assertEqual(entry["preprocessing_profile_id"], LEGACY)
        self.assertEqual(entry["preprocessing_compatibility"], "legacy_protocol")
        self.assertEqual(entry["channel_names"], result["output"]["channel_names"])
        np.testing.assert_array_equal(samples.read_array(samples.registered_path(entry)), self.data(result))

    def test_unknown_eval_code_is_kept_without_fabricating_labels(self):
        _, result = self.run_profile(events=np.array([[1000, 0, 783]], dtype=np.int64))
        self.assertEqual(result["events"][0]["cue_code"], 783)
        self.assertNotIn("label", result["events"][0])
        self.assertEqual(result["output"]["shape"], [1, 22, 1000])

    def test_original_event_codes_are_sorted_and_noncue_events_ignored(self):
        events = np.array([[2200, 0, 772], [250, 0, 768], [1000, 0, 769]], dtype=np.int64)
        row, result = self.run_profile(events=events)
        self.assertEqual(row["event_count"], 2)
        self.assertEqual([event["source_sample"] for event in result["events"]], [1000, 2200])

    def test_generic_dataset_uses_explicit_custom_event_mapping(self):
        row, result = self.run_profile(dataset="MI_GENERIC", metadata={"cue_event_codes": [42]},
                                       events=np.array([[1000, 0, 42]], dtype=np.int64))
        self.assertEqual(row["event_count"], 1)
        self.assertEqual(result["events"][0]["cue_code"], 42)

    def test_2b_preserves_three_bipolar_channels_and_notes_model_boundary(self):
        signal, names = self.signal(channels=3)
        _, result = self.run_profile(dataset="BCI_IV_2b", signal=signal, names=names,
                                     events=np.array([[1000, 0, 770]], dtype=np.int64))
        self.assertEqual(result["output"]["shape"], [1, 3, 1000])
        self.assertIn("22通道", result["scope"])
        self.assertFalse(samples.active_entries()[result["sample_id"]]["compatible"])

    def test_max_trials_selects_first_actual_events_and_rejects_invalid_values(self):
        row, result = self.run_profile(max_trials=1)
        self.assertEqual(result["total_cue_events"], 2)
        self.assertEqual(result["output"]["shape"][0], 1)
        for value in (0, 289, True, 1.5, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                service.preprocess_recording(row["recording_id"], LEGACY, value)

    def test_incomplete_and_nonfinite_windows_are_reported_not_filled(self):
        signal, names = self.signal()
        signal[0, 2300] = np.nan
        events = np.array([[1000, 0, 769], [2200, 0, 770], [4000, 0, 771]], dtype=np.int64)
        _, result = self.run_profile(signal=signal, names=names, events=events)
        self.assertEqual(result["output"]["shape"][0], 1)
        self.assertEqual([v["reason"] for v in result["qc"]["dropped_trials"]],
                         ["nonfinite_window", "incomplete_window"])
        self.assertTrue(np.isfinite(self.data(result)).all())

    def test_all_invalid_windows_leave_no_artifact_or_registered_sample(self):
        row = self.register(events=np.array([[4000, 0, 769]], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "没有生成样本"):
            service.preprocess_recording(row["recording_id"], LEGACY)
        self.assertEqual(samples.active_entries(), {})
        self.assertFalse((service.ROOT / "artifacts").exists())

    def test_qc_measures_flat_channels_large_trials_and_mu_beta(self):
        t = np.arange(1000) / 250
        x = np.zeros((2, 3, 1000))
        x[:, 0] = 10e-6 * np.sin(2 * np.pi * 10 * t)
        x[:, 1] = 200e-6 * np.sin(2 * np.pi * 20 * t)
        qc = service._qc(x, 250)
        self.assertEqual(qc["flat_channel_indices"], [2])
        self.assertEqual(qc["over_200uv_trial_indices"], [0, 1])
        self.assertAlmostEqual(qc["rms_uv_per_channel"][0], 10 / np.sqrt(2), places=8)
        self.assertGreater(qc["mean_band_psd_sum"]["mu"], 0)
        self.assertGreater(qc["mean_band_psd_sum"]["beta"], 0)

    def test_mu_beta_filter_suppresses_50hz_and_keeps_10hz(self):
        signal, names = self.signal()
        t = np.arange(signal.shape[-1]) / 250
        signal[:22] = 10e-6 * (np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t))
        _, result = self.run_profile("mi-mu-beta-v1", signal=signal, names=names)
        x = self.data(result)[0, 0]
        t_window = t[1000:2000]
        amp = lambda freq: abs(np.dot(x, np.exp(-2j * np.pi * freq * t_window))) * 2 / len(x)
        self.assertGreater(amp(10), 8e-6)
        self.assertLess(amp(50), 1e-7)
        self.assertEqual(result["profile"]["model_compatibility"], "requires_validation")
        self.assertIn("双向滤波", result["scope"])

    def test_eog_regression_removes_correlated_eye_signal_preserves_mu(self):
        signal, names = self.signal()
        t = np.arange(signal.shape[-1]) / 250
        eye = 100e-6 * np.sin(2 * np.pi * 6 * t)
        signal[-1] = eye
        # Zero-mean spatial patterns survive CAR, so this checks regression
        # rather than passing because the reference alone removed the signal.
        brain_pattern = np.linspace(1, -1, 22)
        eye_pattern = np.cos(np.arange(22) * 2 * np.pi / 22)
        signal[:22] = (brain_pattern[:, None] * 10e-6 * np.sin(2 * np.pi * 10 * t)
                       + .5 * eye_pattern[:, None] * eye)
        _, result = self.run_profile("mi-eog-regression-v1", signal=signal, names=names)
        x = self.data(result)[0, 0]
        amplitude = lambda f: abs(np.dot(x, np.exp(-2j * np.pi * f * t[1000:2000]))) * 2 / len(x)
        self.assertLess(amplitude(6), 1e-7)
        self.assertGreater(amplitude(10), 8e-6)
        details = result["qc"]["method_details"]
        self.assertGreater(details["fit_valid_samples"], 1000)
        self.assertEqual(len(details["regression_coefficients"]), 22)

    def test_eog_regression_requires_real_eye_channel(self):
        signal, names = self.signal(eog=False)
        row = self.register(signal=signal, names=names)
        with self.assertRaisesRegex(ValueError, "真实EOG"):
            service.preprocess_recording(row["recording_id"], "mi-eog-regression-v1")
        self.assertEqual(samples.active_entries(), {})

    def test_sampling_rate_is_not_silently_changed(self):
        row = self.register(sfreq=200)
        with self.assertRaisesRegex(ValueError, "250Hz"):
            service.preprocess_recording(row["recording_id"], LEGACY)

    def test_original_record_hash_tamper_is_detected(self):
        row = self.register()
        path = service._path(service.recording(row["recording_id"]))
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "哈希不一致"):
            service.preprocess_recording(row["recording_id"], LEGACY)

    def test_artifact_hash_and_identity_tamper_are_detected(self):
        _, result = self.run_profile()
        path = service.data_path(result["artifact_id"])
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "哈希不一致"):
            service.data_path(result["artifact_id"])
        result_path = path.parent / "result.json"
        result["artifact_id"] = "f" * 32
        result_path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "身份不一致"):
            service.read_artifact(path.parent.name)

    def test_unregistered_or_path_like_ids_never_become_sources(self):
        for identity in ("../outside", "0" * 32, "a" * 31, "G" * 32):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                service.recording(identity)
            with self.assertRaises(ValueError):
                service.read_artifact(identity)

    def test_record_storage_path_cannot_escape_upload_directory(self):
        row = self.register()
        private = service.recording(row["recording_id"])
        external = self.root / "outside.npz"
        external.write_bytes(b"not-a-source")
        private["storage_name"] = "../../outside.npz"
        store.write("mi_recording", private["id"], private)
        with self.assertRaisesRegex(ValueError, "不可读取"):
            service.preprocess_recording(private["id"], LEGACY)

    def test_metadata_rejects_unknown_fields_bool_codes_and_guessed_positions(self):
        for metadata in ({"labels": [1]}, {"cue_event_codes": [True]},
                         {"channel_positions": {"EEG-0": [float("nan"), 0, 0]}},
                         {"channel_mapping": {"EEG-0": 1}}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.register(metadata=metadata)
        self.assertEqual(store.all_records("mi_recording"), [])

    def test_duplicate_channel_names_events_and_unknown_mapping_rejected(self):
        signal, names = self.signal()
        duplicated = names.copy()
        duplicated[1] = duplicated[0]
        cases = (
            {"signal": signal, "names": duplicated},
            {"events": np.array([[1000, 0, 769], [1000, 0, 770]], dtype=np.int64)},
            {"metadata": {"channel_mapping": {"missing": "C3"}}},
            {"metadata": {"eog_channels": ["missing"]}},
        )
        for kwargs in cases:
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.register(**kwargs)
        self.assertEqual(list((service.ROOT / "recordings").glob("*")), [])

    def test_invalid_npz_events_units_and_missing_signal_fail_cleanly(self):
        for kwargs in ({"events": np.array([[1000., 0., 769.]])},
                       {"events": np.array([[1000, 1, 769]], dtype=np.int64)},
                       {"unit": "mV"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.register(**kwargs)
        blob = BytesIO()
        np.savez_compressed(blob, data=np.zeros((1, 22, 1000)))
        blob.seek(0)
        with self.assertRaisesRegex(ValueError, "signal"):
            service.register_recording(blob, "wrong.npz", "BCI_IV_2a")
        self.assertEqual(store.all_records("mi_recording"), [])

    def test_npz_expansion_and_forged_signal_header_are_bounded(self):
        blob = BytesIO()
        with zipfile.ZipFile(blob, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("padding", b"0" * 8192)
        blob.seek(0)
        with patch.object(service, "MAX_SIGNAL_BYTES", 4096), self.assertRaisesRegex(ValueError, "展开"):
            service.register_recording(blob, "expansion.npz", "BCI_IV_2a")
        header = BytesIO()
        np.lib.format.write_array_header_1_0(header, {"shape": (22, 100_000_000),
                                                    "fortran_order": False, "descr": "<f8"})
        forged = BytesIO()
        with zipfile.ZipFile(self.blob()) as source, zipfile.ZipFile(forged, "w") as archive:
            for name in source.namelist():
                archive.writestr(name, header.getvalue() if name == "signal.npy" else source.read(name))
        forged.seek(0)
        with self.assertRaisesRegex(ValueError, "声明体积"):
            service.register_recording(forged, "forged.npz", "BCI_IV_2a")
        self.assertEqual(store.all_records("mi_recording"), [])

    def test_npz_metadata_headers_and_integer_overflow_cannot_trigger_large_allocation(self):
        malicious = (
            ("signal.npy", (64, 2**62), "<f8"),
            ("events.npy", (2**61, 3), "<i8"),
            ("ch_names.npy", (23,), "|O"),
            ("sfreq.npy", (2**61,), "<f8"),
            ("unit.npy", (), "<U1000000"),
        )
        for target, shape, descr in malicious:
            header = BytesIO()
            np.lib.format.write_array_header_1_0(header, {"shape": shape, "fortran_order": False, "descr": descr})
            blob = BytesIO()
            with zipfile.ZipFile(self.blob()) as source, zipfile.ZipFile(blob, "w") as archive:
                for name in source.namelist():
                    archive.writestr(name, header.getvalue() if name == target else source.read(name))
            blob.seek(0)
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "数组头"):
                service.register_recording(blob, "metadata-bomb.npz", "BCI_IV_2a")
        self.assertEqual(store.all_records("mi_recording"), [])

    def test_bad_annotations_and_acquisition_boundaries_reject_crossing_epochs(self):
        import mne
        original_open = service._open

        def annotate(path, dataset_id, metadata):
            raw, events, picks = original_open(path, dataset_id, metadata)
            raw.set_annotations(mne.Annotations([4.5, 9.5], [.1, 0.0], ["BAD_artifact", "32766"]))
            return raw, events, picks

        events = np.array([[1000, 0, 769], [2200, 0, 770], [3500, 0, 771]], dtype=np.int64)
        with patch.object(service, "_open", side_effect=annotate):
            _, result = self.run_profile(events=events)
        self.assertEqual([v["reason"] for v in result["qc"]["dropped_trials"]],
                         ["bad_annotation", "acquisition_boundary"])
        self.assertEqual([v["source_sample"] for v in result["events"]], [3500])

    def test_empty_wrong_extension_and_upload_limit_leave_no_files(self):
        for blob, name in ((BytesIO(), "empty.npz"), (BytesIO(b"x"), "code.py")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                service.register_recording(blob, name, "BCI_IV_2a")
        with patch.object(service, "MAX_FILE_BYTES", 16), self.assertRaisesRegex(ValueError, "上传超过"):
            self.register()
        self.assertEqual(list((service.ROOT / "recordings").glob("*")), [])

    def test_output_memory_limit_fails_before_sample_registration(self):
        row = self.register()
        with patch.object(service, "MAX_OUTPUT_BYTES", 100), self.assertRaisesRegex(ValueError, "96MiB"):
            service.preprocess_recording(row["recording_id"], LEGACY)
        self.assertEqual(samples.active_entries(), {})
        self.assertFalse((service.ROOT / "artifacts").exists())

    def test_memory_admission_rejects_busy_host_and_respects_estimate(self):
        fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=512 * 1024**2))
        with patch.dict("sys.modules", {"psutil": fake_psutil}):
            with self.assertRaisesRegex(ValueError, "可用内存不足"):
                self.memory_guard(10)
            fake_psutil.virtual_memory = lambda: SimpleNamespace(available=800 * 1024**2)
            with self.assertRaisesRegex(ValueError, "可用内存不足"):
                self.memory_guard(1024 * 1024**2)
            fake_psutil.virtual_memory = lambda: SimpleNamespace(available=2 * 1024**3)
            self.memory_guard(1024 * 1024**2)

    def test_bandpass_does_not_spread_a_jump_across_acquisition_boundary(self):
        import mne
        signal = np.zeros((2, 4500))
        signal[:, 2250:] = 1e-3
        raw = mne.io.RawArray(signal, mne.create_info(["C3", "C4"], 250, "eeg"), verbose="ERROR")
        raw.set_annotations(mne.Annotations([9.0], [0.0], ["32766"]))
        try:
            valid = np.ones(raw.n_times, dtype=bool)
            service._filter(raw, np.array([0, 1]), [8, 32], valid)
            self.assertLess(np.abs(raw.get_data()).max(), 1e-15)
            self.assertTrue(valid.all())
        finally:
            raw.close()

    def test_nan_gap_filter_segments_remain_separate_and_short_spans_invalidated(self):
        import mne
        signal = np.zeros((2, 4500))
        signal[:, 2250:] = 1e-3
        raw = mne.io.RawArray(signal, mne.create_info(["C3", "C4"], 250, "eeg"), verbose="ERROR")
        try:
            valid = np.ones(raw.n_times, dtype=bool)
            valid[2240:2250] = False
            valid[10:20] = False
            service._filter(raw, np.array([0, 1]), [8, 32], valid)
            self.assertFalse(valid[:20].any())
            self.assertLess(np.abs(raw.get_data()[:, valid]).max(), 1e-15)
        finally:
            raw.close()

    def test_single_execution_lock_rejects_overlap_and_releases_on_error(self):
        with service._exclusive():
            with self.assertRaisesRegex(ValueError, "一次仅处理一个"):
                self.register()
        with self.assertRaisesRegex(RuntimeError, "test"):
            with service._exclusive():
                raise RuntimeError("test")
        self.assertEqual(self.register()["event_count"], 2)

    def test_cross_process_lock_fails_without_waiting_then_succeeds_after_release(self):
        child = """import json, sys, time
from pathlib import Path
from app.agent.skills.mi_preprocessing import service
service.ROOT = Path(sys.argv[1])
started = time.perf_counter()
try:
    with service._exclusive():
        outcome = {'status': 'acquired', 'elapsed': time.perf_counter() - started}
except ValueError as exc:
    print(json.dumps({'status': 'blocked', 'elapsed': time.perf_counter() - started, 'message': str(exc)}))
    sys.exit(23)
print(json.dumps(outcome))
"""
        def compete():
            return subprocess.run([sys.executable, "-c", child, str(service.ROOT)],
                                  cwd=Path(__file__).resolve().parents[1],
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                  capture_output=True, text=True, encoding="utf-8", timeout=15)

        with service._exclusive():
            blocked = compete()
            self.assertEqual(blocked.returncode, 23, blocked.stdout + blocked.stderr)
            outcome = json.loads(blocked.stdout.strip().splitlines()[-1])
            self.assertEqual(outcome["status"], "blocked")
            self.assertIn("另一服务进程", outcome["message"])
            self.assertLess(outcome["elapsed"], 2, "file lock must reject rather than wait")
        released = compete()
        self.assertEqual(released.returncode, 0, released.stdout + released.stderr)
        outcome = json.loads(released.stdout.strip().splitlines()[-1])
        self.assertEqual(outcome["status"], "acquired")
        self.assertLess(outcome["elapsed"], 2)

    def test_registration_storage_failure_rolls_back_original_file(self):
        with patch.object(store, "write", side_effect=RuntimeError("storage failed")):
            with self.assertRaisesRegex(RuntimeError, "storage failed"):
                self.register()
        self.assertEqual(list((service.ROOT / "recordings").glob("*")), [])

    def test_late_artifact_failure_rolls_back_only_new_sample_and_files(self):
        row = self.register()
        untouched = {"id": "existing", "value": "preserve"}
        store.write("sample", "existing", untouched)
        with patch.object(service, "_qc", side_effect=RuntimeError("QC failed")):
            with self.assertRaisesRegex(RuntimeError, "QC failed"):
                service.preprocess_recording(row["recording_id"], LEGACY)
        self.assertEqual(store.all_records("sample"), [untouched])
        self.assertEqual(list((service.ROOT / "artifacts").glob("*")), [])
        self.assertEqual(list(samples.SAMPLE_ROOT.glob("*")), [])
        self.assertTrue(service._path(service.recording(row["recording_id"])).is_file())

    def test_unknown_electrode_positions_are_not_filled_from_order(self):
        row = self.register()
        private = service.recording(row["recording_id"])
        raw, _, picks = service._open(service._path(private), private["dataset_id"], private["metadata"])
        try:
            self.assertFalse(service._has_positions(raw, picks))
        finally:
            raw.close()
        profile = next(p for p in profile_catalog.PROFILES if p["id"] == "mi-iclabel-v1")
        # Ensure this guard is verified even when optional neural deps are absent.
        with patch.object(service, "get_profile", return_value=profile):
            with self.assertRaisesRegex(ValueError, "真实位置"):
                service.preprocess_recording(row["recording_id"], profile["id"])

    def test_explicit_standard_name_mapping_attaches_real_montage(self):
        signal, names = self.signal(channels=3)
        row = self.register(dataset="BCI_IV_2b", signal=signal, names=names,
                            metadata={"channel_mapping": {"EEG-0": "C3", "EEG-1": "Cz", "EEG-2": "C4"}})
        private = service.recording(row["recording_id"])
        raw, _, picks = service._open(service._path(private), private["dataset_id"], private["metadata"])
        try:
            self.assertTrue(service._has_positions(raw, picks))
            self.assertEqual([raw.ch_names[p] for p in picks], ["C3", "Cz", "C4"])
        finally:
            raw.close()

    def test_catalog_keeps_research_only_methods_out_of_executable_profiles(self):
        catalog = service.catalog()
        self.assertEqual(catalog["limits"]["concurrency"], 1)
        self.assertIn(LEGACY, {p["id"] for p in catalog["profiles"]})
        for method in catalog["methods"]:
            self.assertEqual(method["status"], "research_only")
            with self.assertRaisesRegex(ValueError, "未登记"):
                profile_catalog.get_profile(method["id"])

    def optional_fixture(self):
        """Small full-rank spatial EEG with true named standard electrodes."""
        names = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
                 "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
        rng = np.random.default_rng(97)
        t = np.arange(14500) / 250
        sources = rng.normal(scale=2e-6, size=(22, len(t)))
        for i in range(22):
            sources[i] += 8e-6 * np.sin(2 * np.pi * (8 + i) * t + i / 5)
        signal = rng.normal(size=(22, 22)) @ sources / np.sqrt(22)
        events = np.array([[1250 + 1000 * i, 0, 769 + i % 4] for i in range(12)], dtype=np.int64)
        return {"signal": signal, "names": names, "events": events}

    @unittest.skipUnless(find_spec("autoreject") is not None, "optional autoreject package absent")
    def test_real_optional_autoreject_marks_without_changing_input(self):
        fixture = self.optional_fixture()
        _, result = self.run_profile("mi-autoreject-qc-v1", max_trials=12, **fixture)
        details = result["qc"]["method_details"]
        self.assertFalse(details["signal_modified"])
        self.assertEqual(details["n_jobs"], 1)
        self.assertTrue(all(np.isfinite(v) and v > 0 for v in details["thresholds_v"].values()))
        self.assertTrue(np.isfinite(details["channel_labels"]).all())
        expected = np.array([fixture["signal"][:, e[0]:e[0] + 1000] for e in fixture["events"]])
        np.testing.assert_array_equal(self.data(result), expected)

    @unittest.skipUnless(find_spec("mne_icalabel") is not None and find_spec("torch") is not None,
                         "optional ICLabel/PyTorch packages absent")
    def test_real_optional_iclabel_computes_finite_output_and_saves_ica_state(self):
        import mne
        _, result = self.run_profile("mi-iclabel-v1", max_trials=12, **self.optional_fixture())
        details = result["qc"]["method_details"]
        self.assertTrue(np.isfinite(self.data(result)).all())
        self.assertEqual(result["output"]["shape"], [12, 22, 1000])
        self.assertEqual(details["reference"], "common_average")
        self.assertEqual(len(details["component_labels"]), 15)
        self.assertTrue(np.isfinite(details["component_scores"]).all())
        self.assertTrue(np.isfinite(details["ica_unmixing_matrix"]).all())
        self.assertEqual(result["state_files"][0]["file"], "decomposition-ica.fif")
        path = service.ROOT / "artifacts" / result["artifact_id"] / "decomposition-ica.fif"
        self.assertEqual(service._hash(path), result["state_files"][0]["sha256"])
        restored = mne.preprocessing.read_ica(path, verbose="ERROR")
        self.assertEqual(restored.n_components_, 15)


if __name__ == "__main__":
    unittest.main()
