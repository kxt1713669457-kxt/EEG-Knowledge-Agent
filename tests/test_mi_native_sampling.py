"""Native-rate MI epochs preserve the source rate, channel count and event map.

Fixtures are continuous numeric records or small real FIF containers written to
temporary directories; no dataset download, model weight or service is used.
"""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.agent.skills.mi_preprocessing import catalog, service
from app.workspace import samples, store


NATIVE = "mi-native-cue4s-v1"


class MINativeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for owner, name, value in ((service, "ROOT", self.root / "mi"),
                                   (samples, "SAMPLE_ROOT", self.root / "samples"),
                                   (store, "DB_PATH", self.root / "workspace.sqlite3")):
            item = patch.object(owner, name, value)
            item.start()
            self.addCleanup(item.stop)
        memory = patch.object(service, "_memory_guard")
        memory.start()
        self.addCleanup(memory.stop)

    def fixture(self, sfreq=160, channels=64):
        t = np.arange(round(16 * sfreq)) / sfreq
        x = np.array([(1 + i / channels) * 10e-6 * np.sin(2 * np.pi * 10 * t)
                      for i in range(channels)])
        names = [f"EEG-{i}" for i in range(channels)]
        events = np.array([[round(4 * sfreq), 0, 1], [round(10 * sfreq), 0, 2]], dtype=np.int64)
        return x, names, events

    def register(self, sfreq=160, channels=64, metadata=None, signal=None):
        x, names, events = self.fixture(sfreq, channels)
        blob = BytesIO()
        np.savez_compressed(blob, signal=x if signal is None else signal, ch_names=names,
                            events=events, sfreq=sfreq, unit="V")
        blob.seek(0)
        row = service.register_recording(blob, "continuous.npz", "MI_GENERIC", json.dumps(
            {"cue_event_codes": [1, 2]} if metadata is None else metadata))
        return row, x, events

    def data(self, result):
        with np.load(service.data_path(result["artifact_id"]), allow_pickle=False) as arrays:
            return arrays["data"]

    def fif(self):
        import mne
        x, names, _ = self.fixture()
        raw = mne.io.RawArray(x, mne.create_info(names, 160, "eeg"), verbose="ERROR")
        raw.set_annotations(mne.Annotations([0, 4, 10], [4, 4, 4], ["T0", "T1", "T2"]))
        path = self.root / "source_raw.fif"
        raw.save(path, fmt="double", overwrite=False, verbose="ERROR")
        raw.close()
        return path, x

    def register_fif(self, path, mapping, codes):
        metadata = {"annotation_event_map": mapping, "cue_event_codes": codes}
        with path.open("rb") as stream:
            return service.register_recording(stream, path.name, "MI_GENERIC", json.dumps(metadata))

    def test_three_native_profiles_are_independent_of_original_seven_protocols(self):
        rows = catalog.profiles()
        native = [p for p in rows if p["sampling_policy"] == "native"]
        old = [p for p in rows if p["sampling_policy"] == "fixed_250" and not p.get("family")]
        self.assertEqual(len(native), 3)
        self.assertEqual(len(old), 7)
        self.assertTrue(all(p["model_compatibility"] == "requires_validation" for p in native))
        self.assertEqual({p["method"] for p in native}, {"identity", "bandpass"})

    def test_160hz_64channels_identity_preserves_all_samples_and_no_classification(self):
        row, x, events = self.register()
        source_hash = row["sha256"]
        result = service.preprocess_recording(row["recording_id"], NATIVE, 2)
        expected = np.array([x[:, e[0]:e[0] + 640] for e in events])
        np.testing.assert_array_equal(self.data(result), expected)
        self.assertEqual(result["output"]["shape"], [2, 64, 640])
        self.assertEqual(result["output"]["sampling_rate_hz"], 160)
        self.assertEqual(result["protocol"]["sampling_policy"], "native")
        self.assertEqual(result["protocol"]["points"], 640)
        self.assertIsNone(result["protocol"]["resample"])
        self.assertIsNone(result["protocol"]["normalization"])
        self.assertEqual(result["source_sha256"], source_hash)
        self.assertEqual(service._hash(service._path(service.recording(row["recording_id"]))), source_hash)
        sample = samples.active_entries()[result["sample_id"]]
        self.assertFalse(sample["compatible"])
        self.assertEqual(sample["shape"], [2, 64, 640])
        self.assertEqual(sample["sampling_rate_hz"], 160)
        self.assertEqual(result["profile"]["model_compatibility"], "requires_validation")
        self.assertIn("不兼容", result["scope"])
        self.assertNotIn("model", result)
        text = service.artifact_document(result)["text"]
        self.assertIn("160Hz", text)
        self.assertNotIn("，250Hz，V", text)

    def test_native_profiles_use_four_seconds_at_each_supported_rate_without_resampling(self):
        for sfreq in (100, 128, 160, 250, 500, 1000):
            with self.subTest(sfreq=sfreq):
                row, x, events = self.register(sfreq=sfreq)
                result = service.preprocess_recording(row["recording_id"], NATIVE, 1)
                points = round(4 * sfreq)
                self.assertEqual(result["output"]["shape"], [1, 64, points])
                self.assertEqual(result["protocol"]["sampling_rate_hz"], sfreq)
                self.assertIsNone(result["protocol"]["resample"])
                np.testing.assert_array_equal(self.data(result)[0], x[:, events[0, 0]:events[0, 0] + points])

    def test_original_seven_profiles_still_reject_160hz_before_any_optional_processing(self):
        row, _, _ = self.register()
        for profile in catalog.PROFILES:
            if profile.get("sampling_policy") == "native" or profile.get("family"):
                continue
            with self.subTest(profile=profile["id"]), patch.object(service, "get_profile", return_value=profile):
                with self.assertRaisesRegex(ValueError, "固定250Hz"):
                    service.preprocess_recording(row["recording_id"], profile["id"], 2)
        self.assertEqual(samples.active_entries(), {})

    def test_fractional_four_second_sample_count_is_not_silently_rounded(self):
        row, _, _ = self.register(sfreq=160.1)
        with self.assertRaisesRegex(ValueError, "整数采样点"):
            service.preprocess_recording(row["recording_id"], NATIVE, 1)
        self.assertEqual(samples.active_entries(), {})

    def test_native_record_rate_outside_admission_range_is_rejected(self):
        for sfreq in (99, 1001):
            with self.subTest(sfreq=sfreq), self.assertRaisesRegex(ValueError, "100.*1000"):
                self.register(sfreq=sfreq)

    def test_160hz_filter_profiles_keep_native_channels_and_attenuate_50hz(self):
        x, _, _ = self.fixture()
        t = np.arange(x.shape[-1]) / 160
        x[:] = 10e-6 * (np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t))
        for profile in ("mi-native-mu-beta-v1", "mi-native-broadband-v1"):
            with self.subTest(profile=profile):
                row, _, events = self.register(signal=x)
                result = service.preprocess_recording(row["recording_id"], profile, 2)
                window = self.data(result)[0, 0]
                time_window = t[events[0, 0]:events[0, 0] + 640]
                amplitude = lambda freq: 2 * abs(np.dot(window, np.exp(-2j * np.pi * freq * time_window))) / len(window)
                self.assertGreater(amplitude(10), 8e-6)
                # 4–40Hz intentionally has a gentler transition at 50Hz than
                # 8–32Hz; both must remove at least 95% of this contaminant.
                self.assertLess(amplitude(50), 5e-7)
                self.assertEqual(result["output"]["shape"], [2, 64, 640])
                self.assertIsNone(result["protocol"]["resample"])

    def test_real_fif_string_annotations_map_to_declared_codes_and_keep_source_positions(self):
        path, x = self.fif()
        source_hash = service._hash(path)
        row = self.register_fif(path, {"T1": 12, "T2": 42}, [12, 42])
        self.assertEqual(row["event_count"], 2)
        result = service.preprocess_recording(row["recording_id"], NATIVE, 2)
        self.assertEqual([e["cue_code"] for e in result["events"]], [12, 42])
        self.assertEqual([e["source_sample"] for e in result["events"]], [640, 1600])
        self.assertEqual(result["protocol"]["annotation_event_map"], {"T1": 12, "T2": 42})
        np.testing.assert_array_equal(self.data(result)[0], x[:, 640:1280])
        self.assertEqual(service._hash(path), source_hash)

    def test_string_annotations_without_explicit_map_do_not_get_fabricated_internal_codes(self):
        path, _ = self.fif()
        with path.open("rb") as stream, self.assertRaisesRegex(ValueError, "未找到运动想象"):
            service.register_recording(stream, path.name, "MI_GENERIC", '{"cue_event_codes":[1,2]}')
        self.assertEqual(store.all_records("mi_recording"), [])

    def test_unmatched_annotation_mapping_is_explicit_failure_with_upload_rollback(self):
        path, _ = self.fif()
        with self.assertRaisesRegex(ValueError, "未包含映射"):
            self.register_fif(path, {"missing": 99}, [99])
        self.assertEqual(list((service.ROOT / "recordings").glob("*")), [])
        self.assertTrue(path.is_file())

    def test_annotation_map_strict_types_size_and_names(self):
        invalid = ({}, None, [], {"T1": True}, {"T1": 1.0}, {"T1": "1"},
                   {"T1": -1}, {"T1": 65536}, {"": 1}, {" ": 1},
                   {"x" * 81: 1}, {"T\n1": 1}, {f"T{i}": i for i in range(33)})
        for mapping in invalid:
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, "annotation_event_map"):
                service._metadata(json.dumps({"annotation_event_map": mapping}))
        self.assertEqual(service._metadata('{"annotation_event_map":{"T1":0,"T2":65535}}')["annotation_event_map"],
                         {"T1": 0, "T2": 65535})

    def test_npz_event_codes_are_already_explicit_and_do_not_accept_annotation_map(self):
        with self.assertRaisesRegex(ValueError, "连续NPZ使用events"):
            self.register(metadata={"annotation_event_map": {"T1": 1, "T2": 2}, "cue_event_codes": [1, 2]})
        self.assertEqual(store.all_records("mi_recording"), [])


if __name__ == "__main__":
    unittest.main()
