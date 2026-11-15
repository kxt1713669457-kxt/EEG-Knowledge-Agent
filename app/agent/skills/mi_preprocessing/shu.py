"""SHU-MI's released, already epoched sessions -> versioned CTNet/DWDaNet splits.

This is a data preparation adapter, not a continuous-raw reader or a classifier.
Only the two audited local preprocessing protocols are supported. Original uploads
and the user's CxxT/E files are never overwritten or executed.
"""
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import time
import uuid

import numpy as np
from scipy.io import loadmat, savemat, whosmat

from app.workspace import samples, store


MAX_FILE_BYTES = 128 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**2
MAX_DECODED_BYTES = 512 * 1024**2
SPLIT_IDS = {"shu-session1to4-vs5-v1", "shu-mixed-session-stratified70-v1"}
NAME = re.compile(r"^sub-(\d{3})_ses-(0[1-5])_task_motorimagery_eeg\.mat$")
CHANNELS = ["FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
            "CZ", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6",
            "PZ", "P3", "P4", "T5", "T6", "PO3", "PO4", "OZ", "O1", "O2"]
SOURCES = {
    "dataset": "https://figshare.com/articles/software/shu_dataset/19228725",
    "sampling_and_reference": "https://ndownloader.figshare.com/files/34166160",
    "channel_order_and_unit": "https://ndownloader.figshare.com/files/34166154",
    "label_semantics": "https://ndownloader.figshare.com/files/34166166",
    "representation": "https://ndownloader.figshare.com/files/36729006",
}
ORIGINAL_SCRIPTS = {
    "shu-session1to4-vs5-v1": {
        "file": "preprocessing_for_shuMi_direct.py",
        "sha256": "9caa3611062a585a64fd3c0f4bf5d2665bcec4d9e62cb56ab5e71bee5282f018"},
    "shu-mixed-session-stratified70-v1": {
        "file": "preprocessing_for_shuMi_no_cross_session.py",
        "sha256": "326ccba479e6ed2c6cafc249e71cd714d2886dc29c973f0dc1d0b07dea66abc5"},
}


def _split_id(value):
    if not isinstance(value, str) or value not in SPLIT_IDS:
        raise ValueError("SHU仅支持session1–4/5或混合session分层70/30两个已登记划分版本。")
    return value


def _header(path):
    """Check MAT variable headers before loading potentially compressed arrays."""
    try:
        rows = whosmat(path)
    except (OSError, ValueError, TypeError, NotImplementedError) as exc:
        raise ValueError("SHU文件必须是可读取的MAT数值文件（不支持v7.3/HDF5）。") from exc
    names = [name for name, _, _ in rows]
    if len(names) != len(set(names)):
        raise ValueError("SHU MAT含重复变量名。")
    entries = {name: (tuple(shape), kind) for name, shape, kind in rows}
    if not {"data", "labels"}.issubset(entries):
        raise ValueError("SHU源session MAT必须明确包含data和labels，不能猜变量名或接收派生Cxx文件。")
    shape, kind = entries["data"]
    if (kind not in {"single", "double"} or len(shape) != 3 or shape[0] < 1
            or shape[1:] != (32, 1000)):
        raise ValueError("SHU data必须是浮点[试次,32,1000]，不猜轴、不二次切段。")
    label_shape, label_kind = entries["labels"]
    if label_kind not in {"single", "double", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"} or label_shape not in {(1, shape[0]), (shape[0], 1)}:
        raise ValueError("SHU labels必须是与试次数一致的1×N或N×1实数标签。")
    # Four float32-sized copies cover loaded sessions, concatenation and split
    # outputs; add double input conversion and label/index overhead conservatively.
    peak = math.prod(shape) * (24 if kind == "double" else 16) + shape[0] * 128
    if peak > MAX_DECODED_BYTES:
        raise ValueError("SHU声明的解码/处理内存估计超过512MiB。")
    return {"shape": list(shape), "dtype": kind, "estimated_peak_bytes": peak}


def _load_session(path):
    description = _header(path)
    try:
        values = loadmat(path, variable_names=["data", "labels"])
    except (OSError, ValueError, TypeError, NotImplementedError) as exc:
        raise ValueError("SHU MAT数据读取失败。") from exc
    data, labels = values["data"], values["labels"]
    if (not isinstance(data, np.ndarray) or data.dtype.kind != "f"
            or list(data.shape) != description["shape"] or not np.isfinite(data).all()):
        raise ValueError("SHU data需为有限实浮点数，形状必须与声明一致。")
    if (not isinstance(labels, np.ndarray) or labels.dtype.kind not in "fiu"
            or labels.size != len(data) or not np.isfinite(labels).all()
            or not np.isin(labels, [1, 2]).all()):
        raise ValueError("SHU labels仅接受真实1/2标签；不自动重映射、不输入模型。")
    return np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.int64).reshape(-1)


def _remove_records(pairs):
    with store.transaction() as connection:
        for kind, identity in pairs:
            connection.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, identity))


def _clean_folder(folder, names):
    """Only remove files owned by this call; never recurse into source datasets."""
    if folder.exists():
        for name in names:
            (folder / name).unlink(missing_ok=True)
        folder.rmdir()


def register_sessions(items, split_id="shu-session1to4-vs5-v1"):
    from . import service
    split_id = _split_id(split_id)
    if not isinstance(items, list) or len(items) != 5:
        raise ValueError("请一次登记同一被试session1–5的5份SHU源MAT。")
    parsed = []
    for stream, filename in items:
        match = NAME.fullmatch(filename) if isinstance(filename, str) else None
        if match is None or not 1 <= int(match[1]) <= 25:
            raise ValueError("SHU文件名须为sub-001_ses-01_task_motorimagery_eeg.mat格式，被试1–25、session1–5。")
        parsed.append((int(match[1]), int(match[2]), stream, filename))
    if len({r[0] for r in parsed}) != 1 or {r[1] for r in parsed} != set(range(1, 6)):
        raise ValueError("SHU必须为同一被试且session1–5各一份，不接受缺失或重复session。")
    parsed.sort(key=lambda row: row[1])
    identity = uuid.uuid4().hex
    folder = service.ROOT / "shu-recordings" / identity
    owned = []
    with service._exclusive():
        service._memory_guard(768 * 1024**2)
        folder.mkdir(parents=True, exist_ok=False)
        try:
            sources, total, estimate, trials = [], 0, 0, 0
            for subject, session, stream, filename in parsed:
                path = folder / filename
                owned.append(filename)
                size = 0
                with path.open("xb") as output:
                    for block in iter(lambda: stream.read(1024**2), b""):
                        if not isinstance(block, bytes):
                            raise ValueError("SHU上传流必须是二进制。")
                        size += len(block)
                        total += len(block)
                        if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                            raise ValueError("SHU单文件及5份合计上传均不得超过128MiB。")
                        output.write(block)
                if not size:
                    raise ValueError("SHU源session文件不能为空。")
                header = _header(path)
                estimate += header["estimated_peak_bytes"]
                if estimate > MAX_DECODED_BYTES:
                    raise ValueError("SHU五session合计解码/处理内存估计超过512MiB。")
                service._memory_guard(max(768 * 1024**2, estimate * 2))
                data, labels = _load_session(path)
                trials += len(data)
                sources.append({"file": filename, "storage_name": filename,
                                "sha256": service._hash(path), "session": session,
                                "shape": list(data.shape), "bytes": size, "dtype": header["dtype"], "prepared_dtype": "float32",
                                "class_counts": {str(c): int(np.sum(labels == c)) for c in [1, 2]}})
                del data, labels
            aggregate = sha256("\n".join(f"{x['session']}:{x['sha256']}" for x in sources).encode()).hexdigest()
            row = {"id": identity, "recording_id": identity, "file": f"SHU sub-{parsed[0][0]:03d} sessions1-5",
                   "input_kind": "shu_epoched_sessions", "dataset_id": "SHU_MI", "subject": parsed[0][0],
                   "channels": 32, "channel_names": CHANNELS.copy(), "sampling_rate_hz": 250,
                   "duration_seconds": trials * 4, "event_count": trials, "unit": "uV", "bytes": total,
                   "metadata": {"split_id": split_id, "evidence": SOURCES.copy(),
                                "representation": "author-released epoched MI windows; not continuous amplifier raw",
                                "original_reference": "M1 unipolar", "channel_order_evidence": "official channels.tsv order; no per-MAT channel metadata"},
                   "source_files": sources, "sha256": aggregate, "created_at": store.now(), "deleted_at": None}
            store.write("mi_recording", identity, row)
            return row
        except Exception:
            _remove_records([("mi_recording", identity)])
            _clean_folder(folder, owned)
            raise


def _indices(labels, subject, split_id, sessions):
    if split_id == "shu-session1to4-vs5-v1":
        return np.flatnonzero(sessions < 5), np.flatnonzero(sessions == 5), None
    rng = np.random.RandomState(2026 + subject)
    train, test = [], []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("SHU分层划分每类至少需要2个源试次。")
        rng.shuffle(indices)
        count = max(1, min(int(round(len(indices) * .30)), len(indices) - 1))
        test.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    train, test = np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test, 2026 + subject


def preprocess(row, profile, max_trials=32):
    """Called by the parent service while its cross-process execution lock is held."""
    from . import service
    started = time.perf_counter()
    if type(max_trials) is not int or not 1 <= max_trials <= service.MAX_TRIALS:
        raise ValueError("max_trials需为1至288整数。")
    if row.get("input_kind") != "shu_epoched_sessions" or profile.get("family") not in {"dwdanet", "dwtnet"}:
        raise ValueError("SHU源session仅支持已登记DWDaNet/DWTNet原数据准备方案。")
    if profile.get("method") not in {None, "identity", "shu_sessions", "shu_split"} or profile.get("band"):
        raise ValueError("SHU原数据准备方案不额外滤波、去伪迹或归一化。")
    identity = service._identity(row["recording_id"])
    split_id = _split_id(row["metadata"]["split_id"])
    subject = row.get("subject")
    if type(subject) is not int or not 1 <= subject <= 25:
        raise ValueError("SHU被试身份无效。")
    sources = sorted(row["source_files"], key=lambda source: source["session"])
    if len(sources) != 5 or [s["session"] for s in sources] != list(range(1, 6)):
        raise ValueError("SHU源记录必须完整包含session1–5。")
    base = (service.ROOT / "shu-recordings" / identity).resolve()
    if base.parent != (service.ROOT / "shu-recordings").resolve():
        raise ValueError("SHU源路径超出登记目录。")
    estimate = 0
    for source in sources:
        path = (base / source["storage_name"]).resolve()
        if path.parent != base or not path.is_file() or service._hash(path) != source["sha256"]:
            raise ValueError("SHU源文件不存在或哈希不一致。")
        header = _header(path)
        if header["shape"] != source["shape"]:
            raise ValueError("SHU源文件形状与登记信息不一致。")
        estimate += header["estimated_peak_bytes"]
    expected = sha256("\n".join(f"{x['session']}:{x['sha256']}" for x in sources).encode()).hexdigest()
    if expected != row["sha256"] or estimate > MAX_DECODED_BYTES:
        raise ValueError("SHU组合源哈希或512MiB处理内存限制不通过。")
    service._memory_guard(max(768 * 1024**2, estimate * 2))
    chunks, label_chunks, mapping = [], [], []
    for source in sources:
        data, labels = _load_session(base / source["storage_name"])
        chunks.append(data)
        label_chunks.append(labels)
        mapping.extend({"subject": subject, "session": source["session"], "trial_index": i,
                        "source_file": source["file"], "source_sha256": source["sha256"]} for i in range(len(data)))
    all_data = np.concatenate(chunks)
    del chunks, data
    all_labels = np.concatenate(label_chunks)
    del label_chunks, labels
    sessions = np.asarray([m["session"] for m in mapping])
    train_indices, test_indices, seed = _indices(all_labels, subject, split_id, sessions)
    # Only the inference sample is converted to V. Split exports retain the
    # original float32 microvolt convention used by the audited training scripts.
    inference_indices = test_indices[:max_trials]
    output = np.asarray(all_data[inference_indices] * np.float32(1e-6), dtype=np.float32)
    sample_id = None
    sample = None
    artifact_id = uuid.uuid4().hex
    folder = service.ROOT / "artifacts" / artifact_id
    folder.mkdir(parents=True, exist_ok=False)
    owned = ["epochs.npz", "result.json", "train.mat", "test.mat"]
    try:
        for role, indices in [("train", train_indices), ("test", test_indices)]:
            savemat(folder / f"{role}.mat", {"data": all_data[indices], "label": all_labels[indices, None]},
                    do_compression=False)
        np.savez_compressed(folder / "epochs.npz", data=output)
        with (folder / "epochs.npz").open("rb") as stream:
            sample = samples.register(stream, f"shu-sub-{subject:03d}-test.npz", 250, "V", "SHU_MI")
        sample_id = sample["sample_id"]
        notes = ["源MAT为作者已切好的4秒运动想象试次，不冒称连续raw、不再次按提示切段。",
                 "train.mat/test.mat保留原始float32微伏uV；推理样本仅执行uV×1e-6→V，没有滤波、重参考或归一化。",
                 "原模型mean/std如以uV拟合，接收V输入时须同步换算；本工具没有拟合或应用分类归一化。",
                 "当前无已核验的32通道SHU分类权重；形状正确不代表已完成识别或取得准确率。",
                 "DWDaNet与DWTNet使用相同原始数据准备流程，family仅区分来源能力，不虚构数值差异。"]
        if seed is not None:
            notes.append("此版本混合同一被试所有session后分层随机划分，不能解释为跨session泛化评估。")
        scope = " ".join(notes)
        sample.update(preprocessing_artifact_id=artifact_id, preprocessing_profile_id=profile["id"],
                      preprocessing_compatibility="requires_validation", channel_names=CHANNELS.copy(),
                      source_recording_sha256=row["sha256"], preprocessing_scope=scope)
        store.write("sample", sample_id, sample)
        split = {"id": split_id, "seed": seed, "test_ratio": .30 if seed is not None else None,
                 "source_order": "session1..5, then original zero-based trial order",
                 "train_indices": train_indices.tolist(), "test_indices": test_indices.tolist(),
                 "train_source_trials": [mapping[i] for i in train_indices],
                 "test_source_trials": [mapping[i] for i in test_indices],
                 "labels_used_for_split_only": True, "labels_used_as_model_input": False}
        qc = service._qc(output, 250)
        result = {"artifact_id": artifact_id, "sample_id": sample_id, "kind": "mi_preprocessing",
                  "created_at": store.now(), "recording_id": identity, "source_file": row["file"],
                  "source_sha256": row["sha256"], "source_files": sources, "dataset_id": "SHU_MI", "profile": profile,
                  "protocol": {"representation": "author_epoched_sessions", "source_unit": "uV", "output_unit": "V",
                               "sampling_rate_hz": 250, "points": 1000, "window_seconds": 4,
                               "channel_order": CHANNELS.copy(), "evidence": SOURCES.copy(),
                               "reference": "recorded M1 unipolar", "band_hz": None, "filter": None,
                               "baseline": None, "normalization": None, "normalization_not_applied": True,
                               "resample": None, "labels_used": False, "split": split,
                               "audited_original_script": ORIGINAL_SCRIPTS[split_id],
                               "workflow_family": profile["family"],
                               "workflow_contract_sha256": profile.get("workflow_contract_sha256"),
                               "normalization_stage": "classifier_adapter_only_not_applied_here",
                               "augmentation_stage": "training_only_not_applied_at_inference"},
                  "output": {"shape": list(output.shape), "sampling_rate_hz": 250, "unit": "V",
                             "channel_names": CHANNELS.copy(), "sha256": service._hash(folder / "epochs.npz")},
                  "exports": {role: {"file": f"{role}.mat", "shape": [len(indices), 32, 1000], "unit": "uV",
                                    "sha256": service._hash(folder / f"{role}.mat")}
                              for role, indices in [("train", train_indices), ("test", test_indices)]},
                  "events": [mapping[i] for i in inference_indices], "requested_trials": max_trials,
                  "total_cue_events": len(all_data), "qc": {"before": qc, "after": qc,
                      "dropped_trials": [], "warnings": notes, "method_details": {"signal_modified": False,
                      "unit_conversion": "uV*1e-6 -> V for inference only", "normalization_applied": False}},
                  "versions": {"numpy": np.__version__, "scipy": version("scipy")},
                  "implementation_sha256": service._hash(Path(__file__)),
                  "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "scope": scope,
                  "artifact_url": f"/workspace/mi-preprocessing/artifacts/{artifact_id}",
                  "data_url": f"/workspace/mi-preprocessing/artifacts/{artifact_id}/data"}
        result["split_exports"] = [{"name": result["exports"][role]["file"],
                                    "shape": result["exports"][role]["shape"], "unit": "uV",
                                    "sha256": result["exports"][role]["sha256"]}
                                   for role in ("train", "test")]
        (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        index = {k: result[k] for k in ("artifact_id", "source_file", "sample_id", "created_at", "profile")}
        index.update(recording_id=identity, shape=list(output.shape), dataset_id="SHU_MI", split_id=split_id)
        store.write("mi_artifact", artifact_id, index)
        return result
    except Exception:
        pairs = [("mi_artifact", artifact_id)]
        if sample_id:
            pairs.append(("sample", sample_id))
        _remove_records(pairs)
        if sample is not None:
            samples.registered_path(sample).unlink(missing_ok=True)
        _clean_folder(folder, owned)
        raise
