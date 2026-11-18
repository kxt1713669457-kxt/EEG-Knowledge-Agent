"""Bounded, reproducible continuous-recording -> MI epochs pipeline.

Original uploads are immutable. No labels, arbitrary code, external paths, downloads,
or fitted classifier normalization enter this pipeline. Heavy imports are lazy.
"""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
import time
import uuid
import warnings
import zipfile

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from app.agent.skills.mi_preprocessing.catalog import get_profile, profiles, RESEARCH_METHODS
from app.shared.runtime.logger import PROJECT_ROOT
from app.workspace import store, samples

ROOT = PROJECT_ROOT / "output/runtime/mi-preprocessing"
MAX_FILE_BYTES = 128 * 1024**2
MAX_SIGNAL_BYTES = 256 * 1024**2
MAX_OUTPUT_BYTES = 96 * 1024**2
MAX_TRIALS = 288
DATASETS = {"BCI_IV_2a", "BCI_IV_2b", "MI_GENERIC", "SHU_MI", "OPENBMI_MI"}
EXTENSIONS = {".gdf", ".edf", ".bdf", ".fif", ".npz"}
_execution_lock = Lock()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("记录/产物ID格式不合法。")
    return value


def _memory_guard(estimated_bytes):
    # CPU-only, one record at a time. Enforce a conservative admission bound.
    try:
        import psutil
        if psutil.virtual_memory().available < max(768 * 1024**2, estimated_bytes):
            raise ValueError("可用内存不足；请减少试次数或释放内存后再试。")
    except ImportError:
        pass  # Static size limits still apply without the optional memory probe.


@contextmanager
def _exclusive():
    if not _execution_lock.acquire(blocking=False):
        raise ValueError("已有运动想象预处理任务正在运行；为限制内存，本机一次仅处理一个记录。")
    handle = None
    locked = False
    try:
        # Both import/query APIs can serve this router in separate processes.
        ROOT.mkdir(parents=True, exist_ok=True)
        handle = (ROOT / "execution.lock").open("a+b")
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError:
            raise ValueError("另一服务进程正在预处理；为限制内存，本机一次仅处理一个记录。") from None
        yield
    finally:
        if handle is not None:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        _execution_lock.release()


def _metadata(text):
    if not isinstance(text, str) or len(text) > 64000:
        raise ValueError("元信息JSON过大或格式不合法。")
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        raise ValueError("元信息必须是JSON对象。") from None
    allowed = {"cue_event_codes", "eog_channels", "channel_mapping", "channel_positions", "annotation_event_map"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("元信息仅支持cue_event_codes、eog_channels、channel_mapping、channel_positions、annotation_event_map。")
    annotation_map = value.get("annotation_event_map")
    if "annotation_event_map" in value and (not isinstance(annotation_map, dict) or not 1 <= len(annotation_map) <= 32 or any(
        not isinstance(name, str) or not name.strip() or len(name) > 80 or any(ord(c) < 32 for c in name)
        or type(code) is not int or not 0 <= code <= 65535 for name, code in annotation_map.items()
    )):
        raise ValueError("annotation_event_map需为1至32个真实注释名（1至80字符）到0至65535整数事件码的映射。")
    codes = value.get("cue_event_codes")
    if codes is not None and (not isinstance(codes, list) or not 1 <= len(codes) <= 16
                             or any(type(c) is not int or not 0 <= c <= 65535 for c in codes)):
        raise ValueError("cue_event_codes需为1至16个整数原始事件码。")
    eog = value.get("eog_channels", [])
    if not isinstance(eog, list) or len(eog) > 8 or any(not isinstance(c, str) or len(c) > 80 for c in eog):
        raise ValueError("eog_channels需为真实眼电通道名列表。")
    mapping = value.get("channel_mapping", {})
    if not isinstance(mapping, dict) or len(mapping) > 64 or any(
        not isinstance(k, str) or not isinstance(v, str) or not k or not v or len(k) > 80 or len(v) > 80
        for k, v in mapping.items()
    ):
        raise ValueError("channel_mapping需为原始通道名到标准名的映射。")
    positions = value.get("channel_positions", {})
    if not isinstance(positions, dict) or len(positions) > 64:
        raise ValueError("channel_positions需为通道名到米制三维坐标的映射。")
    for name, xyz in positions.items():
        if not isinstance(name, str) or not isinstance(xyz, list) or len(xyz) != 3 or any(
            type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 1 for v in xyz
        ):
            raise ValueError("电极位置需为有限的米制[x,y,z]，不得猜测未知通道位置。")
    return value


def _read_npz(path):
    import mne
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_SIGNAL_BYTES:
            raise ValueError("连续NPZ展开超过256MiB限制。")
        if "signal.npy" not in archive.namelist():
            raise ValueError("连续NPZ必须含signal[通道,时间]；已切段data请使用原样本入口。")
        required = {"signal", "sfreq", "ch_names", "events", "unit"}
        names_in_zip = archive.namelist()
        if len(names_in_zip) != len(set(names_in_zip)) or not {v + ".npy" for v in required}.issubset(names_in_zip):
            raise ValueError("连续NPZ缺少必需数组或含重复条目。")
        for name in required:
            with archive.open(name + ".npy") as header:
                version = np.lib.format.read_magic(header)
                if version not in {(1, 0), (2, 0)}:
                    raise ValueError("连续NPZ仅接受标准NPY v1/v2数组。")
                shape, _, dtype = (np.lib.format.read_array_header_1_0(header) if version == (1, 0)
                                   else np.lib.format.read_array_header_2_0(header))
            byte_count = math.prod(shape) * dtype.itemsize
            valid_header = (dtype.kind in "fiu" and len(shape) == 2 and 1 <= shape[0] <= 64 and shape[1] > 0
                            and math.prod(shape) * 8 <= MAX_SIGNAL_BYTES) if name == "signal" else (
                dtype.kind in "fiu" and shape == () if name == "sfreq" else
                dtype.kind == "U" and len(shape) == 1 and 1 <= shape[0] <= 64 and dtype.itemsize <= 320 if name == "ch_names" else
                dtype.kind in "iu" and len(shape) == 2 and shape[1] == 3 and 1 <= shape[0] <= 20000 if name == "events" else
                dtype.kind == "U" and shape == () and dtype.itemsize <= 16)
            if not valid_header or byte_count > MAX_SIGNAL_BYTES or byte_count > archive.getinfo(name + ".npy").file_size:
                raise ValueError("连续NPZ数组头的类型、形状或声明体积超限。")
    with np.load(path, allow_pickle=False) as arrays:
        if not {"signal", "sfreq", "ch_names", "events", "unit"}.issubset(arrays.files):
            raise ValueError("连续NPZ需含signal、sfreq、ch_names、events、unit（V/uV）。")
        sfreq = float(arrays["sfreq"].item())
        names = arrays["ch_names"].tolist()
        signal = np.asarray(arrays["signal"], dtype=np.float64)
        events = arrays["events"]
        unit = str(arrays["unit"].item())
        if not isinstance(names, list) or len(names) != signal.shape[0] or any(
            not isinstance(c, str) or not c or len(c) > 80 for c in names
        ) or len(set(names)) != len(names):
            raise ValueError("ch_names必须与signal通道一一对应且唯一。")
        if not np.isfinite(sfreq) or not 100 <= sfreq <= 1000 or unit not in {"V", "uV"}:
            raise ValueError("连续NPZ需明确100–1000Hz采样率及V/uV单位。")
        if events.ndim != 2 or events.shape[1] != 3 or events.dtype.kind not in "iu" or len(events) > 20000:
            raise ValueError("events必须是整数[N,3]：原始采样位置、0、原始事件码。")
        if np.any(events[:, 0] < 0) or np.any(events[:, 0] >= signal.shape[1]) or np.any(events[:, 1] != 0) or np.any(events[:, 2] < 0):
            raise ValueError("事件位置/原始事件码无效。")
        if unit == "uV":
            signal *= 1e-6
        raw = mne.io.RawArray(signal, mne.create_info(names, sfreq, ch_types="eeg"), verbose="ERROR")
        events = np.asarray(events[np.argsort(events[:, 0], kind="stable")], dtype=int)
    return raw, events


def _open(path, dataset_id, metadata):
    import mne
    if path.suffix == ".npz":
        if metadata.get("annotation_event_map"):
            raise ValueError("连续NPZ使用events中的原始整数码；annotation_event_map仅适用于带字符串注释的原生记录。")
        raw, events = _read_npz(path)
    else:
        reader = {".gdf": mne.io.read_raw_gdf, ".edf": mne.io.read_raw_edf,
                  ".bdf": mne.io.read_raw_bdf, ".fif": mne.io.read_raw_fif}[path.suffix]
        raw = reader(path, preload=False, verbose="ERROR")
        # MNE's default numeric annotation IDs are not the original GDF codes.
        numeric = {str(a): int(a) for a in raw.annotations.description if re.fullmatch(r"\d+", str(a))}
        event_map = metadata.get("annotation_event_map") or numeric
        try:
            events, _ = mne.events_from_annotations(raw, event_id=event_map, verbose="ERROR") if event_map else (np.empty((0, 3), int), {})
        except ValueError:
            raw.close()
            raise ValueError("记录未包含映射指定的注释事件；请核对annotation_event_map。") from None
    if not 1 <= len(raw.ch_names) <= 64 or raw.n_times * len(raw.ch_names) * 8 > MAX_SIGNAL_BYTES:
        raw.close()
        raise ValueError("记录超过64通道或256MiB解码上限；请先按会话拆分。")
    eog_names = set(metadata.get("eog_channels", []))
    if not eog_names.issubset(raw.ch_names):
        raw.close()
        raise ValueError("声明的EOG通道不在原始记录中。")
    eog_names.update(c for c, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eog" or "eog" in c.casefold())
    if eog_names:
        raw.set_channel_types({c: "eog" for c in eog_names}, verbose="ERROR")
    mapping = metadata.get("channel_mapping", {})
    if not set(mapping).issubset(raw.ch_names) or len(set(mapping.get(c, c) for c in raw.ch_names)) != len(raw.ch_names):
        raw.close()
        raise ValueError("通道映射含未知通道或产生重复名称。")
    if mapping:
        raw.rename_channels(mapping)
    positions = metadata.get("channel_positions", {})
    if positions:
        if not set(positions).issubset(raw.ch_names):
            raw.close()
            raise ValueError("电极位置含未知通道。")
        raw.set_montage(mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head"), on_missing="ignore", verbose="ERROR")
    montage_name = "colin27_1020" if "colin27_1020" in mne.channels.get_builtin_montages() else "standard_1020"
    if not positions and (mapping or all(c in mne.channels.make_standard_montage(montage_name).ch_names
                        for c, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg")):
        raw.set_montage(montage_name, on_missing="ignore", verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
    expected = {"BCI_IV_2a": 22, "BCI_IV_2b": 3}.get(dataset_id)
    if not len(picks) or expected and len(picks) != expected:
        raw.close()
        raise ValueError("EEG通道数与数据集不符；必须显式区分EEG/EOG，不自动补通道。")
    if dataset_id == "OPENBMI_MI" and (len(picks) != 62 or raw.info["sfreq"] != 1000):
        raw.close()
        raise ValueError("OpenBMI requires 62 EEG channels and 1000 Hz.")
    codes = metadata.get("cue_event_codes") or ([1, 2] if dataset_id == "OPENBMI_MI" else [769, 770, 783] if dataset_id == "BCI_IV_2b" else [769, 770, 771, 772, 783])
    events = events[np.isin(events[:, 2], codes)]
    if not len(events):
        raw.close()
        raise ValueError("未找到运动想象提示事件；通用数据需在元信息声明cue_event_codes，不能猜时间窗。")
    if len(np.unique(events[:, 0])) != len(events):
        raw.close()
        raise ValueError("同一采样位置存在重复提示事件，需要先核对事件协议。")
    return raw, events, picks


def recording(identity):
    row = store.read("mi_recording", _identity(identity))
    if not row or row.get("deleted_at"):
        raise ValueError("原始记录未登记或已移出目录。")
    return row


def _path(row):
    path = (ROOT / "recordings" / row["storage_name"]).resolve()
    if path.parent != (ROOT / "recordings").resolve() or not path.is_file():
        raise ValueError("原始记录不可读取。")
    return path


def catalog():
    rows = [{k: v for k, v in row.items() if k not in {"storage_name", "metadata"}}
            for row in store.all_records("mi_recording") if not row.get("deleted_at")]
    artifacts = sorted(store.all_records("mi_artifact"), key=lambda r: r.get("created_at", ""))[-100:]
    return {"profiles": profiles(), "recordings": rows[-100:], "artifacts": artifacts, "methods": list(RESEARCH_METHODS),
            "limits": {"max_upload_bytes": MAX_FILE_BYTES, "max_decoded_bytes": MAX_SIGNAL_BYTES,
                       "max_trials": MAX_TRIALS, "default_trials": 32, "concurrency": 1, "device": "cpu"}}


def register_recording(stream, filename, dataset_id, metadata_json="{}"):
    if dataset_id not in DATASETS:
        raise ValueError("请选择BCI_IV_2a、BCI_IV_2b或MI_GENERIC。")
    if dataset_id == "SHU_MI":
        raise ValueError("SHU源文件已切段；请使用SHU五session入口，不按连续原始记录重新切段。")
    ext = Path(filename or "").suffix.lower()
    if ext not in EXTENSIONS:
        raise ValueError("原始记录仅接受GDF/EDF/BDF/FIF/连续NPZ。")
    metadata = _metadata(metadata_json)
    with _exclusive():
        _memory_guard(768 * 1024**2)
        identity = uuid.uuid4().hex
        folder = ROOT / "recordings"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (identity + ext)
        try:
            size = 0
            with path.open("xb") as target:
                for block in iter(lambda: stream.read(1024**2), b""):
                    size += len(block)
                    if size > MAX_FILE_BYTES:
                        raise ValueError("上传超过128MiB限制；请按记录/会话拆分。")
                    target.write(block)
            if not size:
                raise ValueError("原始记录不能为空。")
            raw, events, picks = _open(path, dataset_id, metadata)
            try:
                row = {"id": identity, "recording_id": identity, "file": Path(filename).name,
                       "storage_name": path.name, "dataset_id": dataset_id, "metadata": metadata,
                       "channels": len(picks), "channel_names": [raw.ch_names[p] for p in picks],
                       "eog_channels": [c for c, k in zip(raw.ch_names, raw.get_channel_types()) if k == "eog"],
                       "sampling_rate_hz": float(raw.info["sfreq"]), "duration_seconds": round(raw.n_times / raw.info["sfreq"], 3),
                       "event_count": len(events), "unit": "V", "sha256": _hash(path), "bytes": size,
                       "acquisition_filter": {"highpass_hz": float(raw.info["highpass"]), "lowpass_hz": float(raw.info["lowpass"])},
                       "created_at": store.now(), "deleted_at": None}
                store.write("mi_recording", identity, row)
                return {k: v for k, v in row.items() if k not in {"storage_name", "metadata"}}
            finally:
                raw.close()
        except Exception:
            path.unlink(missing_ok=True)
            raise


def _qc(data, sfreq):
    # Bounded epoch data only; no full-record FFT or extra full-size copies.
    rms = np.sqrt(np.mean(np.square(data), axis=(0, 2)))
    ptp = np.max(np.ptp(data, axis=-1), axis=0)
    _, psd = welch(data, fs=sfreq, nperseg=min(500, data.shape[-1]), axis=-1)
    freq = np.fft.rfftfreq(min(500, data.shape[-1]), d=1 / sfreq)
    powers = {name: float(np.mean(np.sum(psd[..., (freq >= lo) & (freq < hi)], axis=-1)))
              for name, lo, hi in (("mu", 8, 13), ("beta", 13, 32), ("line50", 49, 51))}
    return {"rms_uv_per_channel": (rms * 1e6).tolist(), "max_peak_to_peak_uv_per_channel": (ptp * 1e6).tolist(),
            "flat_channel_indices": np.flatnonzero(ptp < 1e-12).tolist(),
            "over_200uv_trial_indices": np.flatnonzero(np.any(np.ptp(data, axis=-1) > 200e-6, axis=1)).tolist(),
            "mean_band_psd_sum": powers, "finite": bool(np.isfinite(data).all()),
            "note": "幅值阈值仅为可疑标记；频谱值是PSD频点和，不是分类准确率或去噪真值。"}


def _valid_spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _filter(raw, picks, band, valid):
    if band[1] >= raw.info["sfreq"] / 2:
        raise ValueError("采样率不满足该方案的频率上限，未自动重采样。")
    sos = butter(4, band, btype="bandpass", fs=raw.info["sfreq"], output="sos")
    # Never filter across NaN gaps or acquisition run boundaries.
    boundaries = {0, raw.n_times}
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        if _is_boundary(description):
            boundaries.add(int(round((onset - raw.first_time) * raw.info["sfreq"])))
    for start, stop in _valid_spans(valid):
        limits = sorted({start, stop, *(b for b in boundaries if start < b < stop)})
        for lo, hi in zip(limits, limits[1:]):
            if hi - lo < 32:
                valid[lo:hi] = False
                continue
            raw._data[picks, lo:hi] = sosfiltfilt(sos, raw._data[picks, lo:hi], axis=-1)
    with raw.info._unlock():
        raw.info["highpass"] = max(float(raw.info["highpass"]), band[0])
        raw.info["lowpass"] = min(float(raw.info["lowpass"]), band[1])


def _is_boundary(description):
    return str(description).casefold() in {"32766", "boundary", "bad boundary", "edge boundary"}


def _has_positions(raw, picks):
    return all(np.isfinite(raw.info["chs"][p]["loc"][:3]).all() and
               np.linalg.norm(raw.info["chs"][p]["loc"][:3]) > 0 for p in picks)


def _process(raw, picks, profile, valid):
    import mne
    method = profile["method"]
    extra = {}
    if method == "eog_regression":
        eog = mne.pick_types(raw.info, eeg=False, eog=True, exclude=[])
        if not len(eog):
            raise ValueError("眼电回归需要真实EOG通道；不能把额前EEG猜成EOG。")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        fit = mne.io.RawArray(raw._data[:, valid], raw.info.copy(), verbose="ERROR")
        regression = mne.preprocessing.EOGRegression(picks=list(picks), picks_artifact=list(eog), proj=False).fit(fit)
        regression.apply(fit, copy=False)
        raw._data[np.ix_(picks, np.flatnonzero(valid))] = fit.get_data(picks=picks)
        extra = {"regression_coefficients": regression.coef_.tolist(), "fit_valid_samples": int(valid.sum()),
                 "reference": "common_average", "fit_and_apply_centering": "same valid samples; acquisition gaps excluded",
                 "fit_scope": "本次离线记录的有效片段，无标签；非跨试次在线评估。"}
    elif method == "pyprep_qc":
        if len(picks) < 16:
            raise ValueError("PyPREP方案至少需16个EEG通道；不对3通道2b套空间坏道检测。")
        from pyprep import NoisyChannels
        probe = mne.io.RawArray(raw._data[picks][:, valid], mne.pick_info(raw.info.copy(), picks), verbose="ERROR")
        noisy = NoisyChannels(probe, random_state=97, ransac=False)
        noisy.find_all_bads(ransac=False)
        extra = {"bad_channels": [str(c) for c in noisy.get_bads()], "ransac": False,
                 "signal_modified": False, "fit_scope": "本次离线片段；不使用标签，不插值/删除通道。"}
    elif method == "iclabel":
        if len(picks) < 16 or not _has_positions(raw, picks):
            raise ValueError("ICA+ICLabel需至少16个EEG通道及真实位置；请先提供明确通道映射/位置。")
        if raw.info["sfreq"] <= 200:
            raise ValueError("ICLabel方案需支持1–100Hz，当前采样率不足；不自动上采样。")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        _filter(raw, picks, [1.0, 100.0], valid)
        # ICA fit data is finite and limited to 120 seconds; fixed seed, single CPU path.
        fit_samples = np.flatnonzero(valid)[:int(120 * raw.info["sfreq"])]
        fit = mne.io.RawArray(raw._data[:, fit_samples], raw.info.copy(), verbose="ERROR")
        ica = mne.preprocessing.ICA(n_components=min(15, len(picks) - 1), method="infomax",
            fit_params={"extended": True}, random_state=97, max_iter=128)
        ica.fit(fit, picks=list(picks), decim=2, verbose="ERROR")
        from mne_icalabel import label_components
        labels = label_components(fit, ica, method="iclabel")
        excluded = [i for i, (label, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
                    if label in {"eye blink", "muscle artifact", "heart beat"} and float(prob) >= .90]
        ica.apply(raw, exclude=excluded, verbose="ERROR")
        extra = {"component_labels": labels["labels"], "component_scores": np.asarray(labels["y_pred_proba"]).tolist(),
                 "excluded_components": excluded, "threshold": .90, "seed": 97, "fit_samples": len(fit_samples),
                 "fit_scope": "本次离线记录前至多120秒有效数据；未使用任务标签。",
                 "reference": "common_average", "ica_fit_band_hz": [1, 100],
                 "ica_unmixing_matrix": ica.unmixing_matrix_.tolist(), "ica_mixing_matrix": ica.mixing_matrix_.tolist()}
        extra["_ica_state"] = ica
    if profile["band"]:
        _filter(raw, picks, profile["band"], valid)
    return extra


def preprocess_recording(recording_id, profile_id, max_trials=32):
    if type(max_trials) is not int or not 1 <= max_trials <= MAX_TRIALS:
        raise ValueError("max_trials需为1至288的整数；建议先运行32试次。")
    profile = get_profile(profile_id)
    if profile["method"] == "iclabel" and max_trials > 32:
        raise ValueError("ICA+ICLabel单次最多32试次；请分批并保留独立处理版本。")
    with _exclusive():
        row = recording(recording_id)
        if profile.get("supported_datasets") and row["dataset_id"] not in profile["supported_datasets"]:
            raise ValueError("此原模型流程仅适配2a、2b及已登记SHU session，不适配其他数据域。")
        if row.get("input_kind") == "shu_epoched_sessions":
            if not profile.get("family"):
                raise ValueError("已切段SHU仅使用DWDaNet/DWTNet原流程；不进入连续记录的cue切段方案。")
            from app.agent.skills.mi_preprocessing import shu
            _memory_guard(1536 * 1024**2)
            return shu.preprocess(row, profile, max_trials)
        path = _path(row)
        if _hash(path) != row["sha256"]:
            raise ValueError("原始记录哈希不一致，停止执行。")
        started = time.perf_counter()
        _memory_guard(1024 * 1024**2)
        raw, events, picks = _open(path, row["dataset_id"], row["metadata"])
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                return _run(raw, events, picks, row, profile, max_trials, started, caught)
        finally:
            raw.close()


def _run(raw, events, picks, row, profile, max_trials, started, caught=()):
    import mne
    sfreq = float(raw.info["sfreq"])
    sampling_policy = profile.get("sampling_policy", "fixed_250")
    if sampling_policy == "native":
        if not np.isfinite(sfreq) or not 100 <= sfreq <= 1000:
            raise ValueError("原生采样率方案需100至1000Hz；未自动重采样。")
        points = round(4 * sfreq)
        if not np.isclose(4 * sfreq, points, rtol=0, atol=1e-6):
            raise ValueError("原采样率下4秒不对应整数采样点；请登记独立时间窗协议，不自动取整改变时长。")
    else:
        if not np.isclose(sfreq, 250):
            raise ValueError("当前MI方案固定250Hz/4秒；未静默重采样，请选择独立原生采样率方案。")
        points = 1000
    selected = events[:max_trials]
    if len(selected) * len(picks) * points * 8 > MAX_OUTPUT_BYTES:
        raise ValueError("输出试次数超过96MiB内存/文件限制。")
    # Inspect only requested windows, not every sample in a full recording.
    before, kept, dropped = [], [], []
    for index, event in enumerate(selected):
        offset = int(event[0]) - raw.first_samp
        if offset < 0 or offset + points > raw.n_times:
            dropped.append({"event_index": index, "source_sample": int(event[0]), "reason": "incomplete_window"})
            continue
        trial = raw.get_data(picks=picks, start=offset, stop=offset + points)
        if not np.isfinite(trial).all():
            dropped.append({"event_index": index, "source_sample": int(event[0]), "reason": "nonfinite_window"})
            continue
        # Match MNE Epochs default BAD annotation rejection without using class labels.
        t0 = event[0] / sfreq
        if any(_is_boundary(desc) and t0 < onset < t0 + 4 for onset, desc in
               zip(raw.annotations.onset, raw.annotations.description)):
            dropped.append({"event_index": index, "source_sample": int(event[0]), "reason": "acquisition_boundary"})
            continue
        bad_overlap = any(str(desc).lower().startswith("bad") and onset < t0 + 4 and onset + duration > t0
                          for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description))
        if bad_overlap:
            dropped.append({"event_index": index, "source_sample": int(event[0]), "reason": "bad_annotation"})
            continue
        before.append(trial)
        kept.append({"event_index": index, "source_sample": int(event[0]), "cue_code": int(event[2]), "start_seconds": float(event[0] / sfreq)})
    if not kept:
        raise ValueError("所选窗口均不完整或含非有限数值/BAD标记，没有生成样本。")
    before = np.stack(before)
    output = before.copy()
    extra = {}
    method = profile["method"]
    if method not in {"identity", "autoreject_qc"}:
        first = max(0, kept[0]["source_sample"] - raw.first_samp - int(5 * sfreq))
        last = min(raw.n_times, kept[-1]["source_sample"] - raw.first_samp + points + int(5 * sfreq))
        decoded = (last - first) * len(raw.ch_names) * 8
        _memory_guard(decoded * 5 + before.nbytes * 4)
        # raw is private to this request; crop before loading to avoid deepcopying a full NPZ recording.
        work = raw.crop(tmin=first / sfreq, tmax=(last - 1) / sfreq).load_data(verbose="ERROR")
        try:
            valid = np.isfinite(work._data).all(axis=0)
            # NaN acquisition gaps are excluded from fitting and filtering, never filled as EEG epochs.
            work._data[:, ~valid] = 0
            extra = _process(work, picks, profile, valid)
            for i, event in enumerate(kept):
                offset = event["source_sample"] - work.first_samp
                if not valid[offset:offset + points].all():
                    raise ValueError("处理窗口跨越无效/过短记录段；请选其他记录范围。")
                output[i] = work.get_data(picks=picks, start=offset, stop=offset + points)
        finally:
            work.close()
    elif method == "autoreject_qc":
        if len(kept) < 12 or len(picks) < 16 or not _has_positions(raw, picks):
            raise ValueError("AutoReject需至少12试次、16通道及真实电极位置；未猜位置或删试次。")
        from autoreject import AutoReject
        epochs = mne.EpochsArray(output, mne.pick_info(raw.info.copy(), picks), tmin=0, baseline=None, verbose="ERROR")
        detector = AutoReject(n_interpolate=[1], consensus=[.5, 1.0], cv=3, n_jobs=1, random_state=97, verbose=False)
        detector.fit(epochs)
        log = detector.get_reject_log(epochs)
        extra = {"bad_trial_indices": np.flatnonzero(log.bad_epochs).tolist(), "channel_labels": log.labels.tolist(),
                 "thresholds_v": {k: float(v) for k, v in detector.threshes_.items()},
                 "signal_modified": False, "seed": 97, "cv": 3, "n_jobs": 1,
                 "fit_scope": "本次离线批次自适应质检；模型评估时需另外明确训练/测试拟合边界。"}
    if not np.isfinite(output).all():
        raise ValueError("预处理产生非有限数值，未登记模型输入。")
    notes = ["输出单位V；不使用标签、不执行分类归一化、不自动重采样。",
             "当前为离线研究流程；双向滤波不能称为在线因果算法。"]
    if profile["model_compatibility"] != "legacy_protocol":
        notes.append("此处理改变现有模型训练输入分布；形状兼容不等于效果已验证，请与原协议基线比较。")
    if row["dataset_id"] == "BCI_IV_2b":
        notes.append("保持2b原双极EEG顺序；当前内置22通道分类权重不适配3通道输出。")
    compatibility = profile["model_compatibility"]
    if row["dataset_id"] == "MI_GENERIC":
        compatibility = "requires_validation"
        notes.append("通用数据的事件与训练域未验证；即使无额外处理且形状相同，也不代表复现2a/2b协议。")
    if sampling_policy == "native":
        notes.append(f"保留原生{sfreq:g}Hz及{len(picks)}个真实EEG通道，4秒对应{points}点；不补造或裁减EEG通道。")
        if len(picks) != 22 or points != 1000 or not np.isclose(sfreq, 250):
            notes.append("当前内置22通道/250Hz/1000点分类权重不兼容此输出；仅完成预处理，未训练或运行新分类器。")
    ica_state = extra.pop("_ica_state", None)
    artifact_id = uuid.uuid4().hex
    folder = ROOT / "artifacts" / artifact_id
    folder.mkdir(parents=True, exist_ok=False)
    sample_id = None
    try:
        if ica_state is not None:
            ica_state.save(folder / "decomposition-ica.fif", overwrite=False, verbose="ERROR")
        np.savez_compressed(folder / "epochs.npz", data=output)
        with (folder / "epochs.npz").open("rb") as stream:
            sample = samples.register(stream, row["file"] + ".npz", sfreq, "V", row["dataset_id"])
        sample_id = sample["sample_id"]
        sample.update(preprocessing_artifact_id=artifact_id, preprocessing_profile_id=profile["id"],
                      preprocessing_compatibility=compatibility,
                      channel_names=[raw.ch_names[p] for p in picks], source_recording_sha256=row["sha256"],
                      preprocessing_scope=" ".join(notes))
        store.write("sample", sample_id, sample)
        result = {"artifact_id": artifact_id, "sample_id": sample_id, "kind": "mi_preprocessing", "created_at": store.now(),
                  "recording_id": row["recording_id"], "source_file": row["file"], "source_sha256": row["sha256"],
                  "dataset_id": row["dataset_id"], "profile": {**profile, "model_compatibility": compatibility},
                  "protocol": {"cue_relative_seconds": [0, 4], "endpoint": "exclusive", "points": points,
                               "sampling_policy": sampling_policy,
                               "sampling_rate_hz": sfreq, "reference": "common_average" if method in {"iclabel", "eog_regression"} else "recorded",
                               "band_hz": profile["band"], "filter": "4th-order Butterworth SOS zero-phase; separate finite runs" if profile["band"] else None,
                               "baseline": None, "normalization": None, "resample": None, "labels_used": False,
                               "channel_mapping": row["metadata"].get("channel_mapping", {}),
                               "annotation_event_map": row["metadata"].get("annotation_event_map", {})},
                  "output": {"shape": list(output.shape), "sampling_rate_hz": sfreq, "unit": "V",
                             "channel_names": [raw.ch_names[p] for p in picks], "sha256": _hash(folder / "epochs.npz")},
                  "events": kept, "requested_trials": max_trials, "total_cue_events": len(events),
                  "qc": {"before": _qc(before, sfreq), "after": _qc(output, sfreq), "dropped_trials": dropped,
                         "warnings": notes, "method_details": extra},
                  "versions": {package: package_version(package.replace("_", "-"))
                               for package in sorted({"mne", "numpy", "scipy", *profile["packages"]})},
                  "execution_code_sha256": _hash(Path(__file__)),
                  "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "scope": " ".join(notes),
                  "artifact_url": f"/workspace/mi-preprocessing/artifacts/{artifact_id}",
                  "data_url": f"/workspace/mi-preprocessing/artifacts/{artifact_id}/data"}
        if profile.get("family"):
            result["protocol"].update(workflow_family=profile["family"],
                workflow_contract_sha256=profile["workflow_contract_sha256"],
                normalization_stage="classifier_adapter_only_not_applied_here",
                augmentation_stage="training_only_not_applied_at_inference")
        result["qc"]["warnings"].extend(dict.fromkeys(str(w.message)[:600] for w in caught))
        if ica_state is not None:
            result["state_files"] = [{"file": "decomposition-ica.fif", "sha256": _hash(folder / "decomposition-ica.fif")}]
        (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        store.write("mi_artifact", artifact_id, {k: result[k] for k in ("artifact_id", "source_file", "sample_id", "created_at", "profile")})
        return result
    except Exception:
        if sample_id:
            # Roll back only the sample created by this failed run, never existing uploads.
            with store.transaction() as connection:
                connection.execute("DELETE FROM records WHERE kind='sample' AND id=?", (sample_id,))
            (samples.SAMPLE_ROOT / sample["storage_name"]).unlink(missing_ok=True)
        for name in ("epochs.npz", "result.json", "decomposition-ica.fif"):
            (folder / name).unlink(missing_ok=True)
        folder.rmdir()
        raise


def read_artifact(identity):
    path = ROOT / "artifacts" / _identity(identity) / "result.json"
    if not path.is_file():
        raise ValueError("预处理产物不存在。")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("artifact_id") != identity or result.get("kind") != "mi_preprocessing":
        raise ValueError("预处理产物身份不一致。")
    return result


def data_path(identity):
    result = read_artifact(identity)
    path = ROOT / "artifacts" / identity / "epochs.npz"
    if not path.is_file() or _hash(path) != result["output"]["sha256"]:
        raise ValueError("预处理数据不存在或哈希不一致。")
    return path


def split_path(identity, partition):
    if partition not in {"train", "test"}:
        raise ValueError("划分只能是train或test。")
    result = read_artifact(identity)
    filename = partition + ".mat"
    export = next((item for item in result.get("split_exports", []) if item.get("name") == filename), None)
    path = ROOT / "artifacts" / identity / filename
    if export is None or not path.is_file() or _hash(path) != export.get("sha256"):
        raise ValueError("该划分文件不存在或哈希不一致。")
    return path


def artifact_document(result):
    source_description = ("SHU作者已分段session数据；未执行连续cue切段。" if result.get("dataset_id") == "SHU_MI" else
                          f"原始提示事件{result['total_cue_events']}个，本次保留{result['output']['shape'][0]}个；")
    return {"type": "analysis_result", "artifact_id": result["artifact_id"], "sample_id": result["sample_id"],
            "artifact_kind": "mi_preprocessing",
            "title": "运动想象预处理 · " + result["profile"]["name"], "source_url": result["artifact_url"],
            "text": (f"真实运动想象预处理产物：{result['artifact_id']}。原件{result['source_file']}，SHA256={result['source_sha256']}。"
                     f"方案{result['profile']['id']}；输出sample_id={result['sample_id']}，形状{result['output']['shape']}，{result['output']['sampling_rate_hz']:g}Hz，V。"
                     f"{source_description}"
                     f"拒绝窗口{len(result['qc']['dropped_trials'])}个。{result['scope']}本工具未执行分类，未产生准确率。")}
