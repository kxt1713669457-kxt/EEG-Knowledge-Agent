"""Chat EEG uploads are inert files until a scoped Agent registration tool runs."""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import hashlib
import json
import math
from pathlib import Path
import re
from threading import Lock
import uuid
import zipfile

import numpy as np
from scipy.io import whosmat

from app.shared.runtime.logger import PROJECT_ROOT
from app.workspace import store

ROOT = PROJECT_ROOT / "output/runtime/eeg-attachments"
MAX_BYTES = 128 * 1024**2
MAX_FILES = 5
EXTENSIONS = {".gdf", ".edf", ".bdf", ".fif", ".npz", ".npy", ".mat"}
_selected = ContextVar("task_eeg_attachments", default=())
_registration_lock = Lock()


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("数值数组只支持NPY v1/v2格式。")
    if dtype.hasobject or dtype.kind not in "fiuUS" or math.prod(shape) * dtype.itemsize > 256 * 1024**2:
        raise ValueError("数组类型或声明的展开大小超限；不接受对象数组。")
    return shape, dtype


def _header(path):
    ext = path.suffix
    if ext == ".mat":
        variables = {name: (shape, kind) for name, shape, kind in whosmat(path)}
        shape, kind = variables.get("data", ((), ""))
        if len(shape) not in {2, 3} or kind not in {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"} or not math.prod(shape) or math.prod(shape) * 8 > MAX_BYTES:
            raise ValueError("MAT需包含有限大小的二维/三维实数data数组；不支持v7.3/HDF5或结构体。")
        return "epoched", {"shape": list(shape), "dtype": kind, "sampling_rate_hz": None, "unit": None,
                            "metadata_needed": ["sampling_rate_hz", "unit", "dataset_id"],
                            "note": "data轴序需确认为[试次,通道,采样点]或[通道,采样点]，不按形状猜测。"}
    if ext == ".npy":
        with path.open("rb") as stream:
            shape, dtype = _array_header(stream)
            if stream.tell() + math.prod(shape) * dtype.itemsize > path.stat().st_size:
                raise ValueError("NPY数组体被截断。")
        if len(shape) not in {2, 3} or dtype.kind not in "fiu" or not math.prod(shape):
            raise ValueError("NPY需为二维/三维实数试次数组。")
        return "epoched", {"shape": list(shape), "dtype": str(dtype), "sampling_rate_hz": None, "unit": None,
                            "metadata_needed": ["sampling_rate_hz", "unit", "dataset_id"]}
    if ext == ".npz":
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 32 or len({item.filename for item in entries}) != len(entries) or sum(item.file_size for item in entries) > 256 * 1024**2:
                raise ValueError("NPZ展开大小、条目数或重复数组名不合法。")
            headers = {}
            for item in entries:
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.npy", item.filename):
                    raise ValueError("NPZ仅接受命名数值数组。")
                with archive.open(item) as stream:
                    shape, dtype = _array_header(stream)
                    if stream.tell() + math.prod(shape) * dtype.itemsize > item.file_size:
                        raise ValueError("NPZ数组体被截断。")
                headers[item.filename[:-4]] = (shape, dtype)
            if "signal" in headers and "data" in headers:
                raise ValueError("NPZ同时包含signal与data，无法确定连续记录或试次，请分开上传。")
            key = "signal" if "signal" in headers else "data" if "data" in headers else None
            if key is None:
                raise ValueError("NPZ需包含连续signal或已分段data数组。")
            shape, dtype = headers[key]
            if dtype.kind not in "fiu" or not math.prod(shape) or len(shape) not in ({2} if key == "signal" else {2, 3}):
                raise ValueError("NPZ信号形状或类型不合法。")
            summary = {"shape": list(shape), "dtype": str(dtype), "array_names": sorted(headers),
                       "sampling_rate_hz": None, "unit": None}
            if key == "signal":
                if not {"sfreq", "ch_names", "events", "unit"}.issubset(headers):
                    raise ValueError("连续NPZ还需sfreq、ch_names、events和unit。")
                for scalar in ("sfreq", "unit"):
                    if math.prod(headers[scalar][0]) != 1 or headers[scalar][1].itemsize > 100:
                        raise ValueError("连续NPZ采样率/单位必须是小型标量。")
                with np.load(path, allow_pickle=False) as arrays:
                    summary.update(sampling_rate_hz=float(arrays["sfreq"].item()), unit=str(arrays["unit"].item()))
                if not math.isfinite(summary["sampling_rate_hz"]) or not 100 <= summary["sampling_rate_hz"] <= 1000 or summary["unit"] not in {"V", "uV"}:
                    raise ValueError("连续NPZ需100–1000Hz和明确V/uV单位。")
            else:
                summary["metadata_needed"] = ["sampling_rate_hz", "unit", "dataset_id"]
        return "continuous" if key == "signal" else "epoched", summary
    import mne
    reader = {".gdf": mne.io.read_raw_gdf, ".edf": mne.io.read_raw_edf,
              ".bdf": mne.io.read_raw_bdf, ".fif": mne.io.read_raw_fif}[ext]
    raw = reader(path, preload=False, verbose="ERROR")
    try:
        if not 1 <= len(raw.ch_names) <= 64 or len(raw.ch_names) * raw.n_times * 8 > 256 * 1024**2:
            raise ValueError("记录超过64通道或256MiB解码上限，请按会话拆分。")
        return "continuous", {"channels_including_non_eeg": len(raw.ch_names), "channel_names": raw.ch_names,
                              "channel_types": raw.get_channel_types(), "sampling_rate_hz": float(raw.info["sfreq"]),
                              "duration_seconds": round(raw.n_times / raw.info["sfreq"], 3),
                              "annotation_descriptions": sorted(set(map(str, raw.annotations.description)))[:32],
                              "unit": "V", "unit_note": "MNE按文件校准后输出V；未执行预处理。",
                              "metadata_needed": ["dataset_id", "event_protocol_for_generic_data"]}
    finally:
        raw.close()


def public(row):
    result = {key: row[key] for key in ("attachment_id", "filename", "size", "representation", "header_summary", "sha256", "created_at")}
    registered = []
    for identity in list(dict.fromkeys(row.get("registrations", {}).values()))[-10:]:
        kind = "sample" if identity.startswith("sample-") else "mi_recording"
        record = store.read(kind, identity)
        if record and not record.get("deleted_at"):
            registered.append({key: record[key] for key in ("sample_id", "recording_id", "dataset_id", "sampling_rate_hz", "unit", "shape", "channels") if key in record})
    result["registrations"] = registered
    return result


def attachment_path(row):
    path = (ROOT / row["storage_name"]).resolve()
    if path.parent != ROOT.resolve() or not path.is_file():
        raise ValueError("脑电附件原件不存在。")
    return path


def attachment_record(identity):
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity):
        raise ValueError("脑电附件身份格式不合法。")
    row = store.read("eeg_attachment", identity)
    if row is None or row.get("deleted_at"):
        raise ValueError("脑电附件不存在，请重新上传。")
    if _hash(attachment_path(row)) != row["sha256"]:
        raise ValueError("脑电附件哈希变化，拒绝读取。")
    return row


def upload(items):
    if not 1 <= len(items) <= MAX_FILES:
        raise ValueError("每轮支持1至5份脑电文件。")
    ROOT.mkdir(parents=True, exist_ok=True)
    rows, owned, total = [], [], 0
    try:
        for stream, filename in items:
            name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1][:180]
            ext = Path(name).suffix.lower()
            if ext not in EXTENSIONS:
                raise ValueError("脑电附件仅支持GDF、EDF、BDF、FIF、MAT、NPY、NPZ。")
            identity = uuid.uuid4().hex
            path = ROOT / (identity + ext)
            owned.append(path)
            size = 0
            with path.open("xb") as target:
                for block in iter(lambda: stream.read(1024**2), b""):
                    size += len(block)
                    total += len(block)
                    if size > MAX_BYTES or total > MAX_BYTES:
                        raise ValueError("本次脑电附件合计不能超过128MiB。")
                    target.write(block)
            if not size:
                raise ValueError("脑电附件不能为空。")
            try:
                representation, summary = _header(path)
            except ValueError:
                raise
            except Exception:
                raise ValueError("无法读取脑电文件头，请核对格式；MAT不支持v7.3/HDF5。") from None
            row = {"id": identity, "attachment_id": identity, "filename": name, "storage_name": path.name,
                   "size": size, "representation": representation, "header_summary": summary,
                   "sha256": _hash(path), "created_at": store.now(), "registrations": {}}
            rows.append(row)
        with store.transaction() as connection:
            for row in rows:
                store.write("eeg_attachment", row["id"], row, connection)
        return {"attachments": [public(row) for row in rows]}
    except Exception:
        for path in owned:
            path.unlink(missing_ok=True)
        raise


def selected_attachments():
    return [public(attachment_record(row["attachment_id"])) for row in _selected.get()]


@contextmanager
def attachment_scope(identities):
    if len(identities) > MAX_FILES or len(set(identities)) != len(identities):
        raise ValueError("脑电附件最多5份且不能重复。")
    rows = tuple(attachment_record(identity) for identity in identities)
    if sum(row["size"] for row in rows) > MAX_BYTES:
        raise ValueError("本轮脑电附件合计超过128MiB。")
    token = _selected.set(rows)
    try:
        yield
    finally:
        _selected.reset(token)


def inspect_attachment(attachment_id):
    if attachment_id not in {row["attachment_id"] for row in _selected.get()}:
        raise ValueError("脑电附件不属于本轮选择范围。")
    return public(attachment_record(attachment_id))


def _scoped_row(identity):
    inspect_attachment(identity)
    return attachment_record(identity)


def _register(row, kind, parameters, action):
    """Reuse the same attachment + explicit protocol across clarification turns."""
    key = hashlib.sha256(json.dumps([kind, parameters], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    with _registration_lock:
        row = attachment_record(row["attachment_id"])
        previous = row.get("registrations", {}).get(key)
        if previous:
            existing = store.read(kind, previous)
            if existing and not existing.get("deleted_at"):
                return existing
        result = action()
        row.setdefault("registrations", {})[key] = result.get("sample_id") or result["recording_id"]
        store.write("eeg_attachment", row["id"], row)
        return result


def register_recording(attachment_id, dataset_id, metadata=None):
    from app.agent.skills.mi_preprocessing import service
    row = _scoped_row(attachment_id)
    if row["representation"] != "continuous":
        raise ValueError("附件为已分段试次，请登记样本；SHU五会话使用专门划分流程。")
    metadata = metadata or {}
    def action():
        with attachment_path(row).open("rb") as stream:
            return service.register_recording(stream, row["filename"], dataset_id, json.dumps(metadata, ensure_ascii=False))
    return _register(row, "mi_recording", [dataset_id, metadata], action)


def register_sample(attachment_id, sampling_rate_hz, unit, dataset_id):
    from app.workspace import samples
    row = _scoped_row(attachment_id)
    if row["representation"] != "epoched":
        raise ValueError("连续原始记录须先预处理切段，不能直接登记为模型试次。")
    if not isinstance(sampling_rate_hz, (float, int)) or isinstance(sampling_rate_hz, bool) or not math.isfinite(sampling_rate_hz) or not 0 < sampling_rate_hz <= 100000 or unit not in {"V", "uV"} or not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("登记样本需要明确采样率、V/uV单位和数据集，不能从幅值或形状猜测。")
    def action():
        with attachment_path(row).open("rb") as stream:
            return samples.register(stream, row["filename"], sampling_rate_hz, unit, dataset_id)
    return _register(row, "sample", [sampling_rate_hz, unit, dataset_id], action)


def register_shu_sessions(attachment_ids, split_id):
    from app.agent.skills.mi_preprocessing import shu
    if len(attachment_ids) != 5 or len(set(attachment_ids)) != 5:
        raise ValueError("SHU需要同一被试的五份不同会话MAT。")
    rows = sorted((_scoped_row(identity) for identity in attachment_ids), key=lambda row: row["filename"])
    def action():
        with ExitStack() as stack:
            return shu.register_sessions([(stack.enter_context(attachment_path(row).open("rb")), row["filename"]) for row in rows], split_id)
    return _register(rows[0], "mi_recording", ["SHU_MI", split_id, [row["sha256"] for row in rows]], action)
