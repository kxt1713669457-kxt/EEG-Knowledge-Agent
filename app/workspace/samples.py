"""显式登记数值型EEG样本；论文文档与波形样本使用不同入口。"""
import hashlib
from pathlib import Path
import uuid
import math

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatReadError

from app.shared.runtime.logger import PROJECT_ROOT
from app.workspace import store

SAMPLE_ROOT = PROJECT_ROOT / "output/runtime/registered-samples"
SAMPLE_EXTENSIONS = {".mat", ".npy", ".npz"}
MAX_SAMPLE_BYTES = 128 * 1024 * 1024


def read_array(path):
    try:
        return _read_array(path)
    except (OSError, EOFError, NotImplementedError, MatReadError):
        raise ValueError("无法读取样本格式；MAT需为可读取的数值data（不支持v7.3/HDF5），或使用NPY/NPZ数值数组。") from None


def _read_array(path):
    path = Path(path)
    if path.suffix == ".mat":
        names = {name: (shape, kind) for name, shape, kind in whosmat(path)}
        if "data" not in names or names["data"][1] in {"cell", "struct", "object", "char"}:
            raise ValueError("MAT必须包含数值型data，不能是对象、结构体或文字。")
        if len(names["data"][0]) not in {2, 3} or math.prod(names["data"][0]) * 8 > MAX_SAMPLE_BYTES:
            raise ValueError("MAT声明的形状或展开体积超出样本限制。")
        array = loadmat(path, variable_names=["data"])["data"]
    elif path.suffix == ".npy":
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    elif path.suffix == ".npz":
        import zipfile
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_SAMPLE_BYTES:
                raise ValueError("样本解压后超过128MB限制。")
            if "data.npy" not in archive.namelist():
                raise ValueError("NPZ需要data数组。")
            with archive.open("data.npy") as header:
                version = np.lib.format.read_magic(header)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(header)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(header)
                else:
                    raise ValueError("请将样本保存为标准数值NPZ（NPY v1/v2）。")
                size = math.prod(shape) * dtype.itemsize
                if dtype.kind not in "fiu" or len(shape) not in {2, 3} or size > MAX_SAMPLE_BYTES or size > archive.getinfo("data.npy").file_size:
                    raise ValueError("NPZ数组头的类型、形状或体积不符合数值样本限制。")
        with np.load(path, allow_pickle=False) as arrays:
            if "data" not in arrays:
                raise ValueError("NPZ需要data数组。")
            array = arrays["data"]
    else:
        raise ValueError("样本只接受MAT、NPY、NPZ数值数组，不接受论文或普通文档。")
    mapping = array if isinstance(array, np.memmap) else None
    try:
        if array.dtype.kind not in "fiu" or array.ndim not in {2, 3} or not array.size:
            raise ValueError("样本必须是非空实数矩阵：[通道,采样点]或[试次,通道,采样点]。")
        if array.nbytes > MAX_SAMPLE_BYTES:
            raise ValueError("样本数组超过128MB限制。")
        if mapping is not None:
            array = np.array(mapping, copy=True)
    finally:
        if mapping is not None:
            mapping._mmap.close()  # Windows失败清理前释放文件句柄。
    return array[np.newaxis] if array.ndim == 2 else array


def active_entries():
    return {r["id"]: r for r in store.all_records("sample") if not r.get("deleted_at")}


def registered_path(entry):
    path = (SAMPLE_ROOT / entry["storage_name"]).resolve()
    if not path.is_relative_to(SAMPLE_ROOT.resolve()):
        raise ValueError("登记样本路径超出允许范围。")
    return path


def register(stream, filename, sampling_rate_hz, unit, dataset_id):
    ext = Path(filename or "").suffix.lower()
    if ext not in SAMPLE_EXTENSIONS:
        raise ValueError("样本仅支持.mat/.npy/.npz；PDF、Word等请到知识库资料导入。")
    if sampling_rate_hz <= 0 or sampling_rate_hz > 100000 or unit not in {"V", "uV"}:
        raise ValueError("需要明确的采样率与V/uV单位。")
    identity = "sample-" + uuid.uuid4().hex[:20]
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_ROOT / (identity + ext)
    digest, size = hashlib.sha256(), 0
    try:
        with path.open("xb") as target:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                if size > MAX_SAMPLE_BYTES:
                    raise ValueError("样本文件超过128MB限制。")
                digest.update(block);target.write(block)
        data = read_array(path)
        if not np.isfinite(data).all():
            raise ValueError("样本含NaN或无穷值，未登记。")
        compatible = data.shape[1:] == (22, 1000) and sampling_rate_hz == 250
        entry = {"id": identity, "sample_id": identity, "dataset_id": dataset_id[:100] or "用户登记数据",
                 "file": Path(filename).name, "storage_name": path.name, "sampling_rate_hz": sampling_rate_hz,
                 "unit": unit, "compatible": compatible, "shape": list(data.shape), "sha256": digest.hexdigest(),
                 "created_at": store.now(), "deleted_at": None,
                 "compatibility_note": "可用于当前22通道、250Hz、4秒适配器" if compatible else "已登记；当前适配要求22通道、250Hz、每试次1000采样点，不自动补通道或猜时间窗。"}
        return store.write("sample", identity, entry)
    except Exception:
        path.unlink(missing_ok=True)
        raise
