"""Executable profiles and separately labelled research leads, with primary sources."""
from importlib.util import find_spec
import hashlib
import json
from pathlib import Path

MNE_DOC = "https://mne.tools/stable/api/preprocessing.html"
PROFILES = (
    {"id": "dwdanet-original-v1", "name": "DWDaNet · 我的原数据处理流程", "method": "identity", "family": "dwdanet",
     "description": "复用原2a/2b提示后4秒无额外滤波流程；SHU按明确session划分整理已切段MAT。保留原模型全源标量归一化配方，归一化仍由具体模型适配器执行一次。",
     "supported_datasets": ["BCI_IV_2a", "BCI_IV_2b", "SHU_MI"],
     "band": None, "packages": ["mne"], "model_compatibility": "legacy_protocol",
     "source": "local:original_workflows/dwdanet.json"},
    {"id": "dwtnet-original-v1", "name": "DWTNet · 我的原数据处理流程", "method": "identity", "family": "dwtnet",
     "description": "复用原2a/2b/SHU数据准备流程，原DWT用于训练增强，不在推理前额外扰动。数据切段与DWDaNet共用相同协议，来源和训练配方独立留档。",
     "supported_datasets": ["BCI_IV_2a", "BCI_IV_2b", "SHU_MI"],
     "band": None, "packages": ["mne"], "model_compatibility": "legacy_protocol",
     "source": "local:original_workflows/dwtnet.json"},
    {"id": "bci-cue4s-legacy-v1", "name": "原模型协议 · 提示后4秒", "method": "identity",
     "description": "250Hz，提示后1000点，保留原记录参考和EEG顺序；无额外滤波、去均值、重采样。2a/2b原处理协议。",
     "band": None, "packages": ["mne"], "model_compatibility": "legacy_protocol",
     "source": "local:preprocessing_for_2a.py / preprocessing_for_2b.py"},
    {"id": "mi-mu-beta-v1", "name": "MI μ/β · 8–32Hz", "method": "bandpass",
     "description": "连续片段四阶Butterworth双向8–32Hz；保留原参考，4秒试次。研究方案，需验证现有权重效果。",
     "band": [8.0, 32.0], "packages": ["mne"], "model_compatibility": "requires_validation",
     "source": "https://moabb.neurotechx.com/docs/generated/moabb.paradigms.MotorImagery.html"},
    {"id": "mi-broadband-v1", "name": "MI 宽带 · 4–40Hz", "method": "bandpass",
     "description": "连续片段四阶Butterworth双向4–40Hz；保留原参考，4秒试次，供离线研究比较。",
     "band": [4.0, 40.0], "packages": ["mne"], "model_compatibility": "requires_validation", "source": MNE_DOC},
    {"id": "mi-eog-regression-v1", "name": "MI 眼电回归 · 4–40Hz", "method": "eog_regression",
     "description": "平均参考后使用真实EOG通道拟合MNE眼电回归，再4–40Hz滤波；保存回归系数，区别于原模型参考。",
     "band": [4.0, 40.0], "packages": ["mne"], "model_compatibility": "requires_validation",
     "source": "https://mne.tools/stable/generated/mne.preprocessing.EOGRegression.html"},
    {"id": "mi-pyprep-qc-v1", "name": "PyPREP 坏道质检 · 只标记", "method": "pyprep_qc",
     "description": "检查NaN/平直、幅值、高频噪声、相关性；关闭RANSAC，不插值、不删通道、不改变原模型输入。至少16个EEG通道。",
     "band": None, "packages": ["mne", "pyprep"], "model_compatibility": "legacy_protocol",
     "source": "https://pyprep.readthedocs.io/en/stable/generated/pyprep.NoisyChannels.html"},
    {"id": "mi-autoreject-qc-v1", "name": "AutoReject 试次质检 · 只标记", "method": "autoreject_qc",
     "description": "学习峰峰值阈值并记录坏试次；不剔除/插值。需至少12个试次、16个EEG通道及真实电极位置。",
     "band": None, "packages": ["mne", "autoreject"], "model_compatibility": "legacy_protocol",
     "source": "https://autoreject.github.io/stable/generated/autoreject.AutoReject.html"},
    {"id": "mi-iclabel-v1", "name": "ICA + ICLabel · 可选去伪迹", "method": "iclabel",
     "description": "真实电极位置、平均参考、1–100Hz、extended Infomax；仅移除≥0.90的眼动/肌电/心电分量，最后4–40Hz。最多32试次/120秒ICA拟合。",
     "band": [4.0, 40.0], "packages": ["mne", "mne_icalabel", "torch"], "model_compatibility": "requires_validation",
     "source": "https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html"},
    {"id": "mi-native-cue4s-v1", "name": "原生采样率 · 提示后4秒", "method": "identity",
     "description": "保留100–1000Hz原采样率、全部真实EEG通道及原参考；提示后4秒，无滤波/重采样。不能自动接入现有22通道250Hz权重。",
     "sampling_policy": "native", "band": None, "packages": ["mne"], "model_compatibility": "requires_validation",
     "source": "https://physionet.org/content/eegmmidb/1.0.0/"},
    {"id": "mi-native-mu-beta-v1", "name": "原生采样率 MI μ/β · 8–32Hz", "method": "bandpass",
     "description": "保留原采样率/真实EEG通道/参考，连续有效片段8–32Hz四阶双向滤波，提示后4秒；不自动适配分类权重。",
     "sampling_policy": "native", "band": [8.0, 32.0], "packages": ["mne"], "model_compatibility": "requires_validation",
     "source": MNE_DOC},
    {"id": "mi-native-broadband-v1", "name": "原生采样率 MI 宽带 · 4–40Hz", "method": "bandpass",
     "description": "保留原采样率/真实EEG通道/参考，连续有效片段4–40Hz四阶双向滤波，提示后4秒；不自动适配分类权重。",
     "sampling_policy": "native", "band": [4.0, 40.0], "packages": ["mne"], "model_compatibility": "requires_validation",
     "source": MNE_DOC},
)

RESEARCH_METHODS = (
    {"id": "eegoar-net", "name": "EEGOAR-Net（2025）", "status": "research_only",
     "description": "作者提供去眼电预训练权重；示例128Hz、1秒、64模板通道与掩码，尚未适配当前250Hz数据，未自动下载权重。",
     "source": "https://github.com/dmarcos97/EEGOAR-Net"},
    {"id": "faar", "name": "FAAR（2026，MI伪迹筛选）", "status": "research_only",
     "description": "运动想象质量评分与自适应试次剔除；未核到作者公开实现，未冒称已复现。",
     "source": "https://arxiv.org/html/2605.12408v1"},
    {"id": "asr", "name": "ASR / MEEGkit", "status": "research_only",
     "description": "需干净参考校准、多通道与秩检查；后续独立比较，不默认对3通道2b重建。",
     "source": "https://nbara.github.io/python-meegkit/modules/meegkit.asr.html"},
    {"id": "mirepnet", "name": "MIRepNet / Euclidean Alignment", "status": "research_only",
     "description": "属于MI表征与分布对齐方法，非通用去伪迹器；需独立模型输入协议，不补造真实通道。",
     "source": "https://braindecode.org/stable/generated/braindecode.models.MIRepNet.html"},
)


def profiles():
    result = []
    for profile in PROFILES:
        missing = [package for package in profile["packages"] if find_spec(package) is None]
        missing_manifest = profile.get("family") and not (Path(__file__).parent / "original_workflows" / (profile["family"] + ".json")).is_file()
        result.append({**profile, "sampling_policy": profile.get("sampling_policy", "fixed_250"), "ready": not missing and not missing_manifest,
                       "reason": "缺少依赖：" + ", ".join(missing) if missing else
                       "原流程来源清单尚未就绪。" if missing_manifest else "依赖就绪；记录条件仍由执行前核验。"})
    return result


def get_profile(identity):
    profile = next((p for p in profiles() if p["id"] == identity), None)
    if profile is None:
        raise ValueError("未登记的运动想象预处理方案；研究候选不能执行。")
    if not profile["ready"]:
        raise ValueError(profile["reason"])
    if profile.get("family"):
        path = Path(__file__).parent / "original_workflows" / (profile["family"] + ".json")
        if not path.is_file():
            raise ValueError("原数据处理流程来源清单尚未就绪。")
        blob = path.read_bytes()
        profile["workflow_contract"] = json.loads(blob.decode("utf-8"))
        profile["workflow_contract_sha256"] = hashlib.sha256(blob).hexdigest()
    return profile
