"""固定输入契约的CBraMod预训练编码器；不使用标签、不创建随机分类头。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
import time
import uuid

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import butter, lfilter, resample

from app.shared.runtime.logger import PROJECT_ROOT


MODEL_ID = "cbramod-pretrained"
MODEL_REVISION = "500543c7e30bda1b22bfd51a49301b238dee21fd"
WEIGHTS_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
ARTIFACT_ROOT = PROJECT_ROOT / "output/runtime/eeg-artifacts"
SAMPLES = {
    "bci2a-a01e": {"dataset_id": "BCI_IV_2a", "file": "A01E.mat", "relative_path": "BCI_IV_2a/A01E.mat",
                   "sampling_rate_hz": 250, "unit": "V", "compatible": True},
    "bci2b-b01e": {"dataset_id": "BCI_IV_2b", "file": "B01E.mat", "relative_path": "BCI_IV_2b/B01E.mat",
                   "sampling_rate_hz": 250, "unit": "V", "compatible": False},
}
_model = None
_model_lock = Lock()


def sample_entries():
    from app.workspace.samples import active_entries
    return {**SAMPLES, **active_entries()}


class EEGSkillError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_path(sample_id):
    entry = sample_entries().get(sample_id)
    if entry is None:
        raise EEGSkillError("unknown_sample", "样本未登记；只能选择服务端样本目录中的ID。")
    if entry.get("storage_name"):
        from app.workspace.samples import registered_path
        return registered_path(entry)
    return PROJECT_ROOT / "corpus/eeg/datasets/raw" / entry["relative_path"]


def model_catalog():
    samples = []
    for sample_id, entry in sample_entries().items():
        path = sample_path(sample_id)
        shapes = dict((name, list(shape)) for name, shape, _ in whosmat(path)) if path.exists() and path.suffix == ".mat" else {}
        samples.append({"sample_id": sample_id, **{k:v for k,v in entry.items() if k not in {"relative_path", "storage_name", "sha256"}},
                        "shape": entry.get("shape") or shapes.get("data"), "available": path.exists(),
                        "user_registered": bool(entry.get("storage_name")),
                        "compatibility_note": entry.get("compatibility_note") or "仅验证22通道2a的当前适配；不代表编码器架构本身只能接受22通道。"})
    from app.agent.skills import manager
    models = manager.catalog()
    for sample in samples:
        sample['compatible_model_ids'] = [m['id'] for m in models if m.get('enabled') and m.get('ready') and sample.get('available') and sample.get('shape') and sample['shape'][1:]==[m['input']['channels'],m['input']['points']] and sample['sampling_rate_hz']==m['input']['sampling_rate_hz']]
        sample['compatible'] = bool(sample['compatible_model_ids'])
    return {"models": [{"model_id": MODEL_ID, "category": "pretrained_eeg_encoder", "classification_available": False,
        "weights_revision": MODEL_REVISION, "input_shape": [1,22,4,200], "output": "真实表征及统计，不是类别/置信度",
        "source_url": "https://github.com/wjq-learning/CBraMod", **m} if m["id"] == MODEL_ID else m for m in models], "samples": samples}


def load_trial(sample_id, trial_index):
    path = sample_path(sample_id)
    entry = sample_entries()[sample_id]
    if not entry["compatible"]:
        raise EEGSkillError("incompatible_sample", "当前适配器只验证22通道2a；该数据集尚未适配，不能直接代入。")
    if not path.is_file():
        raise EEGSkillError("sample_unavailable", "登记的样本文件当前不可读取。")
    if entry.get("preprocessing_artifact_id"):
        from app.agent.skills.mi_preprocessing import service as mi_service
        source = mi_service.read_artifact(entry["preprocessing_artifact_id"])
        if source["sample_id"] != sample_id or file_sha256(path) != source["output"]["sha256"]:
            raise EEGSkillError("preprocessing_integrity", "派生样本与预处理产物身份或哈希不一致。")
    # 只读data变量，label不进入推理过程。
    from app.workspace.samples import read_array
    data = read_array(path) if entry.get("storage_name") else loadmat(path, variable_names=["data"])["data"]
    if data.ndim != 3 or data.shape[1:] != (22,1000):
        raise EEGSkillError("incompatible_shape", "当前样本必须为试次×22通道×1000采样点。")
    if type(trial_index) is not int or not 0 <= trial_index < data.shape[0]:
        raise EEGSkillError("trial_out_of_range", "试次下标超出该文件实际试次数。")
    trial = np.asarray(data[trial_index], dtype=np.float64)
    if not np.isfinite(trial).all():
        raise EEGSkillError("nonfinite_signal", "波形含NaN或无穷值，未进入模型。")
    if entry["unit"] == "uV":
        trial = trial / 1_000_000.0
    return trial, {"sample_id":sample_id, "dataset_id":entry["dataset_id"], "file":entry["file"],
        "trial_index":trial_index, "trial_index_base":0, "source_shape":list(data.shape),
        "sampling_rate_hz":entry["sampling_rate_hz"], "unit":entry["unit"],
        "channel_order":"文件中的通道顺序；未登记通道名称时不凭模型补写名称。" if entry.get("storage_name") else "原GDF的EEG选择顺序；既有MAT未存通道名称，不凭模型补写名称。",
        "source_file_sha256":file_sha256(path), "labels_used_as_model_input":False,
        **({"preprocessing_artifact_id": entry["preprocessing_artifact_id"],
            "source_recording_sha256": entry.get("source_recording_sha256"),
            "channel_names": entry.get("channel_names"), "preprocessing_scope": entry.get("preprocessing_scope")}
           if entry.get("preprocessing_artifact_id") else {})}


def preprocess_trial(trial):
    if trial.shape != (22,1000) or not np.isfinite(trial).all():
        raise EEGSkillError("incompatible_shape", "预处理要求有限的22×1000矩阵，不自动填补或裁切。")
    microvolts = trial * 1_000_000.0
    referenced = microvolts - microvolts.mean(axis=0, keepdims=True)
    b, a = butter(5, [0.3,40], btype="bandpass", fs=250)
    filtered = lfilter(b, a, referenced, axis=-1)
    scaled = resample(filtered, 800, axis=-1) / 100.0
    tensor = np.ascontiguousarray(scaled.reshape(1,22,4,200), dtype=np.float32)
    if not np.isfinite(tensor).all():
        raise EEGSkillError("nonfinite_preprocessing", "预处理产生非有限数值，未进入模型。")
    profile = {"profile_id":"bci2a-epoched-cbramod-v1", "original_sampling_rate_hz":250,
        "model_sampling_rate_hz":200, "source_unit":"V", "unit_conversion":"V→uV，再除以100",
        "reference":"逐采样点22个EEG通道共同平均参考", "filter":"5阶Butterworth 0.3–40Hz，lfilter",
        "resampling":"scipy.signal.resample，1000→800采样点", "window_seconds":4,
        "patches":4, "points_per_patch":200, "input_shape":list(tensor.shape),
        "adaptation_boundary":"复用已切好的4秒MAT，在试次内滤波；作者在连续试次过滤后裁窗，滤波边缘并非完全相同，不冒称复现实验协议。"}
    return tensor, profile


def _get_model():
    global _model
    if _model is None:
        import torch
        from .vendor.cbramod import CBraMod
        path = PROJECT_ROOT / "models/eeg/cbramod/pretrained_weights.pth"
        if not path.is_file() or file_sha256(path) != WEIGHTS_SHA256:
            raise EEGSkillError("model_unavailable", "CBraMod权重缺失或与固定版本不符，请先运行模型准备脚本。")
        weights = torch.load(path, map_location="cpu", weights_only=True)
        candidate = CBraMod()
        candidate.load_state_dict(weights, strict=True)
        # 作者示例同样移除重建投影获取表示；不新建未经训练的分类头。
        candidate.proj_out = torch.nn.Identity()
        candidate.eval()
        _model = candidate
    return _model


def run_encoder(input_array):
    import torch
    with _model_lock, torch.inference_mode():
        output = _get_model()(torch.from_numpy(input_array)).cpu().numpy()
    if output.shape != (1,22,4,200) or not np.isfinite(output).all():
        raise EEGSkillError("invalid_model_output", "模型输出形状或数值不符合当前编码器契约。")
    return np.asarray(output, dtype=np.float32)


def analyze_sample(sample_id, trial_index, model_id=MODEL_ID):
    if model_id != MODEL_ID:
        from .classifier import analyze_sample as classify
        return classify(sample_id, trial_index, model_id)
    from app.agent.skills import manager
    try:
        manager.get_skill(model_id)
    except ValueError as exc:
        raise EEGSkillError("unknown_model", str(exc)) from None
    started = time.perf_counter()
    trial, sample = load_trial(sample_id, trial_index)
    model_input, preprocessing = preprocess_trial(trial)
    features = run_encoder(model_input)
    pooled = features.mean(axis=(1,2))[0]
    artifact_id = uuid.uuid4().hex
    folder = ARTIFACT_ROOT / artifact_id
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/"features.npz", patch_features=features, mean_pooled=pooled)
    provenance = json.loads((PROJECT_ROOT / "models/eeg/cbramod/provenance.json").read_text(encoding="utf-8"))
    result = {"artifact_id":artifact_id, "created_at":datetime.now(timezone.utc).isoformat(), "sample":sample,
        "model":{"model_id":MODEL_ID,"category":"pretrained_eeg_encoder","weights_revision":MODEL_REVISION,
                 "weights_sha256":WEIGHTS_SHA256,"code_revision":provenance["code_revision"],
                 "classification_head_available":False,"device":"cpu","precision":"float32"},
        "preprocessing":preprocessing, "output":{"feature_shape":list(features.shape),"pooled_shape":list(pooled.shape),
            "pooling":"对通道与时间片取均值，得到200维摘要；这不是分类头",
            "pooled_l2_norm":float(np.linalg.norm(pooled)),"feature_mean":float(features.mean()),
            "feature_std":float(features.std()),"features_sha256":hashlib.sha256(features.tobytes()).hexdigest(),
            "prediction":None,"confidence":None,"classification_available":False},
        "input_sha256":hashlib.sha256(model_input.tobytes()).hexdigest(),
        "elapsed_ms":round((time.perf_counter()-started)*1000,3),
        "scope":"本次为真实预训练表征，未接经过训练的MI任务头，不能推断左手/右手类别、置信度或总体准确率。",
        "artifact_url":f"/agent/eeg/artifacts/{artifact_id}", "features_url":f"/agent/eeg/artifacts/{artifact_id}/features"}
    temporary=folder/"result.tmp"
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(folder/"result.json")
    return result


def read_artifact(artifact_id):
    if not isinstance(artifact_id,str) or not re.fullmatch(r"[0-9a-f]{32}",artifact_id):
        raise EEGSkillError("invalid_artifact_id","执行产物ID格式不合法。")
    path=ARTIFACT_ROOT/artifact_id/"result.json"
    if not path.is_file():
        raise EEGSkillError("artifact_not_found","该执行产物不存在；不能凭会话文字编造结果。")
    result=json.loads(path.read_text(encoding="utf-8"))
    if result.get("artifact_id") != artifact_id:
        raise EEGSkillError("artifact_identity_mismatch","执行产物身份不一致，停止读取。")
    return result


def artifact_document(result):
    if result["model"]["model_id"] != MODEL_ID:
        from .classifier import artifact_document as classify_document
        return classify_document(result)
    model, sample, prep, output = (result[key] for key in ("model","sample","preprocessing","output"))
    text="\n\n".join([
        f"本次真实EEG模型执行产物，artifact_id={result['artifact_id']}。",
        f"模型为CBraMod预训练EEG编码器，model_id={model['model_id']}，权重版本{model['weights_revision']}；运行CPU、float32。",
        f"样本{sample['sample_id']}，数据集{sample['dataset_id']}，文件{sample['file']}，trial_index={sample['trial_index']}（从0计数）；标签未用于模型输入。",
        f"原始单试次形状[22,1000]，采样率250Hz，登记单位{sample.get('unit', prep.get('source_unit', '未记录'))}（uV输入先换算为V）；适配输入为{prep['input_shape']}，200Hz、4个1秒时间片，每片200个时间采样点。这是模型输入时间轴，不是输出特征轴。",
        "适配执行V转uV再除100、共同平均参考、0.3–40Hz带通及重采样。既有MAT在试次内滤波，与作者先滤连续信号再裁窗有边缘差异。",
        f"完整特征张量形状{output['feature_shape']}，轴依次为样本、通道、时间片、隐藏特征维度；末维200是学习到的隐藏特征维度，不表示200个时间采样点。输入与输出形状数值相同不代表轴含义相同。feature_mean={output['feature_mean']:.8g}、feature_std={output['feature_std']:.8g}属于完整张量，不是池化摘要标准差。",
        f"均值池化摘要形状{output['pooled_shape']}，pooled_l2_norm={output['pooled_l2_norm']:.8g}；当前产物未记录池化摘要的标准差，不得用feature_std替代。",
        "没有经过训练的MI分类任务头，分类预测不可用，prediction=null、confidence=null；不能据此判断左手或右手，也不能把表示范数解释成概率或准确率。",
        "该EEG向量用于信号表征，和RAG中BGE-M3的文献文本向量不同；波形/表征未存入Milvus。",
    ])
    return {"type":"analysis_result","artifact_id":result["artifact_id"],"title":"CBraMod真实单试次表征产物",
            "text":text,"source_url":result["artifact_url"],"model_id":MODEL_ID,"sample_id":sample["sample_id"],"trial_index":sample["trial_index"],"analysis_output":dict(output)}
