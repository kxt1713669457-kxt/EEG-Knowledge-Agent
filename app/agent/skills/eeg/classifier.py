"""用户已训练本地网络的数值推理；分类来自真实logits，不让VLM代算。"""
from datetime import datetime, timezone
import hashlib
import json
from threading import Lock
import time
import uuid

import numpy as np

from app.agent.skills.eeg import service
from app.agent.skills import manager

_models, _guard = {}, Lock()


def load_network(skill, *, weights=None):
    import torch
    if skill["architecture"] == "dwda_multiscale_v1":
        from .vendor.local_dwdanet import EEGTransformer
    else:
        from .vendor.local_dwtnet import EEGTransformer
    channels = skill["input"]["channels"]
    if weights is None:
        path = manager.WEIGHT_ROOT / skill["id"] / "weights.pt"
        if not path.is_file() or service.file_sha256(path) != skill["weights_sha256"]:
            raise ValueError("skill权重缺失或完整性检查不通过。")
        weights = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or not weights or not all(isinstance(v, torch.Tensor) for v in weights.values()):
        raise ValueError("只接受纯张量state_dict。")
    candidate = EEGTransformer(heads=2, emb_size=16, depth=6, database_type={22:"A",3:"B",32:"C"}[channels],
        eeg1_f1=8, eeg1_D=2, eeg1_kernel_size=(32,64,128), eeg1_pooling_size1=8,
        eeg1_pooling_size2=8, eeg1_dropout_rate=.25, flatten_eeg1=240)
    candidate.load_state_dict(weights, strict=True)
    return candidate.eval()


def analyze_sample(sample_id, trial_index, model_id):
    import torch
    started = time.perf_counter()
    try:
        skill = manager.get_skill(model_id)
        if skill["adapter"] != "eeg_classifier": raise ValueError("不是分类skill。")
        contract = skill["input"]
        entry = service.sample_entries().get(sample_id)
        if entry is None: raise ValueError("样本未登记。")
        if entry.get("preprocessing_artifact_id"):
            from app.agent.skills.mi_preprocessing import service as mi_service
            source = mi_service.read_artifact(entry["preprocessing_artifact_id"])
            if source["sample_id"] != sample_id:
                raise ValueError("预处理来源与样本身份不一致。")
            if service.file_sha256(service.sample_path(sample_id)) != source["output"]["sha256"]:
                raise ValueError("派生样本数据与预处理产物哈希不一致。")
        from app.workspace.samples import read_array
        path = service.sample_path(sample_id)
        data = read_array(path)
        if data.shape[1:] != (contract["channels"], contract["points"]) or entry["sampling_rate_hz"] != contract["sampling_rate_hz"]:
            raise ValueError("样本通道数、采样率或试次长度与此skill不兼容；不自动补通道或裁窗。")
        if type(trial_index) is not int or not 0 <= trial_index < len(data): raise ValueError("试次下标越界。")
        trial = np.asarray(data[trial_index], dtype=np.float32)
        if not np.isfinite(trial).all(): raise ValueError("波形含非有限数值。")
        if entry["unit"] == "uV": trial = trial / 1_000_000
        norm = skill["normalization"]
        tensor = np.ascontiguousarray((trial - norm["mean"]) / norm["std"], dtype=np.float32)[None,None]
        if not np.isfinite(tensor).all(): raise ValueError("归一化后数值无效。")
        with _guard, torch.inference_mode():
            cache_key = (model_id, skill["weights_sha256"])
            if cache_key not in _models: _models[cache_key] = load_network(skill)
            features, logits = _models[cache_key](torch.from_numpy(tensor))
            features, scores = features.cpu().numpy(), torch.softmax(logits, dim=-1)[0].cpu().numpy()
        if scores.shape != (len(skill["labels"]),) or not np.isfinite(scores).all(): raise ValueError("分类输出不符合标签契约。")
    except (ValueError, OSError) as exc:
        raise service.EEGSkillError("skill_input_or_model_unavailable", str(exc)) from None
    index = int(scores.argmax()); artifact_id = uuid.uuid4().hex
    folder = service.ARTIFACT_ROOT / artifact_id;folder.mkdir(parents=True, exist_ok=False)
    pooled = features.reshape(-1,features.shape[-1]).mean(axis=0)
    np.savez_compressed(folder/"features.npz", features=features, logits=logits.cpu().numpy(), probabilities=scores)
    result = {"artifact_id":artifact_id, "created_at":datetime.now(timezone.utc).isoformat(),
        "sample":{"sample_id":sample_id,"dataset_id":entry["dataset_id"],"file":entry["file"],"trial_index":trial_index,
                  "sampling_rate_hz":entry["sampling_rate_hz"],"unit":entry["unit"],"source_shape":list(data.shape),
                  "source_file_sha256":service.file_sha256(path),"labels_used_as_model_input":False},
        "model":{"model_id":model_id,"name":skill["name"],"category":"trained_eeg_classifier","weights_sha256":skill["weights_sha256"],
                 "weights_revision":skill["weights_sha256"],"classification_head_available":True,"device":"cpu","precision":"float32"},
        "preprocessing":{"input_shape":list(tensor.shape),"profile_id":model_id,"normalization":norm,
                         "unit_conversion":"uV输入换为V；沿用已分段4秒原始顺序；不使用标签/不拟合此试次"},
        "output":{"prediction":skill["labels"][index],"class_index":index,"classification_available":True,
                  "confidence":None,"softmax_score":float(scores[index]),"score_calibrated":False,
                  "class_scores":[{"label":label,"score":float(score)} for label,score in zip(skill["labels"],scores)],
                  "feature_shape":list(features.shape),"pooled_shape":list(pooled.shape),"pooled_l2_norm":float(np.linalg.norm(pooled)),
                  "feature_mean":float(features.mean()),"feature_std":float(features.std())},
        "input_sha256":hashlib.sha256(tensor.tobytes()).hexdigest(),"elapsed_ms":round((time.perf_counter()-started)*1000,3),
        "scope":"该次真实训练模型的预测，不是人工真值；softmax未校准，不等于正确率。"+contract.get("subject_scope", "")+" "+contract.get("channel_order", "")+" "+norm["note"]+" "+entry.get("preprocessing_scope", ""),
        "artifact_url":f"/agent/eeg/artifacts/{artifact_id}","features_url":f"/agent/eeg/artifacts/{artifact_id}/features"}
    if entry.get("preprocessing_artifact_id"):
        result["sample"].update(preprocessing_artifact_id=entry["preprocessing_artifact_id"],
                                source_recording_sha256=entry.get("source_recording_sha256"),
                                channel_names=entry.get("channel_names"))
        result["preprocessing"]["upstream_profile_id"] = entry.get("preprocessing_profile_id")
    temporary = folder/"result.tmp";temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");temporary.replace(folder/"result.json")
    return result


def artifact_document(result):
    output = result["output"]
    text = (f"真实数值EEG分类执行产物，artifact_id={result['artifact_id']}。\n模型：{result['model']['name']}，"
            f"model_id={result['model']['model_id']}，权重SHA256={result['model']['weights_sha256']}。\n"
            f"样本{result['sample']['sample_id']}，trial_index={result['sample']['trial_index']}（从0计数）；标签未用于模型输入。\n"
            f"该模型本次预测类别：{output['prediction']}；softmax分数={output['softmax_score']:.6g}，未经校准，不是置信度/准确率，不代表真值。\n"
            f"各类别分数：{json.dumps(output['class_scores'],ensure_ascii=False)}。\n"
            f"输入形状：{result['preprocessing']['input_shape']}；真实表征形状：{output['feature_shape']}。\n"
            f"适用边界：{result['scope']}\n此结果来自数值信号模型，不能反过来说仅凭截图就完成了分类。")
    return {"type":"analysis_result","artifact_id":result["artifact_id"],"title":result["model"]["name"]+"单试次预测",
            "text":text,"source_url":result["artifact_url"],"model_id":result["model"]["model_id"],
            "sample_id":result["sample"]["sample_id"],"trial_index":result["sample"]["trial_index"],"analysis_output":dict(output)}
