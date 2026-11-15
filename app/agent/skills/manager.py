"""应用内EEG skills目录；可启停/移除/登记受支持适配器，不执行上传的Python。"""
import hashlib
import json
import math
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Literal

from app.shared.runtime.logger import PROJECT_ROOT
from app.workspace import store

CATALOG_ROOT = PROJECT_ROOT / "app/agent/skills/catalog"
WEIGHT_ROOT = PROJECT_ROOT / "models/eeg/local-skills"


class SignalContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channels: Literal[3, 22, 32]
    sampling_rate_hz: Literal[250] = 250
    points: Literal[1000] = 1000
    unit: Literal["V"] = "V"
    dataset: str = Field(min_length=1, max_length=100)
    subject_scope: str = Field(default="用户登记模型，适用域需另行验证。", max_length=300)
    channel_order: str = Field(default="须与原训练MAT通道顺序一致；未登记通道名时不声称已自动核验顺序。", max_length=500)


class Normalization(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean: float
    std: float = Field(gt=0)
    method: str = Field(max_length=100)
    note: str = Field(max_length=600)

    @field_validator("mean", "std")
    @classmethod
    def finite(cls, value):
        if not math.isfinite(value):
            raise ValueError("归一化参数必须是有限数值。")
        return value


class SkillManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    name: str = Field(min_length=1, max_length=80)
    adapter: Literal["eeg_classifier"] = "eeg_classifier"
    architecture: Literal["dwda_multiscale_v1", "dwt_multiscale_v4"]
    description: str = Field(min_length=1, max_length=1000)
    enabled: bool = True
    input: SignalContract
    labels: list[str] = Field(min_length=2, max_length=4)
    normalization: Normalization
    weights_sha256: str = Field(default="", pattern=r"^(|[0-9a-f]{64})$")
    instructions: str = Field(default="", max_length=4000)
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def label_contract(self):
        if len(self.labels) != (4 if self.input.channels == 22 else 2):
            raise ValueError("分类标签数量与已支持的网络头不匹配。")
        if len(set(self.labels)) != len(self.labels) or any(not v.strip() or len(v) > 40 for v in self.labels):
            raise ValueError("标签须唯一且明确。")
        return self


def all_entries():
    rows = {"cbramod-pretrained": {"id": "cbramod-pretrained", "name": "CBraMod · 预训练表征", "adapter": "cbramod",
        "description": "22通道2a单试次表征，没有训练好的MI分类头。", "enabled": True,
        "input": {"channels":22, "sampling_rate_hz":250, "points":1000}, "labels": []}}
    for path in sorted(CATALOG_ROOT.glob("*.json")):
        row = SkillManifest.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
        rows[row["id"]] = row
    for override in store.all_records("skill"):
        identity = override["id"]
        if override.get("manifest"):
            rows[identity] = override["manifest"]
        if identity in rows:
            rows[identity].update(enabled=override.get("enabled", True), deleted_at=override.get("deleted_at"))
    return rows


def get_skill(identity, *, require_enabled=True):
    row = all_entries().get(identity)
    if row is None or row.get("deleted_at"):
        raise ValueError("skill未登记或已移除，不能执行。")
    if require_enabled and not row.get("enabled"):
        raise ValueError("该skill已停用，不能执行。")
    return row


def public_skill(row):
    path = WEIGHT_ROOT / row["id"] / "weights.pt"
    ready = (PROJECT_ROOT / "models/eeg/cbramod/pretrained_weights.pth").is_file() if row["adapter"] == "cbramod" else path.is_file()
    return {k:row.get(k) for k in ("id", "name", "description", "enabled", "input", "labels", "architecture", "weights_sha256")} | {
        "model_id":row["id"], "ready":ready, "classification_available":row["adapter"] == "eeg_classifier",
        "category":"trained_eeg_classifier" if row["adapter"] == "eeg_classifier" else "pretrained_eeg_encoder",
        "normalization_note":row.get("normalization", {}).get("note"),
        "scope":row.get("input", {}).get("subject_scope", "预训练表征，非MI分类。")}


def catalog():
    return [public_skill(row) for row in all_entries().values() if not row.get("deleted_at")]


def preprocessing_catalog():
    from app.agent.skills.mi_preprocessing.catalog import profiles
    return [{"id": row["id"], "name": row["name"], "description": row["description"],
             "ready": row["ready"], "category": "eeg_preprocessing", "tools": ["register_eeg_recording", "preprocess_eeg_recording", "get_eeg_preprocessing"],
             "scope": row.get("scope") or row.get("model_compatibility") or "预处理与质量检查；识别须另外匹配就绪模型。",
             "skill_document": "app/agent/skills/mi_preprocessing/SKILL.md"} for row in profiles()]


def workflow_instructions():
    root = Path(__file__).parent
    return "\n\n".join((root / folder / "SKILL.md").read_text(encoding="utf-8") for folder in ("mi_preprocessing", "eeg_inference"))


def update_skill(identity, *, enabled=None, deleted=None):
    row = get_skill(identity, require_enabled=False)
    previous = store.read("skill", identity) or {"id":identity}
    if enabled is not None: previous["enabled"] = enabled
    if deleted is not None: previous["deleted_at"] = store.now() if deleted else None
    previous["updated_at"] = store.now()
    store.write("skill", identity, previous)
    return public_skill(row | previous)


def register_skill(manifest_text, stream):
    manifest = SkillManifest.model_validate_json(manifest_text)
    if manifest.id == "cbramod-pretrained" or manifest.id in all_entries():
        raise ValueError("skill ID已存在，请使用新ID；已有产物保持原模型版本。")
    blob = stream.read(32 * 1024 * 1024 + 1)
    if not blob or len(blob) > 32 * 1024 * 1024:
        raise ValueError("纯参数权重须在32MB以内。")
    digest = hashlib.sha256(blob).hexdigest()
    if manifest.weights_sha256 and manifest.weights_sha256 != digest:
        raise ValueError("权重SHA256与清单不符。")
    from io import BytesIO
    from app.agent.skills.eeg.classifier import load_network
    import torch
    # 新上传只接收张量state_dict，旧全模型文件应先用本地准备脚本明确转换。
    try:
        import zipfile
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            if sum(row.file_size for row in archive.infolist()) > 128*1024*1024:
                raise ValueError("权重展开体积超限。")
        weights = torch.load(BytesIO(blob), map_location="cpu", weights_only=True)
        candidate = load_network(manifest.model_dump(), weights=weights)
        with torch.inference_mode():
            _, logits = candidate(torch.zeros(1,1,manifest.input.channels,1000))
        if logits.shape != (1,len(manifest.labels)) or not torch.isfinite(logits).all():
            raise ValueError()
    except Exception:
        raise ValueError("权重不是该适配器可严格加载的纯参数，或输出与标签不符；未登记。") from None
    folder = WEIGHT_ROOT / manifest.id
    folder.mkdir(parents=True, exist_ok=False)
    values = manifest.model_dump() | {"weights_sha256":digest}
    try:
        (folder / "weights.pt").write_bytes(blob)
        store.write("skill", manifest.id, {"id":manifest.id, "manifest":values, "enabled":manifest.enabled, "created_at":store.now()})
    except Exception:
        (folder / "weights.pt").unlink(missing_ok=True)
        folder.rmdir()  # 本次新建且只含本次权重的目录，不递归删除。
        raise
    return public_skill(values)
