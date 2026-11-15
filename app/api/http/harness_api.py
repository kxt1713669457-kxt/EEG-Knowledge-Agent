"""研究工作台的本机连接、项目、样本与文件夹入口。"""
from pathlib import Path
from typing import Literal
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.api.http.workspace_account import require_local_page
from app.workspace import store, connections, files, samples, quota, media, eeg_attachments
from app.agent.skills import manager as skill_manager
from app.shared.runtime.query_tasks import query_tasks

router = APIRouter(prefix="/workspace", tags=["工作台"], dependencies=[Depends(require_local_page)])


@router.get("/skills")
def list_skills():
    return {"items":skill_manager.catalog(), "preprocessing_items":skill_manager.preprocessing_catalog()}


class SkillChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


@router.patch("/skills/{identity}")
def enable_skill(identity: str, payload: SkillChange):
    return call(skill_manager.update_skill, identity, enabled=payload.enabled)


@router.delete("/skills/{identity}")
def remove_skill(identity: str):
    return call(skill_manager.update_skill, identity, deleted=True)


@router.post("/skills", status_code=201)
def register_skill(manifest: str = Form(...), weights: UploadFile = File(...)):
    return call(skill_manager.register_skill, manifest, weights.file)


@router.post("/images", status_code=201)
def upload_image(file: UploadFile = File(...)):
    return call(media.register_image, file.file, file.filename)


@router.post("/eeg-attachments", status_code=201)
def upload_eeg_attachments(files: list[UploadFile] = File(...)):
    return call(eeg_attachments.upload, [(file.file, file.filename) for file in files])


@router.get("/images/{identity}")
def read_image(identity: str):
    row = call(media.image_record, identity)
    return FileResponse(call(media.image_path,row), media_type=row["mime"], headers={"X-Content-Type-Options":"nosniff"})


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


class MIPreprocessingRun(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    recording_id: str = Field(min_length=3, max_length=80, pattern=r"^[a-z0-9-]+$")
    profile_id: str = Field(min_length=3, max_length=80, pattern=r"^[a-z0-9-]+$")
    max_trials: int = Field(default=32, ge=1, le=288)


@router.get("/mi-preprocessing/catalog")
def mi_preprocessing_catalog():
    from app.agent.skills.mi_preprocessing import service
    return call(service.catalog)


@router.post("/mi-preprocessing/recordings", status_code=201)
def mi_register_recording(file: UploadFile = File(...),
                          dataset_id: Literal["BCI_IV_2a", "BCI_IV_2b", "MI_GENERIC"] = Form(...),
                          metadata_json: str = Form("{}")):
    from app.agent.skills.mi_preprocessing import service
    # 同步路由在线程池运行；上传按流交给服务端，避免整份原始记录驻留内存。
    return call(service.register_recording, file.file, file.filename, dataset_id, metadata_json)


@router.post("/mi-preprocessing/run")
def mi_preprocess_recording(payload: MIPreprocessingRun):
    from app.agent.skills.mi_preprocessing import service
    return call(service.preprocess_recording, payload.recording_id, payload.profile_id, payload.max_trials)


@router.post("/mi-preprocessing/shu-sessions", status_code=201)
def mi_register_shu_sessions(files: list[UploadFile] = File(...),
        split_id: Literal["shu-session1to4-vs5-v1", "shu-mixed-session-stratified70-v1"] = Form("shu-session1to4-vs5-v1")):
    from app.agent.skills.mi_preprocessing import shu
    if len(files) != 5:
        raise HTTPException(400, "SHU原流程需要同一被试session1至5共五份已分段MAT。")
    return call(shu.register_sessions, [(item.file, item.filename) for item in files], split_id)


@router.get("/mi-preprocessing/artifacts/{artifact_id}")
def mi_preprocessing_artifact(artifact_id: str):
    from app.agent.skills.mi_preprocessing import service
    return call(service.read_artifact, artifact_id)


@router.get("/mi-preprocessing/artifacts/{artifact_id}/data")
def mi_preprocessing_data(artifact_id: str):
    from app.agent.skills.mi_preprocessing import service
    return FileResponse(call(service.data_path, artifact_id), media_type="application/octet-stream",
                        filename=f"mi-{artifact_id}.npz", headers={"X-Content-Type-Options": "nosniff"})


@router.get("/mi-preprocessing/artifacts/{artifact_id}/split/{partition}")
def mi_preprocessing_split(artifact_id: str, partition: Literal["train", "test"]):
    from app.agent.skills.mi_preprocessing import service
    return FileResponse(call(service.split_path, artifact_id, partition), media_type="application/octet-stream",
                        filename=f"mi-{artifact_id}-{partition}.mat", headers={"X-Content-Type-Options": "nosniff"})


def ensure_idle(identity):
    if query_tasks.active_task(identity):
        raise HTTPException(409, "会话仍在处理，请结束后再归档或删除。")


@router.get("/connections")
def connections_list():
    default = connections.default_connection()
    cached = (store.read("model_cache", "default") or {}).get("models", [])
    default["models"] = cached or [{"id": default["default_model"], "name": default["default_model"],
                                   **connections.model_capabilities(default, default["default_model"] or "")}]
    rows = [default] + [r for r in store.all_records("connection") if not r.get("deleted_at")]
    return {"items": [connections.public_connection(r) for r in rows], "presets": connections.PRESETS}


@router.post("/connections", status_code=201)
def connection_create(payload: connections.ConnectionInput):
    return call(connections.save_connection, payload)


@router.put("/connections/{identity}")
def connection_edit(identity: str, payload: connections.ConnectionInput):
    return call(connections.save_connection, payload, identity)


@router.delete("/connections/{identity}")
def connection_remove(identity: str):
    if identity == "default":
        raise HTTPException(400, "项目默认连接保留在.env中。")
    row = call(connections.get_connection, identity)
    # 删除配置而非历史任务；后端运行中的请求持有独立快照。
    row.update(deleted_at=store.now(), encrypted_key="")
    store.write("connection", identity, row)
    store.write("quota_auth", identity, {})
    return {"status": "removed"}


@router.post("/connections/{identity}/models")
def refresh_connection_models(identity: str):
    return {"models": call(connections.list_models, identity), "checked_at": store.now()}


@router.get("/connections/{identity}/capabilities")
def connection_capabilities(identity: str, model: str):
    row = call(connections.get_connection, identity)
    if not re.fullmatch(connections.MODEL_PATTERN, model):
        raise HTTPException(400, "请输入有效的模型ID。")
    return connections.model_capabilities(row, model)


@router.get("/connections/{identity}/quota")
def connection_quota(identity: str, model: str):
    return call(quota.read_quota, identity, model)


@router.post("/connections/{identity}/authorize")
def authorize_connection(identity: str):
    return call(quota.start_authorization, identity)


@router.get("/authorizations/{identity}")
def read_authorization(identity: str):
    return call(quota.authorization_status, identity)


@router.delete("/authorizations/{identity}")
def cancel_authorization(identity: str):
    return call(quota.cancel_authorization, identity)


class SessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    title: str = Field(default="", max_length=80)


class SessionChange(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str | None = Field(default=None, min_length=1, max_length=80)
    project_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    archived: bool | None = None
    deleted: bool | None = None


class ProjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=60)


@router.get("/sessions")
def session_list(view: Literal["recent", "archived", "deleted"] = "recent"):
    rows = store.all_records("session")
    rows = [r for r in rows if (bool(r.get("deleted_at")) if view == "deleted" else
            not r.get("deleted_at") and bool(r.get("archived")) == (view == "archived"))]
    return {"items": sorted(rows, key=lambda r:r.get("updated_at", ""), reverse=True)[:200],
            "projects": [r for r in store.all_records("project") if not r.get("deleted_at")]}


@router.post("/sessions")
def session_register(payload: SessionInput):
    return store.ensure_session(payload.id, payload.title)


@router.patch("/sessions/{identity}")
def session_update(identity: str, payload: SessionChange):
    row = store.read("session", identity)
    if not row:
        raise HTTPException(404, "会话未登记。")
    changes = payload.model_dump(exclude_unset=True)
    if "archived" in changes or "deleted" in changes:
        ensure_idle(identity)
    if changes.get("project_id"):
        project = store.read("project", changes["project_id"])
        if not project or project.get("deleted_at"):
            raise HTTPException(400, "项目文件夹不存在。")
    if "title" in changes:
        if changes["title"] is None:
            raise HTTPException(400, "标题不能为空。")
        row.update(title=changes.pop("title"), manual_title=True)
    if "deleted" in changes:
        row["deleted_at"] = store.now() if changes.pop("deleted") else None
    row.update(changes, updated_at=store.now())
    return store.write("session", identity, row)


@router.post("/projects", status_code=201)
def create_project(payload: ProjectInput):
    return store.new_record("project", name=payload.name, deleted_at=None)


@router.patch("/projects/{identity}")
def rename_project(identity: str, payload: ProjectInput):
    row = store.read("project", identity)
    if not row or row.get("deleted_at"):
        raise HTTPException(404, "项目不存在。")
    row.update(name=payload.name, updated_at=store.now())
    return store.write("project", identity, row)


@router.get("/samples")
def samples_list():
    from app.agent.skills.eeg.service import model_catalog
    return model_catalog()


@router.post("/samples", status_code=201)
def register_sample(file: UploadFile = File(...), sampling_rate_hz: float = Form(...), unit: str = Form(...), dataset_id: str = Form("用户登记数据")):
    return call(samples.register, file.file, file.filename, sampling_rate_hz, unit, dataset_id)


@router.delete("/samples/{identity}")
def unregister_sample(identity: str):
    row = store.read("sample", identity)
    if not row:
        raise HTTPException(400, "只能取消自行登记的样本；内置样本保留。")
    # 不删除原件和既有artifact，已有产物仍可追问。
    row["deleted_at"] = store.now()
    store.write("sample", identity, row)
    return {"status": "unregistered"}


@router.get("/folders")
def list_folders():
    return {"items": [r for r in store.all_records("folder") if not r.get("deleted_at")]}


@router.post("/folders", status_code=201)
def choose_folder(uploads: list[UploadFile] = File(...), paths: list[str] = Form(...), project_id: str | None = Form(None)):
    if project_id:
        row = store.read("project", project_id)
        if not row or row.get("deleted_at"):
            raise HTTPException(400, "项目文件夹不存在。")
    return call(files.import_folder, uploads, paths, project_id=project_id)


@router.get("/folders/{identity}/files")
def folder_files(identity: str):
    return {"workspace": call(files.workspace, identity), "files": call(files.list_files, identity)}


@router.get("/files/{identity}")
def preview_file(identity: str):
    row = call(files.file_record, identity)
    return {k:v for k,v in row.items() if k != "storage_name"}


@router.get("/files/{identity}/download")
def original_file(identity: str):
    row = call(files.file_record, identity)
    return FileResponse(call(files.stored_path, row), filename=row["name"])
