"""查询任务快照。单文档版本更新，不把数据库状态保存称作图执行恢复。"""
from copy import deepcopy
from datetime import datetime, timezone
import os

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.write_concern import WriteConcern

TERMINAL_STATUSES = {"completed", "failed", "interrupted"}
APPLICATION = "eeg_knowledge_agent"


class TaskStoreUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("任务状态存储暂不可用，本次操作未确认，请稍后查询任务状态。")


class MongoTaskRepository:
    storage = "mongo"

    def __init__(self, collection):
        self.collection = collection
        self.session_heads = collection.database.get_collection("query_session_heads", write_concern=collection.write_concern)

    @classmethod
    def connect(cls):
        client = MongoClient(os.environ["MONGO_URL"], serverSelectionTimeoutMS=4000,
                             connectTimeoutMS=4000, socketTimeoutMS=5000)
        try:
            client.admin.command("ping")
            collection = client[os.environ["MONGO_DB_NAME"]].get_collection(
                "query_tasks", write_concern=WriteConcern(w=1, j=True, wtimeout=4000))
            collection.create_index([("session_id", 1), ("created_at", -1)])
            repository = cls(collection)
            repository.client = client
            return repository
        except Exception:
            client.close()
            raise TaskStoreUnavailable() from None

    def save(self, record):
        updated = deepcopy(record)
        previous = updated.get("storage_version", 0)
        updated.update(_id=updated["task_id"], storage_version=previous + 1, application=APPLICATION)
        try:
            if previous == 0:
                self.collection.insert_one(updated)
            else:
                result = self.collection.replace_one({"_id": updated["task_id"], "storage_version": previous,
                                                     "owner_id": updated["owner_id"], "application": APPLICATION}, updated)
                if result.matched_count != 1:
                    raise TaskStoreUnavailable()
            if previous == 0 and updated.get("mode") in {"rag", "agent", "chat"}:
                self.session_heads.update_one({"_id": APPLICATION + ":" + updated["session_id"]},
                    {"$set": {"task_id": updated["task_id"]}}, upsert=True)
            elif updated.get("mode") == "history_clear" and updated["status"] == "completed":
                self.session_heads.update_one({"_id": APPLICATION + ":" + updated["session_id"]},
                    {"$set": {"task_id": None}}, upsert=True)
        except PyMongoError:
            raise TaskStoreUnavailable() from None
        return {key: value for key, value in updated.items() if key != "_id"}

    def get(self, task_id):
        try:
            return self.collection.find_one({"_id": task_id, "application": APPLICATION}, {"_id": 0})
        except PyMongoError:
            raise TaskStoreUnavailable() from None

    def latest(self, session_id):
        try:
            head = self.session_heads.find_one({"_id": APPLICATION + ":" + session_id})
            if head is not None:
                return self.get(head["task_id"]) if head.get("task_id") else None
            # 本段早期已落库、尚无会话定位记录的任务，保留只读兼容。
            scope = {"application": APPLICATION, "session_id": session_id}
            cleared = self.collection.find_one({**scope, "mode": "history_clear", "status": "completed"},
                                               {"_id": 0}, sort=[("created_at", -1)])
            selector = {**scope, "mode": {"$in": ["rag", "agent", "chat"]}}
            if cleared:
                selector["created_at"] = {"$gt": cleared["finished_at"]}
            return self.collection.find_one(selector,
                                            {"_id": 0}, sort=[("created_at", -1)])
        except PyMongoError:
            raise TaskStoreUnavailable() from None

    def interrupt_previous(self, owner_id):
        now = datetime.now(timezone.utc).isoformat()
        try:
            return self.collection.update_many(
                {"application": APPLICATION, "status": {"$nin": list(TERMINAL_STATUSES)}, "owner_id": {"$ne": owner_id}},
                {"$set": {"status": "interrupted", "phase": "finished", "accepting_updates": False,
                          "finished_at": now, "updated_at": now, "running_list": [],
                          "error": {"code": "service_restarted", "message": "查询服务已重启，本轮执行中断。已保存的记录和分析产物仍可查询；系统未自动重跑。"}},
                 "$inc": {"storage_version": 1}}).modified_count
        except PyMongoError:
            raise TaskStoreUnavailable() from None
