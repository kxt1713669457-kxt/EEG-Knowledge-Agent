"""旧业务 HTTP 用例的真实登录夹具：每个客户端独立临时认证库，不关闭认证。

仅同步 unittest 使用；并发认证/双进程边界在 test_workspace_auth 中单独覆盖。
"""
from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.workspace import auth


class AuthenticatedClient(TestClient):
    def __init__(self, *args, **kwargs):
        self._auth_temp = tempfile.TemporaryDirectory()
        self._auth_path = Path(self._auth_temp.name) / "auth.sqlite3"
        with patch.object(auth, "DB_PATH", self._auth_path):
            token, _ = auth.setup("business-test", "temporary-business-test-password", "test-only")
        headers = {"Origin": "http://127.0.0.1:8001", "Authorization": f"Bearer {token}", **kwargs.pop("headers", {})}
        super().__init__(*args, headers=headers, **kwargs)

    def request(self, *args, **kwargs):
        with patch.object(auth, "DB_PATH", self._auth_path):
            return super().request(*args, **kwargs)

    def close(self):
        super().close()
        self._auth_temp.cleanup()
