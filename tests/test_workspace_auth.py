"""真实签名/密码哈希/临时 SQLite/两个入口的认证回归，不触碰本机账号。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import jwt

from app.api.http.auth_api import install_auth, LOCAL_ORIGINS
from app.workspace import auth
from app.workspace.secrets import reveal

ORIGIN = "http://127.0.0.1:8001"
PASSWORD = "test-only-correct-password"


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patch = patch.object(auth, "DB_PATH", Path(self.temp.name) / "auth.sqlite3")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        app = FastAPI()
        install_auth(app)

        @app.get("/private")
        def private(): return {"data": "private"}

        @app.post("/private")
        def mutate(): return {"saved": True}

        @app.get("/html")
        def html(): return {"workbench": True}

        self.client = TestClient(app, client=("127.0.0.1", 52000), headers={"Origin": ORIGIN})
        self.addCleanup(self.client.close)

    def setup_account(self):
        response = self.client.post("/auth/setup", json={"username": "Researcher", "password": PASSWORD})
        self.assertEqual(response.status_code, 200, response.text)
        return self.client.cookies.get(auth.COOKIE_NAME)

    def signed(self, **updates):
        with auth.transaction() as db:
            secret = reveal(db.execute("SELECT secret FROM settings").fetchone()[0])
        payload = jwt.decode(self.client.cookies.get(auth.COOKIE_NAME), secret, algorithms=["HS256"], audience=auth.AUDIENCE)
        payload.update(updates)
        return jwt.encode(payload, secret, algorithm="HS256")

    def test_first_setup_and_existing_workspace_does_not_require_default_password(self):
        self.assertEqual(self.client.get("/auth/status").json(), {"setup_required": True})
        self.setup_account()
        self.assertEqual(self.client.get("/auth/status").json(), {"setup_required": False})
        self.assertEqual(self.client.get("/auth/me").json()["user"]["username"], "researcher")
        self.assertEqual(self.client.get("/private").status_code, 200)

    def test_password_is_argon2id_and_signing_key_not_plain(self):
        self.setup_account()
        with auth.transaction() as db:
            row = db.execute("SELECT * FROM users").fetchone()
            secret = db.execute("SELECT secret FROM settings").fetchone()[0]
        self.assertTrue(row["password_hash"].startswith("$argon2id$"))
        self.assertNotIn(PASSWORD, auth.DB_PATH.read_bytes().decode("latin1"))
        self.assertNotEqual(secret, reveal(secret))
        me = self.client.get("/auth/me").text
        self.assertNotIn("password_hash", me)
        self.assertNotIn("jti", me)

    def test_second_setup_cannot_create_another_user_or_replace_credentials(self):
        token = self.setup_account()
        self.assertEqual(self.client.post("/auth/setup", json={"username": "another", "password": PASSWORD}).status_code, 409)
        self.assertEqual(auth.authenticate(token)["username"], "researcher")

    def test_racing_setup_creates_one_account(self):
        def create(index):
            try:
                return auth.setup(f"person{index}", PASSWORD, f"local-{index}")[1]["username"]
            except auth.AuthError as exc:
                return exc.status
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(create, [1, 2]))
        self.assertEqual(results.count(409), 1)
        with auth.transaction() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM users").fetchone()[0], 1)

    def test_setup_remote_client_rejected(self):
        client = TestClient(self.client.app, client=("192.0.2.3", 50000), headers={"Origin": ORIGIN})
        self.addCleanup(client.close)
        self.assertEqual(client.post("/auth/setup", json={"username": "remote", "password": PASSWORD}).status_code, 403)
        self.assertFalse(auth.configured())

    def test_password_and_username_limits(self):
        for payload in [{"username": "abc", "password": "short"}, {"username": "bad name", "password": PASSWORD},
                        {"username": "abc", "password": "a" * 129}, {"username": "abc", "password": PASSWORD, "role": "admin"}]:
            self.assertEqual(self.client.post("/auth/setup", json=payload).status_code, 422)

    def test_validation_errors_never_echo_password_or_body(self):
        password = "q7Q?!"
        response = self.client.post("/auth/setup", json={"username": "valid", "password": password})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('"input"', response.text)
        self.assertNotIn(password, response.text)
        self.assertEqual(self.client.post("/auth/setup", json={"username": "valid", "password": " " * 12}).status_code, 422)

    def test_login_case_insensitive_and_generic_failure(self):
        self.setup_account()
        wrong = self.client.post("/auth/login", json={"username": "researcher", "password": "wrong"})
        unknown = self.client.post("/auth/login", json={"username": "unknown", "password": "wrong"})
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(wrong.json(), unknown.json())
        self.assertEqual(self.client.post("/auth/login", json={"username": "RESEARCHER", "password": PASSWORD}).status_code, 200)

    def test_failed_login_rate_limit_shared_in_db(self):
        self.setup_account()
        for _ in range(8):
            self.client.post("/auth/login", json={"username": "researcher", "password": "wrong"})
        response = self.client.post("/auth/login", json={"username": "researcher", "password": PASSWORD})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "300")

    def test_cookie_http_only_same_site_and_no_token_json(self):
        response = self.client.post("/auth/setup", json={"username": "researcher", "password": PASSWORD})
        cookie = response.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=lax", cookie)
        self.assertIn("Path=/", cookie)
        self.assertNotIn("access_token", response.json())
        self.assertEqual(response.headers["cache-control"].split(",")[0], "no-store")

    def test_https_cookie_secure(self):
        with TestClient(self.client.app, base_url="https://127.0.0.1:8001", client=("127.0.0.1", 50000),
                        headers={"Origin": ORIGIN}) as client:
            response = client.post("/auth/setup", json={"username": "researcher", "password": PASSWORD})
            self.assertIn("Secure", response.headers["set-cookie"])

    def test_missing_expired_tampered_wrong_audience_wrong_issuer_and_version(self):
        token = self.setup_account()
        for bad in [None, "broken", token + "a", self.signed(exp=int(time.time()) - 60),
                    self.signed(aud="other-api"), self.signed(iss="other-issuer"), self.signed(ver=999), self.signed(jti="unknown")]:
            with self.assertRaises(auth.AuthError): auth.authenticate(bad)

    def test_unsigned_and_wrong_algorithm_rejected(self):
        self.setup_account()
        with auth.transaction() as db:
            secret = reveal(db.execute("SELECT secret FROM settings").fetchone()[0])
        payload = jwt.decode(self.client.cookies.get(auth.COOKIE_NAME), secret, algorithms=["HS256"], audience=auth.AUDIENCE)
        for token in [jwt.encode(payload, "", algorithm="none"), jwt.encode(payload, secret, algorithm="HS384")]:
            with self.assertRaises(auth.AuthError): auth.authenticate(token)

    def test_database_disabled_identity_even_with_valid_jwt(self):
        token = self.setup_account()
        with auth.transaction() as db: db.execute("UPDATE users SET active=0")
        with self.assertRaises(auth.AuthError): auth.authenticate(token)

    def test_logout_revokes_copied_token_without_revoking_other_device(self):
        token = self.setup_account()
        other, _ = auth.login("researcher", PASSWORD, "other-device")
        self.assertEqual(self.client.post("/auth/logout").status_code, 200)
        with self.assertRaises(auth.AuthError): auth.authenticate(token)
        self.assertEqual(auth.authenticate(other)["username"], "researcher")
        self.assertEqual(self.client.get("/private").status_code, 401)

    def test_password_change_revokes_all_sessions_and_old_password(self):
        token = self.setup_account()
        other, _ = auth.login("researcher", PASSWORD, "other-device")
        response = self.client.post("/auth/password", json={"current_password": PASSWORD, "new_password": "changed-test-password"})
        self.assertEqual(response.status_code, 200)
        for old in [token, other]:
            with self.assertRaises(auth.AuthError): auth.authenticate(old)
        self.assertEqual(self.client.post("/auth/login", json={"username": "researcher", "password": PASSWORD}).status_code, 401)
        self.assertEqual(self.client.post("/auth/login", json={"username": "researcher", "password": "changed-test-password"}).status_code, 200)

    def test_wrong_current_password_preserves_login(self):
        token = self.setup_account()
        self.assertEqual(self.client.post("/auth/password", json={"current_password": "wrong", "new_password": "changed-test-password"}).status_code, 400)
        self.assertEqual(auth.authenticate(token)["username"], "researcher")

    def test_cookie_mutation_rejects_foreign_and_missing_origin(self):
        self.setup_account()
        self.assertEqual(self.client.post("/private", headers={"Origin": "https://evil.example"}).status_code, 403)
        self.client.headers.pop("origin")
        self.assertEqual(self.client.post("/private").status_code, 403)
        self.assertEqual(self.client.get("/private").status_code, 200)

    def test_bearer_supported_and_not_allowed_to_fall_back_to_cookie(self):
        token = self.setup_account()
        self.client.headers.pop("origin")
        self.assertEqual(self.client.post("/private", headers={"Authorization": f"Bearer {token}"}).status_code, 200)
        self.assertEqual(self.client.get("/private", headers={"Authorization": "Bearer invalid"}).status_code, 401)

    def test_cross_port_credentials_cors_and_foreign_preflight(self):
        self.setup_account()
        response = self.client.get("/private", headers={"Origin": "http://127.0.0.1:8000"})
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://127.0.0.1:8000")
        self.assertEqual(response.headers["Access-Control-Allow-Credentials"], "true")
        for origin in ["https://evil.example", "http://127.0.0.1:9999"]:
            response = self.client.options("/private", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
            self.assertEqual(response.status_code, 400)
            self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    def test_html_redirect_api_401_and_login_page_public(self):
        response = self.client.get("/html", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/login?next=%2Fhtml")
        self.assertEqual(self.client.get("/private").status_code, 401)
        self.assertEqual(self.client.get("/login").status_code, 200)

    def test_auth_survives_new_connection_to_store(self):
        token = self.setup_account()
        self.assertEqual(auth.authenticate(token)["username"], "researcher")
        other = TestClient(self.client.app, headers={"Authorization": f"Bearer {token}"})
        self.addCleanup(other.close)
        self.assertEqual(other.get("/private").status_code, 200)

    def test_image_proxy_rejects_foreign_original_and_traversal_before_storage(self):
        from app.api.http import source_images
        base = source_images.minio_gateway.build_image_url("test", "figure.png")
        self.assertTrue(source_images.image_object_key(base).endswith("test/figure.png"))
        invalid = ["https://evil.example/figure.png", base + "?x=1", base.replace("figure.png", "../source.pdf"),
                   base.replace("figure.png", "%2e%2e/source.png"), base.replace("figure.png", "%252e%252e/source.png"),
                   base.replace("figure.png", "script.svg"), base.replace("figure.png", "file.html")]
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError): source_images.image_object_key(url)

    def test_private_bucket_initialization_never_restores_public_policy(self):
        from app.shared.clients import minio_utils
        client = MagicMock()
        client.bucket_exists.return_value = True
        minio_utils._create_bucket_ready(client)
        client.delete_bucket_policy.assert_called_once_with(minio_utils.minio_config.bucket_name)
        client.set_bucket_policy.assert_not_called()
        client.make_bucket.assert_not_called()

    def test_image_proxy_streams_authenticated_object_and_closes_response(self):
        from app.api.http import source_images, query_server
        token = self.setup_account()
        response = MagicMock()
        response.stream.return_value = [b"real-test-image-bytes"]
        client = TestClient(query_server.app, headers={"Authorization": f"Bearer {token}"})
        self.addCleanup(client.close)
        url = source_images.minio_gateway.build_image_url("test", "figure.png")
        with patch.object(source_images.minio_gateway, "read_original", return_value=response) as read:
            result = client.get("/sources/image", params={"url": url})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content, b"real-test-image-bytes")
        self.assertEqual(result.headers["content-type"], "image/png")
        read.assert_called_once_with(source_images.image_object_key(url))
        response.close.assert_called_once()
        response.release_conn.assert_called_once()

    def test_real_query_and_import_routes_require_login(self):
        from app.api.http import query_server, import_server
        routes = {query_server.app: [("GET", "/workspace/connections"), ("GET", "/workspace/account"),
                     ("GET", "/history/any"), ("GET", "/tasks/any"), ("GET", "/stream/any"),
                     ("GET", "/sources/image"),
                     ("GET", "/agent/eeg/catalog"), ("GET", "/workspace/images/any"),
                     ("GET", "/agent/eeg/artifacts/any/features"), ("GET", "/sources/original/any/pdf"),
                     ("POST", "/query"), ("POST", "/chat/tasks"), ("POST", "/agent/tools/execute"),
                     ("DELETE", "/workspace/skills/any"), ("GET", "/openapi.json")],
                  import_server.app: [("POST", "/upload"), ("GET", "/status/any"), ("GET", "/original/any")]}
        for app, paths in routes.items():
            # 不进入 TestClient 上下文，不触发业务 lifespan / 生产任务仓库恢复。
            client = TestClient(app, headers={"Origin": ORIGIN})
            self.addCleanup(client.close)
            for method, path in paths:
                with self.subTest(path=path): self.assertEqual(client.request(method, path).status_code, 401)
            self.assertEqual(client.get("/health").status_code, 200)

    def test_two_api_apps_share_login_and_logout_revocation(self):
        from app.api.http import query_server, import_server
        query_client = TestClient(query_server.app, client=("127.0.0.1", 50000), headers={"Origin": ORIGIN})
        self.addCleanup(query_client.close)
        response = query_client.post("/auth/setup", json={"username": "researcher", "password": PASSWORD})
        self.assertEqual(response.status_code, 200)
        token = query_client.cookies.get(auth.COOKIE_NAME)
        import_client = TestClient(import_server.app, headers={"Origin": "http://127.0.0.1:8000"},
                                   cookies={auth.COOKIE_NAME: token})
        self.addCleanup(import_client.close)
        self.assertEqual(import_client.get("/auth/me").status_code, 200)
        self.assertEqual(import_client.post("/auth/logout").status_code, 200)
        self.assertEqual(query_client.get("/auth/me").status_code, 401)


if __name__ == "__main__": unittest.main()
