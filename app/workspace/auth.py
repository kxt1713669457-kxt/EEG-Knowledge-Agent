"""个人工作台认证：一个普通账号，Argon2id + JWT + 可撤销登录会话。

账号库与现有资料库分离；两个 API 共享此库，不把已有资料误当成多租户数据。
"""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import hashlib
import os
import secrets
import sqlite3
import time
import uuid

import jwt
from pwdlib import PasswordHash

from app.shared.runtime.logger import PROJECT_ROOT
from app.workspace.secrets import protect, reveal

DB_PATH = Path(os.environ.get("EEG_AUTH_DB_PATH") or PROJECT_ROOT / "output/runtime/auth.sqlite3")
COOKIE_NAME = "eeg_access"
ISSUER = "eeg-workspace"
AUDIENCE = "eeg-workspace-api"
TOKEN_SECONDS = 8 * 60 * 60
PASSWORDS = PasswordHash.recommended()


class AuthError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


@contextmanager
def transaction():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), secret TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS users (singleton INTEGER PRIMARY KEY CHECK(singleton=1), id TEXT UNIQUE NOT NULL, username TEXT NOT NULL, password_hash TEXT NOT NULL, token_version INTEGER NOT NULL DEFAULT 1, active INTEGER NOT NULL DEFAULT 1)")
        db.execute("CREATE TABLE IF NOT EXISTS sessions (jti TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires INTEGER NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS attempts (bucket TEXT PRIMARY KEY, started INTEGER NOT NULL, count INTEGER NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        if not db.execute("SELECT 1 FROM settings WHERE id=1").fetchone():
            db.execute("INSERT INTO settings VALUES(1,?)", (protect(secrets.token_urlsafe(48)),))
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def configured():
    with transaction() as db:
        return db.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def public_user(row):
    return {"id": row["id"], "username": row["username"]}


def _rate_limit(db, client):
    # 不信任客户端自行填写的 X-Forwarded-For；失败限制跨两个 API 进程共享。
    bucket = hashlib.sha256(client.encode()).hexdigest()
    now = int(time.time())
    row = db.execute("SELECT * FROM attempts WHERE bucket=?", (bucket,)).fetchone()
    if row and now - row["started"] < 300:
        if row["count"] >= 8:
            raise AuthError(429, "尝试次数过多，请五分钟后重试。")
        db.execute("UPDATE attempts SET count=count+1 WHERE bucket=?", (bucket,))
    else:
        db.execute("INSERT OR REPLACE INTO attempts VALUES(?,?,1)", (bucket, now))
    db.execute("DELETE FROM attempts WHERE started < ?", (now - 600,))
    return bucket


@lru_cache(maxsize=1)
def _dummy_hash():
    return PASSWORDS.hash(secrets.token_urlsafe(24))


def _issue(db, user):
    now, jti = int(time.time()), uuid.uuid4().hex
    secret = reveal(db.execute("SELECT secret FROM settings WHERE id=1").fetchone()[0])
    claims = {"sub": user["id"], "ver": user["token_version"], "jti": jti,
              "iss": ISSUER, "aud": AUDIENCE, "iat": now, "nbf": now, "exp": now + TOKEN_SECONDS}
    token = jwt.encode(claims, secret, algorithm="HS256")
    db.execute("DELETE FROM sessions WHERE expires <= ?", (now,))
    db.execute("INSERT INTO sessions VALUES(?,?,?)", (jti, user["id"], now + TOKEN_SECONDS))
    return token, public_user(user)


def setup(username, password, client):
    with transaction() as db:
        _rate_limit(db, client)
    password_hash = PASSWORDS.hash(password)
    with transaction() as db:
        if db.execute("SELECT 1 FROM users").fetchone():
            raise AuthError(409, "研究空间已经设置账号，请直接登录。")
        db.execute("INSERT INTO users(singleton,id,username,password_hash) VALUES(1,?,?,?)",
                   (uuid.uuid4().hex, username.casefold(), password_hash))
        db.execute("DELETE FROM attempts WHERE bucket=?", (hashlib.sha256(client.encode()).hexdigest(),))
        user = db.execute("SELECT * FROM users WHERE singleton=1").fetchone()
        return _issue(db, user)


def login(username, password, client):
    with transaction() as db:
        bucket = _rate_limit(db, client)
        user = db.execute("SELECT * FROM users WHERE singleton=1").fetchone()
    # 用户名不匹配也验证一个哈希，不将账号是否存在暴露成不同错误。
    matched = bool(user and secrets.compare_digest(user["username"], username.casefold()))
    valid = PASSWORDS.verify(password, user["password_hash"] if matched else _dummy_hash())
    if not valid or not matched or not user["active"]:
        raise AuthError(401, "账号或密码不正确。")
    with transaction() as db:
        current = db.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
        if not current or not current["active"] or current["token_version"] != user["token_version"]:
            raise AuthError(401, "账号状态已变化，请重新登录。")
        db.execute("DELETE FROM attempts WHERE bucket=?", (bucket,))
        return _issue(db, current)


def authenticate(token):
    if not token or len(token) > 4096:
        raise AuthError(401, "请先登录研究空间。")
    with transaction() as db:
        secret = reveal(db.execute("SELECT secret FROM settings WHERE id=1").fetchone()[0])
        try:
            claims = jwt.decode(token, secret, algorithms=["HS256"], issuer=ISSUER, audience=AUDIENCE,
                                options={"require": ["sub", "jti", "ver", "exp", "iat", "nbf", "iss", "aud"]})
            if (not isinstance(claims["sub"], str) or not isinstance(claims["jti"], str)
                    or type(claims["ver"]) is not int):
                raise ValueError("invalid claims")
        except (jwt.PyJWTError, ValueError, TypeError):
            raise AuthError(401, "登录已失效，请重新登录。") from None
        user = db.execute("SELECT * FROM users WHERE id=?", (claims["sub"],)).fetchone()
        session = db.execute("SELECT * FROM sessions WHERE jti=? AND user_id=? AND expires>?",
                             (claims["jti"], claims["sub"], int(time.time()))).fetchone()
        if not user or not user["active"] or user["token_version"] != claims["ver"] or not session:
            raise AuthError(401, "登录已失效，请重新登录。")
        return {**public_user(user), "jti": claims["jti"]}


def logout(user):
    with transaction() as db:
        db.execute("DELETE FROM sessions WHERE jti=? AND user_id=?", (user["jti"], user["id"]))


def change_password(user, current_password, new_password, client):
    with transaction() as db:
        _rate_limit(db, client)
        row = db.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    if not row or not PASSWORDS.verify(current_password, row["password_hash"]):
        raise AuthError(400, "当前密码不正确。")
    password_hash = PASSWORDS.hash(new_password)
    with transaction() as db:
        # 在提交修改时复查登录未被并发登出或另一条修改密码请求撤销。
        if not db.execute("SELECT 1 FROM sessions WHERE jti=? AND user_id=? AND expires>?",
                          (user["jti"], user["id"], int(time.time()))).fetchone():
            raise AuthError(401, "登录已失效，请重新登录。")
        db.execute("UPDATE users SET password_hash=?, token_version=token_version+1 WHERE id=?",
                   (password_hash, user["id"]))
        db.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))
