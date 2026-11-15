"""两个 API 统一认证入口；Cookie 兼容 EventSource、图片和原件下载。"""
import ipaddress
from urllib.parse import quote

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

from app.shared.runtime.logger import PROJECT_ROOT
from app.workspace import auth

LOCAL_ORIGINS = [f"{scheme}://{host}:{port}" for scheme in ("http", "https") for host in ("127.0.0.1", "localhost", "[::1]")
                 for port in (8000, 8001, 18000, 18001)]
router = APIRouter(prefix="/auth", tags=["登录"])


class LoginInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=1, max_length=128, repr=False)


class SetupInput(LoginInput):
    password: str = Field(min_length=10, max_length=128, repr=False)

    @field_validator("password")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("密码不能全部是空白字符")
        return value


class PasswordInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(min_length=1, max_length=128, repr=False)
    new_password: str = Field(min_length=10, max_length=128, repr=False)

    @field_validator("new_password")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("密码不能全部是空白字符")
        return value


def error(exc):
    return JSONResponse({"detail": exc.message}, status_code=exc.status,
                        headers={"Cache-Control": "no-store", **({"Retry-After": "300"} if exc.status == 429 else {})})


def client_host(request):
    return request.client.host if request.client else "unknown"


def local_setup(request):
    try:
        local = ipaddress.ip_address(client_host(request)).is_loopback
    except ValueError:
        local = False
    if not local:
        raise HTTPException(403, "首次设置账号需要在本机打开研究空间。")


def signed_response(request, token, user):
    response = JSONResponse({"user": user, "expires_in": auth.TOKEN_SECONDS}, headers={"Cache-Control": "no-store"})
    response.set_cookie(auth.COOKIE_NAME, token, max_age=auth.TOKEN_SECONDS, httponly=True,
                        secure=request.url.scheme == "https", samesite="lax", path="/")
    return response


@router.get("/status")
def status():
    return JSONResponse({"setup_required": not auth.configured()}, headers={"Cache-Control": "no-store"})


@router.post("/setup", status_code=200)
def setup(payload: SetupInput, request: Request):
    local_setup(request)
    token, user = auth.setup(payload.username, payload.password, client_host(request))
    return signed_response(request, token, user)


@router.post("/login")
def login(payload: LoginInput, request: Request):
    token, user = auth.login(payload.username, payload.password, client_host(request))
    return signed_response(request, token, user)


@router.get("/me")
def me(request: Request):
    from app.infra.object_storage.minio_gateway import minio_gateway
    return {"user": {k: request.state.user[k] for k in ("id", "username")},
            "media": {"image_base": minio_gateway.build_image_url("", "").rstrip("/") + "/"}}


@router.post("/logout")
def logout(request: Request):
    auth.logout(request.state.user)
    response = JSONResponse({"message": "已退出登录。"})
    response.delete_cookie(auth.COOKIE_NAME, path="/", httponly=True, samesite="lax")
    return response


@router.post("/password")
def password(payload: PasswordInput, request: Request):
    auth.change_password(request.state.user, payload.current_password, payload.new_password, client_host(request))
    response = JSONResponse({"message": "密码已更新，请重新登录。"})
    response.delete_cookie(auth.COOKIE_NAME, path="/", httponly=True, samesite="lax")
    return response


class AuthMiddleware:
    """ASGI 门禁覆盖所有业务入口；不缓冲 SSE，不修改后台任务。"""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope, receive)
        path, method = request.url.path, request.method
        public = path in {"/login", "/health", "/auth/status", "/auth/setup", "/auth/login"}
        origin = request.headers.get("origin")
        bearer = request.headers.get("authorization", "")
        if method not in {"GET", "HEAD", "OPTIONS"}:
            if origin and origin not in LOCAL_ORIGINS:
                return await JSONResponse({"detail": "请求来源不被允许。"}, status_code=403)(scope, receive, send)
            if not origin and not bearer.startswith("Bearer "):
                return await JSONResponse({"detail": "缺少可信请求来源。"}, status_code=403)(scope, receive, send)
        if not public and method != "OPTIONS":
            token = bearer[7:] if bearer.startswith("Bearer ") else request.cookies.get(auth.COOKIE_NAME)
            try:
                user = await run_in_threadpool(auth.authenticate, token)
                scope.setdefault("state", {})["user"] = user
            except auth.AuthError as exc:
                if path in {"/html", "/import/html", "/"}:
                    response = RedirectResponse("/login?next=" + quote(path, safe=""), status_code=303)
                else:
                    response = error(exc)
                return await response(scope, receive, send)

        async def safe_send(message):
            if message["type"] == "http.response.start":
                headers = [(key, value) for key, value in message.get("headers", []) if key.lower() != b"cache-control"]
                names = {key.lower() for key, _ in headers}
                headers.append((b"cache-control", b"no-store"))
                for key, value in [(b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"same-origin")]:
                    if key not in names:
                        headers.append((key, value))
                message = {**message, "headers": headers}
            await send(message)
        return await self.app(scope, receive, safe_send)


def install_auth(app: FastAPI):
    app.include_router(router)

    @app.get("/login", include_in_schema=False)
    def login_page():
        return FileResponse(PROJECT_ROOT / "app/resources/html/login.html", media_type="text/html")

    @app.exception_handler(auth.AuthError)
    async def authentication_error(request, exc):
        return error(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        if request.url.path in {"/auth/login", "/auth/setup", "/auth/password"}:
            # FastAPI 默认 422 会回显 input；认证输入不能回显密码或请求正文。
            return JSONResponse({"detail": [{k: item[k] for k in ("loc", "msg", "type")}
                                            for item in exc.errors()]}, status_code=422)
        return await request_validation_exception_handler(request, exc)

    app.add_middleware(AuthMiddleware)
    # 显式来源 + credentials，导入页和查询页可共用 HttpOnly Cookie。
    app.add_middleware(CORSMiddleware, allow_origins=LOCAL_ORIGINS, allow_credentials=True,
                       allow_methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
                       allow_headers=["Content-Type", "Authorization"])
