"""把旧的 MinIO 配图地址映射为登录后的读取；不是任意 URL 抓取代理。"""
from mimetypes import guess_type
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from minio.error import S3Error

from app.infra.config.providers import infra_config
from app.infra.object_storage.minio_gateway import minio_gateway

router = APIRouter()


def image_object_key(url):
    settings = infra_config.minio_config
    expected = urlsplit(("https://" if settings.minio_secure else "http://") + settings.endpoint)
    target = urlsplit(url)
    prefix = "/" + settings.bucket_name + "/"
    if (target.scheme != expected.scheme or target.netloc != expected.netloc or target.query or target.fragment
            or target.username or target.password or not target.path.startswith(prefix)):
        raise ValueError("图片地址不属于本项目对象存储")
    key = unquote(target.path[len(prefix):])
    allowed_prefix = settings.minio_img_dir.strip("/") + "/"
    if (not key.startswith(allowed_prefix) or "\\" in key or "%" in key or "\x00" in key
            or any(part in {".", "..", ""} for part in key.split("/"))
            or PurePosixPath(key).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}):
        raise ValueError("无效的配图身份或格式")
    return key


@router.get("/sources/image")
def source_image(url: str = Query(max_length=4096)):
    try:
        key = image_object_key(url)
    except (ValueError, TypeError):
        raise HTTPException(400, "无效的本项目配图地址。") from None
    try:
        response = minio_gateway.read_original(key)
    except S3Error as exc:
        raise HTTPException(404 if exc.code in {"NoSuchKey", "NoSuchObject"} else 502, "配图不存在或暂不可读取。") from None
    except Exception:
        raise HTTPException(502, "配图读取失败，请稍后重试。") from None

    def content():
        try:
            yield from response.stream(amt=64 * 1024)
        finally:
            response.close()
            response.release_conn()
    return StreamingResponse(content(), media_type=guess_type(key)[0] or "application/octet-stream",
                             headers={"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox; default-src 'none'"})
