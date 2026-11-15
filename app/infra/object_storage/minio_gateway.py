import re
from mimetypes import guess_type
from pathlib import Path

from app.shared.clients.minio_utils import get_minio_client
from app.infra.config.providers import infra_config

class MinioGateway:
    # 提供获取桶名称的函数
    @property
    def bucket_name(self):
        return infra_config.minio_config.bucket_name

    # 提供获取图片前缀的函数
    @property
    def minio_img_dir(self):
        return infra_config.minio_config.minio_img_dir

    # 提供获取minio_client的函数
    @property
    def minio_client(self):
        return get_minio_client()

    def archive_original(self, local_file_path: str | Path, file_hash: str) -> str:
        """按内容身份归档原件；同内容重试使用相同对象键。"""
        if not re.fullmatch(r"[0-9a-f]{64}", file_hash):
            raise ValueError("无效的文件内容指纹")
        source = Path(local_file_path)
        object_key = f"originals/{file_hash}/source{source.suffix.lower()}"
        self.minio_client.fput_object(
            self.bucket_name, object_key, str(source),
            content_type=guess_type(source.name)[0] or "application/octet-stream",
            metadata={"file-sha256": file_hash},
        )
        return object_key

    def read_original(self, object_key: str):
        return self.minio_client.get_object(self.bucket_name, object_key)

    # minio上传文件不会返回访问地址 -> 我们拼接访问地址  http https  端点  桶 对象名
    # 封装一个拼接访问地址的函数
    def build_image_url(self, stem: str, image_name: str):
        """
        image_name = upload-images / 文件名 / 图片名称.png
        :param object_name:
        :return:
        """
        image_url = ("https://" if infra_config.minio_config.minio_secure else "http://") + (f"{infra_config.minio_config.endpoint}"
            f"/{infra_config.minio_config.bucket_name}{infra_config.minio_config.minio_img_dir}/{stem}/{image_name}")
        return image_url
minio_gateway=MinioGateway()
