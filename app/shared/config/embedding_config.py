"""
Embedding 配置模块，负责读取向量模型相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_str


@dataclass
class EmbeddingConfig:
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool
    # BGE-M3模型输入容量；编码前检查，避免SDK默认512静默截断。
    max_input_tokens: int = 8192

embedding_config = EmbeddingConfig(
    bge_m3_path=env_str("BGE_M3_PATH"),
    bge_m3=env_str("BGE_M3"),
    bge_device=env_str("BGE_DEVICE"),
    bge_fp16=env_bool("BGE_FP16"),
)
