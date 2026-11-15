"""
Reranker 配置模块，负责读取重排模型相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_str

@dataclass
class RerankerConfig:
    bge_reranker_large: str
    bge_reranker_device: str
    bge_reranker_fp16: bool
    enabled: bool
    batch_size: int = 2
    cpu_threads: int = 4

reranker_config = RerankerConfig(
    # 旧变量名保留兼容；新配置可以选择经过验证的BGE系列重排模型。
    bge_reranker_large=env_str("BGE_RERANKER_MODEL") or env_str("BGE_RERANKER_LARGE"),
    bge_reranker_device=env_str("BGE_RERANKER_DEVICE"),
    bge_reranker_fp16=env_bool("BGE_RERANKER_FP16"),
    # large 模型在 CPU 上需要数 GB 额外内存；默认关闭可避免进程被原生 OOM/访问冲突直接杀死。
    enabled=env_bool("RERANK_ENABLED", False),
    batch_size=max(1, int(env_str("BGE_RERANKER_BATCH_SIZE", "2"))),
    cpu_threads=max(1, int(env_str("BGE_RERANKER_CPU_THREADS", "4"))),
)
