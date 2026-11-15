"""
工具模块，负责提供 reranker 相关的辅助能力。
"""
from FlagEmbedding import FlagReranker
import threading
import torch

from app.shared.config.reranker_config import reranker_config
from app.shared.runtime.logger import logger

_reranker_model: FlagReranker | None = None
_model_lock = threading.Lock()


def get_reranker_model() -> FlagReranker:
    """
    获取重排模型单例对象。

    Returns:
        FlagReranker: 初始化完成的重排模型实例。
    """
    global _reranker_model
    if _reranker_model is None:
        with _model_lock:
            if _reranker_model is None:
                if reranker_config.bge_reranker_device == "cpu":
                    torch.set_num_threads(reranker_config.cpu_threads)
                logger.info("开始初始化重排模型")
                _reranker_model = FlagReranker(
                    model_name_or_path=reranker_config.bge_reranker_large,
                    devices=reranker_config.bge_reranker_device,
                    use_fp16=reranker_config.bge_reranker_fp16,
                    batch_size=reranker_config.batch_size,
                    max_length=512,
                    query_max_length=512,
                )
                logger.success("重排模型初始化成功")
    return _reranker_model
