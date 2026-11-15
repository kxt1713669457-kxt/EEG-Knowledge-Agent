"""
工具模块，负责提供 embedding 相关的辅助能力。
"""
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import threading

from app.shared.config.embedding_config import embedding_config
from app.shared.runtime.logger import logger

_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_DEFAULT_EMBEDDING_DEVICE = "cpu"
_bge_m3_ef: BGEM3EmbeddingFunction | None = None
_model_lock = threading.Lock()
_encoding_lock = threading.Lock()


def _load_bge_m3_ef() -> BGEM3EmbeddingFunction:
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    # 从环境变量加载配置，无配置则使用默认值
    # 本地有可以使用本地地址！ 没有使用 "BAAI/bge-m3" 会自动下载！ 如果云端部署也可以使用url地址！
    model_name = embedding_config.bge_m3_path or embedding_config.bge_m3 or _DEFAULT_EMBEDDING_MODEL
    device = embedding_config.bge_device or _DEFAULT_EMBEDDING_DEVICE
    use_fp16 = embedding_config.bge_fp16

    # 打印模型初始化配置，便于问题排查
    logger.info(
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # 稠密向量L2归一化；稀疏向量保留词项权重。显式指定输入上限。
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True,
            query_max_length=embedding_config.max_input_tokens,
            passage_max_length=embedding_config.max_input_tokens,
        )
        logger.success("BGE-M3初始化成功：稠密向量L2归一化，稀疏词项权重保留，输入上限显式配置")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
        raise  # 向上抛出异常，由调用方处理


def get_bge_m3_ef() -> BGEM3EmbeddingFunction:
    # 首次普通/HyDE并行抵达时，也只初始化一份权重。
    with _model_lock:
        return _load_bge_m3_ef()


def generate_embeddings(texts: list[str]) -> dict[str, list]:
    """
    为文本列表生成混合向量；稠密向量L2归一化，稀疏向量保留词项权重。
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")
    if any(not isinstance(text, str) for text in texts):
        logger.warning("生成向量入参不合法，texts中存在非字符串内容")
        raise ValueError("参数texts必须是字符串列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # SDK的fast tokenizer在encode_documents中会改变截断/填充设置。
        # 计数也读取同一实例，因此计数与实际编码必须在同一把进程内锁内完成。
        with _encoding_lock:
            for index, text in enumerate(texts):
                token_count = len(model.model.tokenizer.encode(text, add_special_tokens=True))
                if token_count > embedding_config.max_input_tokens:
                    raise ValueError(f"第{index}条Embedding输入有{token_count}个token，超过"
                                     f"{embedding_config.max_input_tokens}；需要先细分，不能静默截断")
            # 输出转换在锁外进行，不扩大共享模型临界区。
            embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
        # # 把模型输出的 CSR 稀疏矩阵 ，按“每条文本一行”拆成 {特征索引: 权重} 字典
        # # - indices ：非零元素的“列号（特征ID）”
        # # - data ：对应列号的权重值
        # # - indptr ：每一行在 indices/data 里的起止位置指针
        # # 数据示例:
        # # indices = [3, 8, 20, 1, 9]
        # # data    = [0.7, 0.2, 0.1, 0.6, 0.4]  -> milvus -> 稠密向量 [1024] 稀疏向量 : {index:值,index:值}
        # # indptr  = [0, 3, 5]
        # # 获取对应的数据
        # # - 第0条文本用 0:3 => indices=[3,8,20] , data=[0.7,0.2,0.1]
        # # - 第1条文本用 3:5 => indices=[1,9] , data=[0.6,0.4]
        for i in range(len(texts)):
            # 提取第i个文本的稀疏向量索引：np.int64 → Python int（满足字典key可哈希要求）
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 提取第i个文本的稀疏向量权重：np.float32 → Python float（适配JSON序列化/接口返回）
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 构造{特征索引: 模型词项权重}的稀疏向量字典
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
        result = {
            # embeddings["dense"] = [[1稠密向量],[2稠密向量],[...]  -> 1024]
            # sparse的每条结果是词表特征ID到权重的映射，不是1024维稠密列表。
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse  # 字典列表，不假定稀疏词项权重L2归一化
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


