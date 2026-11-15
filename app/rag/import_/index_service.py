from pymilvus import DataType

from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import step_log
from app.shared.runtime.logger import logger
from app.rag.import_.chunk_metadata import require_doc_id, processing_profile
from app.rag.import_.index_publication import publish_document_rows
from app.rag.import_.item_name_service import delete_and_insert_item_name, prepared_milvus_item_name_collection


def require_embeddings_content(state: ImportGraphState) -> list[dict]:
    """
    获取并校验带向量的切片数据。

    功能：
    1. 从 state 中获取 embeddings_content
    2. 校验数据是否存在
    3. 校验数据类型是否正确
    4. 返回可直接写入 Milvus 的切片数据

    :param state: LangGraph 导入流程状态
    :return: 带向量字段的切片列表
    """

    # 1. 从 state 中获取已经生成向量的切片
    embeddings_content = state.get("embeddings_content")

    # 2. 判断是否存在
    if not embeddings_content:
        raise ValueError(
            "embeddings_content 为空，无法写入 Milvus，"
            "请先执行向量生成节点"
        )

    # 3. 判断最外层数据类型
    if not isinstance(embeddings_content, list):
        raise TypeError(
            f"embeddings_content 应该是 list 类型，"
            f"实际类型为: {type(embeddings_content).__name__}"
        )

    # 4. 判断列表里面是不是 dict
    for index, chunk in enumerate(embeddings_content):
        if not isinstance(chunk, dict):
            raise TypeError(
                f"embeddings_content[{index}] 应该是 dict 类型，"
                f"实际类型为: {type(chunk).__name__}"
            )

    logger.info(
        f"embeddings_content 校验成功，共 {len(embeddings_content)} 条切片"
    )

    return embeddings_content


@step_log("prepare_chunks_collection")
def prepare_chunks_collection() -> None:
    """
    准备 Milvus 切片集合
    功能: 检查集合是否存在，不存在则创建 schema 和索引
    :return: 无返回值
    """
    # 获取 Milvus 客户端
    milvus_client = milvus_gateway.milvus_client

    # 获取集合名称（从配置中读取）
    collection_name = milvus_gateway.chunk_collection_name

    # 如果集合已存在，直接返回，无需重复创建
    if milvus_client.has_collection(collection_name=collection_name):
        return

    # ==================== 创建 Schema ====================
    # 创建 schema，启用自动 ID 和动态字段
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

    # 添加主键字段: chunk_id，INT64 类型，自增
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 添加文件标题字段: VARCHAR 类型，最大长度 512
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)

    # 添加切片标题字段: VARCHAR 类型，最大长度 512
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)

    # 添加父标题字段: VARCHAR 类型，最大长度 512
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)

    # 新集合使用INT32；已有集合维持原schema。
    schema.add_field(field_name="part", datatype=DataType.INT32)

    # 添加内容字段: VARCHAR 类型，最大长度 65535（支持长文本）
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

    # 添加稠密向量字段: FLOAT_VECTOR 类型，维度 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

    # 添加稀疏向量字段: SPARSE_FLOAT_VECTOR 类型
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # ========================= 创建索引 =========================
    # 准备索引参数
    index_params = milvus_client.prepare_index_params()

    # 新集合的稠密索引使用HNSW，按COSINE相似度检索；已有集合不自动重建。
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={
            "M": 64,  # 每个点最大的链接数量
            "efConstruction": 100  # 考虑链接的范围 在这个范围内确定 M
        }
    )
    # 为稀疏向量创建索引：使用 SPARSE_INVERTED_INDEX，算法为 DAAT_MAXSCORE
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    # 创建集合并应用索引
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)



def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """按内容身份发布切片，核验后同步主体索引，回传可提交回执的结果。"""
    embeddings_content = require_embeddings_content(state)
    metadata = state.get("document_metadata") or {}
    doc_id = require_doc_id(metadata)
    if not state.get("item_name"):
        raise ValueError("缺少主体名称，不能完成索引发布")
    prepare_chunks_collection()
    client = milvus_gateway.milvus_client
    if client is None:
        raise ConnectionError("Milvus不可用，无法发布文档索引")
    result = publish_document_rows(client, milvus_gateway.chunk_collection_name, embeddings_content, doc_id)
    prepared_milvus_item_name_collection()
    # 放到切片发布后，避免Embedding失败时提前制造只有主体、没有正文的索引。
    delete_and_insert_item_name(state["item_name"], state.get("file_title", ""), metadata)
    for chunk, chunk_id in zip(embeddings_content, result["ids"]):
        chunk["chunk_id"] = chunk_id
    state["index_metadata"] = {**result, "action": "indexed", "doc_id": doc_id, "profile": processing_profile()}
    logger.info(f"文档索引核验完成，共{result['chunk_count']}个切片")
    return state
