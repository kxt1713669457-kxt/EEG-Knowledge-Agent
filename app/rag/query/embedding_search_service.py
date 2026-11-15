from app.infra.llm.providers import llm_providers
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.rag.query.config import LOCAL_BRANCH_TOPK, LOCAL_VECTOR_CANDIDATE_TOPK
from app.rag.query.query_normalization import expand_eeg_retrieval_terms
from app.rag.query.query_understanding import topic_filter, validate_retrieval_input
from app.rag.import_.chunk_metadata import SOURCE_FIELDS, source_fields

def get_data_and_validates(state: QueryGraphState) -> tuple[list[str], str]:
    return validate_retrieval_input(state)


def search_chunk_vectors(vectors: dict, item_names: list[str]):
    """复用同一编码结果；主题无命中最多回退一次，不重复加载或编码。"""
    def search(names):
        requests = milvus_gateway.create_requests(
            dense_vector=vectors["dense"][0], sparse_vector=vectors["sparse"][0],
            expr=topic_filter(names), limit=LOCAL_VECTOR_CANDIDATE_TOPK,
        )
        results = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.chunk_collection_name, reqs=requests,
            ranker_weights=(0.6, 0.4), norm_score=True, limit=LOCAL_BRANCH_TOPK,
            output_fields=["chunk_id", "file_title", "title", "parent_title", "part", "item_name", "content", *SOURCE_FIELDS],
        )
        # 网关用None表示请求异常；[[]]才是正常查询后的零命中。
        if results is None:
            raise RuntimeError("Milvus检索请求失败，不能当作零命中或继续主题回退")
        return results
    if 1<len(item_names)<=4:
        # 同一编码按已确认主体分组召回；有界交错保留候选，避免一个对象占满全部额度。
        groups=[(search([name]) or [[]])[0] for name in item_names]
        if any(groups):
            merged=[];seen=set()
            for rank in range(LOCAL_BRANCH_TOPK):
                for group in groups:
                    if rank>=len(group):continue
                    hit=group[rank];entity=hit.get('entity') or {}
                    uid=entity.get('chunk_uid') or entity.get('chunk_id') or hit.get('id') or hit.get('chunk_id')
                    if uid is None:raise RuntimeError('多主体候选缺少身份，不能去重')
                    if uid not in seen:seen.add(uid);merged.append(hit)
                    if len(merged)==LOCAL_BRANCH_TOPK:return merged
            return merged
        results=search([])  # 全组正常零命中才开放回退，异常仍上抛。
        return results[0] if results else []
    results = search(item_names)
    if item_names and not (results and results[0]):
        logger.info("主题过滤无命中，复用向量回退开放检索")
        results = search([])
    return results[0] if results else []


def search_by_milvus(item_names: list[str], rewritten_query: str):
    vectors = llm_providers.generate_embeddings([expand_eeg_retrieval_terms(rewritten_query)])
    return search_chunk_vectors(vectors, item_names)


def deal_milvus_list(milvus_list):
    # {id / chunk_id,distance,entity:{}} -> {}
    embedding_chunks = []
    for item in milvus_list:
        entity = item.get("entity",{})
        embedding_chunks.append({
            **source_fields(entity),
            "chunk_id": entity.get("chunk_id") , # item.get("id") or item.get("chunk_id")
            "score": item.get("distance",0.0),
            "title":entity.get("title"),
            "file_title":entity.get("file_title"),
            "parent_title":entity.get("parent_title"),
            "part":entity.get("part"),
            "item_name":entity.get("item_name"),
            "content":entity.get("content"),
            "source": "milvus" # 直接查询 或者假设性查询 milvus 网络检索 web
        })
    return embedding_chunks


def search_by_embedding(state: QueryGraphState) -> list[dict]:
    """
    向量检索服务：
    1. 根据改写后的问题和可选主题范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 返回真实文档切片，由节点回写 embedding_chunks
    """
    #1.获取并校验参数(state)->
    item_names,rewritten_query=get_data_and_validates(state)
    # 2. 进行向量的混合+条件检索(item_names, rewritten_query) -> [ {id/chunk_id:x,distance:0.9,entity:{输出的field}} ]
    milvus_list = search_by_milvus(item_names, rewritten_query)
    # 3. 进行结果的统一格式化处理( [ {id/chunk_id:x,distance:0.9,entity:{输出的field}} ]) -> 
    embedding_chunks=deal_milvus_list(milvus_list)
    return embedding_chunks
