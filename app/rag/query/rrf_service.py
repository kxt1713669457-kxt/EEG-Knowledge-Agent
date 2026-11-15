from copy import deepcopy

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import NODE_RRF_LIMIT_TOP, NODE_RRF_K
from app.shared.runtime.logger import logger
from app.rag.query.coverage_selection import select_with_subject_coverage


def get_data_and_validates(state: QueryGraphState):
    # 1.获取参数
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])
    # 2.非空校验
    if not embedding_chunks and not hyde_embedding_chunks and not state.get("web_search_docs"):
        logger.error("embedding_chunks 与 hyde_embedding_chunks 均为空，无法进行 RRF 融合")
        raise ValueError("embedding_chunks 与 hyde_embedding_chunks 均为空，无法进行 RRF 融合")
    # 3. 返回结果
    return embedding_chunks, hyde_embedding_chunks


def use_by_rrf(rrf_list: list, top: int = NODE_RRF_LIMIT_TOP,
               k: int = NODE_RRF_K, *, source_names: list[str] | None = None) -> list[dict]:
    """按各路名次累加 weight/(k+rank)，再去重排序截取。

    同一路的重复切片只投一次票；跨路出现可以累加。k是排名平滑常数，
    top才是返回条数。输出独立拷贝，原始召回分数留在各路输入和贡献记录中。
    """
    scores = {}
    chunks = {}
    contributions = {}
    for branch_index, (weight, ranked_chunks) in enumerate(rrf_list):
        branch = source_names[branch_index] if source_names else f"branch_{branch_index + 1}"
        seen = set()
        for chunk in ranked_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id is None:
                raise ValueError("本地RRF切片缺少chunk_id，不能合并成同一条证据")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank = len(seen)
            contribution = weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            chunks.setdefault(chunk_id, chunk)
            contributions.setdefault(chunk_id, []).append({
                "branch": branch, "rank": rank, "weight": weight,
                "retrieval_score": chunk.get("score"), "contribution": contribution,
            })

    fused = []
    for chunk_id, score in scores.items():
        chunk = deepcopy(chunks[chunk_id])
        chunk.update(score=score, rrf_score=score,
                     rrf_contributions=contributions[chunk_id])
        fused.append(chunk)
    # 同分时保留首次出现顺序，方便复现；不是额外的语义相关性判断。
    fused.sort(key=lambda chunk: chunk["rrf_score"], reverse=True)
    return fused[:top]



def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 融合普通与HyDE两路本地排名，网页留给后续重排适配
    2. 使用名次而不直接相加原始检索分数
    3. 按配置截取融合候选，保留原文来源及各路贡献
    4. 回写 rrf_chunks
    """
    embedding_chunks, hyde_embedding_chunks = get_data_and_validates(state)
    rrf_chunks = use_by_rrf(
        [(1.0, embedding_chunks), (1.0, hyde_embedding_chunks)],
        source_names=["ordinary", "hyde"],
        top=len(embedding_chunks)+len(hyde_embedding_chunks),
    )
    rrf_chunks=select_with_subject_coverage(rrf_chunks,state.get('item_names') or [],NODE_RRF_LIMIT_TOP)
    state['rrf_chunks'] = rrf_chunks
    return state
