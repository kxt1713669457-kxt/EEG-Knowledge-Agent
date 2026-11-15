from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_providers
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.rag.query.query_normalization import expand_eeg_retrieval_terms
from app.rag.query.query_understanding import validate_retrieval_input
from app.rag.query.embedding_search_service import search_chunk_vectors
from app.rag.import_.chunk_metadata import SOURCE_FIELDS, source_fields

def get_data_and_validates(state: QueryGraphState) -> tuple[list[str], str]:
    return validate_retrieval_input(state)


def search_by_milvus(item_names: list[str], rewritten_query: str, answer: str):
    vectors = llm_providers.generate_embeddings([expand_eeg_retrieval_terms(rewritten_query) + ": " + answer])
    return search_chunk_vectors(vectors, item_names)


def deal_milvus_list(milvus_list):
    # {id / chunk_id,distance,entity:{}} -> {}
    hybe_embedding_chunks = []
    for item in milvus_list:
        entity = item.get("entity",{})
        hybe_embedding_chunks.append({
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
    return hybe_embedding_chunks


def call_llm_answer(rewritten_query) -> str:
    # 1. 获取模型客户端对象
    llm_client = llm_providers.chat()
    # 2. 加载提示词字符串
    hyde_prompt_text = load_prompt(  "hyde_prompt",rewritten_query=rewritten_query)
    # 3. 封装提示词Message
    messages = [
        HumanMessage(
            content= hyde_prompt_text
        )
    ]
    # 4. 封装调用链
    chains = llm_client | StrOutputParser()
    # 5. 调用获取结果
    answer = chains.invoke(messages)
    return answer



def search_by_hyde(state: QueryGraphState) -> list[dict]:
    """
    HyDE分支：生成假设文本，将问题与假设拼接后编码，召回真实切片。
    假设不是事实证据，不加入返回的文档列表。
    """
    #1.获取并校验参数(state)->
    item_names,rewritten_query=get_data_and_validates(state)
    # 假设只作检索输入，不能作为返回证据。
    hypothesis = call_llm_answer(rewritten_query)
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("HyDE生成文本为空，本次分支未完成")
    hypothesis = hypothesis.strip()
    # 2. 进行向量的混合+条件检索(item_names, rewritten_query) -> [ {id/chunk_id:x,distance:0.9,entity:{输出的field}} ]
    milvus_list = search_by_milvus(item_names, rewritten_query, hypothesis)
    # 3. 进行结果的统一格式化处理( [ {id/chunk_id:x,distance:0.9,entity:{输出的field}} ]) ->
    hybe_embedding_chunks=deal_milvus_list(milvus_list)
    return hybe_embedding_chunks
