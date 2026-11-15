from app.process.query.agent.state import execution_id
import sys

from app.shared.runtime.logger import node_log
from app.rag.query.rrf_service import fuse_by_rrf
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_rrf")
def node_rrf(state):
    """
    节点功能：Reciprocal Rank Fusion
    融合普通与 HyDE 两路本地召回；Web 结果在 rerank 服务中加入。
    """
    add_running_task(execution_id(state), sys._getframe().f_code.co_name, state.get("is_stream"))
    from app.workspace.files import selected_documents
    attachments = selected_documents(state.get("rewritten_query") or state.get("original_query") or "")
    has_retrieval = any(state.get(key) for key in ("embedding_chunks", "hyde_embedding_chunks", "web_search_docs"))
    if not has_retrieval and not attachments:
        if state.get("web_search_status", {}).get("status") == "failed":
            answer = "本次联网检索未完成，知识库也未召回可用证据。请稍后重试或补充相关论文资料。"
        else:
            answer = "本次知识库和联网检索均未返回可用证据，暂时无法据此回答。请补充相关论文或明确查询范围。"
        update = {"rrf_chunks": [], "answer": answer}
    else:
        update = {"rrf_chunks": (fuse_by_rrf(state)["rrf_chunks"] if has_retrieval else []) + attachments}
    add_done_task(execution_id(state), sys._getframe().f_code.co_name, state.get("is_stream"))
    return update
