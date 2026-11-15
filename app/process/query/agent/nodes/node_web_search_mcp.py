from app.process.query.agent.state import execution_id
import sys

from app.shared.runtime.logger import node_log
from app.rag.query.web_search_service import search_by_web
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    add_running_task(
        execution_id(state),
        sys._getframe().f_code.co_name,
        state["is_stream"]
    )

    result_state = search_by_web(state)

    add_done_task(
        execution_id(state),
        sys._getframe().f_code.co_name,
        state["is_stream"]
    )

    return {
        "web_search_docs": result_state.get("web_search_docs", []),
        "web_search_status": result_state["web_search_status"],
    }
