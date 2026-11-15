from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.docx_parse_service import parse_docx_to_markdown
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_docx_to_md")
def node_docx_to_md(state: ImportGraphState) -> ImportGraphState:
    add_running_task(state["task_id"], "node_docx_to_md")
    state = parse_docx_to_markdown(state)
    add_done_task(state["task_id"], "node_docx_to_md")
    return state
