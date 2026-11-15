from app.agent.tools.schemas import ToolPayload, ToolError
from app.workspace.files import selected_documents


def read_workspace_files(arguments):
    documents = selected_documents(arguments.query)
    allowed = {d["file_id"] for d in documents}
    if not set(arguments.file_ids).issubset(allowed) or not allowed:
        return ToolPayload(status="failed", error=ToolError(code="file_not_selected", message="文件不在本轮用户已选资料中，不能读取任意文件或路径。"))
    chosen = [d for d in documents if d["file_id"] in arguments.file_ids]
    return ToolPayload(status="ok", data={"documents": chosen, "scope": "用户已选文件的相关正文窗口；未执行目录中的代码，不宣称已阅读全文。"})
