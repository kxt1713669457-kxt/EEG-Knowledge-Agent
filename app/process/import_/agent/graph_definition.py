"""导入图统一拓扑；线上与隔离检查共用格式路由。"""
from collections.abc import Callable, Mapping

from langgraph.graph import END, StateGraph
from app.process.import_.agent.state import ImportGraphState

IMPORT_NODE_NAMES = (
    "node_entry", "node_pdf_to_md", "node_docx_to_md", "node_md_img",
    "node_document_split", "node_item_name_recognition", "node_bge_embedding", "node_import_milvus",
)


def node_entry_after(state: ImportGraphState) -> str:
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    if state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    if state.get("is_docx_read_enabled"):
        return "node_docx_to_md"
    raise ValueError("导入文件没有可用的格式解析器")


def build_import_graph_builder(nodes: Mapping[str, Callable]) -> StateGraph:
    builder = StateGraph(ImportGraphState)
    for name in IMPORT_NODE_NAMES:
        builder.add_node(name, nodes[name])
    builder.set_entry_point("node_entry")
    builder.add_conditional_edges("node_entry", node_entry_after, {
        name: name for name in ("node_md_img", "node_pdf_to_md", "node_docx_to_md")
    })
    builder.add_edge("node_pdf_to_md", "node_md_img")
    builder.add_edge("node_docx_to_md", "node_md_img")
    builder.add_edge("node_md_img", "node_document_split")
    builder.add_edge("node_document_split", "node_item_name_recognition")
    builder.add_edge("node_item_name_recognition", "node_bge_embedding")
    builder.add_edge("node_bge_embedding", "node_import_milvus")
    builder.add_edge("node_import_milvus", END)
    return builder
