from app.process.query.agent.graph_definition import (
    after_node_item_name_confirm,
    build_query_graph_builder,
)
from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp


# 线上使用真实业务节点；离线检查只替换节点实现，复用同一份图定义。
query_graph_builder = build_query_graph_builder({
    "node_item_name_confirm": node_item_name_confirm,
    "node_search_embedding": node_search_embedding,
    "node_search_embedding_hyde": node_search_embedding_hyde,
    "node_web_search_mcp": node_web_search_mcp,
    "node_rrf": node_rrf,
    "node_rerank": node_rerank,
    "node_answer_output": node_answer_output,
})
query_app = query_graph_builder.compile()
