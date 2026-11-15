"""MCP联网补充：区分请求失败、正常空结果与可用网页摘要。"""
import asyncio
from datetime import datetime, timezone
import json
import time
from urllib.parse import urlsplit

from agents.mcp import MCPServerStreamableHttp

from app.infra.config.providers import infra_config
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import MCP_SEARCH_COUNT, MCP_TOTAL_TIMEOUT_SECONDS
from app.shared.runtime.logger import logger


class MCPResultError(ValueError):
    """记录稳定错误类别，不把远端原始报错或鉴权信息传给前端。"""


def get_data_and_validate(state) -> str:
    query = state.get("rewritten_query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("改写问题为空，不能执行联网检索")
    return query.strip()


async def open_ai_mcp(rewritten_query):
    server = MCPServerStreamableHttp(
        name="EEG web search",
        params={
            "url": infra_config.mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp_config.api_key}"},
            "timeout": 15,
            "sse_read_timeout": 20,
        },
        client_session_timeout_seconds=20,
        cache_tools_list=True,
        max_retry_attempts=1,
    )
    try:
        # connect和call_tool共享请求预算；清理仍在同一协程内完成。
        async with asyncio.timeout(MCP_TOTAL_TIMEOUT_SECONDS):
            await server.connect()
            return await server.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": rewritten_query, "count": MCP_SEARCH_COUNT},
            )
    finally:
        try:
            await server.cleanup()
        except Exception as exc:
            logger.warning("MCP连接清理异常：{}", type(exc).__name__)


def extract_pages(result):
    if result is None:
        raise MCPResultError("missing_response")
    if getattr(result, "isError", False):
        raise MCPResultError("tool_error")
    structured = getattr(result, "structuredContent", None)
    candidates = [structured] if structured is not None else []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            candidates.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    for payload in candidates:
        if not isinstance(payload, dict) or "pages" not in payload:
            continue
        if payload.get("status") in ("error", "failed"):
            raise MCPResultError("tool_error")
        if not isinstance(payload["pages"], list):
            raise MCPResultError("invalid_response")
        return payload["pages"]
    raise MCPResultError("invalid_response")


def normalize_pages(pages, retrieved_at):
    documents, seen, rejected = [], set(), 0
    for page in pages:
        if not isinstance(page, dict):
            rejected += 1
            continue
        url, snippet, title = (page.get(key) for key in ("url", "snippet", "title"))
        if not isinstance(url, str) or not isinstance(snippet, str) or not snippet.strip():
            rejected += 1
            continue
        url = url.strip()
        try:
            parsed = urlsplit(url)
        except ValueError:
            rejected += 1
            continue
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            rejected += 1
            continue
        if url in seen:
            continue
        seen.add(url)
        documents.append({"title": title.strip() if isinstance(title, str) and title.strip() else url,
                          "url": url, "snippet": snippet.strip(), "source": "web",
                          "retrieved_at": retrieved_at})
    if pages and not documents:
        raise MCPResultError("invalid_response")
    return documents[:MCP_SEARCH_COUNT], rejected


def search_by_web(state: QueryGraphState) -> QueryGraphState:
    query = get_data_and_validate(state)
    started = time.perf_counter()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    status = {"status": "failed", "reason": "", "retrieved_at": retrieved_at,
              "returned_count": 0, "rejected_count": 0}
    documents = []
    try:
        result = asyncio.run(open_ai_mcp(query))
        documents, rejected = normalize_pages(extract_pages(result), retrieved_at)
        status.update(status="ok" if documents else "empty", returned_count=len(documents), rejected_count=rejected)
    except TimeoutError:
        status["reason"] = "timeout"
    except MCPResultError as exc:
        status["reason"] = str(exc)
    except Exception as exc:
        status["reason"] = "request_error"
        logger.warning("MCP请求异常类型：{}", type(exc).__name__)
    status["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    if status["status"] == "failed":
        logger.warning("MCP联网降级：{}", status["reason"])
    state["web_search_docs"] = documents
    state["web_search_status"] = status
    return state
