"""
工具模块，负责提供 sse 相关的辅助能力。
"""
import json
import queue
import asyncio
from threading import RLock
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号


# Key是单次执行task_id；历史函数名保留供导入流程和旧独立测试使用。
_session_stream: Dict[str, queue.Queue] = {}
_stream_consumers = {}
_stream_lock = RLock()


class StreamAlreadyConnected(RuntimeError):
    pass

def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列"""
    with _stream_lock:
        return _session_stream.get(session_id)

def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列"""
    with _stream_lock:
        # 同一执行的重复初始化不丢掉已排队事件。
        if session_id not in _session_stream:
            _session_stream[session_id] = queue.Queue()
        return _session_stream[session_id]

def remove_sse_queue(session_id: str, *, expected_queue=None):
    """移除指定 session 的队列"""
    with _stream_lock:
        current = _session_stream.get(session_id)
        if expected_queue is None or current is expected_queue:
            _session_stream.pop(session_id, None)
        if expected_queue is None or _stream_consumers.get(session_id) is expected_queue:
            _stream_consumers.pop(session_id, None)


def claim_sse_queue(task_id):
    """HTTP返回前占用单消费者，避免两个连接把同一事件队列分食。"""
    with _stream_lock:
        stream_queue = _session_stream.get(task_id)
        if stream_queue is None:
            return None
        if task_id in _stream_consumers:
            raise StreamAlreadyConnected(task_id)
        _stream_consumers[task_id] = stream_queue
        return stream_queue

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    # print(f"[SSE] Packing event: {event}, payload: {payload[:50]}...")
    return f"event: {event}\ndata: {payload}\n\n"

def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """
    通过 session_id 推送事件
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        # print(f"[SSE] Pushing to session {session_id}: {event}")
        stream_queue.put({"event": event, "data": data})
    else:
        print(f"[SSE] Warning: No queue found for session {session_id} when pushing {event}")

async def sse_generator(session_id: str, request: Request, *, stream_queue=None):
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse
    """
    print(f"[SSE] Generator started for session: {session_id}")
    if stream_queue is None:
        stream_queue = claim_sse_queue(session_id)
    if stream_queue is None:
        # 如果没有对应的队列，直接结束
        print(f"[SSE] Error: Queue not found for session {session_id}. Available sessions: {list(_session_stream.keys())}")
        return

    loop = asyncio.get_running_loop()
    try:
        # 发送连接建立信号
        print(f"[SSE] Sending ready signal for {session_id}")
        yield _sse_pack("ready", {"task_id": session_id})

        while True:
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                print(f"[SSE] Client disconnected: {session_id}")
                print("-----------------------断开连接--------------------")
                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                # {event:"process" : data: {任务状态 / 已完成节点 / 进行中节点}}
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue

            event = msg.get("event")
            data = msg.get("data")
            
            # print(f"[SSE] Yielding event {event} for {session_id}")

            # 特殊关闭事件
            if event == "__close__":
                print(f"[SSE] Closing signal received for {session_id}")
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        print(f"[SSE] Client disconnected (Cancelled/Reset/Pipe): {session_id}")
        # 生成器被取消/对端断开：静默退出
        return
    except Exception as e:
        print(f"[SSE] Exception in generator for {session_id}: {e}")
    finally:
        print(f"[SSE] Generator finished for {session_id}")
        # 清理资源
        remove_sse_queue(session_id, expected_queue=stream_queue)
