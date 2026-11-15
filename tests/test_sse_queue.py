"""SSE 会话队列的回归测试。

运行：
    .\\.venv\\Scripts\\python.exe tests\\test_sse_queue.py
"""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared.utils.sse_utils import (
    SSEEvent,
    create_sse_queue,
    get_sse_queue,
    push_to_session,
    remove_sse_queue,
)


class SseQueueTest(unittest.TestCase):
    session_id = "test-sse-queue"

    def tearDown(self) -> None:
        remove_sse_queue(self.session_id)

    def test_queue_accepts_progress_before_client_connects(self) -> None:
        queue = create_sse_queue(self.session_id)
        push_to_session(self.session_id, SSEEvent.PROGRESS, {"status": "processing"})

        self.assertIs(get_sse_queue(self.session_id), queue)
        self.assertEqual(
            queue.get_nowait(),
            {"event": SSEEvent.PROGRESS, "data": {"status": "processing"}},
        )

    def test_queue_transmits_error_then_close(self) -> None:
        queue = create_sse_queue(self.session_id)
        push_to_session(self.session_id, SSEEvent.ERROR, {"error": "模拟失败"})
        push_to_session(self.session_id, SSEEvent.CLOSE, {})

        self.assertEqual(queue.get_nowait()["event"], SSEEvent.ERROR)
        self.assertEqual(queue.get_nowait()["event"], SSEEvent.CLOSE)


if __name__ == "__main__":
    unittest.main()
