"""上传、文本保全和请求收尾回归；图执行和外部服务调用均用替身隔离。"""
import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient

from app.api.http import import_server, query_server
from app.infra.object_storage import minio_gateway as minio_module
from app.rag.import_ import enrich_markdown_images as image_module
from app.rag.import_.entry_service import resolve_input_file
from app.rag.import_.split_service import _split_long_chunk, refine_chunks, split_chunks_document
from app.shared.clients import mongo_history_utils
from app.workspace import auth
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, remove_sse_queue, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, add_done_task, clear_task, get_task_status, update_task_status,
)


class HistoryInitializationTest(unittest.TestCase):
    def load_isolated_module(self):
        spec = importlib.util.spec_from_file_location("_history_import_probe", mongo_history_utils.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("pymongo.MongoClient") as client:
            spec.loader.exec_module(module)
            client.assert_not_called()
        return module

    def test_import_does_not_connect_and_first_use_reuses_initialized_tool(self):
        module = self.load_isolated_module()
        tool = object()
        with patch.object(module, "HistoryMongoTool", return_value=tool) as create:
            self.assertIs(module.get_history_mongo_tool(), tool)
            self.assertIs(module.get_history_mongo_tool(), tool)
            create.assert_called_once_with()

    def test_initialization_failure_is_exposed_and_next_call_can_retry(self):
        module = self.load_isolated_module()
        tool = object()
        with patch.object(module, "HistoryMongoTool", side_effect=[ConnectionError("offline"), tool]):
            with self.assertRaises(ConnectionError):
                module.get_history_mongo_tool()
            self.assertIsNone(module._history_mongo_tool)
            self.assertIs(module.get_history_mongo_tool(), tool)


class AuthenticatedApiTest(unittest.TestCase):
    """业务回归使用临时账号和真实签名，不能关闭生产门禁或改写本机账号。"""
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        auth_path = patch.object(auth, "DB_PATH", Path(temp.name) / "auth.sqlite3")
        auth_path.start()
        self.addCleanup(auth_path.stop)
        self.auth_token, _ = auth.setup("regression", "temporary-regression-password", "local-test")

    def api_client(self, app):
        client = TestClient(app, headers={"Origin": "http://127.0.0.1:8001", "Authorization": f"Bearer {self.auth_token}"})
        self.addCleanup(client.close)
        return client


class ImportApiTest(AuthenticatedApiTest):
    def test_batch_upload_saves_each_file_and_returns_real_ids(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(import_server, "PROJECT_ROOT", Path(directory)), \
                patch.object(import_server, "invoke_import_graph") as invoke:
            response = self.api_client(import_server.app).post("/upload", files=[
                ("files", ("../first.MD", b"first document", "text/markdown")),
                ("files", ("second.pdf", b"second document", "application/pdf")),
            ])
            self.assertEqual(response.status_code, 200, response.text)
            task_ids = response.json()["task_ids"]
            self.assertEqual(len(set(task_ids)), 2)
            self.assertEqual(invoke.call_count, 2)
            for call, task_id, content in zip(invoke.call_args_list, task_ids, [b"first document", b"second document"]):
                saved = Path(call.kwargs["local_file_path"])
                self.assertEqual(saved.parent.name, task_id)
                self.assertTrue(saved.is_relative_to(Path(directory)))
                self.assertEqual(saved.read_bytes(), content)
                self.assertEqual(get_task_status(task_id), "pending")
                clear_task(task_id)

    def test_invalid_batch_is_rejected_before_saving_any_file(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(import_server, "PROJECT_ROOT", Path(directory)), \
                patch.object(import_server, "invoke_import_graph") as invoke:
            response = self.api_client(import_server.app).post("/upload", files=[
                ("files", ("good.md", b"valid", "text/markdown")),
                ("files", ("unsupported.exe", b"invalid", "application/octet-stream")),
            ])
            self.assertEqual(response.status_code, 400)
            self.assertEqual(list(Path(directory).iterdir()), [])
            invoke.assert_not_called()

    def test_empty_file_list_has_clear_error(self):
        with self.assertRaises(HTTPException) as caught:
            import_server.uploads(BackgroundTasks(), [])
        self.assertEqual(caught.exception.status_code, 400)

    def test_completed_status_matches_schema_and_unknown_task_is_404(self):
        task_id = "regression-completed-import"
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        add_done_task(task_id, "node_document_split")
        try:
            client = self.api_client(import_server.app)
            response = client.get(f"/status/{task_id}")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["done_list"], ["文档切分"])
            self.assertEqual(response.json()["running_list"], [])
            self.assertEqual(client.get("/status/regression-unknown-import").status_code, 404)
        finally:
            clear_task(task_id)

    def test_both_page_urls_serve_the_existing_page(self):
        client = self.api_client(import_server.app)
        for url in ("/html", "/import/html"):
            with self.subTest(url=url):
                response = client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.headers["content-type"])


class ImportContentTest(unittest.TestCase):
    def test_text_before_first_heading_is_preserved(self):
        chunks = split_chunks_document("重要开头\n# 第一节\n章节正文", "EEG文档")
        self.assertEqual([chunk["content"] for chunk in chunks], ["重要开头", "# 第一节\n章节正文"])
        self.assertEqual(chunks[0]["title"], "EEG文档")

    def test_heading_in_code_fence_is_not_a_section(self):
        text = "# 标题\n```python\n# 代码注释\nprint(1)\n```\n结束"
        chunks = split_chunks_document(text, "EEG文档")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], text)

    def test_untitled_long_text_keeps_its_first_character(self):
        text = "重要开头。" + "后续内容。" * 250
        chunks = refine_chunks(split_chunks_document(text, "EEG文档"))
        self.assertIn("重要开头。", chunks[0]["content"])
        self.assertTrue(all(chunk["title"] and chunk["parent_title"] for chunk in chunks))
        legacy_chunks = _split_long_chunk({"title": None, "file_title": "EEG文档", "content": text})
        self.assertIn("重要开头。", legacy_chunks[0]["content"])

    def test_uppercase_extensions_select_the_correct_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix, field, content in ((".MD", "is_md_read_enabled", b"# EEG"),
                                            (".PDF", "is_pdf_read_enabled", b"%PDF-1.4\n")):
                with self.subTest(suffix=suffix):
                    source = Path(directory) / ("document" + suffix)
                    source.write_bytes(content)
                    self.assertTrue(resolve_input_file({"local_file_path": str(source)})[field])

    def test_no_image_input_skips_vlm_and_object_storage(self):
        for mode in ("absent", "empty_directory", "file"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                md_path = Path(directory) / "sample.md"
                md_path.write_text("原始正文", encoding="utf-8")
                images = Path(directory) / "images"
                if mode == "empty_directory":
                    images.mkdir()
                elif mode == "file":
                    images.write_text("not a directory", encoding="utf-8")
                with patch.object(image_module, "summarize_images") as summarize, \
                        patch.object(image_module, "upload_images_and_replace") as upload:
                    result = image_module.enrich_markdown_images({"md_path": str(md_path)})
                self.assertEqual(result["md_content"], "原始正文")
                self.assertEqual(result["md_path"], str(md_path))
                summarize.assert_not_called()
                upload.assert_not_called()

    def test_http_and_https_urls_keep_the_complete_object_path(self):
        for secure, scheme in ((False, "http"), (True, "https")):
            config = SimpleNamespace(minio_config=SimpleNamespace(
                minio_secure=secure, endpoint="localhost:9000", bucket_name="kb", minio_img_dir="/images",
            ))
            with self.subTest(secure=secure), patch.object(minio_module, "infra_config", config):
                self.assertEqual(minio_module.minio_gateway.build_image_url("doc", "figure.png"),
                                 f"{scheme}://localhost:9000/kb/images/doc/figure.png")


class QueryLifecycleTest(AuthenticatedApiTest):
    def tearDown(self):
        remove_sse_queue("regression-query")
        clear_task("regression-query")

    def test_success_stream_finishes_without_browser_close(self):
        create_sse_queue("regression-query")
        with patch.object(query_server.query_app, "invoke", return_value={"answer": "answer", "image_urls": []}):
            query_server.invoke_query_graph("regression-query", "question", True)

        async def consume():
            request = MagicMock()
            async def connected():
                return False
            request.is_disconnected = connected
            return [event async for event in sse_generator("regression-query", request)]

        events = asyncio.run(asyncio.wait_for(consume(), timeout=3))
        self.assertTrue(any("event: final" in event for event in events))
        self.assertEqual(get_task_status("regression-query"), TASK_STATUS_COMPLETED)

    def test_failure_stream_emits_error_then_close(self):
        queue = create_sse_queue("regression-query")
        with patch.object(query_server.query_app, "invoke", side_effect=RuntimeError("simulated failure")):
            query_server.invoke_query_graph("regression-query", "question", True)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait()["event"])
        self.assertEqual(events[-2:], [SSEEvent.ERROR, SSEEvent.CLOSE])
        self.assertEqual(get_task_status("regression-query"), TASK_STATUS_FAILED)

    def test_blank_query_is_rejected_before_graph_or_stream_queue(self):
        client = self.api_client(query_server.app)
        with patch.object(query_server.query_app, "invoke") as invoke, \
                patch.object(query_server, "create_sse_queue") as create_queue:
            for query in ("", " \t\n ", None, ["EEGNet"]):
                with self.subTest(query=query):
                    response = client.post("/query", json={"query": query, "is_stream": True})
                    self.assertEqual(response.status_code, 422)
            invoke.assert_not_called()
            create_queue.assert_not_called()

    def test_normalized_query_reaches_graph(self):
        with patch.object(query_server.query_app, "invoke", return_value={"answer": "替身回答", "image_urls": []}) as invoke:
            response = self.api_client(query_server.app).post("/query", json={
                "query": "  EEGNet 的特点？\n", "session_id": " regression-query ",
            })
        self.assertEqual(response.status_code, 200)
        state = invoke.call_args.args[0]
        self.assertEqual(state["original_query"], "EEGNet 的特点？")
        self.assertEqual(state["session_id"], "regression-query")

    def test_unknown_stream_is_404(self):
        response = self.api_client(query_server.app).get("/stream/day01-missing-stream")
        self.assertEqual(response.status_code, 404)

    def test_history_limit_is_bounded_before_repository_call(self):
        client = self.api_client(query_server.app)
        with patch.object(query_server.history_repository, "list_recent", return_value=[]) as recent:
            for limit in (0, -1, 101):
                with self.subTest(limit=limit):
                    self.assertEqual(client.get(f"/history/day01?limit={limit}").status_code, 422)
            recent.assert_not_called()
            self.assertEqual(client.get("/history/day01?limit=1").status_code, 200)
            recent.assert_called_once_with(session_id="day01", limit=1)


class HistoryOrderTest(unittest.TestCase):
    def test_latest_messages_return_in_chronological_order(self):
        cursor = MagicMock()
        cursor.sort.return_value = cursor
        cursor.limit.return_value = cursor
        cursor.__iter__.return_value = iter([{"ts": 3}, {"ts": 2}])
        mongo_tool = MagicMock()
        mongo_tool.chat_message.find.return_value = cursor
        with patch.object(mongo_history_utils, "get_history_mongo_tool", return_value=mongo_tool):
            result = mongo_history_utils.get_recent_messages("session", limit=2)
        self.assertEqual([item["ts"] for item in result], [2, 3])
        cursor.sort.assert_called_once_with("ts", -1)
        cursor.limit.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
