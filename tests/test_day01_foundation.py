"""不依赖 Docker、模型或联网的请求契约、状态隔离和真实图编排检查。"""
import unittest

from pydantic import ValidationError

from app.api.schema.query_schema import QueryRequestSchema
from app.process.import_.agent.state import create_default_state
from app.process.query.agent.state import create_query_default_state
from scripts.day01_diagnose import run_query_case


class RequestContractTest(unittest.TestCase):
    def test_empty_or_non_string_query_is_rejected(self):
        for query in ("", " \t\n ", None, 123, ["EEGNet"]):
            with self.subTest(query=query), self.assertRaises(ValidationError):
                QueryRequestSchema(query=query)

    def test_query_and_session_are_normalized(self):
        request = QueryRequestSchema(query="  EEGNet 的特点？\n", session_id="  eeg-session  ")
        self.assertEqual(request.query, "EEGNet 的特点？")
        self.assertEqual(request.session_id, "eeg-session")

    def test_missing_null_and_blank_sessions_mean_new_session(self):
        for params in ({}, {"session_id": None}, {"session_id": " \t"}):
            with self.subTest(params=params):
                self.assertIsNone(QueryRequestSchema(query="EEGNet？", **params).session_id)


class StateIsolationTest(unittest.TestCase):
    def test_query_nested_lists_do_not_leak_across_requests(self):
        first = create_query_default_state()
        first["embedding_chunks"].append({"content": "first"})
        first["history"].append({"text": "first"})
        second = create_query_default_state()
        self.assertEqual(second["embedding_chunks"], [])
        self.assertEqual(second["history"], [])

    def test_import_chunks_do_not_leak_across_tasks(self):
        first = create_default_state()
        first["chunks"].append({"content": "first"})
        first["embeddings_content"].append({"dense_vector": [0.1]})
        self.assertEqual(create_default_state()["chunks"], [])
        self.assertEqual(create_default_state()["embeddings_content"], [])


class QueryTopologyTest(unittest.TestCase):
    def test_retrieval_branches_join_once_before_rerank(self):
        result = run_query_case("retrieval")
        self.assertEqual(result["status"], "completed")
        names = [item["node"] for item in result["trace"]]
        self.assertEqual(names.count("node_rrf"), 1)
        for branch in ("node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"):
            self.assertLess(names.index(branch), names.index("node_rrf"))
        self.assertLess(names.index("node_rrf"), names.index("node_rerank"))
        self.assertEqual(names[-1], "node_answer_output")
        self.assertEqual(result["rrf_count"], 1)

    def test_explicit_clarification_bypasses_retrieval(self):
        result = run_query_case("clarification")
        self.assertEqual(result["status"], "completed")
        self.assertEqual([item["node"] for item in result["trace"]],
                         ["node_item_name_confirm", "node_answer_output"])
        self.assertIn("具体指", result["answer"])

    def test_open_retrieval_still_joins_three_branches(self):
        result = run_query_case("open_retrieval")
        self.assertEqual(result["status"], "completed")
        names = [item["node"] for item in result["trace"]]
        self.assertEqual(names.count("node_rrf"), 1)
        for branch in ("node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"):
            self.assertLess(names.index(branch), names.index("node_rrf"))

    def test_node_failure_prevents_success_answer(self):
        result = run_query_case("node_failure")
        self.assertEqual(result["status"], "failed")
        self.assertIn("HyDE", result["error"])
        self.assertFalse(any(item["node"] in {"node_rrf", "node_rerank", "node_answer_output"}
                             for item in result["trace"]))


if __name__ == "__main__":
    unittest.main()
