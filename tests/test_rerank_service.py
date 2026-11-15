"""Rerank 到答案节点之间关键数据契约的回归测试。

运行：
    .\\.venv\\Scripts\\python.exe tests\\test_rerank_service.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.query import rerank_service


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> list[int]:
        return [ord(char) for char in (text or "")]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(value) for value in ids)

    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2


class FakeReranker:
    tokenizer = FakeTokenizer()

    def compute_score(self, pairs: list[list[str]], normalize: bool = True, **kwargs) -> list[float]:
        return [0.9 - index * 0.1 for index in range(len(pairs))]


class RerankServiceTest(unittest.TestCase):
    def test_rerank_returns_a_list_when_web_search_is_empty(self) -> None:
        state = {
            "rewritten_query": "HAK 180 如何安全放置？",
            "rrf_chunks": [
                {"chunk_id": 1, "title": "安全要求", "content": "设备应放在平整稳固的台面上。"},
                {"chunk_id": 2, "title": "注意事项", "content": "使用前检查设备稳定性。"},
            ],
            "web_search_docs": [],
        }
        fake_reranker = FakeReranker()

        with (
            patch.object(rerank_service.reranker_config, "enabled", True),
            patch.object(rerank_service.llm_providers, "reranker_model", return_value=fake_reranker),
        ):
            result = rerank_service.rerank_documents(state)

        self.assertIsInstance(result["reranked_docs"], list)
        self.assertEqual(len(result["reranked_docs"]), 2)
        self.assertEqual(result["reranked_docs"][0]["chunk_id"], 1)

    def test_dynamic_limit_never_returns_none(self) -> None:
        docs = [{"score": 0.9}, {"score": 0.7}, {"score": 0.6}]

        result = rerank_service.dyn_limit_reranker_docs(docs)

        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 2)

    def test_disabled_model_uses_rrf_scores_without_loading_model(self) -> None:
        state = {
            "rewritten_query": "测试问题",
            "rrf_chunks": [
                {"chunk_id": 1, "title": "低分", "content": "内容一", "score": 0.01},
                {"chunk_id": 2, "title": "高分", "content": "内容二", "score": 0.03},
            ],
            "web_search_docs": [],
        }

        with (
            patch.object(rerank_service.reranker_config, "enabled", False),
            patch.object(rerank_service.llm_providers, "reranker_model") as model_loader,
        ):
            result = rerank_service.rerank_documents(state)

        model_loader.assert_not_called()
        self.assertEqual([doc["chunk_id"] for doc in result["reranked_docs"]], [2, 1])


if __name__ == "__main__":
    unittest.main()
