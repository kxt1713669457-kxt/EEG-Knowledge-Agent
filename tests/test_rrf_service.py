"""RRF 融合服务的无外部依赖回归测试。

运行：
    .\\.venv\\Scripts\\python.exe tests\\test_rrf_service.py
"""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.process.query.agent.state import create_query_default_state
from app.rag.query.rrf_service import fuse_by_rrf, use_by_rrf


def chunk(chunk_id: int) -> dict:
    return {"chunk_id": chunk_id, "score": 0.0, "content": f"chunk {chunk_id}"}


class RrfServiceTest(unittest.TestCase):
    def test_fuse_by_rrf_combines_two_retrieval_lists(self) -> None:
        state = create_query_default_state(
            embedding_chunks=[chunk(1), chunk(2)],
            hyde_embedding_chunks=[chunk(2), chunk(3)],
        )

        result = fuse_by_rrf(state)

        self.assertEqual([item["chunk_id"] for item in result["rrf_chunks"]], [2, 1, 3])
        self.assertGreater(result["rrf_chunks"][0]["score"], result["rrf_chunks"][1]["score"])

    def test_use_by_rrf_requires_weighted_lists(self) -> None:
        result = use_by_rrf([(0.5, [chunk(1)]), (1.0, [chunk(2)])])

        self.assertEqual([item["chunk_id"] for item in result], [2, 1])

    def test_fuse_by_rrf_allows_one_retrieval_source_to_be_empty(self) -> None:
        state = create_query_default_state(
            embedding_chunks=[chunk(1)],
            hyde_embedding_chunks=[],
        )

        result = fuse_by_rrf(state)

        self.assertEqual([item["chunk_id"] for item in result["rrf_chunks"]], [1])

    def test_web_only_results_can_reach_reranker(self) -> None:
        state = create_query_default_state(web_search_docs=[{"content": "web evidence", "url": "https://example.org"}])
        result = fuse_by_rrf(state)
        self.assertEqual(result["rrf_chunks"], [])
        self.assertEqual(result["web_search_docs"], state["web_search_docs"])

    def test_all_sources_empty_still_raise(self) -> None:
        with self.assertRaises(ValueError):
            fuse_by_rrf(create_query_default_state())


if __name__ == "__main__":
    unittest.main()
