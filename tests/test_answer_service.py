"""答案收尾逻辑的回归测试。"""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.query.answer_service import extract_text_image_url


class AnswerServiceTest(unittest.TestCase):
    def test_generated_image_block_uses_actual_evidence_urls(self) -> None:
        state = {"answer": "有依据的说明\n\n【图片】\n<https://example.com/wrong-hash.png>",
                 "reranked_docs": [{"url": None, "text": "![图](https://example.com/actual-hash.png)"}]}
        extract_text_image_url(state)
        self.assertIn("actual-hash.png", state["answer"])
        self.assertNotIn("wrong-hash.png", state["answer"])

    def test_extract_image_urls_accepts_extension_set_and_query_string(self) -> None:
        state = {
            "reranked_docs": [
                {"url": "https://example.com/manual.jpg?token=1", "text": "正文"},
                {"url": "https://example.com/article", "text": "![示意图](https://example.com/a.png)"},
                {"url": None, "text": None},
            ]
        }

        extract_text_image_url(state)

        self.assertEqual(
            state["image_urls"],
            ["https://example.com/manual.jpg?token=1", "https://example.com/a.png"],
        )


if __name__ == "__main__":
    unittest.main()
