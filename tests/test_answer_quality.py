"""冻结真实缺陷：stop+EE/标题/半句、漏结果、过期引用与有界纠正。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from app.agent import conversation
from app.rag.query import answer_service, evidence_service
from app.rag.query.answer_quality import (answer_issues, requested_length, output_budget,
                                         text_size, IncompleteAnswerError)


class Client:
    def __init__(self, responses): self.responses=iter(responses);self.calls=[];self.bindings=[]
    def bind(self, **kwargs): self.bindings.append(kwargs);return self
    def bind_tools(self, declarations): self.declarations=declarations;return self
    def invoke(self, messages): self.calls.append(messages);return next(self.responses)


def reply(text, reason="stop"):
    return AIMessage(content=text,response_metadata={"finish_reason":reason},
                     usage_metadata={"input_tokens":10,"output_tokens":5,"total_tokens":15})


def history_source():
    return {"role":"assistant","task_id":"old-research-task","text":"依据真实论文给出的旧回答。 [E9]",
            "citation_status":{"status":"valid"},"answer_audit":{"conversation":{"rag_invoked":False}},"evidence_sources":[{
                "text":"EEGNet的实际原文夹具：一个小型网络，参数量较少。",
                "type":"web","url":"https://example.org/eegnet-paper","doc_id":"a"*64,"chunk_uid":"b"*64,"source_type":"pdf",
                "object_key":"originals/"+"a"*64+"/source.pdf","title":"EEGNet paper"}]}


class AnswerQualityTests(unittest.TestCase):
    def run_chat(self, responses, question="解释卷积网络的方法", history=None, review=None):
        client=Client(responses)
        with patch.object(conversation.llm_providers,"chat",return_value=client), \
             patch.object(conversation.history_repository,"list_recent",return_value=history or []), \
             patch.object(conversation,"review_answer_support",side_effect=review):
            result=conversation.run_conversation(question,"quality-test","x"*32,1)
        return result,client

    def test_real_ee_response_is_not_a_complete_answer_even_with_stop(self):
        self.assertIn("label_instead_of_answer",answer_issues("eegnet这个框架结果如何","EE","stop"))

    def test_real_heading_only_response_is_not_complete(self):
        self.assertIn("heading_only",answer_issues("结果呢","### EEGNet","stop"))

    def test_real_summary_intro_does_not_cover_requested_length(self):
        answer="### EEGNet\n\n研究背景\n"+"这是一段背景介绍。"*25
        self.assertIn("requested_length_not_covered",answer_issues("写一千字左右的总结，包括结果",answer,"stop"))
        self.assertIn("requested_length_not_covered",answer_issues("写1000字总结","文"*690,"stop"))

    def test_unclosed_generated_code_is_not_a_complete_answer(self):
        self.assertIn("unclosed_code_block",answer_issues("解释流程","下面是流程。\n```mermaid\nflowchart TD\nA --> B","stop"))

    def test_half_sentence_and_unfinished_heading_are_rejected(self):
        self.assertIn("unfinished_sentence",answer_issues("结果呢","EEGNet在四个具有代表性的脑电数据集中进行了BCI"))
        self.assertIn("unfinished_heading",answer_issues("解释方法","下面说明方法。\n\n###"))

    def test_real_short_parameter_and_yes_no_answers_remain_valid(self):
        self.assertEqual(answer_issues("采样率是多少？","250 Hz","stop"),[])
        self.assertEqual(answer_issues("只回答是或否：可以吗","否","stop"),[])
        self.assertEqual(answer_issues("你好","你好！","stop"),[])
        self.assertEqual(answer_issues("解释处理步骤","- 首先读取输入数据\n- 然后执行模型推理\n- 最后核对输出与依据","stop"),[])

    def test_explicit_brief_name_answer_is_preserved(self):
        self.assertEqual(answer_issues("解释后只返回名称","EEGNet","stop"),[])

    def test_length_and_filtered_response_are_not_success(self):
        self.assertEqual(answer_issues("你好","你好","length"),["output_limit"])
        self.assertEqual(answer_issues("你好","你好","content_filter"),["invalid_finish_reason"])

    def test_length_parse_and_output_reservation(self):
        self.assertEqual(requested_length("一千字左右总结"),(1000,"characters"))
        self.assertEqual(requested_length("写1500 words"),(1500,"words"))
        self.assertEqual(output_budget("一千字左右总结"),2112)
        self.assertEqual(output_budget("一万字总结",2400),2400)  # 不识别未支持的数量而瞎分配预算。
        self.assertEqual(output_budget("100000字总结"),4096)

    def test_quality_repair_is_one_extra_call_and_records_both_attempts(self):
        (_,result),client=self.run_chat([reply("EE"),reply("卷积网络通过滑动卷积核提取局部特征，并通过后续网络组合成更高层表示。")])
        self.assertEqual(len(client.calls),2)
        self.assertGreater(len(result["answer"]),20)

    def test_two_bad_answers_fail_and_keep_raw_audit(self):
        with self.assertRaises(IncompleteAnswerError) as caught:
            self.run_chat([reply("EE"),reply("### EEGNet")])
        self.assertEqual(len(caught.exception.audit["generation_attempts"]),2)
        self.assertEqual(caught.exception.audit["generation_attempts"][0]["raw_answer"],"EE")

    def test_length_stops_without_retry_and_preserves_audit(self):
        with self.assertRaises(conversation.ContextBudgetError) as caught:
            self.run_chat([reply("被截断的草稿","length")])
        self.assertEqual(len(caught.exception.audit["generation_attempts"]),1)

    def test_history_incomplete_assistant_is_not_fed_back_as_an_example(self):
        history=[{"role":"user","text":"结果呢"},{"role":"assistant","text":"### EEGNet"},
                 {"role":"user","text":"结果如何"},{"role":"assistant","text":"EE"}]
        past=conversation._past_messages(history)
        self.assertFalse(any(isinstance(m,AIMessage) for m in past))

    def test_old_reference_numbers_are_removed_from_history_text(self):
        past=conversation._past_messages([{"role":"assistant","text":"这是旧回答的完整句子。 [E9]"}])
        self.assertNotIn("[E9]",past[0].content)

    def test_only_real_source_snapshot_is_reused_not_history_answer(self):
        row=history_source();sources=conversation._cached_sources([row],"EEGNet结果如何")
        self.assertEqual(len(sources),1)
        self.assertEqual(sources[0]["text"],row["evidence_sources"][0]["text"])
        self.assertNotEqual(sources[0]["text"],row["text"])
        self.assertEqual(sources[0]["reused_from_task_id"],row["task_id"])

    def test_other_named_model_does_not_receive_eegnet_snapshot(self):
        self.assertEqual(conversation._cached_sources([history_source()],"Transformer结果如何"),[])

    def test_invalid_source_identity_and_invalid_review_are_not_reused(self):
        row=history_source();row["citation_status"]["status"]="invalid"
        self.assertEqual(conversation._cached_sources([row],"结果呢"),[])
        row=history_source();row["evidence_sources"][0]["type"]="milvus";row["evidence_sources"][0]["object_key"]="guessed-name.pdf"
        self.assertEqual(conversation._cached_sources([row],"结果呢"),[])

    def test_null_legacy_citation_status_does_not_crash(self):
        row=history_source();row["citation_status"]=None
        self.assertEqual(len(conversation._cached_sources([row],"结果呢")),1)

    def test_snapshot_sources_are_renumbered_for_current_round_and_reviewed(self):
        seen=[]
        def review(state): seen.append(deepcopy(state));state["citation_status"]["support_check"]="model_review"
        (state,result),client=self.run_chat([reply("EEGNet采用紧凑设计，原文片段说明该网络参数量较少。 [E1]")],
            question="EEGNet结果如何",history=[history_source()],review=review)
        self.assertEqual(result["citation_status"]["status"],"valid")
        self.assertEqual(len(seen),1)
        self.assertEqual(state["answer_audit"]["conversation"]["rag_invoked"],False)
        self.assertEqual(state["evidence_sources"][0]["evidence_id"],"E1")
        self.assertTrue(state["answer_audit"]["generation_attempts"])

    def test_old_unknown_reference_is_repaired_and_not_hidden_as_success(self):
        (_,result),_=self.run_chat([reply("这是一个有用的方法。 [E999]"),reply("卷积通过局部连接和共享参数提取特征，这里没有新增查询到的文献依据。")])
        self.assertNotIn("[E999]",result["answer"])

    def test_requested_search_not_actually_called_cannot_claim_completed_search(self):
        with self.assertRaises(IncompleteAnswerError) as caught:
            self.run_chat([reply("我已经搜索，发现了相关信息。")],question="帮我上网搜一下")
        self.assertIn("requested_search_not_executed",caught.exception.reasons)

    def test_rag_generation_has_bounded_repair_too(self):
        client=Client([reply("### EEGNet"),reply("EEGNet采用深度卷积和可分离卷积来减少网络参数。 [E1]")])
        state={"original_query":"解释EEGNet的方法","is_stream":False}
        with patch.object(answer_service.llm_providers,"chat",return_value=client):
            answer_service.call_llm_deal_answer(state,"按原文解释EEGNet的方法。")
        self.assertEqual(len(client.calls),2)
        self.assertEqual(state["generation_usage"]["total_tokens"],30)
        self.assertEqual(len(state["answer_audit"]["generation_attempts"]),2)

    def test_citation_validation_preserves_generation_audit(self):
        state={"answer":"已支持的事实。 [E1]","evidence_sources":[{"evidence_id":"E1","text":"实际依据"}],
               "answer_audit":{"generation_attempts":[{"raw_answer":"EE"}]}}
        evidence_service.validate_answer_citations(state)
        self.assertEqual(state["answer_audit"]["generation_attempts"][0]["raw_answer"],"EE")

    def test_source_review_sections_are_closed_labels_and_remain_quote_checked(self):
        from app.rag.query import answer_support_service as review
        source={"evidence_id":"E1","text":"EEGNet是紧凑网络。参数量较少。"}
        payload={"verdict":"supported","claims":[
            {"claim":"EEGNet是紧凑网络。","evidence_id":"E1","quote_id":"E1.Q1","section":"method"},
            {"claim":"参数量较少。","evidence_id":"E1","quote_id":"E1.Q1","section":"results"}]}
        client=Client([reply(json.dumps(payload,ensure_ascii=False))])
        state={"original_query":"总结EEGNet论文","answer":"EEGNet是紧凑网络，参数量较少。 [E1]",
               "evidence_sources":[source],"citation_status":{"status":"valid"}}
        with patch.object(review.llm_providers,"chat",return_value=client):review.review_answer_support(state)
        self.assertEqual(state["citation_status"]["status"],"valid")
        self.assertIn("## 模型设计",state["answer"])
        self.assertIn("## 实验结果",state["answer"])
        self.assertTrue(state["citation_status"]["quotes_checked"])

    def test_source_review_cannot_hide_new_factual_claims_in_headings(self):
        from app.rag.query.answer_support_service import SupportedClaim
        with self.assertRaises(ValueError):SupportedClaim.model_validate({"claim":"结论","evidence_id":"E1","quote_id":"E1.Q1","section":"准确率99%"})

    def test_long_review_preserves_supported_draft_block_without_resummarizing(self):
        from app.rag.query import answer_support_service as review
        text="EEGNet是紧凑网络，其参数量较少。 [E1]"
        payload={"verdict":"supported","claims":[],"blocks":[{"block_id":"B1","quote_ids":["E1.Q1"],"section":"results"}]}
        client=Client([reply(json.dumps(payload))])
        state={"original_query":"写1000字总结","answer":text,"evidence_sources":[{"evidence_id":"E1","text":"EEGNet是紧凑网络，其参数量较少。"}],"citation_status":{"status":"valid"}}
        with patch.object(review.llm_providers,"chat",return_value=client):review.review_answer_support(state)
        self.assertEqual(state["answer"],text)
        self.assertEqual(state["citation_status"]["status"],"valid")

    def test_unknown_draft_block_cannot_be_added_to_verified_answer(self):
        from app.rag.query import answer_support_service as review
        client=Client([reply(json.dumps({"verdict":"supported","claims":[],"blocks":[{"block_id":"B99","quote_ids":["E1.Q1"]}]}))])
        state={"original_query":"写1000字总结","answer":"已支持的句子。 [E1]","evidence_sources":[{"evidence_id":"E1","text":"已支持的句子。"}],"citation_status":{"status":"valid"}}
        with patch.object(review.llm_providers,"chat",return_value=client):review.review_answer_support(state)
        self.assertEqual(state["citation_status"]["status"],"invalid")
        self.assertIn("复核未完成",state["answer"])

    def test_reviewed_block_requires_quotes_for_all_its_current_sources(self):
        from app.rag.query import answer_support_service as review
        client=Client([reply(json.dumps({"verdict":"supported","claims":[],"blocks":[{"block_id":"B1","quote_ids":["E1.Q1"]}]}))])
        state={"original_query":"写1000字总结","answer":"事实甲。 [E1] 事实乙。 [E2]","evidence_sources":[{"evidence_id":"E1","text":"事实甲。"},{"evidence_id":"E2","text":"事实乙。"}],"citation_status":{"status":"valid"}}
        with patch.object(review.llm_providers,"chat",return_value=client):review.review_answer_support(state)
        self.assertEqual(state["citation_status"]["status"],"invalid")


if __name__=="__main__":unittest.main()
