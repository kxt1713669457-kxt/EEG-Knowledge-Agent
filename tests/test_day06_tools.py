"""工具调用边界：无效参数不进服务，正常空结果与失败分开，来源不能由模型编造。"""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock, PropertyMock, patch
import unittest

from tests.auth_client import AuthenticatedClient as TestClient
from pydantic import ValidationError

from app.agent.tools import rag_tools, registry
from app.agent.tools.schemas import (
    CompareEvidenceInput, DocumentContextInput, KnowledgeSearchInput, ToolPayload,
    WebSearchInput,
)
from app.api.http.query_server import app
from app.rag.query import answer_service


DOC = 'a' * 64
UID = 'b' * 64


def row(index=1, revision='batch1', uid=UID):
    return {'doc_id': DOC, 'chunk_uid': uid, 'chunk_id': index + 10, 'chunk_index': index,
            'index_revision': revision, 'content': 'sampling frequency 250 Hz',
            'source_type': 'pdf', 'object_key': f'originals/{DOC}/source.pdf',
            'title': 'Recording', 'file_title': '2b_description'}


class ToolRegistryTest(unittest.TestCase):
    def execute_with(self, arguments, result=None, error=None):
        handler = Mock(return_value=ToolPayload(status='empty') if result is None else result,
                       side_effect=error)
        definitions = {'search_knowledge': replace(registry.TOOLS['search_knowledge'], handler=handler)}
        result = registry.execute_tool('search_knowledge', arguments, definitions=definitions)
        return result, handler

    def test_unknown_tool_does_not_run_any_handler(self):
        handler = Mock()
        result = registry.execute_tool('delete_collection', {}, definitions={
            'search_knowledge': replace(registry.TOOLS['search_knowledge'], handler=handler)})
        self.assertEqual(result.error.code, 'unknown_tool')
        handler.assert_not_called()

    def test_invalid_inputs_never_enter_handler(self):
        for args in ({}, {'query': '  '}, {'query': 'x', 'top_k': 11},
                     {'query': 'x', 'top_k': '5'}, {'query': 'x', 'top_k': True},
                     {'query': 'x', 'collection_name': 'other'}, None):
            with self.subTest(args=args):
                result, handler = self.execute_with(args)
                self.assertEqual(result.error.code, 'invalid_arguments')
                handler.assert_not_called()

    def test_defaults_applied_and_empty_is_not_failure(self):
        result, handler = self.execute_with({'query': ' EEG '})
        self.assertEqual(handler.call_args.args[0].query, 'EEG')
        self.assertEqual(handler.call_args.args[0].top_k, 10)
        self.assertEqual(result.status, 'empty')
        self.assertIsNone(result.error)
        self.assertEqual(len(result.tool_call_id), 32)

    def test_none_output_is_failure_not_empty(self):
        handler = Mock(return_value=None)
        definition = replace(registry.TOOLS['search_knowledge'], handler=handler)
        result = registry.execute_tool('search_knowledge', {'query': 'x'},
                                       definitions={'search_knowledge': definition})
        self.assertEqual(result.status, 'failed')

    def test_internal_error_text_is_not_exposed(self):
        result, _ = self.execute_with({'query': 'x'}, error=ConnectionError('private-provider-url'))
        self.assertEqual(result.status, 'failed')
        self.assertTrue(result.error.retryable)
        self.assertNotIn('private-provider-url', result.model_dump_json())

    def test_schema_and_runtime_share_limits(self):
        tool = next(t for t in registry.tool_declarations() if t['function']['name'] == 'search_knowledge')
        schema = tool['function']['parameters']
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['properties']['top_k']['maximum'], 10)
        self.assertNotIn('collection_name', schema['properties'])

    def test_output_contract_rejects_error_in_ok_result(self):
        with self.assertRaises(ValidationError):
            ToolPayload(status='ok', error={'code': 'x', 'message': 'bad'})

    def test_http_lists_tools_and_rejects_unknown_name(self):
        client = TestClient(app)
        self.assertEqual({row['function']['name'] for row in client.get('/agent/tools').json()['tools']},
                         {'search_knowledge', 'search_web', 'get_document_context', 'compare_evidence',
                          'analyze_eeg_sample', 'get_eeg_analysis', 'read_workspace_files', 'inspect_uploaded_image', 'read_public_document',
                          'preprocess_eeg_recording', 'get_eeg_preprocessing', 'analyze_preprocessed_eeg',
                          'inspect_eeg_attachment', 'register_eeg_recording', 'register_eeg_sample'})
        response = client.post('/agent/tools/execute', json={'name': 'shell', 'arguments': {}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['error']['code'], 'unknown_tool')

    def test_http_distinguishes_envelope_422_and_tool_argument_failure(self):
        client = TestClient(app)
        self.assertEqual(client.post('/agent/tools/execute', json={'name': 'search_knowledge'}).status_code, 422)
        result = client.post('/agent/tools/execute', json={'name': 'search_knowledge', 'arguments': {'query': 'x', 'top_k': 99}})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['error']['code'], 'invalid_arguments')


class RagToolsTest(unittest.TestCase):
    def test_zero_local_hits_skip_rrf_and_reranker(self):
        with patch.object(rag_tools.embedding_search_service, 'search_by_embedding', return_value=[]), \
                patch.object(rag_tools.hyde_search_service, 'search_by_hyde', return_value=[]), \
                patch.object(rag_tools.rerank_service, 'rerank_documents') as rerank:
            result = rag_tools.search_knowledge(KnowledgeSearchInput(query='EEG'))
        self.assertEqual(result.status, 'empty')
        rerank.assert_not_called()

    def test_knowledge_reuses_services_and_applies_output_cap(self):
        before = row()
        def rerank(state):
            state['reranked_docs'] = [{'text': 'first', 'doc_id': DOC, 'chunk_uid': UID},
                                     {'text': 'second', 'doc_id': DOC, 'chunk_uid': 'c'*64},
                                     {'text': 'legacy without identity'}]
            state['rerank_status'] = {'mode': 'model'}
        with patch.object(rag_tools.embedding_search_service, 'search_by_embedding', return_value=[before]), \
                patch.object(rag_tools.hyde_search_service, 'search_by_hyde', return_value=[]), \
                patch.object(rag_tools.rrf_service, 'fuse_by_rrf') as fuse, \
                patch.object(rag_tools.rerank_service, 'rerank_documents', side_effect=rerank):
            result = rag_tools.search_knowledge(KnowledgeSearchInput(query='EEG', top_k=1))
        self.assertEqual(len(result.data['documents']), 1)
        self.assertEqual(result.data['rerank_status']['mode'], 'model')
        self.assertEqual(result.data['legacy_without_identity_excluded'], 1)
        fuse.assert_called_once()
        self.assertEqual(before, row())

    def test_web_empty_and_failure_are_distinct(self):
        for status, expected in [('empty', 'empty'), ('failed', 'failed')]:
            def search(state):
                state.update(web_search_docs=[], web_search_status={'status': status, 'reason': 'timeout'})
            with patch.object(rag_tools.web_search_service, 'search_by_web', side_effect=search):
                self.assertEqual(rag_tools.search_web(WebSearchInput(query='EEG')).status, expected)

    def test_context_neighbors_are_same_publication_and_sorted(self):
        client = Mock()
        client.query.side_effect = [[row()], [row(2, uid='c'*64), row(1), row(0, uid='d'*64)]]
        with patch.object(rag_tools.milvus_gateway.__class__, 'milvus_client', new_callable=PropertyMock, return_value=client):
            result = rag_tools.get_document_context(DocumentContextInput(doc_id=DOC, chunk_uid=UID))
        self.assertIn('index_revision == "batch1"', client.query.call_args.kwargs['filter'])
        self.assertEqual([d['chunk_index'] for d in result.data['documents']], [0, 1, 2])
        self.assertEqual(result.data['documents'][0]['index_revision'], 'batch1')

    def test_web_filters_explicit_dataset_conflict_before_output_cap(self):
        def search(state):
            state.update(web_search_status={'status': 'ok'}, web_search_docs=[
                {'title': 'BCI 2a', 'snippet': '22 EEG channels', 'url': 'https://example.com/2a'},
                {'title': 'BCI 2b', 'snippet': '3 EEG channels', 'url': 'https://example.com/2b'}])
        with patch.object(rag_tools.web_search_service, 'search_by_web', side_effect=search):
            result = rag_tools.search_web(WebSearchInput(query='BCI 2b', top_k=1))
        self.assertEqual(result.data['documents'][0]['title'], 'BCI 2b')
        self.assertEqual(result.data['explicit_conflicts_excluded'], 1)

    def test_context_missing_is_empty(self):
        client = Mock()
        client.query.return_value = []
        with patch.object(rag_tools.milvus_gateway.__class__, 'milvus_client', new_callable=PropertyMock, return_value=client):
            result = rag_tools.get_document_context(DocumentContextInput(doc_id=DOC, chunk_uid=UID))
        self.assertEqual(result.status, 'empty')

    def test_context_refuses_mixed_or_changed_publication(self):
        client = Mock()
        client.query.side_effect = [[row()], [row(revision='other')]]
        with patch.object(rag_tools.milvus_gateway.__class__, 'milvus_client', new_callable=PropertyMock, return_value=client):
            with self.assertRaises(RuntimeError):
                rag_tools.get_document_context(DocumentContextInput(doc_id=DOC, chunk_uid=UID))

    def test_comparison_takes_only_real_source_references(self):
        with self.assertRaises(ValidationError):
            CompareEvidenceInput(question='compare', sources=[{'doc_id': DOC, 'chunk_uid': UID, 'text': 'fake'},
                                                             {'doc_id': DOC, 'chunk_uid': 'c'*64}])

    def test_comparison_rejects_duplicate_sources(self):
        with self.assertRaises(ValidationError):
            CompareEvidenceInput(question='compare', sources=[{'doc_id': DOC, 'chunk_uid': UID}]*2)

    def test_comparison_does_not_persist_internal_chat_message(self):
        args = CompareEvidenceInput(question='compare', sources=[{'doc_id': DOC, 'chunk_uid': UID},
                                                                {'doc_id': DOC, 'chunk_uid': 'c'*64}])
        def generate(state, **kwargs):
            state.update(answer='answer', citations=[], citation_status={'status': 'insufficient'})
        with patch.object(rag_tools, 'read_document_context', return_value=[{'text': 'real'}]), \
                patch.object(rag_tools.answer_service, 'generate_answer', side_effect=generate) as generate_mock:
            result = rag_tools.compare_evidence(args)
        self.assertFalse(generate_mock.call_args.kwargs['persist_history'])
        self.assertEqual(result.status, 'empty')

    def test_answer_history_default_preserved_and_internal_mode_can_skip(self):
        with patch.object(answer_service.history_repository, 'save_message') as save:
            answer_service.generate_answer({'answer': 'clarify', 'is_stream': False})
            self.assertEqual(save.call_count, 1)
            answer_service.generate_answer({'answer': 'clarify', 'is_stream': False}, persist_history=False)
            self.assertEqual(save.call_count, 1)


if __name__ == '__main__':
    unittest.main()
