"""图片来源、失败处理及论文结构保真回归；外部模型和对象存储用替身。"""
import re
import tempfile
import unittest
import numpy as np
from scipy.sparse import csr_array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.rag.import_ import enrich_markdown_images as images
from app.rag.import_ import split_service as splitter
from app.rag.import_.markdown_structure import table_parts
from app.shared.model import embedding_utils


class EmbeddingInputBoundaryTest(unittest.TestCase):
    def test_model_adapter_receives_explicit_capacity_instead_of_default_512(self):
        with patch.object(embedding_utils, '_bge_m3_ef', None), \
                patch.object(embedding_utils, 'BGEM3EmbeddingFunction') as factory:
            embedding_utils.get_bge_m3_ef()
        self.assertEqual(factory.call_args.kwargs['passage_max_length'], 8192)
        self.assertEqual(factory.call_args.kwargs['query_max_length'], 8192)

    def test_over_capacity_input_fails_before_encoder_can_truncate(self):
        model = MagicMock()
        model.model.tokenizer.encode.return_value = [1] * 8193
        with patch.object(embedding_utils, 'get_bge_m3_ef', return_value=model), self.assertRaisesRegex(ValueError, '不能静默截断'):
            embedding_utils.generate_embeddings(['long input'])
        model.encode_documents.assert_not_called()

    def test_input_longer_than_512_reaches_encoder_and_sparse_weights_are_preserved(self):
        model = MagicMock()
        model.model.tokenizer.encode.return_value = [1] * 534
        model.encode_documents.return_value = {
            'dense': np.zeros((1, 1024), dtype=np.float32),
            'sparse': csr_array(([2.5], ([0], [2])), shape=(1, 8)),
        }
        with patch.object(embedding_utils, 'get_bge_m3_ef', return_value=model):
            result = embedding_utils.generate_embeddings(['table input'])
        model.encode_documents.assert_called_once_with(['table input'])
        self.assertEqual(len(result['dense'][0]), 1024)
        self.assertEqual(result['sparse'][0], {2: 2.5})


class ImageFidelityTest(unittest.TestCase):
    def test_observed_vlm_timing_and_anatomy_claims_do_not_enter_retrieval_text(self):
        raw = '实验时间线，含Screening与Smiley Feedback。反馈期3至8秒。#1位于受试者右眼外眦。横轴是时间。'
        text = images.retrieval_description(raw)
        self.assertIn('Smiley Feedback', text)
        self.assertNotIn('3至8秒', text)
        self.assertNotIn('外眦', text)

    def test_nested_uppercase_and_similar_names_match_exact_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'images'
            (root / 'a').mkdir(parents=True)
            (root / 'b').mkdir()
            for name in ('a/figure.PNG', 'b/figure.PNG', 'image.png'):
                (root / name).write_bytes(b'image')
            text = '![A](images/a/figure.PNG)\n![B](images/b/figure.PNG)'
            found = images.scan_images(root, text)
            self.assertEqual([row[0] for row in found], ['a/figure.PNG', 'b/figure.PNG'])
            self.assertEqual(images.scan_images(root, '![remote](https://example.com/image.png)'), [])
            # 相似文件名存在却不是被引用的图片，不能误判成功。
            with self.assertRaises(FileNotFoundError):
                images.scan_images(root, '![A](images/other_image.png)')

    def test_missing_referenced_assets_fail_instead_of_silent_success(self):
        with tempfile.TemporaryDirectory() as directory:
            md = Path(directory) / 'paper.md'
            md.write_text('![图](images/missing.png)', encoding='utf-8')
            with self.assertRaises(FileNotFoundError), patch.object(images, 'summarize_images') as vlm:
                images.enrich_markdown_images({'md_path': str(md)})
            vlm.assert_not_called()

    def test_repeated_reference_only_calls_vision_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'images'
            root.mkdir()
            (root / 'one.png').write_bytes(b'image')
            found = images.scan_images(root, '![A](images/one.png)\n![B](images/one.png)')
            self.assertEqual(len(found), 1)

    def test_exact_replacement_retains_body_caption_and_literal_backslash(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'one.png'
            source.write_bytes(b'one')
            client = MagicMock()
            gateway = SimpleNamespace(minio_client=client, bucket_name='kb', minio_img_dir='/images',
                                      build_image_url=lambda ns, key: f'http://localhost/kb/images/{ns}/{key}')
            md = '实验正文\n![A](images/a/one.png)\n原图图注\n![B](images/b/one.png)'
            with patch.object(images, 'minio_gateway', gateway):
                result = images.upload_images_and_replace(md, {'a/one.png': r'连接\1 [模块]'},
                                                          [('a/one.png', str(source), ('', ''))], 'doc')
            self.assertIn('实验正文', result)
            self.assertIn('原图图注', result)
            self.assertIn(r'连接\1 [模块]', result)
            self.assertIn('![B](images/b/one.png)', result)
            self.assertIn('图像说明（模型生成）', result)
            client.remove_objects.assert_not_called()

    def test_upload_failure_is_visible_and_old_assets_are_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'one.png'
            source.write_bytes(b'one')
            client = MagicMock()
            client.fput_object.side_effect = RuntimeError('object storage unavailable')
            gateway = SimpleNamespace(minio_client=client, bucket_name='kb', minio_img_dir='/images')
            with patch.object(images, 'minio_gateway', gateway), self.assertRaises(RuntimeError):
                images.upload_images_and_replace('![A](images/one.png)', {'one.png': '说明'},
                                                 [('one.png', str(source), ('', ''))], 'doc')
            client.remove_objects.assert_not_called()

    def test_changed_image_bytes_get_different_object_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'one.png'
            client = MagicMock()
            gateway = SimpleNamespace(minio_client=client, bucket_name='kb', minio_img_dir='/images',
                                      build_image_url=lambda ns, key: f'http://localhost/{ns}/{key}')
            with patch.object(images, 'minio_gateway', gateway):
                for content in (b'old', b'new'):
                    source.write_bytes(content)
                    images.upload_images_and_replace('![A](images/one.png)', {'one.png': '说明'},
                                                     [('one.png', str(source), ('', ''))], 'doc')
            keys = [call.kwargs['object_name'] for call in client.fput_object.call_args_list]
            self.assertNotEqual(*keys)
            client.remove_objects.assert_not_called()


class StructureFidelityTest(unittest.TestCase):
    def test_repeated_heading_text_does_not_merge_different_section_occurrences(self):
        chunks = [{'content': '# 方法\n第一部分', 'parent_title': '# 方法', 'section_index': 0},
                  {'content': '# 方法\n第二部分', 'parent_title': '# 方法', 'section_index': 2}]
        result = splitter._merge_short_chunks_same_parent_title(chunks)
        self.assertEqual(len(result), 2)

    def test_merging_does_not_guess_overlap_from_coincidentally_matching_letters(self):
        chunks = [{'content':'# EEG\nwith','parent_title':'# EEG','content_kind':'prose'},
                  {'content':'# EEG\nhow to classify','parent_title':'# EEG','content_kind':'prose'}]
        result = splitter._merge_short_chunks_same_parent_title(chunks)
        self.assertEqual(result[0]['content'], '# EEG\nwith\nhow to classify')

    def test_real_source_overlap_is_removed_once_using_offsets(self):
        chunks = [{'content':'# EEG\nABCDEFGHIJ','parent_title':'# EEG','content_kind':'prose',
                   'section_index':0,'_text_start':0,'_text_end':10},
                  {'content':'# EEG\nHIJ KLMNOP','parent_title':'# EEG','content_kind':'prose',
                   'section_index':0,'_text_start':7,'_text_end':17,'_joiner_before':''}]
        result = splitter._merge_short_chunks_same_parent_title(chunks)
        self.assertEqual(result[0]['content'], '# EEG\nABCDEFGHIJ KLMNOP')

    def test_sections_within_maximum_keep_original_text_instead_of_splitting_then_merging(self):
        for size in (610, 750, 1000):
            with self.subTest(size=size):
                prefix = '# EEG\n' + '甲' * 300 + '\n\n'
                text = prefix + '乙' * (size - len(prefix))
                chunks = splitter.refine_chunks(splitter.split_chunks_document(text, 'paper'))
                self.assertEqual([c['content'] for c in chunks], [text])

    def test_section_exceeding_maximum_is_split_without_losing_body_characters(self):
        body = ''.join(chr(0x4e00 + i) for i in range(1800))
        chunks = splitter.refine_chunks(splitter.split_chunks_document('# EEG\n' + body, 'paper'))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c['content']) <= splitter.CHUNK_MAX_SIZE for c in chunks))
        combined = ''.join(c['content'] for c in chunks)
        self.assertTrue(all(character in combined for character in body))

    def test_short_sections_with_different_headings_are_not_combined(self):
        chunks = splitter.refine_chunks(splitter.split_chunks_document(
            '# EEG\n' + '甲' * 300 + '\n# EOG\n' + '乙' * 300, 'paper'))
        self.assertEqual(len(chunks), 2)
        self.assertNotIn('乙', chunks[0]['content'])
        self.assertNotIn('甲', chunks[1]['content'])

    def test_blank_lines_indentation_and_different_fences_are_preserved(self):
        text = '引言\n\n# 方法\n\n````python\n    if True:\n        print(1)\n~~~\n# 这不是标题\n````\n\n# 结果\n结论'
        chunks = splitter.split_chunks_document(text, 'paper')
        self.assertEqual(len(chunks), 3)
        self.assertIn('    if True:\n        print(1)', chunks[1]['content'])
        self.assertIn('# 这不是标题', chunks[1]['content'])
        self.assertIn('\n\n', chunks[1]['content'])

    def test_unbroken_long_text_is_bounded_without_losing_characters(self):
        text = 'A' * 2500
        chunks = splitter.refine_chunks(splitter.split_chunks_document(text, 'paper'))
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(row['content']) <= splitter.CHUNK_MAX_SIZE for row in chunks))
        self.assertTrue(all(set(row['content'].removeprefix('paper\n')) == {'A'} for row in chunks))

    def test_realistic_long_table_splits_rows_and_repeats_header(self):
        header = '<tr><td>ID</td><td>Training</td><td>Evaluation</td></tr>'
        rows = [f'<tr><td>{i}</td><td>B{i:02}01T</td><td>B{i:02}04E</td></tr>' for i in range(1, 20)]
        table = '<table>' + header + ''.join(rows) + '</table>'
        chunks = splitter.refine_chunks(splitter.split_chunks_document('# 数据\n' + table, 'paper'))
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn(header, chunk['content'])
            self.assertEqual(chunk['content'].count('<table>'), 1)
            self.assertEqual(chunk['content'].count('</table>'), 1)
            self.assertLessEqual(len(chunk['content']), splitter.CHUNK_MAX_SIZE)
        for row in rows:
            self.assertEqual(sum(chunk['content'].count(row) for chunk in chunks), 1)

    def test_rowspan_is_kept_intact_instead_of_misassigning_cells(self):
        table = '<table><tr><th>实验</th><th>数值</th></tr><tr><td rowspan="2">MI</td><td>250</td></tr><tr><td>3</td></tr></table>'
        self.assertEqual(table_parts(table, 50), [table])

    def test_image_and_generated_description_stay_in_same_chunk(self):
        image = '![原图](http://localhost/figure.png)\n\n> 图像说明（模型生成）：C3、Cz、C4三个EEG通道。'
        text = '# 设置\n' + ('研究正文。' * 180) + '\n\n' + image + '\n\n' + ('后续正文。' * 150)
        chunks = splitter.refine_chunks(splitter.split_chunks_document(text, 'paper'))
        matches = [c['content'] for c in chunks if 'figure.png' in c['content']]
        self.assertEqual(len(matches), 1)
        self.assertIn('C3、Cz、C4', matches[0])


if __name__ == '__main__':
    unittest.main()
