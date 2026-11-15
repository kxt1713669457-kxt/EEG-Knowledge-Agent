import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MAX_SIZE, CHUNK_MIN
from app.rag.import_.chunk_metadata import attach_chunk_metadata
from app.rag.import_.markdown_structure import structure_blocks, table_parts
from app.shared.runtime.logger import logger, step_log


@step_log("load_markdown_content")
def load_markdown_content(state: ImportGraphState):
    content = state.get('md_content')
    md_path = state.get('md_path')
    if not content and md_path and Path(md_path).is_file():
        content = Path(md_path).read_text(encoding='utf-8-sig')
    if not content or not content.strip():
        raise ValueError('Markdown内容为空，不能切块')
    title = state.get('file_title') or (Path(md_path).stem if md_path else 'default')
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    state.update(file_title=title, md_content=content)
    return content, title


@step_log("split_chunks_document")
def split_chunks_document(md_content, file_title) -> list[dict[str, Any]]:
    """按标题粗切；保留标题前正文、空行和代码缩进，代码围栏内不识别标题。"""
    chunks, lines = [], []
    title, fence = None, None

    def flush():
        content = '\n'.join(lines).strip('\n')
        if content.strip():
            chunks.append({'content': content, 'title': title or file_title,
                           'file_title': file_title, 'section_index': len(chunks)})

    for line in md_content.split('\n'):
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            lines.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            lines.append(line)
        elif re.match(r'^ {0,3}#{1,6}\s+\S', line):
            flush()
            title, lines = line.strip(), [line]
        else:
            lines.append(line)
    flush()
    return chunks


@step_log("_split_long_chunk")
def _split_long_chunk(chunk) -> list[dict[str, Any]]:
    title = chunk.get('title') or ''
    prefix = title + '\n' if title else ''
    content = chunk['content']
    body = content[len(prefix):] if prefix and content.startswith(prefix) else content
    budget = max(1, CHUNK_SIZE - len(prefix))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=budget, chunk_overlap=min(CHUNK_OVERLAP, budget - 1),
        add_start_index=True,
        separators=['\n\n', '\n', '。', '！', '？', '；', '.', '!', '?', ';', ' ', ''],
    )
    result = []
    block_cursor = 0
    for kind, block in structure_blocks(body):
        block_start = body.find(block, block_cursor)
        block_cursor = block_start + len(block)
        if kind == 'prose':
            documents = splitter.create_documents([block])
            parts = [document.page_content for document in documents]
        elif kind == 'table':
            # 真实2b表格在1000字符内保持整体；超长简单表才按行重复表头。
            parts = table_parts(block, max(1, CHUNK_MAX_SIZE - len(prefix)))
        else:
            parts = [block]
        previous_end = None
        for part_index, part in enumerate(parts):
            result.append({**chunk, 'title': f'{title}_第{len(result)+1}部分',
                           'content': prefix + part, 'parent_title': title,
                           'part': len(result)+1, 'content_kind': kind})
            if kind == 'prose':
                start = documents[part_index].metadata['start_index']
                if start < 0 or block[start:start+len(part)] != part:
                    raise ValueError('正文切片位置无法对应原文，不能猜测重叠')
                result[-1].update(_text_start=block_start+start, _text_end=block_start+start+len(part),
                                 _joiner_before=block[previous_end:start] if previous_end is not None and start>=previous_end else '')
                previous_end = start+len(part)
            if len(prefix + part) > CHUNK_MAX_SIZE:
                # 跨行表格、代码等不能为了长度硬拆；显式保留例外供质量检查。
                result[-1]['structure_overflow'] = True
    return result


@step_log("_merge_short_chunks_same_parent_title")
def _merge_short_chunks_same_parent_title(refine_list) -> list[dict[str, Any]]:
    merged = []
    for current in refine_list:
        if not merged:
            merged.append(dict(current))
            continue
        previous = merged[-1]
        parent = previous.get('parent_title')
        if (not parent or parent != current.get('parent_title')
                or previous.get('section_index') != current.get('section_index')
                or len(previous['content']) >= CHUNK_MIN
                or previous.get('content_kind', 'prose') != 'prose'
                or current.get('content_kind', 'prose') != 'prose'):
            merged.append(dict(current))
            continue
        prefix = parent + '\n'
        following = current['content'].removeprefix(prefix)
        # 只按原文位置消除真实重叠，不能把碰巧相同的字母误删。
        positions = all(isinstance(value, int) for value in
                        (previous.get('_text_end'), current.get('_text_start')))
        overlap = max(0, previous['_text_end']-current['_text_start']) if positions else 0
        if overlap and not previous['content'].endswith(following[:overlap]):
            raise ValueError('正文切片重叠与原文位置不符，不能合并')
        joiner = current.get('_joiner_before', '\n') if positions else '\n'
        combined = previous['content'] + joiner + following[overlap:]
        if len(combined) <= CHUNK_MAX_SIZE:
            previous['content'] = combined
            if positions:
                previous['_text_end'] = current['_text_end']
        else:
            merged.append(dict(current))
    return merged


@step_log("refine_chunks")
def refine_chunks(chunks):
    """标题粗切后，<=1000字符保留；超长正文按600/100细分，结构单独保真。"""
    refined = []
    for chunk in chunks:
        if len(chunk['content']) > CHUNK_MAX_SIZE:
            refined.extend(_split_long_chunk(chunk))
        else:
            # 长度规则直接决定保留610/750，不依赖模型临时判断“语义好不好”。
            refined.append(dict(chunk))
    refined = _merge_short_chunks_same_parent_title(refined)
    for chunk in refined:
        for key in ('_text_start', '_text_end', '_joiner_before'):
            chunk.pop(key, None)
        chunk.setdefault('parent_title', chunk.get('title') or 'default_title')
        chunk.setdefault('part', 1)
    return refined


@step_log("backup_chunks_json")
def backup_chunks_json(refine_chunks_list, md_path: str):
    Path(md_path).with_suffix('.json').write_text(
        json.dumps(refine_chunks_list, indent=2, ensure_ascii=False), encoding='utf-8')


@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    content, title = load_markdown_content(state)
    chunks = refine_chunks(split_chunks_document(content, title))
    attach_chunk_metadata(chunks, state.get('document_metadata') or {})
    state['chunks'] = chunks
    backup_chunks_json(chunks, state['md_path'])
    return state
