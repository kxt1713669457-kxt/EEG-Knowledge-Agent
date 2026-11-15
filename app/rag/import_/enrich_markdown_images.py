"""将本地论文插图转为可检索的文字说明，并保留可访问的原图。"""
import base64
import hashlib
import json
import re
from mimetypes import guess_type
from pathlib import Path
from posixpath import normpath
from urllib.parse import unquote, urlsplit

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_providers
from app.infra.object_storage.minio_gateway import minio_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.rate_limit_utils import apply_api_rate_limit

# MinerU行内图片及可选标题；远程URL保持原样。
IMAGE_REFERENCE = re.compile(r'!\[(?P<alt>(?:\\.|[^\]\\\n])*)\]\(\s*(?P<target><[^>\n]+>|[^\s)]+)(?:\s+["\'][^\n]*?["\'])?\s*\)')


def _local_target(target: str) -> str | None:
    parsed = urlsplit(target.strip('<>'))
    if parsed.scheme or parsed.netloc:
        return None
    return unquote(parsed.path).replace('\\', '/')


def retrieval_description(raw: str) -> str:
    """已实测VLM会误读时间边界/人体左右；这些断言不进入检索说明。"""
    precise = re.compile(
        r'\d+(?:\.\d+)?\s*(?:秒|毫秒|分钟|小时|Hz|ms|μV|µV|mV|%|％|s\b)'
        r'|外眦|位于[^。；]*(?:眼|眉|额|头皮|脑)', re.I)
    sentences = re.split(r'(?<=[。；])', ' '.join(raw.split()))
    kept = ''.join(sentence for sentence in sentences if not precise.search(sentence)).strip()
    return kept or '论文插图；精确读数和电极位置请核对原图及正文。'


@step_log("validates_and_get_data")
def validates_and_get_data(state: ImportGraphState):
    md_path = state.get("md_path")
    if not md_path or not Path(md_path).is_file():
        raise FileNotFoundError("缺少可读取的Markdown文件")
    md_path_obj = Path(md_path)
    md_content = md_path_obj.read_text(encoding="utf-8-sig")
    if not md_content.strip():
        raise ValueError("Markdown内容为空")
    state['md_content'] = md_content
    return md_content, md_path_obj.parent / "images", md_path_obj


@step_log("scan_images")
def scan_images(images_path_obj, md_content) -> list[tuple[str, str, tuple[str, str]]]:
    """按完整相对路径匹配引用，区分子目录里的同名图。"""
    image_root = Path(images_path_obj).resolve()
    result, seen = [], set()
    for match in IMAGE_REFERENCE.finditer(md_content):
        target = _local_target(match['target'])
        if target is None:
            continue
        image_path = (image_root.parent / target).resolve()
        if not image_path.is_relative_to(image_root):
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if not image_path.is_file():
            raise FileNotFoundError(f"Markdown引用的图片不存在：{target}")
        if image_path in seen:
            continue
        seen.add(image_path)
        result.append((image_path.relative_to(image_root).as_posix(), str(image_path), (
            md_content[max(0, match.start() - 300):match.start()],
            md_content[match.end():match.end() + 300],
        )))
    return result


@step_log("summarize_images")
def summarize_images(image_content: list[tuple[str, str, tuple[str, str]]], stem: str):
    vision_client = llm_providers.vision_chat()
    summaries = {}
    audit = []
    for image_name, image_path_str, context in image_content:
        apply_api_rate_limit()
        image_text = load_prompt("image_summary", root_folder=stem, image_content=context)
        encoded = base64.b64encode(Path(image_path_str).read_bytes()).decode("ascii")
        message = HumanMessage(content=[
            {"type": "text", "text": image_text},
            {"type": "image_url", "image_url": {"url": f"data:{guess_type(image_name)[0] or 'image/png'};base64,{encoded}"}},
        ])
        summary = (vision_client | StrOutputParser()).invoke([message]).strip()
        if not summary:
            raise ValueError(f"视觉模型未返回图片说明：{image_name}")
        summaries[image_name] = retrieval_description(summary)
        audit.append({'asset': image_name, 'raw_model_output': summary,
                      'retrieval_description': summaries[image_name],
                      'notice': '模型生成的检索辅助；不能作为精确读数依据'})
        logger.info(f"完成图片说明：{image_name}")
    if image_content:
        # 留下原始输出便于审计；正文中仅放受限的检索说明。
        assets_dir = Path(image_content[0][1]).parent
        (assets_dir / 'image_descriptions.json').write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return summaries


@step_log("upload_images_and_replace")
def upload_images_and_replace(md_content: str, summaries_images: dict[str, str],
                              image_content: list[tuple[str, str, tuple[str, str]]], stem: str) -> str:
    """图片按内容寻址；全部上传成功后才替换正文，重建失败不会删旧图。"""
    urls = {}
    for image_name, image_path_str, _ in image_content:
        summary = summaries_images.get(image_name, "").strip()
        if not summary:
            raise ValueError(f"图片缺少说明：{image_name}")
        image_path = Path(image_path_str)
        asset_name = hashlib.sha256(image_path.read_bytes()).hexdigest() + image_path.suffix.lower()
        # 保留当前网关与历史链接使用的对象键约定。
        object_name = f"{minio_gateway.minio_img_dir}/{stem}/{asset_name}"
        minio_gateway.minio_client.fput_object(
            bucket_name=minio_gateway.bucket_name, object_name=object_name,
            file_path=image_path_str, content_type=guess_type(image_path.name)[0] or "application/octet-stream",
        )
        urls[image_name] = minio_gateway.build_image_url(stem, asset_name)

    def replace(match):
        target = _local_target(match['target'])
        if target is None:
            return match.group(0)
        normalized = normpath(target)
        image_key = normalized.removeprefix('./').removeprefix('images/')
        if image_key not in urls:
            return match.group(0)
        description = " ".join(summaries_images[image_key].split())
        return f"![{match['alt'] or '原图'}]({urls[image_key]})\n\n> 图像说明（模型生成）：{description}"

    return IMAGE_REFERENCE.sub(replace, md_content)


@step_log("backup_new_md_content")
def backup_new_md_content(md_content_new, md_path_obj):
    target = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")
    target.write_text(md_content_new, encoding="utf-8")
    return str(target)


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    md_content, images_dir, md_path = validates_and_get_data(state)
    if not images_dir.is_dir():
        if any((_local_target(m['target']) or '').removeprefix('./').startswith('images/')
               for m in IMAGE_REFERENCE.finditer(md_content)):
            raise FileNotFoundError("Markdown引用了images目录中的图片，但图片目录不存在")
        return state
    images = scan_images(images_dir, md_content)
    if not images:
        return state
    namespace = (state.get("document_metadata") or {}).get("doc_id") or md_path.stem
    summaries = summarize_images(images, namespace)
    enhanced = upload_images_and_replace(md_content, summaries, images, namespace)
    state['md_content'] = enhanced
    state['md_path'] = backup_new_md_content(enhanced, md_path)
    return state
