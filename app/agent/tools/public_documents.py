"""有界读取公开学术原件；返回实际页码/摘录范围，搜索摘要不冒充正文。"""
from datetime import datetime, timezone
import ipaddress
import hashlib
import json
from pathlib import Path
import re
import socket
import time
from urllib.parse import urlsplit, urljoin, urlunsplit

import httpx
from bs4 import BeautifulSoup
from app.agent.tools.schemas import ToolPayload

HOSTS = {'arxiv.org', 'export.arxiv.org', 'openreview.net', 'proceedings.neurips.cc',
         'papers.neurips.cc', 'pmc.ncbi.nlm.nih.gov', 'www.bbci.de', 'bbci.de',
         'mne.tools', 'www.nature.com', 'www.frontiersin.org'}
MAX_BYTES = 12 * 1024 * 1024


def checked_url(url):
    parsed = urlsplit(url)
    if parsed.username or parsed.password: raise ValueError('公开原件地址不能带凭证')
    if parsed.scheme == 'http' and parsed.hostname in HOSTS and parsed.port in {None,80}:
        parsed=parsed._replace(scheme='https',netloc=parsed.hostname)
    if (parsed.scheme != 'https' or parsed.hostname not in HOSTS or parsed.port not in {None, 443}
            or parsed.username or parsed.password):
        raise ValueError('只接受受支持学术站点的HTTPS公开原件，不接受内网或带凭证地址')
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError('公开来源解析到了非公网地址')
    path = re.sub(r'^/(?:abs|html)/', '/pdf/', parsed.path) if parsed.hostname in {'arxiv.org', 'export.arxiv.org'} else parsed.path
    if parsed.hostname in {'arxiv.org','export.arxiv.org'} and path.startswith('/pdf/'):
        path=re.sub(r'\.pdf$','',path)
        parsed=parsed._replace(query='')
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ''))


def download(url):
    url = checked_url(url)
    cache = Path(__file__).resolve().parents[3] / 'output/runtime/public-documents'
    identity = hashlib.sha256(url.encode()).hexdigest()
    entry = cache / (identity+'.json')
    if entry.exists() and time.time()-entry.stat().st_mtime < 3600:
        row = json.loads(entry.read_text(encoding='utf8'))
        body = (cache / (identity+'.bin')).read_bytes()
        if len(body)<=MAX_BYTES and hashlib.sha256(body).hexdigest()==row['sha256']:
            return checked_url(row['url']), body, row['retrieved_at']
    deadline = time.monotonic() + 55
    with httpx.Client(timeout=httpx.Timeout(15, connect=2), transport=httpx.HTTPTransport(retries=0), follow_redirects=False) as client:
        for _ in range(4):
            with client.stream('GET', url, headers={'User-Agent': 'EEGResearch/1.0'}) as response:
                if response.is_redirect:
                    url = checked_url(urljoin(url, response.headers['location']))
                    continue
                response.raise_for_status()
                data = bytearray()
                for block in response.iter_bytes():
                    data.extend(block)
                    if len(data) > MAX_BYTES: raise ValueError('原件超过12MB读取上限')
                    if time.monotonic() > deadline: raise TimeoutError('原件读取超时')
                body=bytes(data)
                retrieved_at=datetime.now(timezone.utc).isoformat()
                cache.mkdir(parents=True,exist_ok=True)
                # 缓存只降低重复下载开销，不改变来源获取时间或声称重新联网。
                (cache / (identity+'.bin')).write_bytes(body)
                entry.write_text(json.dumps({'url':url,'retrieved_at':retrieved_at,'sha256':hashlib.sha256(body).hexdigest()}),encoding='utf8')
                return url, body, retrieved_at
    raise ValueError('原件重定向次数过多')


def select_pages(pages, query, max_bytes=18500):
    terms = set(re.findall(r'[A-Za-z][A-Za-z0-9_-]{2,}', query.lower())) - {'the', 'and', 'results', 'accuracy', 'paper'}
    scores = []
    results = bool(re.search(r'结果|性能|准确率|对比|比较|实验|评测|results|accuracy|performance|experiment|table|finding', query, re.I))
    for index, text in enumerate(pages):
        low = text.lower()
        score = sum(min(low.count(word), 4) for word in terms)
        if results:
            score += 40 * len(re.findall(r'accuracies\s*\(%\)\s*in', low))
            score += 8 * len(re.findall(r'table\s+[ivx\d]+[^\n]*(?:accurac|result|performance)', low))
            score += 3 * len(re.findall(r'(?:\d{2}\.\d+\s*±|30%.*finetun|experimental results)', low))
        scores.append((score, index))
    indices = sorted({0, *(index for _, index in sorted(scores, key=lambda pair: (-pair[0], pair[1]))[:3])})
    selected, used = [], 0
    for index in indices:
        text = pages[index].strip()
        if index == 0 and len(text.encode('utf8')) > 2400:
            text = '\n'.join(text.splitlines()[:22])
        size = len(text.encode('utf8'))
        if used + size > max_bytes: continue
        if text:
            selected.append((index + 1, text)); used += size
    return selected


def excerpt_cache_path(url,query):
    category='results' if re.search(r'结果|性能|准确率|对比|比较|实验|评测|results|accuracy|performance|experiment|table|finding',query,re.I) else 'other'
    identity=hashlib.sha256((url+'|'+category).encode()).hexdigest()
    return Path(__file__).resolve().parents[3]/'output/runtime/public-excerpts'/(identity+'.json')


def read_public_document(args):
    url=checked_url(args.url)
    cache_path=excerpt_cache_path(url,args.query)
    if cache_path.exists() and time.time()-cache_path.stat().st_mtime<3600:
        payload=ToolPayload.model_validate_json(cache_path.read_text(encoding='utf8'))
        payload.data['cache_hit']=True
        payload.data['notice']='复用此前实际提取的有限页面，保留原获取时间；不代表本轮重新下载或已读取整篇。'
        return payload
    url, data, retrieved_at = download(url)
    if data.startswith(b'%PDF-'):
        import fitz
        with fitz.open(stream=data, filetype='pdf') as pdf:
            if pdf.page_count > 60: raise ValueError('原件超过60页，请上传文献知识库处理')
            pages = [page.get_text() for page in pdf]
            version=re.search(r'arXiv:\s*(\d{4}\.\d{4,5}v\d+)',pages[0])
        selected, count, scope = select_pages(pages, args.query), len(pages), 'public_pdf_selected_pages'
    else:
        soup = BeautifulSoup(data, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']): tag.decompose()
        main = soup.find('article') or soup.find('main') or soup
        text = main.get_text('\n', strip=True)
        selected, count, scope = select_pages([text], args.query), None, 'public_page_partial_extract'
    docs = [{'type': 'web', 'title': f'公开原件摘录 · 第{page}页' if count else '公开页面摘录',
             'text': text, 'url': url, 'source_url': url, 'page_start': page if count else None,
             'page_end': page if count else None, 'content_scope': scope,
             'retrieval_backend': 'public_document_reader', 'retrieved_at': retrieved_at}
            for page, text in selected]
    payload = ToolPayload(status='ok' if docs else 'empty', data={'documents': docs,
        'content_scope': scope, 'total_pages': count, 'selected_pages': [p for p, _ in selected],
        'cache_hit': False,
        'notice': '仅返回实际提取的页面/段落；图像表格或缺失页面不能据此补猜。'})
    if docs:
        cache_path.parent.mkdir(parents=True,exist_ok=True)
        cache_path.write_text(payload.model_dump_json(),encoding='utf8')
        if data.startswith(b'%PDF-') and version and urlsplit(url).hostname in {'arxiv.org','export.arxiv.org'}:
            alias=excerpt_cache_path('https://arxiv.org/pdf/'+version.group(1),args.query)
            alias.write_text(payload.model_dump_json(),encoding='utf8')
    return payload
