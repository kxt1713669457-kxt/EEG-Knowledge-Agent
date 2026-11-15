"""已登记官方来源的入口查询；真实访问后返回提取片段，未知主题交给MCP。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

CATALOG=Path(__file__).resolve().parents[3]/'corpus/eeg/official-source-catalog.json'
ALLOWED_HOSTS={'www.bbci.de','bbci.de','arxiv.org','openreview.net','proceedings.iclr.cc','mne.tools','moabb.neurotechx.com'}
MAX_BYTES=8*1024*1024


def _key(value):return re.sub(r'[^a-z0-9]+','',value.casefold())


def matched_official_sources(query):
    if not re.search(r'官方|官网|official',query,re.I) or not CATALOG.exists():return []
    if not re.search(r'页面|入口|网址|链接|发表|文档|page|entry|publication|documentation',query,re.I):return []
    records=json.loads(CATALOG.read_text(encoding='utf8'));query_key=_key(query);matched=[]
    for record in records:
        parsed=urlsplit(record['url'])
        if parsed.scheme!='https' or parsed.hostname not in ALLOWED_HOSTS or parsed.port not in {None,443} or parsed.username or parsed.password:
            continue
        strength=max((len(_key(alias)) for alias in record['aliases']if len(_key(alias))>=4 and _key(alias)in query_key),default=0)
        if strength:matched.append((strength,record))
    matched.sort(key=lambda pair:pair[0],reverse=True)
    return [row for _,row in matched[:2]]


def fetch_official_source(record):
    # URL只来自服务端目录，不接受模型提交URL；不跟随重定向到其他主机。
    parsed=urlsplit(record['url'])
    if parsed.scheme!='https' or parsed.hostname not in ALLOWED_HOSTS or parsed.port not in {None,443} or parsed.username or parsed.password:
        raise ValueError('unregistered_official_host')
    with httpx.Client(timeout=8,follow_redirects=False)as client:
        deadline=time.perf_counter()+12
        with client.stream('GET',record['url'])as response:
            response.raise_for_status()
            if response.status_code!=200:raise ValueError('official_source_not_200')
            if int(response.headers.get('content-length','0'))>MAX_BYTES:raise ValueError('official_source_too_large')
            data=bytearray()
            for block in response.iter_bytes():
                data.extend(block)
                if time.perf_counter()>deadline:raise TimeoutError('official_source_deadline')
                if len(data)>MAX_BYTES:raise ValueError('official_source_too_large')
    if bytes(data).startswith(b'%PDF-'):
        import fitz
        with fitz.open(stream=bytes(data),filetype='pdf')as document:
            text=document[0].get_text()
        scope='official_pdf_first_page_extract'
    else:
        soup=BeautifulSoup(bytes(data),'html.parser')
        for tag in soup(['script','style','nav','footer','header']):tag.decompose()
        main=soup.find('main')or soup.find('article')or soup
        text=main.get_text(' ',strip=True);scope='official_page_partial_extract'
    if not text.strip():raise ValueError('official_source_empty')
    if not any(len(_key(alias))>=6 and _key(alias)in _key(text)for alias in record['aliases']):
        raise ValueError('official_source_identity_mismatch')
    return {'title':record['title'],'url':record['url'],'snippet':text[:4000],
            'retrieved_at':datetime.now(timezone.utc).isoformat(),'content_scope':scope,'source':'web',
            'retrieval_backend':'registered_official_source'}


def lookup_official_sources(query):
    docs=[];errors=[]
    for record in matched_official_sources(query):
        try:docs.append(fetch_official_source(record))
        except Exception as exc:errors.append({'source_id':record['source_id'],'error_type':type(exc).__name__})
    return docs,errors
