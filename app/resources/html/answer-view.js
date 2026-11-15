// 只调整展示。后端答案、引用身份、原文复核与历史审计保持原样。
function answerLink(raw){
  try{const u=new URL(raw,API_BASE);if(!['http:','https:'].includes(u.protocol)||u.username||u.password)return '';return window.eegMediaBase&&u.href.startsWith(window.eegMediaBase)?new URL('/sources/image?url='+encodeURIComponent(u.href),API_BASE).href:u.href;}catch(_){return '';}
}

let answerDiagramSequence=0;
function renderAnswerFlowchart(container,source){
  // 只解析小型有向无环流程图；不运行Mermaid/HTML/事件或用户代码，未知语法保留源码。
  const lines=source.trim().split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
  if(!/^(?:flowchart|graph)\s+(?:TD|TB)\s*;?$/.test(lines.shift()||''))return false;
  const nodes=new Map(),edges=[];
  function readNode(text){
    const m=text.match(/^([A-Za-z][A-Za-z0-9_]{0,31})(?:\["([^"\n]{1,100})"\]|\[([^\]\n]{1,100})\]|\("([^"\n]{1,100})"\)|\(([^)\n]{1,100})\)|\{"([^"\n]{1,100})"\}|\{([^}\n]{1,100})\})?/);
    if(!m)return null;const id=m[1],label=m[2]||m[3]||m[4]||m[5]||m[6]||m[7],decision=!!(m[6]||m[7]);
    if(nodes.has(id)&&label&&nodes.get(id).explicit&&nodes.get(id).label!==label)return null;
    if(!nodes.has(id)||label)nodes.set(id,{id,label:label||id,explicit:!!label,decision});
    return{id,rest:text.slice(m[0].length).trim()};
  }
  for(let line of lines){
    if(line.startsWith('%%'))continue;if(line.endsWith(';'))line=line.slice(0,-1).trim();
    let left=readNode(line);if(!left)return false;
    while(left.rest){const arrow=left.rest.match(/^-->(?:\|([^|\n]{1,24})\|)?\s*/);if(!arrow)return false;
      const right=readNode(left.rest.slice(arrow[0].length));if(!right||left.id===right.id)return false;
      edges.push({from:left.id,to:right.id,label:arrow[1]||''});left=right;
    }
  }
  if(nodes.size<2||nodes.size>14||edges.length>20||!edges.length)return false;
  const degree=new Map([...nodes.keys()].map(id=>[id,0])),ranks=new Map([...nodes.keys()].map(id=>[id,0]));
  for(const edge of edges)degree.set(edge.to,degree.get(edge.to)+1);
  const queue=[...nodes.keys()].filter(id=>degree.get(id)===0);let visited=0;
  for(let i=0;i<queue.length;i++){const id=queue[i];visited++;
    for(const edge of edges.filter(e=>e.from===id)){ranks.set(edge.to,Math.max(ranks.get(edge.to),ranks.get(id)+1));degree.set(edge.to,degree.get(edge.to)-1);if(!degree.get(edge.to))queue.push(edge.to);}
  }
  if(visited!==nodes.size)return false;
  const groups=[];for(const node of nodes.values())(groups[ranks.get(node.id)]||=[]).push(node);
  const compact=window.innerWidth<=600,nodeWidth=compact?148:188,nodeHeight=compact?68:44,gap=compact?14:22,step=nodeHeight+(compact?32:24),labelLimit=compact?7:13;
  function wrapLabel(text){const result=[];let line='',units=0;for(const char of text){const size=/[\x00-\x7f]/.test(char)?0.55:1;if(units+size>labelLimit){result.push(line);line='';units=0;}line+=char;units+=size;}if(line)result.push(line);return result;}
  for(const node of nodes.values()){node.lines=wrapLabel(node.label);if(node.lines.length>3)return false;}
  const width=Math.max(compact?250:500,...groups.map(g=>g.length*nodeWidth+(g.length-1)*gap+40)),height=groups.length*step+16,positions=new Map();
  groups.forEach((group,rank)=>{const left=(width-(group.length*nodeWidth+(group.length-1)*gap))/2;group.forEach((node,i)=>positions.set(node.id,{x:left+nodeWidth/2+i*(nodeWidth+gap),y:28+rank*step}));});
  const ns='http://www.w3.org/2000/svg',create=(tag,attrs={})=>{const el=document.createElementNS(ns,tag);for(const [key,value]of Object.entries(attrs))el.setAttribute(key,String(value));return el;};
  const figure=document.createElement('figure');figure.className='answer-flowchart';
  const caption=document.createElement('figcaption');caption.textContent='流程示意';figure.append(caption);
  const scroll=document.createElement('div');scroll.className='answer-diagram-scroll';
  const svg=create('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':'流程示意，包含：'+[...nodes.values()].map(n=>n.label).join('、')});
  svg.style.width=width+'px';svg.style.maxWidth='100%';
  svg.style.setProperty('--answer-diagram-label',compact?'18px':'14px');
  const defs=create('defs'),marker=create('marker',{id:'answer-arrow-'+(++answerDiagramSequence),viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});
  marker.append(create('path',{d:'M 1 1 L 9 5 L 1 9',fill:'none',stroke:'currentColor','stroke-width':1.5}));defs.append(marker);svg.append(defs);
  for(const edge of edges){const from=positions.get(edge.from),to=positions.get(edge.to),y1=from.y+nodeHeight+(nodes.get(edge.from).decision?8:0),y2=to.y-(nodes.get(edge.to).decision?8:0),mid=(y1+y2)/2;
    svg.append(create('path',{d:`M ${from.x} ${y1} V ${mid} H ${to.x} V ${y2-4}`,class:'answer-flow-edge','marker-end':`url(#${marker.id})`}));
    if(edge.label){const label=create('text',{x:(from.x+to.x)/2+7,y:mid-6,class:'answer-edge-label'});label.textContent=edge.label;svg.append(label);}
  }
  for(const node of nodes.values()){const p=positions.get(node.id),group=create('g');group.append(node.decision?create('polygon',{points:`${p.x},${p.y-8} ${p.x+nodeWidth/2+8},${p.y+nodeHeight/2} ${p.x},${p.y+nodeHeight+8} ${p.x-nodeWidth/2-8},${p.y+nodeHeight/2}`,class:'answer-flow-node'}):create('rect',{x:p.x-nodeWidth/2,y:p.y,width:nodeWidth,height:nodeHeight,rx:10,class:'answer-flow-node'}));
    const lineHeight=compact?22:18,text=create('text',{x:p.x,y:p.y+nodeHeight/2,'text-anchor':'middle',class:'answer-flow-label'});
    node.lines.forEach((line,i)=>{const span=create('tspan',{x:p.x,y:p.y+nodeHeight/2-(node.lines.length-1)*lineHeight/2+i*lineHeight+5});span.textContent=line;text.append(span);});group.append(text);svg.append(group);
  }
  scroll.append(svg);figure.append(scroll);const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre'),code=document.createElement('code');
  summary.textContent='查看流程代码';code.textContent=source;pre.append(code);details.append(summary,pre);figure.append(details);container.append(figure);return true;
}

function renderAnswerText(container,text,citations=[]){
  const ids=new Set(citations.map(s=>s.evidence_id).filter(id=>/^[EA]\d+$/.test(id||'')));
  const stripReferences=value=>value.replace(/\[([EA]\d+)\]/g,(raw,id)=>ids.has(id)?'':raw);
  // 只创建已知DOM节点，模型文本不交给innerHTML；代码和普通方括号保持原样。
  function inline(parent,value,depth=0){
    if(depth>6){parent.append(document.createTextNode(value));return;}
    const token=/(`+)([^\n]*?)\1|!\[([^\]]*)\]\(([^\s)]+)\)|\[([^\]]+)\]\(([^\s)]+)\)|\*\*([^\n]+?)\*\*|__([^\n]+?)__|\*([^*\n]+?)\*|\[([EA]\d+)\]/g;
    let last=0,m;
    while((m=token.exec(value))!==null){
      parent.append(document.createTextNode(value.slice(last,m.index)));
      if(m[1]){const code=document.createElement('code');code.textContent=m[2];parent.append(code);}
      else if(m[3]!==undefined){parent.append(document.createTextNode(m[3]||'资料配图'));}
      else if(m[5]!==undefined){const url=answerLink(m[6]);if(url){const a=document.createElement('a');a.href=url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=m[5];parent.append(a);}else{parent.append(document.createTextNode(m[5]));}}
      else if(m[7]||m[8]||m[9]){const tag=document.createElement(m[9]?'em':'strong');inline(tag,m[7]||m[8]||m[9],depth+1);parent.append(tag);}
      else if(!ids.has(m[10]))parent.append(document.createTextNode(m[0]));
      last=token.lastIndex;
    }
    parent.append(document.createTextNode(value.slice(last)));
  }
  const lines=String(text||'').replace(/\r\n/g,'\n').split('\n');
  const cells=line=>line.trim().replace(/^\||\|$/g,'').split(/(?<!\\)\|/).map(c=>c.trim().replace(/\\\|/g,'|').replace(/<br\s*\/?\s*>/gi,'\n'));
  const tableStart=i=>i+1<lines.length&&lines[i].trim().startsWith('|')&&/^\|?[\s:|\-]+\|?$/.test(lines[i+1])&&lines[i+1].includes('-');
  const special=i=>/^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s?|```|~~~)/.test(lines[i])||tableStart(i);
  for(let i=0;i<lines.length;){
    if(!lines[i].trim() || (!stripReferences(lines[i]).trim() && /\[[EA]\d+\]/.test(lines[i]))){i++;continue;}
    const fence=lines[i].match(/^\s*(`{3,}|~{3,})(\S*)/);
    if(fence){
      const pre=document.createElement('pre'),code=document.createElement('code'),body=[];i++;
      while(i<lines.length&&!lines[i].trim().startsWith(fence[1]))body.push(lines[i++]);
      if(i<lines.length)i++;
      if(fence[2].toLowerCase()==='mermaid'&&renderAnswerFlowchart(container,body.join('\n')))continue;
      code.textContent=body.join('\n');pre.append(code);container.append(pre);continue;
    }
    if(tableStart(i)){
      const rows=[cells(lines[i])];i+=2;
      while(i<lines.length&&lines[i].trim().startsWith('|'))rows.push(cells(lines[i++]));
      // 后端复核表格的纯编号列收进来源区，避免空白“来源”列。
      const hidden=new Set(rows[0].map((h,j)=>h==='来源'&&rows.length>1&&rows.slice(1).every(r=>/^\s*(?:\[[EA]\d+\]\s*)+$/.test(r[j]||'')&&!stripReferences(r[j]).trim())?j:-1).filter(j=>j>=0));
      const wrap=document.createElement('div');wrap.className='answer-table-scroll';
      const table=document.createElement('table'),head=document.createElement('thead'),body=document.createElement('tbody');
      rows.forEach((row,n)=>{const tr=document.createElement('tr');row.forEach((value,j)=>{if(hidden.has(j))return;const cell=document.createElement(n?'td':'th');if(!n)cell.scope='col';inline(cell,value);tr.append(cell);});(n?body:head).append(tr);});
      table.append(head,body);wrap.append(table);container.append(wrap);continue;
    }
    const heading=lines[i].match(/^\s*(#{1,6})\s+(.+)$/);
    if(heading){const h=document.createElement('h'+Math.min(4,heading[1].length+1));inline(h,heading[2]);container.append(h);i++;continue;}
    const list=lines[i].match(/^\s*([-*+]|\d+[.)])\s+(.+)$/);
    if(list){
      const ordered=/\d/.test(list[1]),ul=document.createElement(ordered?'ol':'ul');if(ordered)ul.start=parseInt(list[1]);
      while(i<lines.length){const row=lines[i].match(/^\s*([-*+]|\d+[.)])\s+(.+)$/);if(!row||/\d/.test(row[1])!==ordered)break;const li=document.createElement('li');inline(li,row[2]);ul.append(li);i++;while(i<lines.length&&/^\s{2,}\S/.test(lines[i])&&!special(i)){li.append(document.createElement('br'));inline(li,lines[i++].trim());}if(!lines[i]?.trim()&&/^\s*(?:[-*+]|\d+[.)])\s/.test(lines[i+1]||''))i++;}
      container.append(ul);continue;
    }
    if(/^\s*>/.test(lines[i])){const quote=document.createElement('blockquote');while(i<lines.length&&/^\s*>/.test(lines[i])){if(quote.childNodes.length)quote.append(document.createElement('br'));inline(quote,lines[i++].replace(/^\s*>\s?/,''));}container.append(quote);continue;}
    const p=document.createElement('p');inline(p,lines[i++]);
    while(i<lines.length&&lines[i].trim()&&!special(i)){p.append(document.createElement('br'));inline(p,lines[i++]);}
    container.append(p);
  }
}

function renderAnswerWithImages(containerEl,answerText,candidateImageUrls,citations=[],citationStatus={}){
  const parsed=parseAnswerAndImages(answerText);
  containerEl.replaceChildren();
  const text=document.createElement('div');text.className='answer-text';
  renderAnswerText(text,parsed.text||'（已完成，但未返回答案）',citations);containerEl.append(text);
  // 展示真实返回的来源配图，不从模型随意生成的URL下载图片。
  const images=dedupeKeepOrder((Array.isArray(candidateImageUrls)?candidateImageUrls:[]).filter(url=>typeof url==='string'&&isImageUrl(url)).map(answerLink).filter(Boolean));
  if(images.length){
    const gallery=document.createElement('div');gallery.className='answer-images';
    images.forEach((url,index)=>{
      const figure=document.createElement('figure'),link=document.createElement('a'),img=document.createElement('img');
      link.href=url;link.target='_blank';link.rel='noopener noreferrer';link.className='answer-image-link';link.setAttribute('aria-label',`打开资料配图 ${index+1}`);
      img.src=url;img.alt=`来源资料配图 ${index+1}`;img.loading='lazy';img.referrerPolicy='no-referrer';
      const caption=document.createElement('figcaption');caption.textContent=`资料配图 ${index+1} · 点击查看原图`;
      img.addEventListener('error',()=>{img.hidden=true;caption.textContent=`资料配图 ${index+1} 暂时无法加载，点击打开原图`;});
      link.append(img,caption);figure.append(link);gallery.append(figure);
    });containerEl.append(gallery);
  }
  renderCitations(containerEl,citations,citationStatus);
  const actions=document.createElement('div');actions.className='answer-actions';
  const copy=document.createElement('button');copy.type='button';copy.textContent='复制回答';
  copy.addEventListener('click',async()=>{
    try{if(!navigator.clipboard?.writeText)throw new Error();await navigator.clipboard.writeText(text.innerText||text.textContent);copy.textContent='已复制';}
    catch(_){copy.textContent='可选中正文复制';}
  });actions.append(copy);containerEl.append(actions);
}

function renderCitations(containerEl,citations,citationStatus){
  if(!Array.isArray(citations)||!citations.length)return;
  const group=document.createElement('details');group.className='answer-sources';
  const summary=document.createElement('summary');summary.textContent=`查看参考来源 · ${citations.length} 份`;group.append(summary);
  for(const source of citations){
    const card=document.createElement('details');card.className='answer-source-card';card.dataset.evidenceId=source.evidence_id||'';
    const title=document.createElement('summary');title.textContent=source.file_title||source.title||(source.type==='analysis_result'?'脑电分析结果':'来源资料');card.append(title);
    const meta=document.createElement('p');
    const page=source.page_start?`第 ${source.page_start}${source.page_end&&source.page_end!==source.page_start?'–'+source.page_end:''} 页`:'未提供页码';
    meta.textContent=source.type==='analysis_result'?`模型执行结果 · ${source.sample_id||'样本见记录'} · ${source.analysis_output?.classification_available?'训练模型预测，非人工真值':'表征结果不等于分类预测'}`:source.type==='image_observation'?'用户上传图片 · 视觉观察说明':source.type==='web'?`网页资料${source.retrieved_at?' · 获取于 '+source.retrieved_at:''}`:`${source.section||source.title||'原文片段'} · ${page}`;card.append(meta);
    const url=source.source_url&&answerLink(source.source_url);
    if(url){const link=document.createElement('a');link.href=url;link.target='_blank';link.rel='noopener noreferrer';link.textContent=source.type==='analysis_result'?'查看分析结果 ↗':source.type==='web'?'打开网页 ↗':source.type==='image_observation'?'查看原图 ↗':'阅读原件 ↗';card.append(link);}
    const excerpt=document.createElement('pre');excerpt.textContent=source.text||'';card.append(excerpt);group.append(card);
  }
  containerEl.append(group);
}
