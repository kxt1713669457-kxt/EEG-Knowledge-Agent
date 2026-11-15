/* 本机工作台增强。API密钥不写入localStorage，也不进入会话或任务消息。 */
(()=>{
  const $=id=>document.getElementById(id),chat=!!$('input');
  const endpoint=new URL(location.origin);endpoint.port=['18000','18001'].includes(endpoint.port)?'18001':'8001';
  const state={connections:[],presets:{},connectionId:'default',model:'',effort:'auto',workspaceId:null,fileIds:[],files:[],projects:[],sessions:[],view:'recent',ready:false,editing:'default',capsSeq:0,settingsSeq:0,credentialsDirty:false};
  try{const saved=JSON.parse(localStorage.getItem('eeg_workspace_preferences')||'{}');for(const key of ['connectionId','model','effort','workspaceId','fileIds'])if(saved[key]!==undefined)state[key]=saved[key];}catch(_){}
  state.fileIds=Array.isArray(state.fileIds)?state.fileIds.filter(id=>typeof id==='string'&&/^[0-9a-f]{32}$/.test(id)).slice(0,8):[];
  if(typeof state.model!=='string')state.model='';if(typeof state.effort!=='string')state.effort='auto';
  let toastTimer,quotaTimer,authTimer,editAction,authorizationJob=null;
  const isBusy=()=>chat&&typeof activeTurn!=='undefined'&&!!activeTurn;
  const isLoading=()=>chat&&typeof historyLoading!=='undefined'&&historyLoading;
  const prefs=()=>localStorage.setItem('eeg_workspace_preferences',JSON.stringify({connectionId:state.connectionId,model:state.model,effort:state.effort,workspaceId:state.workspaceId,fileIds:state.fileIds}));
  const el=(tag,className,text)=>{const n=document.createElement(tag);if(className)n.className=className;if(text!==undefined)n.textContent=text;return n;};
  function notify(text){clearTimeout(toastTimer);const n=$('workspaceToast');n.textContent=text;n.hidden=false;toastTimer=setTimeout(()=>n.hidden=true,6500);}
  async function api(path,options={}){
    const res=await fetch(endpoint.origin+'/workspace'+path,{...options,signal:options.signal||AbortSignal.timeout(40000)});
    const data=await res.json().catch(()=>({}));
    if(!res.ok){let message=data.detail;if(Array.isArray(message))message='输入格式不符合要求，请检查字段。';throw new Error(typeof message==='string'?message:message?.message||`请求未完成（${res.status}）`);}
    return data;
  }
  const json=(method,body)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  function openDialog(id){document.querySelectorAll('dialog[open]').forEach(d=>{if(d.id!==id)d.close();});const dialog=$(id);if(!dialog.open)dialog.showModal();}
  document.querySelectorAll('[data-close-dialog]').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
  document.querySelectorAll('#modelSettings,#sampleManager,#miPreprocessing,#folderWorkspace,#workspaceEdit').forEach(d=>d.addEventListener('click',e=>{if(e.target===d)d.close();}));
  function currentConnection(){return state.connections.find(c=>c.id===state.connectionId);}
  function options(select,rows,selected){select.replaceChildren();for(const row of rows){const option=el('option','',row.name||row.id);option.value=row.id;select.append(option);}select.value=selected;}

  function updateControls(){
    if(!chat)return;const frozen=isBusy()||isLoading();
    for(const id of ['executionConnection','executionModel','executionEffort'])if($(id))$(id).disabled=frozen||!state.ready;
    document.querySelectorAll('[data-new-chat],.session-open').forEach(b=>b.disabled=frozen);
    if(!state.ready&&!isBusy())$('send').disabled=true;
    $('contextFolderButton')?.setAttribute('title',isBusy()?'当前运行沿用已提交资料，可查看文件内容':'选择工作文件夹与本轮资料');
    $('useFolderFiles').disabled=isBusy();$('pickWorkingFolder').disabled=isBusy();
    $('contextFilesLabel').textContent=state.fileIds.length?`${state.fileIds.length} 份本轮资料`:'工作文件夹';
  }
  function fillConnectionPickers(){
    const rows=state.connections.map(c=>({id:c.id,name:c.name}));
    if(chat)options($('executionConnection'),rows,state.connectionId);
    options($('settingsConnection'),rows,state.editing||state.connectionId);
    if(chat){
      const c=currentConnection();$('executionModel').value=state.model;
      const list=$('executionModels');list.replaceChildren();for(const m of c?.models||[])list.append(el('option','',m.id));
      list.querySelectorAll('option').forEach(o=>o.value=o.textContent);
    }
    updateControls();
  }
  async function updateEfforts(){
    if(!chat)return;const seq=++state.capsSeq,c=currentConnection();if(!c)return;
    let caps=(c.models||[]).find(m=>m.id===state.model);
    try{if(!caps)caps=await api(`/connections/${c.id}/capabilities?model=${encodeURIComponent(state.model)}`);}catch(_){caps={efforts:['auto']};}
    if(seq!==state.capsSeq)return;
    const labels={auto:'自动',off:'不启用思考',low:'轻量推理',medium:'标准推理',high:'深入推理',max:'最大推理',xhigh:'更高推理',minimal:'最小推理',none:'不推理'};
    const efforts=Array.isArray(caps?.efforts)?caps.efforts:['auto'];if(!efforts.includes(state.effort))state.effort='auto';
    options($('executionEffort'),efforts.map(id=>({id,name:labels[id]||id})),state.effort);prefs();updateControls();
  }
  async function loadConnections(){
    const data=await api('/connections');state.connections=data.items||[];state.presets=data.presets||{};
    if(!state.connections.some(c=>c.id===state.connectionId)){state.connectionId='default';state.model='';}
    const c=currentConnection();if(!state.model)state.model=c?.default_model||c?.models?.[0]?.id||'';
    state.ready=!!c;fillConnectionPickers();await updateEfforts();prefs();if(chat)inputMode();
  }
  function showConnection(identity){
    state.credentialsDirty=false;state.settingsSeq++;authorizationJob=null;$('cancelQuotaAuthorization').hidden=true;
    clearTimeout(authTimer);$('authorizeQuota').disabled=false;
    state.editing=identity;const row=state.connections.find(c=>c.id===identity)||{};
    $('settingsConnection').value=identity||'';$('connectionName').value=identity?row.name:'新的模型连接';
    $('connectionProvider').value=row.provider||'bailian';$('connectionBase').value=row.base_url||state.presets.bailian?.base_url||'';
    $('connectionKey').value='';$('connectionKey').placeholder=identity&&identity!=='default'&&row.key_configured?'已保存，留空保留原密钥':'填写自己的API Key，默认密钥不会复制';
    $('connectionDefaultModel').value=row.default_model||'';$('reasoningFormat').value=row.reasoning_format||'auto';
    $('removeConnection').disabled=!identity||identity==='default';$('fetchModels').disabled=!identity;
    $('refreshQuota').disabled=false;
    $('saveConnection').textContent=identity==='default'?'另存为自己的连接':'保存连接';
    $('connectionNotice').textContent=identity==='default'?'默认配置来自项目；填写密钥后可另存，不会改写.env。':'';
    $('connectionBase').readOnly=$('connectionProvider').value!=='custom';
    fillSettingsModels(row.models||[]);renderQuotaPlaceholder(row);clearInterval(quotaTimer);
    if(identity){refreshQuota().catch(()=>{});quotaTimer=setInterval(()=>{if($('modelSettings').open)refreshQuota().catch(()=>{});},30000);}
  }
  function fillSettingsModels(rows){const list=$('settingsModels');list.replaceChildren();for(const row of rows){const option=el('option');option.value=row.id;list.append(option);}}
  function renderQuotaPlaceholder(row){$('liveQuotaTitle').textContent=row.provider==='bailian'?'模型免费额度':'供应商实时额度';$('liveQuotaResult').textContent=state.editing?'正在查询…':'保存连接后可查询。';$('authorizeQuota').hidden=row.provider!=='bailian'||!state.editing;$('quotaAuthorizationLink').hidden=true;$('quotaAuthStatus').textContent='';}
  async function refreshQuota(){
    const seq=state.settingsSeq,id=state.editing,model=$('connectionDefaultModel').value.trim()||state.connections.find(c=>c.id===id)?.default_model;
    if(!id||!model){$('liveQuotaResult').textContent='请先保存连接并填写模型ID。';return;}
    const button=$('refreshQuota');if(button.disabled)return;button.disabled=true;
    try{
      const data=await api(`/connections/${id}/quota?model=${encodeURIComponent(model)}`);if(id!==state.editing||seq!==state.settingsSeq)return;
      const target=$('liveQuotaResult');target.replaceChildren();
      if(data.status==='available'){
        const rows=data.rows||[];if(!rows.length)target.append(el('p','',data.kind==='free_tokens'?'未查到该模型的免费额度；不代表付费余额为零。':'供应商未返回余额数据。'));
        for(const row of rows){const card=el('div','quota-result-row');card.append(el('span','',row.model||row.currency||'余额'),el('strong','',row.remaining===null||row.remaining===undefined?'未提供':`${Number(row.remaining).toLocaleString('zh-CN')}${data.kind==='free_tokens'?' token':' '+(row.currency||'')}`));if(row.expires)card.append(el('small','',`到期 ${row.expires}`));target.append(card);}
        target.append(el('p','fine-print',data.message||''));if(data.checked_at)target.append(el('small','',`更新于 ${new Date(data.checked_at).toLocaleTimeString('zh-CN')}`));
      }else target.textContent=data.message||'额度暂不可读取。';
      $('authorizeQuota').hidden=state.connections.find(c=>c.id===id)?.provider!=='bailian';
    }catch(error){if(seq===state.settingsSeq)$('liveQuotaResult').textContent=error.message+' 未将失败显示为零额度。';}finally{button.disabled=state.credentialsDirty||!state.editing;}
  }
  async function openModelSettings(){try{await loadConnections();openDialog('modelSettings');showConnection(state.connectionId);}catch(error){notify(error.message);openDialog('modelSettings');}}
  document.querySelectorAll('[data-model-settings]').forEach(b=>b.addEventListener('click',openModelSettings));
  $('settingsConnection').addEventListener('change',()=>showConnection($('settingsConnection').value));
  $('useSavedConnection').addEventListener('click',()=>{if(!state.editing||state.credentialsDirty){notify('请先保存密钥或地址的修改。');return;}if(isBusy()){notify('当前任务沿用原配置，结束后可切换。');return;}const row=state.connections.find(c=>c.id===state.editing);if(!row)return;state.connectionId=row.id;state.model=$('connectionDefaultModel').value.trim()||row.default_model||row.models?.[0]?.id||'';state.effort='auto';prefs();fillConnectionPickers();updateEfforts();$('modelSettings').close();notify('已选择 '+row.name+'，下一次发送使用此连接。');});
  for(const id of ['connectionKey','connectionBase','connectionProvider'])for(const event of ['input','change'])$(id).addEventListener(event,()=>{state.credentialsDirty=true;state.settingsSeq++;$('fetchModels').disabled=true;$('refreshQuota').disabled=true;$('authorizeQuota').disabled=true;$('connectionNotice').textContent='请先保存密钥或地址的修改，再读取模型或额度。';});
  $('newConnection').addEventListener('click',()=>showConnection(null));
  $('connectionProvider').addEventListener('change',()=>{const provider=$('connectionProvider').value;$('connectionBase').value=state.presets[provider]?.base_url||'';$('connectionBase').readOnly=provider!=='custom';});
  $('connectionForm').addEventListener('submit',async event=>{
    event.preventDefault();const button=$('saveConnection');button.disabled=true;
    const payload={name:$('connectionName').value.trim(),provider:$('connectionProvider').value,base_url:$('connectionBase').value.trim(),default_model:$('connectionDefaultModel').value.trim(),reasoning_format:$('reasoningFormat').value};
    if($('connectionKey').value)payload.api_key=$('connectionKey').value;
    const identity=state.editing==='default'?null:state.editing;
    try{const row=await api(identity?`/connections/${identity}`:'/connections',json(identity?'PUT':'POST',payload));$('connectionKey').value='';if(!isBusy()){state.connectionId=row.id;state.model=row.default_model;state.effort='auto';prefs();}await loadConnections();showConnection(row.id);$('connectionNotice').textContent=isBusy()?'已保存，可在下一轮选择；当前任务沿用原配置。':'连接已保存，可读取模型或直接填写模型ID使用。';}
    catch(error){$('connectionNotice').textContent=error.message;}finally{button.disabled=false;}
  });
  $('fetchModels').addEventListener('click',async()=>{
    const seq=state.settingsSeq,id=state.editing;if(!id||state.credentialsDirty)return;const button=$('fetchModels');button.disabled=true;$('connectionNotice').textContent='正在向所选接口读取模型…';
    try{const data=await api(`/connections/${id}/models`,{method:'POST'});if(seq!==state.settingsSeq)return;fillSettingsModels(data.models||[]);$('connectionNotice').textContent=`已读取 ${data.models.length} 个模型，可在默认模型或输入框右下角选择。`;await loadConnections();$('settingsConnection').value=id;}
    catch(error){if(seq===state.settingsSeq)$('connectionNotice').textContent=error.message+' 也可按供应商文档手动填写模型ID。';}finally{button.disabled=state.credentialsDirty||!state.editing;}
  });
  $('removeConnection').addEventListener('click',async()=>{if(!state.editing||state.editing==='default'||!confirm('移除这个模型连接？历史回答和当前运行不受影响，密钥需要重新填写才能恢复连接。'))return;try{await api(`/connections/${state.editing}`,{method:'DELETE'});await loadConnections();showConnection('default');notify('连接已移除。');}catch(error){notify(error.message);}});
  $('refreshQuota').addEventListener('click',refreshQuota);
  $('authorizeQuota').addEventListener('click',async()=>{
    const id=state.editing;if(!id)return;$('authorizeQuota').disabled=true;$('quotaAuthStatus').textContent='正在准备官方登录…';
    try{const job=await api(`/connections/${id}/authorize`,{method:'POST'});authorizationJob=job.id;$('cancelQuotaAuthorization').hidden=false;clearTimeout(authTimer);
      async function poll(){try{const data=await api(`/authorizations/${job.id}`);if(state.editing!==id)return;$('quotaAuthStatus').textContent=data.message;
        if(data.authorization_url){const link=$('quotaAuthorizationLink');link.href=data.authorization_url;link.hidden=false;}
        if(data.status==='pending')authTimer=setTimeout(poll,1500);else{$('authorizeQuota').disabled=false;$('quotaAuthorizationLink').hidden=true;$('cancelQuotaAuthorization').hidden=true;if(data.status==='completed')await refreshQuota();}
      }catch(error){$('quotaAuthStatus').textContent=error.message;$('authorizeQuota').disabled=false;}}
      await poll();
    }catch(error){$('quotaAuthStatus').textContent=error.message;$('authorizeQuota').disabled=false;}
  });
  $('cancelQuotaAuthorization').addEventListener('click',async()=>{if(!authorizationJob)return;try{const job=await api('/authorizations/'+authorizationJob,{method:'DELETE'});clearTimeout(authTimer);$('quotaAuthStatus').textContent=job.message;$('quotaAuthorizationLink').hidden=true;$('cancelQuotaAuthorization').hidden=true;$('authorizeQuota').disabled=false;}catch(error){$('quotaAuthStatus').textContent=error.message;}});
  $('modelSettings').addEventListener('close',()=>{clearInterval(quotaTimer);$('connectionKey').value='';});

  // 会话和项目：服务器保存元数据；删除进入可恢复列表，不销毁原任务与产物。
  function openEdit(title,value,action,{move=false}={}){
    $('workspaceEditTitle').textContent=title;$('workspaceEditValue').value=value||'';$('workspaceEditValue').required=!move;
    $('workspaceEditLabel').hidden=move;$('workspaceMoveLabel').hidden=!move;$('workspaceEditNotice').textContent='';
    if(move)options($('workspaceMoveProject'),[{id:'',name:'不放入项目'},...state.projects.map(p=>({id:p.id,name:p.name}))],value||'');
    editAction=()=>action(move?$('workspaceMoveProject').value:$('workspaceEditValue').value.trim());openDialog('workspaceEdit');if(!move)$('workspaceEditValue').focus();
  }
  $('workspaceEditForm').addEventListener('submit',async event=>{event.preventDefault();$('workspaceEditSave').disabled=true;try{await editAction();$('workspaceEdit').close();await loadSessions();}catch(error){$('workspaceEditNotice').textContent=error.message;}finally{$('workspaceEditSave').disabled=false;}});
  async function modifySession(row,change){await api(`/sessions/${encodeURIComponent(row.id)}`,json('PATCH',change));await loadSessions();}
  function rowMenu(label,actions){
    const menu=el('details','row-menu'),summary=el('summary','icon-button','⋯');summary.setAttribute('aria-label','更多操作：'+label);menu.append(summary);
    const body=el('div','row-menu-body');for(const [name,action,danger]of actions){const button=el('button',danger?'danger-text':'',name);button.type='button';button.addEventListener('click',async()=>{menu.open=false;try{await action();}catch(error){notify(error.message);}});body.append(button);}menu.append(body);
    menu.addEventListener('toggle',()=>{if(!menu.open)return;document.querySelectorAll('.row-menu[open]').forEach(other=>{if(other!==menu)other.open=false;});const rect=summary.getBoundingClientRect();body.style.position='fixed';body.style.right='auto';body.style.left=Math.max(8,Math.min(innerWidth-body.offsetWidth-8,rect.right-body.offsetWidth))+'px';body.style.top=(rect.bottom+body.offsetHeight+8<innerHeight?rect.bottom+4:Math.max(8,rect.top-body.offsetHeight-4))+'px';});
    menu.addEventListener('keydown',event=>{if(event.key==='Escape'){menu.open=false;summary.focus();}});return menu;
  }
  function sessionRow(row){
    const wrap=el('div','session-row'+(chat&&row.id===sessionId?' selected':'')),button=el('button','session-open',row.title||'新的研究会话');button.title=row.title;button.disabled=isBusy()||isLoading();
    button.addEventListener('click',async()=>{if(isBusy()||isLoading()){notify('当前任务仍在执行或加载，可先在输入框补充。');return;}if(!chat){localStorage.setItem('kb_session_id',row.id);location.href=endpoint.origin+'/html';return;}if(sessionId!==row.id)window.clearSessionEEG?.();sessionId=row.id;localStorage.setItem('kb_session_id',sessionId);await loadHistory();document.body.classList.remove('nav-open');renderSessions();});
    const actions=[['更名',()=>openEdit('更名会话',row.title,value=>modifySession(row,{title:value}))],['移至项目',()=>openEdit('选择项目',row.project_id,value=>modifySession(row,{project_id:value||null}),{move:true})]];
    if(row.deleted_at)actions.push(['恢复会话',()=>modifySession(row,{deleted:false})]);else actions.push([row.archived?'取消归档':'归档',()=>modifySession(row,{archived:!row.archived})],['移至最近删除',async()=>{await modifySession(row,{deleted:true});notify('已移至最近删除，可在侧栏筛选中恢复；原始记录和分析产物保留。');},true]);
    wrap.append(button,rowMenu(row.title,actions));return wrap;
  }
  function renderSessions(){
    const container=$('recentSessions');container.replaceChildren();
    const search=($('sessionSearch')?.value||'').trim().toLowerCase();
    const matching=state.sessions.filter(r=>(r.title||'').toLowerCase().includes(search));
    const ungrouped=matching.filter(r=>!r.project_id||!state.projects.some(p=>p.id===r.project_id));
    for(const row of ungrouped)container.append(sessionRow(row));
    for(const project of state.projects){
      if(search&&!matching.some(r=>r.project_id===project.id)&&!project.name.toLowerCase().includes(search))continue;
      const group=el('details','project-folder');group.open=true;const header=el('summary','project-name');header.append(el('span','',project.name));group.append(header);
      const rename=el('button','project-rename','⋯');rename.title='更名项目';rename.setAttribute('aria-label','更名项目：'+project.name);rename.addEventListener('click',e=>{e.preventDefault();openEdit('更名项目',project.name,value=>api(`/projects/${project.id}`,json('PATCH',{name:value})));});header.append(rename);
      const rows=matching.filter(r=>r.project_id===project.id);for(const row of rows)group.append(sessionRow(row));if(!rows.length)group.append(el('p','project-empty','还没有研究会话'));container.append(group);
    }
    if(!state.sessions.length&&!state.projects.length)container.append(el('p','sidebar-empty',state.view==='recent'?'还没有研究会话':'这里还没有记录'));
    if(search&&!container.children.length)container.append(el('p','sidebar-empty','没有匹配的会话'));
    updateControls();
  }
  async function loadSessions(){try{const data=await api('/sessions?view='+state.view);state.sessions=data.items||[];state.projects=data.projects||[];renderSessions();}catch(error){notify('会话列表暂不可读取：'+error.message);}}
  $('sessionView')?.addEventListener('change',()=>{state.view=$('sessionView').value;loadSessions();});
  $('sessionSearch')?.addEventListener('input',renderSessions);
  document.addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='k'&&!document.querySelector('dialog[open]')){event.preventDefault();document.body.classList.add('nav-open');$('sessionSearch')?.focus();}});
  $('newProject')?.addEventListener('click',()=>openEdit('新建项目文件夹','',name=>api('/projects',json('POST',{name}))));
  document.addEventListener('click',e=>{document.querySelectorAll('.row-menu[open]').forEach(menu=>{if(!menu.contains(e.target))menu.open=false;});});
  $('recentSessions')?.addEventListener('scroll',()=>document.querySelectorAll('.row-menu[open]').forEach(menu=>menu.open=false));
  window.addEventListener('resize',()=>document.querySelectorAll('.row-menu[open]').forEach(menu=>menu.open=false));
  async function migrateSessions(){
    let rows=[];try{rows=JSON.parse(localStorage.getItem('eeg_recent_sessions')||'[]');}catch(_){}
    if(!Array.isArray(rows))rows=[];if(chat&&!rows.some(r=>r.id===sessionId))rows.unshift({id:sessionId,title:'新的研究会话'});
    for(const row of rows.slice(0,20))if(typeof row.id==='string'&&/^[\w-]{1,128}$/.test(row.id))await api('/sessions',json('POST',{id:row.id,title:String(row.title||'').slice(0,80)}));
    await loadSessions();
  }

  // 样本登记：类型、元信息与实际波形数值全部由后端核验。
  async function refreshSampleRegistry(){
    const data=await api('/samples');const list=$('sampleRegistryList');list.replaceChildren();
    for(const sample of data.samples||[]){const row=el('div','registry-row');row.setAttribute('role','listitem');const body=el('div');body.append(el('strong','',sample.file||sample.sample_id),el('p','',`${sample.sample_id} · ${(sample.shape||[]).join(' × ')||'文件未就绪'} · ${sample.sampling_rate_hz} Hz · ${sample.unit}`),el('small','',sample.compatibility_note||''));const status=el('span','status-tag'+(sample.compatible&&sample.available?' ready':''),sample.compatible&&sample.available?'可分析':'需适配');row.append(body,status);
      if(sample.user_registered){const remove=el('button','text-button','移出登记');remove.addEventListener('click',async()=>{try{await api('/samples/'+sample.sample_id,{method:'DELETE'});await refreshSampleRegistry();if(chat)await loadSampleCatalog();notify('已移出登记，已有分析产物仍可读取。');}catch(error){notify(error.message);}});row.append(remove);}else row.append(el('small','builtin-tag','内置示例'));list.append(row);
    }
  }
  async function openSampleManager(){openDialog('sampleManager');try{await refreshSampleRegistry();}catch(error){$('sampleRegisterNotice').textContent=error.message;}}
  document.querySelectorAll('[data-register-sample]').forEach(b=>b.addEventListener('click',openSampleManager));
  $('refreshSamples').addEventListener('click',()=>refreshSampleRegistry().catch(error=>notify(error.message)));
  $('sampleRegisterForm').addEventListener('submit',async event=>{
    event.preventDefault();const file=$('sampleFile').files[0];if(!file)return;if(!/\.(mat|npy|npz)$/i.test(file.name)){notify('样本只接受MAT、NPY、NPZ，论文请到知识库。');return;}
    const button=$('registerSample');button.disabled=true;$('sampleRegisterNotice').textContent='正在上传并核验数值与形状…';
    const form=new FormData();form.append('file',file);form.append('sampling_rate_hz',$('sampleRate').value);form.append('unit',$('sampleUnit').value);form.append('dataset_id',$('sampleDataset').value.trim()||'用户登记数据');
    try{const result=await api('/samples',{method:'POST',body:form,signal:AbortSignal.timeout(120000)});$('sampleRegisterNotice').textContent=result.compatible?'已登记，可在研究面板选择；尚未运行模型。':result.compatibility_note;await refreshSampleRegistry();if(chat)await loadSampleCatalog();$('sampleFile').value='';}
    catch(error){$('sampleRegisterNotice').textContent=error.message;}finally{button.disabled=false;}
  });

  // 运动想象原始记录：不在浏览器计算波形，不将研究候选冒充已验证的模型协议。
  const mi={catalog:null,result:null,busy:false,loading:false,classificationModels:[]};
  const miOriginalProfiles=new Set(['dwdanet-original-v1','dwtnet-original-v1']);
  const miIsShu=recording=>recording?.dataset_id==='SHU_MI'||recording?.input_kind==='shu_epoched_sessions';
  const miProfileAllowed=(profile,recording)=>!!profile&&(!miIsShu(recording)||miOriginalProfiles.has(profile.id))&&(!recording||!Array.isArray(profile.supported_datasets)||profile.supported_datasets.includes(recording.dataset_id));
  const miSummary=value=>{
    if(value===null||value===undefined)return '未提供';
    if(typeof value==='number')return Number.isFinite(value)?Number(value.toPrecision(5)).toLocaleString('zh-CN'):'无效数值';
    if(typeof value==='boolean')return value?'是':'否';
    if(Array.isArray(value))return !value.length?'0':value.length>12?`${value.length} 项（详见记录）`:value.map(v=>typeof v==='object'?JSON.stringify(v):String(v)).join('、');
    if(typeof value==='object')return JSON.stringify(value).slice(0,180);
    return String(value);
  };
  const miCompatibility=value=>({legacy_protocol:'与原模型预处理协议一致',requires_validation:'需单独验证分类效果',incompatible:'与当前分类模型不兼容'}[value]||'按当前模型目录核验输入条件');
  function miQualityValue(key,value){
    if(value===undefined||value===null)return '未提供';
    if(['rms_uv_per_channel','max_peak_to_peak_uv_per_channel'].includes(key)&&Array.isArray(value)){
      const numbers=value.filter(v=>typeof v==='number'&&Number.isFinite(v)).sort((a,b)=>a-b);if(!numbers.length)return '未提供';
      const middle=Math.floor(numbers.length/2),median=numbers.length%2?numbers[middle]:(numbers[middle-1]+numbers[middle])/2;
      return `最小 ${miSummary(numbers[0])} · 中位 ${miSummary(median)} · 最大 ${miSummary(numbers[numbers.length-1])} µV`;
    }
    if((key.endsWith('_indices')||['flat_channels','bad_channels'].includes(key))&&Array.isArray(value))return value.length?`${value.length}个（下标：${value.slice(0,8).join('、')}${value.length>8?'…':''}）`:'0';
    if(key==='mean_band_psd_sum'&&typeof value==='object')return [['mu','μ'],['beta','β'],['line50','50 Hz']].map(([band,label])=>`${label}：${typeof value[band]==='number'&&Number.isFinite(value[band])?value[band].toExponential(3):'未提供'}`).join('；');
    return miSummary(value);
  }
  function miUpdateControls(){
    const disabled=mi.busy||mi.loading;
    for(const id of ['miUploadRecording','miRecordingFile','miDataset','miMetadata','miShuSplit','miRecordingSelect','miProfileSelect','miMaxTrials','miRefreshCatalog','miArtifactSelect','miRefreshArtifacts'])$(id).disabled=disabled;
    const profile=mi.catalog?.profiles?.find(p=>p.id===$('miProfileSelect').value);
    const recording=mi.catalog?.recordings?.find(r=>r.recording_id===$('miRecordingSelect').value);
    const rateCompatible=profile?.sampling_policy==='native'||Number(recording?.sampling_rate_hz)===250;
    $('miRunPreprocessing').disabled=disabled||!recording||!profile?.ready||!rateCompatible||!miProfileAllowed(profile,recording);
    $('miReadArtifact').disabled=disabled||!$('miArtifactSelect').value;
  }
  function miSelectionInfo(){
    const recording=mi.catalog?.recordings?.find(r=>r.recording_id===$('miRecordingSelect').value);
    $('miRecordingInfo').textContent=recording?(miIsShu(recording)?`SHU-MI · 已分段会话 · ${miSummary(recording.channels)} 通道 · ${miSummary(recording.sampling_rate_hz)} Hz；按登记的训练/测试划分处理，不再次按事件切段。`:`${recording.dataset_id} · ${miSummary(recording.channels)} 通道 · ${miSummary(recording.sampling_rate_hz)} Hz · ${miSummary(recording.duration_seconds)} 秒 · ${miSummary(recording.event_count)} 个事件`):'';
    let profile=mi.catalog?.profiles?.find(p=>p.id===$('miProfileSelect').value),rateNote='';
    for(const option of $('miProfileSelect').options){const candidate=(mi.catalog?.profiles||[]).find(p=>p.id===option.value);option.disabled=!candidate?.ready||!miProfileAllowed(candidate,recording);}
    $('miTrialsLabel').textContent=miIsShu(recording)?'本次最多预览测试试次':'本次最多提取试次';
    if(miIsShu(recording)){
      if(!profile?.ready||!miProfileAllowed(profile,recording)){const original=(mi.catalog?.profiles||[]).find(p=>p.id==='dwdanet-original-v1'&&p.ready)||(mi.catalog?.profiles||[]).find(p=>p.id==='dwtnet-original-v1'&&p.ready);if(original){$('miProfileSelect').value=original.id;profile=original;}}
      rateNote='SHU-MI会话已经切段，按已登记原协议划分，不再次按提示事件切段；标签仅用于划分。试次数上限只限制测试集预览，完整训练/测试集另行导出。';
    }else if(recording&&Number(recording.sampling_rate_hz)!==250){
      if(profile?.sampling_policy!=='native'){
        const native=(mi.catalog?.profiles||[]).find(p=>p.ready&&p.sampling_policy==='native'&&p.method==='identity')||(mi.catalog?.profiles||[]).find(p=>p.ready&&p.sampling_policy==='native');
        if(native){$('miProfileSelect').value=native.id;profile=native;}else rateNote=`记录为${miSummary(recording.sampling_rate_hz)} Hz，当前250 Hz方案不兼容；不会自动重采样。`;
      }
      if(profile?.sampling_policy==='native')rateNote=`当前记录为${miSummary(recording.sampling_rate_hz)} Hz，已选择原生采样率方案；保留采样率，不重采样。`;
    }
    $('miProfileInfo').textContent=profile?[rateNote,profile.description,profile.reason,profile.model_compatibility?`模型兼容性：${miCompatibility(profile.model_compatibility)}`:''].filter(Boolean).join('；'):'';
    miUpdateControls();
  }
  function miRenderMethods(rows){
    const list=$('miMethodsList');list.replaceChildren();
    for(const row of rows||[]){const card=el('div','mi-method-card');card.append(el('strong','',row.name||row.id||'处理方法'));for(const key of ['description','requirements','limitations','status'])if(row[key])card.append(el('p','fine-print',miSummary(row[key])));const urls=[row.source,row.source_url,row.url,...(Array.isArray(row.sources)?row.sources:[])];for(const source of urls){const candidate=typeof source==='object'?source?.url:source;if(typeof candidate!=='string')continue;try{const url=new URL(candidate);if(url.protocol!=='https:')continue;const link=el('a','',`来源：${url.hostname} ↗`);link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';card.append(link);}catch(_){}}list.append(card);}
    if(!list.children.length)list.append(el('p','fine-print','方法适用条件会随处理方案保留在完整记录中。'));
  }
  async function miRefresh(preferredRecording,preferredArtifact){
    mi.loading=true;miUpdateControls();
    try{
      const data=await api('/mi-preprocessing/catalog');mi.catalog=data;
      const previousRecording=preferredRecording||$('miRecordingSelect').value,previousProfile=$('miProfileSelect').value;
      const recordings=(data.recordings||[]).map(r=>({id:r.recording_id,name:`${r.file||r.recording_id} · ${r.dataset_id}`}));
      options($('miRecordingSelect'),[{id:'',name:recordings.length?'请选择原始记录':'尚未登记原始记录'},...recordings],recordings.some(r=>r.id===previousRecording)?previousRecording:'');
      const previousArtifact=preferredArtifact||$('miArtifactSelect').value;
      const artifacts=(data.artifacts||[]).map(r=>({id:r.artifact_id,name:`${r.source_file||r.artifact_id} · ${r.profile?.name||r.profile?.id||'处理方案'} · ${r.created_at?new Date(r.created_at).toLocaleString('zh-CN'):r.artifact_id}`}));
      options($('miArtifactSelect'),[{id:'',name:artifacts.length?'请选择已有产物':'尚无已完成的产物'},...artifacts],artifacts.some(r=>r.id===previousArtifact)?previousArtifact:'');
      const select=$('miProfileSelect');select.replaceChildren();
      for(const profile of data.profiles||[]){const option=el('option','',`${profile.name||profile.id}${profile.ready?'':' · 未就绪'}`);option.value=profile.id;option.disabled=!profile.ready;select.append(option);}
      select.value=(data.profiles||[]).some(p=>p.id===previousProfile&&p.ready)?previousProfile:(data.profiles||[]).find(p=>p.ready)?.id||'';
      const limits=data.limits||{},mb=limits.max_upload_bytes?Math.floor(limits.max_upload_bytes/1048576):null;
      $('miResourceNote').textContent=`一次处理一个记录，默认最多32个试次${mb?`；文件上限 ${mb} MB`:''}。处理参数、质量变化和原始事件对应关系随结果保存。`;
      if(Number.isInteger(limits.max_trials))$('miMaxTrials').max=String(Math.min(288,limits.max_trials));
      miRenderMethods(data.methods);miSelectionInfo();
    }finally{mi.loading=false;miUpdateControls();}
  }
  async function openMiPreprocessing(){openDialog('miPreprocessing');try{await miRefresh();}catch(error){$('miRunNotice').textContent=error.message;}}
  document.querySelectorAll('[data-mi-preprocessing]').forEach(button=>button.addEventListener('click',openMiPreprocessing));
  // 将入口放在聊天侧栏已有的数值样本入口旁；导入页仍可从样本登记进入。
  const sampleEntry=document.querySelector('.context-panel [data-register-sample]');
  if(sampleEntry){const button=el('button','subtle-button','运动想象预处理');button.type='button';button.dataset.miPreprocessing='';button.addEventListener('click',openMiPreprocessing);sampleEntry.parentElement.append(button);}
  $('miRefreshCatalog').addEventListener('click',()=>miRefresh().catch(error=>{$('miRunNotice').textContent=error.message;}));
  $('miRefreshArtifacts').addEventListener('click',()=>miRefresh().catch(error=>{$('miArtifactNotice').textContent=error.message;}));
  $('miArtifactSelect').addEventListener('change',miUpdateControls);
  $('miReadArtifact').addEventListener('click',async()=>{const id=$('miArtifactSelect').value;if(mi.busy||mi.loading||!/^[0-9a-f]{32}$/.test(id||''))return;mi.loading=true;miUpdateControls();$('miArtifactNotice').textContent='正在读取已有产物…';try{const result=await api('/mi-preprocessing/artifacts/'+id);await miRenderResult(result);$('miArtifactNotice').textContent='已读取保存结果；未重新执行预处理或分类。';}catch(error){$('miArtifactNotice').textContent=error.message;}finally{mi.loading=false;miUpdateControls();}});
  $('miRecordingSelect').addEventListener('change',miSelectionInfo);$('miProfileSelect').addEventListener('change',miSelectionInfo);
  $('miDataset').addEventListener('change',()=>{const shu=$('miDataset').value==='SHU_MI';$('miRecordingFile').multiple=shu;$('miRecordingFile').accept=shu?'.mat':'.gdf,.edf,.bdf,.fif,.npz';$('miShuSplitField').hidden=!shu;$('miMetadataField').hidden=shu;$('miRecordingFileHint').textContent=shu?'同时选择同一被试的5份session MAT；数据已分段，250 Hz、µV。服务端会核验会话与划分。':'GDF / EDF / BDF / FIF，或包含连续信号与事件的 NPZ；已切好的 MAT / NPY 请使用“样本登记”。';$('miUploadNotice').textContent='';miSelectionInfo();});
  $('miRecordingForm').addEventListener('submit',async event=>{
    event.preventDefault();if(mi.busy||mi.loading)return;
    const files=Array.from($('miRecordingFile').files||[]),file=files[0],shu=$('miDataset').value==='SHU_MI';if(!file)return;
    if(shu?(files.length!==5||files.some(f=>!/\.mat$/i.test(f.name))):(files.length!==1||!/\.(gdf|edf|bdf|fif|npz)$/i.test(file.name))){$('miUploadNotice').textContent=shu?'请同时选择同一被试的5份session MAT。':'请选择一份GDF、EDF、BDF、FIF或连续NPZ；已切好的试次请使用样本登记。';return;}
    const limit=mi.catalog?.limits?.max_upload_bytes;if(limit&&files.some(f=>f.size>limit)){$('miUploadNotice').textContent=`单份文件超过 ${Math.floor(limit/1048576)} MB 限制。`;return;}
    const form=new FormData();
    if(shu){for(const session of files)form.append('files',session);form.append('split_id',$('miShuSplit').value);}
    else{const metadata=$('miMetadata').value.trim()||'{}';try{const value=JSON.parse(metadata);if(!value||typeof value!=='object'||Array.isArray(value))throw Error();}catch(_){$('miUploadNotice').textContent='补充元信息必须是JSON对象。';return;}form.append('file',file);form.append('dataset_id',$('miDataset').value);form.append('metadata_json',metadata);}
    mi.busy=true;miUpdateControls();$('miUploadNotice').textContent='正在上传并核验通道、事件和资源占用…';
    try{const result=await api(shu?'/mi-preprocessing/shu-sessions':'/mi-preprocessing/recordings',{method:'POST',body:form,signal:AbortSignal.timeout(180000)});$('miUploadNotice').textContent=`已登记 ${result.file||(shu?'5份SHU会话':file.name)}；尚未执行预处理。`;$('miRecordingFile').value='';await miRefresh(result.recording_id);}
    catch(error){$('miUploadNotice').textContent=error.message;}finally{mi.busy=false;miUpdateControls();}
  });
  async function miRenderResult(result){
    mi.result=result;mi.classificationModels=[];$('miResult').hidden=false;
    $('miSplitDownloads').hidden=result.dataset_id!=='SHU_MI'||!Array.isArray(result.split_exports);
    $('miDownloadTrain').disabled=!(result.split_exports||[]).some(row=>row.name==='train.mat');$('miDownloadTest').disabled=!(result.split_exports||[]).some(row=>row.name==='test.mat');
    $('miResultSummary').textContent=`${result.profile?.name||result.profile?.id||'处理方案'} · 输出 ${(result.output?.shape||[]).join(' × ')} · ${miSummary(result.output?.sampling_rate_hz)} Hz · ${result.output?.unit||'V'}；样本ID：${result.sample_id||'未登记'}。`;
    const qc=result.qc||{},before=qc.before||{},after=qc.after||{},target=$('miQualityComparison');target.replaceChildren();
    const table=el('table'),head=el('tr');for(const label of ['质量指标','处理前','处理后'])head.append(el('th','',label));const thead=el('thead');thead.append(head);table.append(thead);const body=el('tbody');
    const labels={finite:'数值有限',finite_fraction:'有限数值比例',rms_uv:'RMS · µV',rms_uV:'RMS · µV',median_rms_uv:'RMS中位数 · µV',rms_uv_per_channel:'各通道 RMS',max_peak_to_peak_uv_per_channel:'各通道最大峰峰值',peak_to_peak_uv:'峰峰值 · µV',flat_channels:'平直通道',bad_channels:'异常通道',flat_channel_indices:'平直通道下标',over_200uv_trial_indices:'峰峰值超过200 µV的试次下标',mean_band_psd_sum:'μ / β / 50 Hz 频带PSD频点和',line_noise_ratio:'工频噪声比例',n_trials:'试次数',n_channels:'通道数',duration_seconds:'时长 · 秒',note:'指标说明'};
    for(const key of [...new Set([...Object.keys(before),...Object.keys(after)])]){const row=el('tr'),label=el('th','',labels[key]||'其他质量记录'),beforeCell=el('td','',miQualityValue(key,before[key])),afterCell=el('td','',miQualityValue(key,after[key]));label.scope='row';beforeCell.dataset.label='处理前';afterCell.dataset.label='处理后';row.append(label,beforeCell,afterCell);body.append(row);}
    if(!body.children.length){const row=el('tr');row.append(el('td','','详见完整质量记录'));body.append(row);}table.append(body);target.append(table);
    const warnings=$('miWarnings');warnings.replaceChildren();const dropped=Array.isArray(qc.dropped_trials)?qc.dropped_trials.length:Number(qc.dropped_trials)||0;
    warnings.append(el('p','fine-print',`未保留窗口：${dropped}个。质量指标变化不等同于识别率提升。`));
    const distinctWarnings=[...new Set((Array.isArray(qc.warnings)?qc.warnings:qc.warnings?[qc.warnings]:[]).map(w=>miSummary(w).trim()).filter(Boolean))];
    for(const warning of distinctWarnings)warnings.append(el('p','fine-print',warning));
    let scope=String(result.scope||'').trim();for(const warning of distinctWarnings)scope=scope.replaceAll(warning,'');scope=scope.trim();
    $('miResultScope').textContent=scope;$('miResultScope').hidden=!scope;$('miReportDetails').textContent=JSON.stringify(result,null,2);
    const button=$('miUseSample'),compatibility=document.querySelector('#miResult > .fine-print');button.disabled=true;button.textContent='正在核验分类模型…';
    compatibility.textContent='仅完成预处理，尚未执行分类。正在核验派生样本与当前模型目录的输入条件。';
    try{
      const response=await fetch(endpoint.origin+'/agent/eeg/catalog',{signal:AbortSignal.timeout(15000)});if(!response.ok)throw Error();const catalog=await response.json();if(mi.result!==result)return;
      const sample=(catalog.samples||[]).find(s=>s.sample_id===result.sample_id),compatibleIds=new Set(sample?.compatible_model_ids||[]);
      mi.classificationModels=sample?.available?(catalog.models||[]).filter(m=>compatibleIds.has(m.id||m.model_id)&&m.classification_available===true&&m.ready===true&&m.enabled===true):[];
      if(mi.classificationModels.length){compatibility.textContent=`已登记为派生样本；当前兼容分类模型：${mi.classificationModels.map(m=>m.name||m.id||m.model_id).join('、')}。仅表示输入条件匹配，处理方案效果仍需验证；尚未执行分类。`;button.textContent='在样本面板中选择';button.disabled=!chat;}
      else{compatibility.textContent='仅完成预处理，尚未执行分类。当前目录中没有与此派生样本兼容且就绪的分类模型；可下载结果，后续登记匹配模型。';button.textContent='暂无兼容分类模型';}
    }catch(_){if(mi.result!==result)return;compatibility.textContent='仅完成预处理，尚未执行分类。暂时无法核验模型目录，请稍后重新读取本产物。';button.textContent='分类模型目录暂不可核验';}
  }
  $('miPreprocessForm').addEventListener('submit',async event=>{
    event.preventDefault();if(mi.busy||mi.loading)return;
    const recording=$('miRecordingSelect').value,profile=$('miProfileSelect').value,max=Number($('miMaxTrials').value);
    if(!recording||!profile||!Number.isInteger(max)||max<1||max>Number($('miMaxTrials').max)){$('miRunNotice').textContent='请选择原始记录、可用方案，并填写范围内的整数试次数。';return;}
    mi.busy=true;miUpdateControls();$('miRunNotice').textContent='正在执行并保存质量记录，请勿重复提交…';
    try{const result=await api('/mi-preprocessing/run',{...json('POST',{recording_id:recording,profile_id:profile,max_trials:max}),signal:AbortSignal.timeout(300000)});await miRenderResult(result);$('miRunNotice').textContent='预处理完成；尚未执行分类模型。';await miRefresh(recording,result.artifact_id);await refreshSampleRegistry();if(chat)await loadSampleCatalog();}
    catch(error){$('miRunNotice').textContent=error.name==='TimeoutError'||error.name==='AbortError'?'请求中断，后端处理可能仍在继续。请刷新目录核对结果，避免立即重复提交。':error.message;}finally{mi.busy=false;miUpdateControls();}
  });
  async function miDownload(kind){
    if(!mi.result)return;const id=mi.result.artifact_id;if(!/^[0-9a-f]{32}$/.test(id||'')){notify('产物ID不符合记录格式。');return;}
    const split=kind==='train'||kind==='test';if(split&&(mi.result.dataset_id!=='SHU_MI'||!(mi.result.split_exports||[]).some(row=>row.name===`${kind}.mat`)))return;
    const button=$({data:'miDownloadData',report:'miDownloadReport',train:'miDownloadTrain',test:'miDownloadTest'}[kind]);button.disabled=true;
    try{const response=await fetch(`${endpoint.origin}/workspace/mi-preprocessing/artifacts/${id}${split?'/split/'+kind:kind==='data'?'/data':''}`,{signal:AbortSignal.timeout(60000)});if(!response.ok){const error=await response.json().catch(()=>({}));throw Error(typeof error.detail==='string'?error.detail:'产物暂不可下载。');}const blob=kind==='report'?new Blob([JSON.stringify(await response.json(),null,2)],{type:'application/json'}):await response.blob();const url=URL.createObjectURL(blob),link=el('a');link.href=url;link.download=split?`mi-${id}-${kind}.mat`:`mi-${id}.${kind==='data'?'npz':'json'}`;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
    catch(error){notify(error.message);}finally{button.disabled=false;}
  }
  $('miDownloadData').addEventListener('click',()=>miDownload('data'));$('miDownloadReport').addEventListener('click',()=>miDownload('report'));
  $('miDownloadTrain').addEventListener('click',()=>miDownload('train'));$('miDownloadTest').addEventListener('click',()=>miDownload('test'));
  $('miUseSample').addEventListener('click',async()=>{if(!mi.result?.sample_id||!chat||!mi.classificationModels.length||$('miUseSample').disabled)return;try{await loadSampleCatalog();const select=$('sampleSelect');if(!Array.from(select.options).some(option=>option.value===mi.result.sample_id&&!option.disabled)){notify('派生样本暂未出现在可用目录，请刷新样本面板。');return;}select.value=mi.result.sample_id;select.dispatchEvent(new Event('change',{bubbles:true}));$('trialIndex').value='0';$('miPreprocessing').close();select.focus();notify('已选择本次派生样本；选择识别模型后发送分析任务。');}catch(error){notify(error.message);}});

  // 文件夹：明确选择、预览和引用；不把文件名当作已经阅读的证据。
  async function loadFolders(){const data=await api('/folders');options($('workingFolderSelect'),[{id:'',name:'选择工作文件夹'},...(data.items||[]).map(r=>({id:r.id,name:`${r.name} · ${r.file_count}份`}))],state.workspaceId||'');}
  function renderFileTree(){
    const tree=$('workingFileTree');tree.replaceChildren();$('folderSelectedCount').textContent=`已选择 ${state.fileIds.length} / ${state.files.length} 份`;
    for(const file of state.files){const row=el('div','working-file');row.setAttribute('role','listitem');const check=el('input');check.type='checkbox';check.checked=state.fileIds.includes(file.id);check.disabled=isBusy();check.setAttribute('aria-label','用于本轮：'+file.path);check.addEventListener('change',()=>{if(check.checked&&state.fileIds.length>=8){check.checked=false;notify('每轮最多8份文件，较多资料请分组或入知识库。');return;}state.fileIds=check.checked?[...state.fileIds,file.id]:state.fileIds.filter(id=>id!==file.id);prefs();renderFileTree();updateControls();});const button=el('button','file-name-button',file.path);button.title=file.path;button.addEventListener('click',()=>previewFile(file.id));row.append(check,button,el('small','',Math.max(1,Math.round(file.size/1024))+' KB'));tree.append(row);}
    if(!state.files.length)tree.append(el('div','empty-state','选择文件夹后，文件将显示在这里。'));
  }
  async function loadFolder(identity){
    if(!identity){state.workspaceId=null;state.files=[];state.fileIds=[];prefs();renderFileTree();updateControls();return;}
    const data=await api(`/folders/${identity}/files`);state.workspaceId=identity;state.files=data.files||[];state.fileIds=state.fileIds.filter(id=>state.files.some(f=>f.id===id));prefs();renderFileTree();updateControls();
  }
  async function previewFile(identity){
    $('previewFileText').textContent='正在读取…';try{const file=await api('/files/'+identity);$('previewFileTitle').textContent=file.path;$('previewFileText').textContent=file.text;$('previewFileOriginal').href=endpoint.origin+'/workspace/files/'+identity+'/download';$('previewFileOriginal').hidden=false;}catch(error){$('previewFileText').textContent=error.message;$('previewFileOriginal').hidden=true;}
  }
  async function openFolders(){openDialog('folderWorkspace');try{await loadFolders();if(state.workspaceId)await loadFolder(state.workspaceId);}catch(error){$('folderStatus').textContent=error.message;}}
  document.querySelectorAll('[data-working-folder]').forEach(b=>b.addEventListener('click',openFolders));
  $('workingFolderSelect').addEventListener('change',()=>{if(isBusy()){notify('本轮运行中，请结束后切换资料。');return;}state.fileIds=[];loadFolder($('workingFolderSelect').value).catch(error=>notify(error.message));});
  $('pickWorkingFolder').addEventListener('click',()=>{if(!isBusy())$('workingFolderInput').click();});
  $('workingFolderInput').addEventListener('change',async()=>{
    const all=Array.from($('workingFolderInput').files||[]),allowed=new Set(['md','txt','pdf','docx','csv','json','yaml','yml','html','htm','py','r','m','toml']);
    const selected=all.filter(f=>{const path=f.webkitRelativePath||f.name;return !path.split('/').some(p=>p.startsWith('.')||['node_modules','venv','__pycache__','models','dist','build'].includes(p))&&allowed.has(f.name.split('.').pop().toLowerCase())&&!/(?:^|[-_.])(?:secret|credentials|private-key)(?:[-_.]|$)/i.test(f.name);});
    if(!selected.length){$('folderStatus').textContent='没有可读文档。波形样本请到“样本登记”，隐藏文件和凭据文件会跳过。';return;}
    if(selected.length>200||selected.reduce((n,f)=>n+f.size,0)>24*1024*1024||selected.some(f=>f.size>8*1024*1024)){$('folderStatus').textContent='请选择较小目录：最多200份，单份8MB、整批24MB。';return;}
    const form=new FormData();for(const file of selected){form.append('uploads',file);form.append('paths',file.webkitRelativePath||file.name);}
    $('pickWorkingFolder').disabled=true;$('folderStatus').textContent=`正在读取 ${selected.length} 份资料…`;
    try{const data=await api('/folders',{method:'POST',body:form,signal:AbortSignal.timeout(120000)});state.workspaceId=data.workspace.id;state.files=data.files;state.fileIds=data.files.slice(0,8).map(f=>f.id);prefs();await loadFolders();renderFileTree();updateControls();$('folderStatus').textContent=`已读取 ${data.files.length} 份，默认选中 ${state.fileIds.length} 份。${all.length-selected.length+(data.skipped||[]).length} 份被跳过。`;
      if(data.skipped?.length){const details=el('details','skipped-files');details.append(el('summary','',`查看 ${data.skipped.length} 份未读取原因`));for(const row of data.skipped)details.append(el('p','',row.path+'：'+row.reason));$('folderStatus').append(details);}
    }catch(error){$('folderStatus').textContent=error.message;}finally{$('pickWorkingFolder').disabled=isBusy();$('workingFolderInput').value='';}
  });
  $('selectFolderFiles').addEventListener('click',()=>{if(isBusy())return;state.fileIds=state.files.slice(0,8).map(f=>f.id);prefs();renderFileTree();updateControls();});
  $('clearFolderFiles').addEventListener('click',()=>{if(isBusy())return;state.fileIds=[];prefs();renderFileTree();updateControls();});
  $('useFolderFiles').addEventListener('click',()=>{prefs();$('folderWorkspace').close();updateControls();if(chat&&state.fileIds.length&&!isBusy()){queryMode.value='agent';inputMode();syncWorkspaceMode();inputEl.focus();}notify(state.fileIds.length?`本轮已选择 ${state.fileIds.length} 份文件；发送后作为资料依据。`:'本轮未附加工作区文件。');});

  if(chat){
    const originalInputMode=inputMode;inputMode=function(){originalInputMode();updateControls();};
    new MutationObserver(updateControls).observe(queryMode,{attributes:true,attributeFilter:['disabled']});
    $('executionConnection').addEventListener('change',()=>{state.connectionId=$('executionConnection').value;const c=currentConnection();state.model=c?.default_model||c?.models?.[0]?.id||'';state.effort='auto';prefs();fillConnectionPickers();updateEfforts();});
    $('executionModel').addEventListener('change',()=>{state.model=$('executionModel').value.trim();state.effort='auto';prefs();updateEfforts();});
    $('executionEffort').addEventListener('change',()=>{state.effort=$('executionEffort').value;prefs();});
    window.getExecutionOptions=()=>({connection_id:state.connectionId,model:$('executionModel').value.trim()||state.model,effort:state.effort,workspace_id:state.workspaceId,file_ids:state.fileIds});
    window.restoreExecutionOptions=options=>{if(!options||isBusy())return;if(state.connections.length&&!state.connections.some(c=>c.id===options.connection_id))return;state.connectionId=options.connection_id;state.model=options.model;state.effort=options.effort||'auto';state.workspaceId=options.workspace_id;state.fileIds=options.file_ids||[];prefs();fillConnectionPickers();updateEfforts();};
    rememberSession=function(title){api('/sessions',json('POST',{id:sessionId,title:String(title||'').slice(0,80)})).then(loadSessions).catch(()=>{});};
    showRecentSessions=()=>loadSessions();
  }
  Promise.allSettled([loadConnections(),migrateSessions(),loadFolders()]).then(results=>{if(results[0].status==='rejected'){state.ready=false;notify('模型配置读取失败，请打开模型设置重试。');}updateControls();});
})();
