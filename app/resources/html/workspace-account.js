// 两个页面使用同一查询服务保存资料；不新建或修改会话、任务。
(()=>{
  const dialog=document.getElementById('workspaceAccount');if(!dialog)return;
  const byId=id=>document.getElementById(id),form=byId('profileForm'),save=byId('saveProfile'),refresh=byId('refreshAccount');
  const origin=new URL(location.origin);origin.port=['18000','18001'].includes(origin.port)?'18001':'8001';
  let busy=false,loaded=false,dirty=false;
  function showProfile(profile){
    for(const [id,key]of [['profileName','name'],['profileRole','role'],['profileFocus','focus']])byId(id).value=profile[key]||'';
    document.querySelectorAll('[data-profile-name]').forEach(el=>el.textContent=profile.name||'个人研究空间');
    document.querySelectorAll('[data-profile-focus]').forEach(el=>el.textContent=profile.focus||'个人研究空间');
    document.querySelectorAll('[data-profile-mark]').forEach(el=>el.textContent=Array.from(profile.name||'N')[0]);
  }
  function clearUsage(){['usageInput','usageOutput','usageTotal'].forEach(id=>byId(id).textContent='—');}
  function displayAccount(data){
    if(!dirty){showProfile(data.profile||{});loaded=true;}
    byId('accountStatus').textContent=data.service?.configured?`${data.service.provider} · 已配置模型密钥`:'模型密钥或服务地址尚未配置';
    byId('accountModel').textContent=data.service?.model||'未设置';byId('accountVision').textContent=data.service?.vision_model||'未设置';
    byId('accountQuota').textContent='需在控制台查看';byId('quotaNotice').textContent=data.quota?.message||'暂时无法读取余额。';
    const link=byId('quotaConsole');link.hidden=true;
    if(data.quota?.console_url==='https://bailian.console.aliyun.com/'){link.href=data.quota.console_url;link.hidden=false;}
    clearUsage();const usage=data.usage||{};
    if(usage.status==='available'){
      for(const [id,key]of [['usageInput','input_tokens'],['usageOutput','output_tokens'],['usageTotal','total_tokens']])byId(id).textContent=Number.isSafeInteger(usage[key])&&usage[key]>=0?usage[key].toLocaleString('zh-CN'):'—';
      byId('usageNotice').textContent=`仅统计已保存回答的生成与复核，不含其他模型调用；实际总用量与余额以供应商账单为准。已读取 ${usage.answer_count||0} 条回答、${usage.calls_with_usage||0} 份调用用量${usage.truncated?'，已达到展示上限':''}。`;
    }else byId('usageNotice').textContent=usage.message||'用量记录暂不可读取，不能据此判断为零用量。';
    const updated=new Date(data.updated_at);byId('accountUpdated').textContent=Number.isNaN(updated.getTime())?'':`更新于 ${updated.toLocaleString('zh-CN')}`;
  }
  async function loadAccount(){
    if(busy)return;busy=true;refresh.disabled=true;byId('accountStatus').textContent='正在读取研究空间…';
    try{const res=await fetch(`${origin.origin}/workspace/account`,{signal:AbortSignal.timeout(7000)});if(!res.ok)throw new Error();displayAccount(await res.json());}
    catch(_){byId('accountStatus').textContent='查询服务未连接，检查服务启动后可点击刷新。';byId('accountQuota').textContent='暂不可获取';byId('quotaNotice').textContent='连接失败不能判定额度已用完。';clearUsage();byId('usageNotice').textContent='用量暂不可读取，请稍后刷新。';byId('quotaConsole').hidden=true;byId('accountModel').textContent='—';byId('accountVision').textContent='—';byId('accountUpdated').textContent='';}
    finally{busy=false;refresh.disabled=false;save.disabled=!loaded;form.querySelectorAll('input').forEach(input=>input.disabled=!loaded);}
  }
  document.querySelectorAll('[data-account]').forEach(button=>button.addEventListener('click',()=>{dialog.showModal();loadAccount();}));
  dialog.querySelector('.close-dialog').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});refresh.addEventListener('click',loadAccount);
  form.addEventListener('input',()=>{dirty=true;byId('profileNotice').textContent='有未保存修改';});
  form.addEventListener('submit',async event=>{
    event.preventDefault();if(!loaded)return;save.disabled=true;const notice=byId('profileNotice');notice.textContent='正在保存…';
    try{const res=await fetch(`${origin.origin}/workspace/profile`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:byId('profileName').value.trim(),role:byId('profileRole').value.trim(),focus:byId('profileFocus').value.trim()}),signal:AbortSignal.timeout(5000)});if(!res.ok)throw new Error();const data=await res.json();showProfile(data.profile);dirty=false;notice.textContent='已保存，下次打开仍可使用。';}
    catch(_){notice.textContent='保存失败，修改仍保留，请稍后重试。';}finally{save.disabled=false;}
  });
})();
