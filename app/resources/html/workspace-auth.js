// 本地两个 API 共用 Cookie；凭证不写入 localStorage，不混入会话、任务或模型参数。
(()=>{'use strict';
  const trusted=new Set(['8000','8001','18000','18001']),nativeFetch=window.fetch.bind(window);let redirecting=false;
  function localRequest(input){try{const url=new URL(typeof input==='string'?input:input.url,location.href);return url.hostname===location.hostname&&trusted.has(url.port)&&url.protocol===location.protocol;}catch(_){return false;}}
  function login(){if(redirecting)return;redirecting=true;location.replace('/login?next='+encodeURIComponent(location.pathname));}
  window.fetch=async(input,options={})=>{const local=localRequest(input),url=new URL(typeof input==='string'?input:input.url,location.href);if(local&&!url.pathname.startsWith('/auth/'))await window.eegAuthReady;const response=await nativeFetch(input,local?{...options,credentials:'include'}:options);if(local&&response.status===401&&!['/auth/login','/auth/password'].includes(url.pathname))login();return response;};
  const $=id=>document.getElementById(id);
  async function request(path,body){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{}),signal:AbortSignal.timeout(12000)});const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'输入格式不符合要求，请检查密码。');return data;}
  window.eegAuthReady=fetch('/auth/me').then(async response=>{if(!response.ok)return;const data=await response.json();window.eegMediaBase=data.media?.image_base||'';if($('signedInUser'))$('signedInUser').textContent=data.user?.username||'已登录';}).catch(()=>{if($('signedInUser'))$('signedInUser').textContent='账号状态暂不可读取';});
  document.addEventListener('DOMContentLoaded',()=>{
    $('logoutAccount')?.addEventListener('click',async()=>{const button=$('logoutAccount');button.disabled=true;try{await request('/auth/logout');login();}catch(exc){$('securityNotice').textContent=exc.message;button.disabled=false;}});
    $('passwordForm')?.addEventListener('submit',async e=>{e.preventDefault();const button=$('savePassword'),notice=$('securityNotice');if($('newPassword').value!==$('confirmNewPassword').value){notice.textContent='两次输入的新密码不一致。';return;}button.disabled=true;notice.textContent='';try{await request('/auth/password',{current_password:$('currentPassword').value,new_password:$('newPassword').value});e.target.reset();login();}catch(exc){notice.textContent=exc.message;}finally{button.disabled=false;}});
  });
  window.addEventListener('pageshow',event=>{if(event.persisted)location.reload();});
})();
