const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('app/resources/html/research-extensions.js','utf8');
const block=source.slice(source.indexOf('  let eegFiles=[]'),source.indexOf('  window.renderUserImageAttachments='));
class Element{constructor(){this.children=[];this.value='';this.disabled=false;} append(...v){this.children.push(...v)} insertBefore(){} prepend(...v){this.children.unshift(...v)} replaceChildren(){this.children=[]} setAttribute(){} addEventListener(){} }
const nodes={input:new Element(),send:new Element()};const composer=new Element();let requests=[];
const context={window:{getExecutionOptions:()=>({model:'test'})},el:()=>new Element(),composer,document:{querySelector:()=>composer},$:id=>nodes[id],busy:()=>false,notice:()=>{},inputMode:()=>{},syncWorkspaceMode:()=>{},queryMode:{value:'rag'},activeTurn:null,images:[],uploading:0,FormData:class{constructor(){this.files=[]}append(k,v){this.files.push(v)}},api:async(path,options)=>{requests.push([path,options.body.files]);return {attachments:[{attachment_id:'eeg1',filename:'sample.npy',size:12}]};}};
vm.createContext(context);vm.runInContext(block,context);
(async()=>{await vm.runInContext("addEEG([{name:'sample.npy',size:12}])",context);assert.equal(requests[0][0],'/eeg-attachments');assert.equal(context.window.canSubmitAttachments(),true);assert.equal(context.queryMode.value,'agent');context.queryMode.value='chat';assert.equal(context.window.canSubmitAttachments(),true);assert.equal(context.queryMode.value,'agent');assert.deepEqual(Array.from(context.window.getExecutionOptions().eeg_attachment_ids),['eeg1']);
// Failed task creation does not invoke any clear; the clarification retains the same ID.
assert.deepEqual(Array.from(context.window.getExecutionOptions().eeg_attachment_ids),['eeg1']);assert.equal(context.window.pendingEEGSnapshot().length,1);
context.window.clearSessionEEG();assert.equal(context.window.pendingEEGSnapshot().length,0);assert.equal(context.window.getExecutionOptions().eeg_attachment_ids,undefined);console.log('EEG attachment lifecycle passed');})().catch(e=>{console.error(e);process.exit(1)});

