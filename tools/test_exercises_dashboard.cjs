/* Actual E1–E3 renderer, without browser or chart dependencies. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('docs/exercises.html','utf8');
const manifest=JSON.parse(fs.readFileSync('docs/data/experiments_manifest.json'));
function element(id){return {id,innerHTML:'',textContent:'',dataset:{},attrs:{},events:{},hidden:false,style:{},classList:{add(){},remove(){}},setAttribute(k,v){this.attrs[k]=v;},getAttribute(k){return this.attrs[k];},removeAttribute(k){delete this.attrs[k];},addEventListener(k,f){this.events[k]=f;},querySelectorAll(){return [];},focus(){}};}
(async()=>{
for(const complete of [false,true]){
 const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],element(m[1])]));
 const backend=['all','scratch','torch'].map(n=>{const e=element(n);e.attrs['data-e2-backend']=n;return e;});
 const token=['all','patch','row','cnn'].map(n=>{const e=element(n);e.attrs['data-e2-tokenizer']=n;return e;});
 const pdf=element('pdf');pdf.dataset.reportPdf='exercises.pdf';
 const document={readyState:'complete',body:{},documentElement:{setAttribute(){}},getElementById:id=>elements[id],querySelectorAll:s=>s==='[data-e2-backend]'?backend:s==='[data-e2-tokenizer]'?token:s==='[data-report-pdf]'?[pdf]:[]};
 const charts={};function Chart(canvas,config){charts[canvas.id]=config;this.destroy=()=>{};this.resize=()=>{};}
 const location={hash:'#e2-section'},window={Chart,addEventListener(){}};
 const storageReads=[];
 const context={window,document,location,history:{pushState(a,b,h){location.hash=h;}},localStorage:{getItem:k=>{storageReads.push(k);return k==='co5085_preview_mode'?'true':null;},setItem(){}},Chart,console,fetch:async()=>{throw Error('file fallback');}};
 vm.createContext(context);for(const file of ['fallback_data','ui'])vm.runInContext(fs.readFileSync(`docs/assets/js/${file}.js`,'utf8'),context);
 const data=window.FALLBACK_EXPERIMENTS_MANIFEST;
 for(const a of ['E1','E2','E3'])for(const e of data.assignments[a].experiments){e.has_full_metrics=complete;if(!complete){e.metrics=null;e.history=[];}}
 if(complete){data.assignments.E1.experiments[0].metrics.test_accuracy=0;data.assignments.E1.experiments[0].history[0].val_accuracy=0;}
 else data.reports['exercises.pdf'].available=false;
 vm.runInContext(fs.readFileSync('docs/assets/js/site.js','utf8'),context);await new Promise(r=>setImmediate(r));
 assert.match(elements['exercises-completion'].textContent,new RegExp(`${complete?13:0}/13`));
 assert.equal((elements['exercises-summary'].innerHTML.match(/<tr>/g)||[]).length,13);
 assert.equal(elements['e2-section'].hidden,false);
 assert.ok(!storageReads.includes('co5085_preview_mode'));
 assert.equal(pdf.attrs['aria-disabled'],String(!complete));
 if(complete){assert.match(elements['e1-table-body'].innerHTML,/0\.00%/);assert.equal(charts['e1-chart-canvas'].data.datasets[0].data[0].y,0);}
 else {assert.ok(!pdf.attrs.href);assert.equal(charts['e1-chart-canvas'].data.datasets.length,0);assert.ok(charts['e2-chart-canvas'].data.datasets.every(d=>d.data.every(v=>v===null)));}
 backend[1].events.click();token[1].events.click();assert.equal((elements['e2-table-body'].innerHTML.match(/<tr>/g)||[]).length,1);
 elements['exercise-tab-e1'].events.click();elements['exercise-tab-e2'].events.click();assert.equal((elements['e2-table-body'].innerHTML.match(/<tr>/g)||[]).length,1);
 elements['theme-toggle-btn'].events.click();assert.equal((elements['e2-table-body'].innerHTML.match(/<tr>/g)||[]).length,1);
 console.log(`PASS E1–E3 ${complete?'full':'missing'}: fallback, zero, real histories, PDF states, filters across tabs/theme`);
}
})().catch(e=>{console.error(e);process.exitCode=1;});
