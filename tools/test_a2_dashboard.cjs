/* Renderer contract; browser behavior is also checked manually. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('docs/a2.html','utf8'),source=fs.readFileSync('docs/assets/js/a2.js','utf8');
const manifest=JSON.parse(fs.readFileSync('docs/data/experiments_manifest.json'));
const siteConfig=JSON.parse(fs.readFileSync('docs/data/site_config.json'));
for(const count of [0,1,2]){
 const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(x=>[x[1],{innerHTML:'',textContent:'',dataset:{},events:{},attrs:{},setAttribute(k,v){this.attrs[k]=v;},addEventListener(k,v){this.events[k]=v;},focus(){}}]));
 const names=['overview','data','paper','training','results'],buttons=names.map(n=>elements['a2-tab-'+n]);
 const handlers={},location={hash:'#paper-analysis'};
 const document={getElementById:id=>elements[id],querySelectorAll:s=>s==='[data-a2-tab]'?buttons:[]};
 const window={addEventListener(k,fn){handlers[k]=fn;}};
 const history={pushState(a,b,hash){location.hash=hash;}};
 const context={window,document,location,history,console};vm.createContext(context);vm.runInContext(fs.readFileSync('docs/assets/js/ui.js','utf8'),context);vm.runInContext(source,context);
 const m=structuredClone(manifest);m.reports['assignment2.pdf']={available:false,is_placeholder:true};m.assignments.A2.experiments.forEach((e,i)=>{e.has_full_metrics=i<count;if(i>=count){e.metrics=null;e.benchmark=null;}});
 window.renderAnimeA2({manifest:m,siteConfig});
 assert.equal(elements['a2-tab-paper'].attrs['aria-selected'],'true');
 assert.match(elements['a2-completion'].textContent,new RegExp(`${count}/2`));
 assert.ok(!elements['a2-deliverables'].innerHTML.includes('href="reports/assignment2.pdf"'));
 assert.match(elements['a2-deliverables'].innerHTML,/Chưa cung cấp/);
 buttons[2].events.keydown({key:'End',preventDefault(){}});assert.equal(location.hash,'#a2-results');
 buttons[4].events.keydown({key:'Home',preventDefault(){}});assert.equal(location.hash,'#a2-overview');
 location.hash='#pipeline-results';handlers.hashchange();assert.equal(elements['a2-panel-results'].hidden,false);
 location.hash='#a2-paper';handlers.popstate();assert.equal(elements['a2-panel-paper'].hidden,false);
 if(count===2){assert.match(elements['a2-conclusion'].textContent,/298,51/);assert.match(elements['a2-conclusion'].textContent,/41,84/);assert.match(elements['a2-conclusion'].textContent,/0,0584/);assert.match(elements['a2-quality-table'].innerHTML,/>0,0000</);assert.match(elements['a2-provenance'].innerHTML,/20000/);}
 else assert.match(elements['a2-conclusion'].textContent,/Chưa đủ/);
 const done=structuredClone(m);done.reports['assignment2.pdf']={available:true,is_placeholder:false,path:'reports/assignment2.pdf'};
 window.renderAnimeA2({manifest:done,siteConfig:{...siteConfig,media:{a2_demo_video_url:'https://example.com/demo'}}});
 assert.match(elements['a2-deliverables'].innerHTML,/Tải báo cáo PDF/);assert.match(elements['a2-deliverables'].innerHTML,/https:\/\/example.com\/demo/);
 console.log('PASS A2 '+count+'/2: metrics, empty states, tabs, legacy links, PDF/video states');
}

// Run the actual shared initializer with fetch rejected, as with local file://.
(async()=>{
 const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(x=>[x[1],{innerHTML:'',textContent:'',dataset:{},events:{},attrs:{},setAttribute(k,v){this.attrs[k]=v;},addEventListener(k,v){this.events[k]=v;},focus(){}}]));
 const buttons=['overview','data','paper','training','results'].map(n=>elements['a2-tab-'+n]);
 const document={readyState:'complete',body:{},documentElement:{setAttribute(){}},getElementById:id=>elements[id],querySelectorAll:s=>s==='[data-a2-tab]'?buttons:[]};
 const window={addEventListener(){}};
 const context={window,document,location:{hash:''},history:{pushState(){}},console,localStorage:{getItem(){return null;},setItem(){}},fetch:async()=>{throw new Error('file:// fetch unavailable');}};
 vm.createContext(context);
 vm.runInContext(fs.readFileSync('docs/assets/js/fallback_data.js','utf8'),context);
 assert.deepEqual(JSON.parse(JSON.stringify(window.FALLBACK_EXPERIMENTS_MANIFEST)),manifest);
 vm.runInContext(fs.readFileSync('docs/assets/js/ui.js','utf8'),context);vm.runInContext(source,context);vm.runInContext(fs.readFileSync('docs/assets/js/site.js','utf8'),context);
 await new Promise(resolve=>setImmediate(resolve));
 assert.match(elements['a2-completion'].textContent,/2\/2/);
 assert.match(elements['a2-conclusion'].textContent,/41,84/);
 console.log('PASS shared initializer: rejected fetch uses identical file:// fallback');
})().catch(error=>{console.error(error);process.exitCode=1;});
