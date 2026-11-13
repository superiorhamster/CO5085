/* Renderer tests with a small DOM double. Real browser behavior is checked separately. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'docs/a1.html'), 'utf8');
const source = fs.readFileSync(path.join(root, 'docs/assets/js/a1.js'), 'utf8');
const base = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/experiments_manifest.json'))).assignments.A1;
function element(id) {
  return { id, innerHTML:'', textContent:'', hidden:false, attrs:{}, events:{}, dataset:{},
    setAttribute(k,v) { this.attrs[k]=v; }, addEventListener(k,fn) { this.events[k]=fn; }, focus() {} };
}
for (const count of [0, 1, 8]) {
  const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], element(m[1])]));
  const tabNames = ['overview','eda','scratch','head_only','partial','full','augmentation'];
  const buttons = tabNames.map(name => { const e=elements['a1-tab-'+name];e.dataset.a1Tab=name;return e; });
  const document = { documentElement:{dataset:{theme:'light'}}, getElementById:id=>elements[id],
    querySelectorAll:selector=>selector==='[data-a1-tab]'?buttons:[] };
  const observed = {}, location = {hash:'#a1-partial'};
  function Chart(target,config) { observed[target.id]=config;this.destroy=()=>{};this.resize=()=>{}; }
  const window = { Chart,addEventListener() {},FALLBACK_EXPERIMENTS_MANIFEST:{reports:{'assignment1.pdf':{is_placeholder:true}}} };
  const context = {window,document,location,history:{pushState(a,b,hash){location.hash=hash;}},console};
  vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(root,'docs/assets/js/ui.js'),'utf8'),context);vm.runInContext(source,context);
  const a1=structuredClone(base);
  a1.experiments.forEach((e,i)=>{
    e.has_full_metrics=i<count;
    e.metrics=i<count?{val_accuracy:.8,test_accuracy:i===0?0:.75,macro_f1:.72,parameters:100,trainable_parameters:10,training_seconds:10,epoch_seconds:1,latency_ms_per_image:2,peak_gpu_memory_mb:100}:null;
  });
  window.renderCifarA1({manifest:{assignments:{A1:a1}}});
  assert.match(elements['a1-overview-message'].innerHTML,/PDF chưa có bản build/);
  window.renderCifarA1({manifest:{assignments:{A1:a1},reports:{'assignment1.pdf':{available:true,is_placeholder:false,pages:22}}}});
  assert.match(elements['a1-overview-message'].innerHTML,/PDF 22 trang/);
  assert.match(elements['a1-completion'].textContent,new RegExp(`${count}/8`));
  assert.match(elements['a1-completion'].textContent,/25 epoch/);
  assert.equal(buttons.length,7);
  assert.match(elements['a1-panel-eda'].innerHTML,/12 cross-split/);
  assert.match(elements['a1-panel-augmentation'].innerHTML,/Noaug test/);
  assert.equal(elements['a1-tab-partial'].attrs['aria-selected'],'true');
  assert.equal(elements['a1-panel-partial'].hidden,false);
  assert.equal((elements['a1-summary'].innerHTML.match(/<tr>/g)||[]).length,8);
  const actual=observed['a1-accuracy'].data.datasets.flatMap(d=>d.data);
  assert.equal(actual.filter(v=>v!==null).length,count);
  if(count)assert.match(elements['a1-summary'].innerHTML,/0\.00%/);
  else assert.ok(actual.every(v=>v===null));
  if(count<8)assert.match(elements['a1-panel-full'].innerHTML,/Chưa đủ hai full run/);
  else assert.match(elements['a1-panel-full'].innerHTML,/DeiT − ResNet/);
  buttons[4].events.keydown({key:'ArrowRight',preventDefault(){}});
  assert.equal(elements['a1-tab-full'].attrs['aria-selected'],'true');
  assert.equal(location.hash,'#a1-full');
  console.log(`PASS dashboard ${count}/8: correct cells, missing values, deltas, deep link and keyboard`);
}

// A committed fallback is a complete render source when fetch/file:// is unavailable.
const fallbackContext={window:{}};
vm.createContext(fallbackContext);
vm.runInContext(fs.readFileSync(path.join(root,'docs/assets/js/fallback_data.js'),'utf8'),fallbackContext);
assert.equal(fallbackContext.window.FALLBACK_EXPERIMENTS_MANIFEST.total_completed_full_experiments,23);
assert.equal(fallbackContext.window.FALLBACK_EXPERIMENTS_MANIFEST.assignments.A1.experiments.length,8);
assert.equal(fallbackContext.window.FALLBACK_EXPERIMENTS_MANIFEST.assignments.A1.augmentation.controls.length,2);
console.log('PASS file fallback: 23 main results and 2 separate augmentation controls');
