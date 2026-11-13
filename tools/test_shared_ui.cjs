/* Dependency-free regression checks for shared tab and image-dialog behavior. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
function element(id) {
  return {id, hidden:false, attrs:{}, handlers:{}, focused:false, isConnected:true,
    setAttribute(k,v){this.attrs[k]=v;}, addEventListener(k,fn){this.handlers[k]=fn;},
    focus(){this.focused=true;}, contains(t){return t?.parent===this;}, scrollIntoView(){},
    close(){this.open=false;this.handlers.close?.();}, showModal(){this.open=true;}, replaceChildren(...children){this.children=children;}};
}
const names=['overview','e1','e2','e3'];
const buttons=names.map(n=>element('tab-'+n)), panels=names.map(n=>element(n==='overview'?'exercises-overview':n+'-section'));
const nested=element('e2-details');nested.parent=panels[2];
const dialog=element('artifact-lightbox'), close=element('lightbox-close-btn'), container=element('lightbox-image-container'), heading=element('lightbox-title');
const elements=[...buttons,...panels,nested,dialog,close,container,heading];
const listeners={}, location={hash:''}, stack=[''];let index=0, activations=0;
const window={addEventListener(k,f){listeners[k]=f;}};
const document={getElementById:id=>elements.find(e=>e.id===id),createElement:()=>element('img')};
const context={window,document,location,history:{pushState(a,b,h){stack.splice(++index);stack.push(location.hash=h);}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(root,'docs/assets/js/ui.js'),'utf8'),context);
const tabs=window.CourseworkUI.tabs({names,button:n=>buttons[names.indexOf(n)],panel:n=>panels[names.indexOf(n)],hash:n=>n==='overview'?'#exercises-overview':'#'+n+'-section',aliases:{'#legacy-e2':'e2'},onActivate(){activations++;}});
assert.equal(tabs.selected,'overview');assert.equal(panels[0].hidden,false);
buttons[2].handlers.click();assert.equal(location.hash,'#e2-section');assert.equal(buttons[2].attrs['aria-selected'],'true');assert.equal(buttons[1].tabIndex,-1);
buttons[2].handlers.keydown({key:'ArrowRight',preventDefault(){}});assert.equal(tabs.selected,'e3');assert.ok(buttons[3].focused);
for(const [key,expected] of [['Home','overview'],['End','e3'],['ArrowRight','overview'],['ArrowLeft','e3']]) {
 buttons[names.indexOf(tabs.selected)].handlers.keydown({key,preventDefault(){}});assert.equal(tabs.selected,expected);
}
location.hash='#e2-details';listeners.hashchange();assert.equal(tabs.selected,'e2');assert.equal(panels[2].hidden,false);
location.hash='#legacy-e2';listeners.hashchange();assert.equal(tabs.selected,'e2');
location.hash=stack[--index];listeners.popstate();assert.equal(tabs.selected,'overview');
location.hash=stack[++index];listeners.popstate();assert.equal(tabs.selected,'e3');
tabs.refresh();assert.equal(tabs.selected,'e3');assert.ok(activations>=10);
window.CourseworkUI.openImage('test.png','Measured confusion matrix',buttons[2]);assert.ok(dialog.open);assert.equal(container.children[0].alt,'Measured confusion matrix');
buttons[2].focused=false;close.handlers.click();assert.ok(buttons[2].focused);assert.equal(dialog.open,false);
const manifest=JSON.parse(fs.readFileSync(path.join(root,'docs/data/experiments_manifest.json')));
assert.equal(manifest.total_completed_full_experiments,23);
for(const assignment of Object.values(manifest.assignments))for(const exp of assignment.experiments)assert.ok(!Object.hasOwn(exp,'preview'));
const site=fs.readFileSync(path.join(root,'docs/assets/js/site.js'),'utf8');
assert.ok(!/co5085_preview|previewMode|STORAGE_PREVIEW/.test(site));
for(const name of ['index','exercises','a1','a2']) {
 const html=fs.readFileSync(path.join(root,`docs/${name}.html`),'utf8');
 assert.ok(html.includes('assets/js/ui.js'));assert.ok(html.includes('assets/css/ui.css'));
 assert.ok(!/preview-toggle|preview-banner|mode-toggle-btn/.test(html));
}
console.log('PASS shared UI: tabs, nested/legacy hashes, keyboard, history, focus restoration and real-only data');
