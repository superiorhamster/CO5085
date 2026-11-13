/* CIFAR-100 dashboard: verified snapshots, never live training telemetry. */
(function () {
  'use strict';
  const strategies = ['scratch', 'head_only', 'partial', 'full'];
  const tabs = ['overview', 'eda', ...strategies, 'augmentation'];
  const labels = { scratch: 'Scratch', head_only: 'Head-only', partial: 'Partial', full: 'Full fine-tune' };
  const names = { resnet18: 'ResNet-18', deit_tiny: 'DeiT-Tiny' };
  const colors = { resnet18: '#0b57d0', deit_tiny: '#b45309' };
  let snapshot, report, selected = 'overview', tabController;
  const charts = {};
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const metric = (e, k) => e?.has_full_metrics && Number.isFinite(e.metrics?.[k]) ? e.metrics[k] : null;
  const fmt = (v, kind) => v == null || !Number.isFinite(v) ? '—' : kind === 'pct' ? `${(v * 100).toFixed(2)}%` : kind === 'int' ? v.toLocaleString('vi-VN') : v.toFixed(3);
  const find = (family, strategy) => snapshot.experiments.find(e => e.family === family && e.strategy === strategy);
  function activate(tab, updateHash = true) {
    if (tabController) tabController.select(tab, {update:updateHash});
    selected = tabController?.selected || 'overview';
  }
  function plot(id, type, data, extra = {}) {
    if (charts[id]) charts[id].destroy();
    if (!window.Chart) return;
    const color = document.documentElement.dataset.theme === 'dark' ? '#cbd5e1' : '#334155';
    charts[id] = new window.Chart(document.getElementById(id), { type, data, options: {
      responsive: true, maintainAspectRatio: false, color,
      plugins: { legend: { labels: { color } } },
      scales: { x: { ticks: { color } }, y: { beginAtZero: true, ticks: { color } } }, ...extra
    } });
  }
  function figure(src, title) {
    if (!src) return `<p class="a1-empty">${esc(title)}: chưa có artifact.</p>`;
    return `<figure><figcaption>${esc(title)}</figcaption><button class="a1-figure-button" data-a1-image="${esc(src)}" data-title="${esc(title)}" aria-label="Phóng to ${esc(title)}"><img src="${esc(src)}" alt="${esc(title)}" loading="lazy"></button></figure>`;
  }
  function overview() {
    const count = snapshot.experiments.filter(e => e.has_full_metrics).length;
    const controls = snapshot.augmentation?.controls?.filter(e => e.has_full_metrics).length || 0;
    document.getElementById('a1-completion').textContent = `${count}/8 cấu hình chính · ${controls}/2 đối chứng · ${snapshot.epochs} epoch`;
    const pdfState = report?.available && !report.is_placeholder ? `Báo cáo PDF ${report.pages} trang đã biên dịch và kiểm tra; MSSV/video còn thiếu.` : 'PDF chưa có bản build được xác minh.';
    document.getElementById('a1-overview-message').innerHTML = `<h3>${count}/8 kết quả đã xác minh</h3><p>CIFAR-100, seed ${snapshot.seed}, ${snapshot.epochs} epoch/run. ${count < 8 ? 'Chưa đủ ma trận; các ô thiếu được giữ trống.' : 'Đủ ma trận CNN/Transformer và ba mức fine-tune.'} ${pdfState}</p>`;
    document.getElementById('a1-matrix').innerHTML = '<div class="a1-axis">Kiến trúc</div>' + strategies.map(s => `<div class="a1-axis">${labels[s]}</div>`).join('') + Object.keys(names).map(f => `<div class="a1-axis">${names[f]}</div>` + strategies.map(s => `<button data-matrix-tab="${s}" aria-label="Xem ${names[f]} ${labels[s]}"><strong>${fmt(metric(find(f,s),'test_accuracy'),'pct')}</strong><span>Test accuracy</span></button>`).join('')).join('');
    document.querySelectorAll('[data-matrix-tab]').forEach(b => b.addEventListener('click', () => { activate(b.dataset.matrixTab); document.getElementById(`a1-tab-${selected}`).focus(); }));
    document.getElementById('a1-summary').innerHTML = snapshot.experiments.map(e => `<tr><th scope="row">${names[e.family]} / ${labels[e.strategy]}</th><td>${fmt(metric(e,'val_accuracy'),'pct')}</td><td>${fmt(metric(e,'test_accuracy'),'pct')}</td><td>${fmt(metric(e,'macro_f1'))}</td><td>${fmt(metric(e,'trainable_parameters'),'int')} / ${fmt(metric(e,'parameters'),'int')}</td><td>${fmt(metric(e,'training_seconds'))} / ${fmt(metric(e,'epoch_seconds'))} s</td><td>${fmt(metric(e,'peak_gpu_memory_mb'))} MB</td><td>${fmt(metric(e,'latency_ms_per_image'))} ms</td></tr>`).join('');
    const series = (key, scale=1) => Object.keys(names).map(f => ({label:names[f],backgroundColor:colors[f],data:strategies.map(s=>{const v=metric(find(f,s),key);return v===null?null:v*scale;})}));
    plot('a1-accuracy','bar',{labels:strategies.map(s=>labels[s]),datasets:series('test_accuracy',100)});
    plot('a1-f1','bar',{labels:strategies.map(s=>labels[s]),datasets:series('macro_f1')});
    plot('a1-cost','scatter',{datasets:Object.keys(names).map(f=>({label:names[f],backgroundColor:colors[f],pointRadius:6,data:strategies.map(s=>find(f,s)).filter(e=>e?.has_full_metrics).map(e=>({x:metric(e,'training_seconds')/60,y:metric(e,'test_accuracy')*100}))}))}, {scales:{x:{title:{display:true,text:'Train đồng thời (phút)'}},y:{title:{display:true,text:'Test accuracy (%)'}}}});
    document.getElementById('a1-observations').innerHTML = Object.keys(names).map(f=>{
      const runs=snapshot.experiments.filter(e=>e.family===f && e.has_full_metrics);
      const winner=runs.reduce((a,b)=>!a || metric(b,'val_accuracy')>metric(a,'val_accuracy')?b:a,null);
      const scratch=metric(find(f,'scratch'),'test_accuracy'),full=metric(find(f,'full'),'test_accuracy');
      return `<article><h3>${names[f]} · quan sát tại seed42</h3><p>${winner?`Tốt nhất theo validation: <strong>${labels[winner.strategy]}</strong>, val ${fmt(metric(winner,'val_accuracy'),'pct')}, test tương ứng ${fmt(metric(winner,'test_accuracy'),'pct')}.`:'Chưa có kết quả.'}</p><p>${scratch!==null&&full!==null?`Pretrained full − scratch: ${((full-scratch)*100).toFixed(2)} điểm phần trăm test accuracy.`:'Chưa đủ scratch và full để so sánh.'} Không suy rộng thành quy luật cho mọi recipe.</p></article>`;
    }).join('');
  }
  function stages(family,strategy) {
    const blocks=family==='resnet18'?['stem','layer1','layer2','layer3','layer4','fc']:['patch_embed','blocks.0–9','blocks.10','blocks.11','norm','head'];
    const open=b=>['scratch','full'].includes(strategy)||(family==='resnet18'?b==='fc'||strategy==='partial'&&b==='layer4':b==='head'||strategy==='partial'&&['blocks.10','blocks.11','norm'].includes(b));
    return `<div class="a1-stage-list" aria-label="Các tầng khả huấn luyện">${blocks.map(b=>`<span data-trainable="${open(b)}">${b} · ${open(b)?'train':'freeze'}</span>`).join('')}</div>`;
  }
  function modelCard(e) {
    const runLabel=labels[e.strategy]+(e.id.endsWith('_noaug')?' · noaug':'');
    const title=`${names[e.family]} / ${runLabel}`;
    if(!e.has_full_metrics)return `<article class="a1-model"><h3>${title}</h3><p>Chưa có full run hợp lệ.</p></article>`;
    const fields=[['Train-eval accuracy','train_accuracy','pct'],['Validation accuracy','val_accuracy','pct'],['Test accuracy','test_accuracy','pct'],['Macro-F1','macro_f1'],['Validation loss','val_loss'],['Test loss','test_loss'],['Best epoch (1-based)','best_epoch','int'],['Trainable parameters','trainable_parameters','int'],['Total parameters','parameters','int'],['Train time (s)','training_seconds'],['Time / epoch (s)','epoch_seconds'],['Peak training VRAM (MB)','peak_gpu_memory_mb']];
    const benchmark=e.benchmark?.measurements || {};
    const pairs=e.confusion?.top_confusions||[];
    const classes=e.class_metrics||[];
    const hardest=[...classes].sort((a,b)=>a.f1-b.f1).slice(0,10);
    return `<article class="a1-model" data-family="${e.family}"><header><div><h3>${names[e.family]}</h3><span class="a1-model-label">${e.family==='resnet18'?'CNN · GPU0':'Transformer · GPU1'} / ${runLabel}</span></div><span class="a1-run-status" data-complete="true">Full run đã xác minh</span></header>${stages(e.family,e.strategy)}<dl class="a1-metrics">${fields.map(([label,key,kind])=>`<div><dt>${label}</dt><dd>${fmt(metric(e,key),kind)}</dd></div>`).join('')}</dl><h4>Inference FP32 · GPU0 · Tesla T4</h4><div class="table-wrapper"><table><thead><tr><th>Batch</th><th>ms/ảnh</th><th>Ảnh/s</th><th>VRAM MB</th></tr></thead><tbody>${[1,32].map(n=>`<tr><th>${n}</th><td>${fmt(benchmark[n]?.latency_ms_per_image)}</td><td>${fmt(benchmark[n]?.images_per_second)}</td><td>${fmt(benchmark[n]?.peak_allocated_memory_mb)}</td></tr>`).join('')}</tbody></table></div><p class="a1-small">Warmup 10 · 50 lần đo; không bao gồm data loading.</p>${figure(e.images?.curves,title+' · learning curves')}${figure(e.images?.confusion_matrix,title+' · confusion matrix 100 lớp')}<h4>Cặp lớp nhầm nhiều nhất</h4><ul>${pairs.slice(0,5).map(p=>`<li>${esc(p.true)} → ${esc(p.predicted)}: ${p.count} ảnh</li>`).join('')}</ul><details><summary>10 lớp có F1 thấp nhất · 100 lớp lớp</summary><div class="table-wrapper"><table><thead><tr><th>Lớp</th><th>Precision</th><th>Recall</th><th>F1</th><th>N</th></tr></thead><tbody>${hardest.map(c=>`<tr><th>${esc(c.class_name)}</th><td>${fmt(c.precision)}</td><td>${fmt(c.recall)}</td><td>${fmt(c.f1)}</td><td>${c.support}</td></tr>`).join('')}</tbody></table></div><p>Đầy đủ 100 lớp: <a href="https://github.com/superiorhamster/CO5085/blob/main/artifacts/A1/full/${esc(e.id)}/class_metrics.csv">xem artifact CSV</a>.</p></details>${figure(e.images?.failures,title+' · dự đoán sai tự tin cao')}</article>`;
  }
  function pair(strategy) {
    const a=find('resnet18',strategy),b=find('deit_tiny',strategy);
    const x=metric(a,'test_accuracy'),y=metric(b,'test_accuracy');
    const delta=x===null||y===null?'Chưa đủ hai full run để so sánh.':`DeiT − ResNet: <strong>${((y-x)*100).toFixed(2)} điểm phần trăm test accuracy</strong>, seed42, ${snapshot.epochs} epoch.`;
    document.getElementById(`a1-panel-${strategy}`).innerHTML=`<div class="a1-delta">${delta}</div><div class="ui-pair">${[a,b].filter(Boolean).map(modelCard).join('')}</div>`;
  }
  function eda() {
    const data=snapshot.eda;
    if(!data){document.getElementById('a1-panel-eda').innerHTML='<p>Chưa có EDA.</p>';return;}
    const stats=data.statistics,duplicates=data.duplicates;
    const rgb=v=>Array.isArray(v)?v.map(x=>x.toFixed(4)).join(' / '):'—';
    const figures=[['eda_distribution','Phân bố100 lớp và ba split'],['eda_color','RGB và độ sáng · chỉ train'],['eda_all_classes','Ảnh train đại diện100 lớp'],['eda_superclasses','Ảnh train trong cùng superclass'],['eda_preprocessing','Ảnh32px, resize 224px và augmentation']];
    document.getElementById('a1-panel-eda').innerHTML=`<h3>Từ dữ liệu đến quyết định xử lý</h3><p>100 lớp cân bằng: 450 train,50 validation và100 test/lớp. Nhãn coarse chỉ dùng để EDA; target là fine label.</p><dl class="a1-metrics"><div><dt>Định dạng</dt><dd>RGB 32×32 · ${esc(stats.dtype)} · [0,255]</dd></div><div><dt>Train RGB mean</dt><dd>${rgb(stats.train_rgb_mean)}</dd></div><div><dt>Train RGB std</dt><dd>${rgb(stats.train_rgb_std)}</dd></div><div><dt>Exact duplicates</dt><dd>${duplicates.duplicate_groups} nhóm · ${duplicates.cross_split_groups} cross-split · ${duplicates.conflicting_label_groups} xung đột nhãn</dd></div></dl><p class="a1-limitations">Giữ nguyên split đã đăng ký, không âm thầm dedup. Cross-split duplicates có thể làm đánh giá lạc quan. Hash pixel không phát hiện mọi near-duplicate hoặc overlap với pretraining. Validation/test chỉ dùng kiểm tra integrity/count/duplicate, không chọn augmentation.</p><div class="a1-eda-grid">${figures.map(([key,title])=>figure(data.images[key],title)).join('')}</div><h3>Quyết định sau EDA</h3><ol><li>Không class weighting/weighted sampler vì train cân bằng.</li><li>Ảnh32px ít chi tiết: crop 32, padding phản chiếu4 và flip 0,5 trước resize 224 bicubic; upscale không tạo thêm thông tin.</li><li>ImageNet normalization cố định cho cả hai model; không ước lượng từ test.</li><li>Đo hiệu quả bằng hai cặp scratch augmented/noaug trong tab Đối chứng.</li></ol>`;
  }
  function augmentation() {
    const aug=snapshot.augmentation;
    if(!aug){document.getElementById('a1-panel-augmentation').innerHTML='<p>Chưa có đối chứng augmentation.</p>';return;}
    document.getElementById('a1-panel-augmentation').innerHTML=`<h3>Crop/flip trước upscale có giúp scratch?</h3><p>Cùng split, seed, khởi tạo, AdamW và ${snapshot.epochs} epoch. Main scratch dùng crop 32/pad 4/flip; noaug chỉ resize 224 và normalize. Chỉ augmentation thay đổi.</p><div class="table-wrapper"><table><caption>Kết quả đo trên cùng test 10.000 ảnh</caption><thead><tr><th>Model</th><th>Augmented test</th><th>Noaug test</th><th>Δ test (điểm %)</th><th>Δ val (điểm %)</th><th>Δ macro-F1</th><th>Δ gap (điểm %)</th></tr></thead><tbody>${Object.keys(names).map(f=>{
      const a=find(f,'scratch'),b=aug.controls.find(e=>e.family===f),d=aug.deltas.find(e=>e.family===f);
      const complete=a?.has_full_metrics&&b?.has_full_metrics&&d;
      return `<tr><th>${names[f]}</th><td>${fmt(metric(a,'test_accuracy'),'pct')}</td><td>${fmt(metric(b,'test_accuracy'),'pct')}</td><td>${complete?fmt(d.delta_test_accuracy*100):'—'}</td><td>${complete?fmt(d.delta_val_accuracy*100):'—'}</td><td>${complete?fmt(d.delta_macro_f1):'—'}</td><td>${complete?fmt(d.delta_generalization_gap*100):'—'}</td></tr>`;
    }).join('')}</tbody></table></div><p>Δ = augmented − noaug; generalization gap = FP32 train-eval accuracy − validation accuracy của cùng best checkpoint. Gap giảm không đồng nghĩa đã chứng minh ý nghĩa thống kê. Kết luận này chỉ áp dụng scratch tại seed42, không mở rộng sang tất cả pretrained strategies.</p>${figure(aug.image,'Delta augmentation · accuracy/F1/gap')}<div class="ui-pair">${aug.controls.map(e=>modelCard(e)).join('')}</div>`;
  }
  window.renderCifarA1 = state => {
    if(!document.getElementById('a1-dashboard')||!state.manifest?.assignments?.A1)return;
    snapshot=state.manifest.assignments.A1;
    report=state.manifest.reports?.['assignment1.pdf'];
    if(snapshot.protocol!=='cifar100_resnet18_deit_v1'){
      document.getElementById('a1-completion').textContent='Protocol A1 chưa tương thích; nhập bundle CIFAR-100.';return;
    }
    overview();eda();strategies.forEach(pair);augmentation();
    document.querySelectorAll('[data-a1-image]').forEach(b=>b.addEventListener('click',()=>
      window.CourseworkUI.openImage(b.dataset.a1Image,b.dataset.title,b)));
    if (!tabController) tabController = window.CourseworkUI.tabs({
      names:tabs, button:n=>document.getElementById('a1-tab-'+n), panel:n=>document.getElementById('a1-panel-'+n), hash:n=>'#a1-'+n,
      onActivate:n=>{selected=n;Object.values(charts).forEach(c=>c.resize());}
    });
    else tabController.refresh();
    selected=tabController.selected;
  };
}());
