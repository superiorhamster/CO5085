/* A2 snapshot renderer: only validated exports, never layout-demo metrics. */
(() => {
  'use strict';
  const tabs=['overview','data','paper','training','results'];
  const aliases={'#paper-analysis':'paper','#pipeline-results':'results','#a2-videos':'overview'};
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=(v,d=2)=>typeof v==='number'&&Number.isFinite(v)?v.toLocaleString('vi-VN',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
  const put=(id,html)=>{const e=document.getElementById(id);if(e)e.innerHTML=html;};
  const text=(id,value)=>{const e=document.getElementById(id);if(e)e.textContent=value;};
  function table(caption,headers,rows){return `<div class="table-wrapper"><table><caption>${esc(caption)}</caption><thead><tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map((x,i)=>i?`<td>${esc(x)}</td>`:`<th scope="row">${esc(x)}</th>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
  let tabController;
  function bindTabs(){
    if (!tabController) tabController=window.CourseworkUI.tabs({
      names:tabs,button:n=>document.getElementById('a2-tab-'+n),panel:n=>document.getElementById('a2-panel-'+n),hash:n=>'#a2-'+n,aliases
    });
    else tabController.refresh();
  }
  function figure(exp,key,caption){
    const path=exp.images?.[key];if(!path)return `<p class="a2-muted">Chưa có hình ${esc(caption)}.</p>`;
    return `<figure><figcaption>${esc(exp.name)} · ${esc(caption)}</figcaption><button class="a2-figure" type="button" data-a2-image="${esc(path)}" data-caption="${esc(exp.name+' — '+caption)}" aria-label="Phóng to ${esc(exp.name+' '+caption)}"><img src="${esc(path)}" alt="${esc(exp.name+' — '+caption)}" loading="lazy" width="512" height="512"></button></figure>`;
  }
  function bindFigures(){document.querySelectorAll('[data-a2-image]').forEach(b=>b.addEventListener('click',()=>
    window.CourseworkUI.openImage(b.dataset.a2Image,b.dataset.caption,b)));}
  function plot(target,series,labels,line=false){
    if(!series.length)return;const w=560,h=280,left=58,bottom=225,top=20,width=480;
    const max=Math.max(1,...series.flatMap(s=>s.values));
    let body=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(labels.title)}"><title>${esc(labels.title)}</title>`;
    for(let i=0;i<=4;i++){const y=bottom-i*(bottom-top)/4;body+=`<line x1="${left}" x2="${left+width}" y1="${y}" y2="${y}" stroke="currentColor" opacity=".15"/><text x="${left-8}" y="${y+4}" text-anchor="end" font-size="11" fill="currentColor">${esc(num(max*i/4,line?2:0))}</text>`;}
    series.forEach((s,j)=>{
      if(line){const points=s.values.map((v,i)=>`${left+(s.steps[i]/20000)*width},${bottom-v/max*(bottom-top)}`).join(' ');body+=`<polyline points="${points}" fill="none" class="${j?'a2-augmented':'a2-baseline'}" style="fill:none" stroke-width="1.8"/>`;}
      else s.values.forEach((v,i)=>{const group=width/s.values.length,bw=group/(series.length+1),x=left+i*group+j*bw;body+=`<rect x="${x}" y="${bottom-v/max*(bottom-top)}" width="${bw-2}" height="${v/max*(bottom-top)}" class="${j?'a2-augmented':'a2-baseline'}"><title>${esc(s.name+' '+labels.x[i]+': '+num(v,0))}</title></rect>`;});
    });
    const xs=line?['0','5k','10k','15k','20k']:labels.x;
    xs.forEach((label,i)=>{const x=line?left+i*width/4:left+(i+.5)*width/xs.length;body+=`<text x="${x}" y="246" text-anchor="middle" font-size="10" fill="currentColor">${esc(label)}</text>`;});
    body+=`<text x="${left}" y="274" font-size="11" fill="currentColor">${esc(line?'Update · loss trung bình 50 bước':labels.unit)}</text></svg>`;
    put(target,body+`<div class="a2-legend">${series.map(s=>`<span>${esc(s.name)}</span>`).join('')}</div>`);
  }
  function resources(state){
    const cfg=state.siteConfig||{},report=state.manifest?.reports?.['assignment2.pdf'],media=cfg.media||{};
    const link=(url,label)=>/^https:\/\//.test(url||'')?`<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)}</a>`:'Chưa cung cấp';
    const pdf=report?.available&&!report.is_placeholder?`<a href="${esc(report.path)}">Tải báo cáo PDF</a>`:'Chưa cung cấp';
    put('a2-deliverables',`<ul><li>Báo cáo PDF: ${pdf}</li><li>Video demo: ${link(media.a2_demo_video_url,'Xem demo')}</li><li>Video trình bày: ${link(media.a2_presentation_video_url,'Xem trình bày')}</li></ul>`);
    document.querySelectorAll('[data-a2-pdf-state]').forEach(e=>{e.innerHTML=pdf==='Chưa cung cấp'?'PDF: Chưa cung cấp':pdf;});
  }
  window.renderAnimeA2=function(state){
    const a=state.manifest?.assignments?.A2;if(!a)return;
    const experiments=a.experiments||[],valid=experiments.filter(e=>e.has_full_metrics),complete=valid.length===2&&valid[0].metrics.metric_protocol_hash===valid[1].metrics.metric_protocol_hash;
    const summary=complete?`FID ${num(valid[0].metrics.fid)} → ${num(valid[1].metrics.fid)}; recall ${num(valid[0].metrics.generative_recall,4)} → ${num(valid[1].metrics.generative_recall,4)}. DiffAugment cải thiện FID/KID ở seed 42, nhưng độ phủ còn thấp.`:'Chưa đủ hai full run tương thích để kết luận.';
    text('a2-home-summary',`${valid.length}/2 full runs. ${summary} PDF/video: xem trạng thái trên dashboard A2.`);
    resources(state);
    if(!document.getElementById('a2-dashboard'))return;
    bindTabs();text('a2-completion',`Huấn luyện ${valid.length}/2 biến thể · ${complete?'20.000 updates/nhánh':'chờ artifact'}`);
    text('a2-conclusion',summary);text('a2-result-discussion',summary+' Các chỉ số phản ánh một lần train và cùng bộ tham chiếu; chưa phải kết luận thống kê.');
    put('a2-overview-pair',experiments.map(e=>`<article><h3>${esc(e.name)}</h3>${e.has_full_metrics?`<p>FID ${num(e.metrics.fid)} · KID ${num(e.metrics.kid_mean,4)} · Recall ${num(e.metrics.generative_recall,4)}</p>`:'<p>Chờ full run hợp lệ.</p>'}${figure(e,'samples','64 latent cố định')}</article>`).join(''));
    const rows=keys=>experiments.map(e=>[e.name,...keys.map(k=>e.has_full_metrics?num(e.metrics[k],k.includes('parameters')?0:4):'Chờ full run')]);
    put('a2-quality-table',table('Cùng 5.000 ảnh sinh và 5.000 test thật; final EMA',['Biến thể','FID ↓','KID mean ↓','KID std','IS mean ↑','IS split std','Precision ↑','Recall ↑'],rows(['fid','kid_mean','kid_std','inception_score_mean','inception_score_split_std','generative_precision','generative_recall'])));
    put('a2-cost-table',table('Chi phí train đo trong các worker',['Biến thể','Train (s)','Updates/s','Ảnh thật/s','Host peak MB','GPU peak MB','Tham số G'],rows(['training_seconds','updates_per_second','real_images_per_second','peak_host_memory_mb','peak_training_gpu_memory_mb','parameters'])));
    put('a2-inference-table',table('Benchmark inference của generator EMA, FP32',['Biến thể','Batch','ms/ảnh','Ảnh/s','GPU peak MB','Thiết bị'],experiments.flatMap(e=>[1,64].map(n=>{const b=e.has_full_metrics?e.benchmark?.[n]:null;return [e.name,n,num(b?.latency_ms_per_image,4),num(b?.images_per_second),num(b?.peak_gpu_memory_mb),b?.device||'Chờ full run'];}))));
    const captions={samples:'64 latent cố định',interpolation:'Nội suy giữa hai latent',curves:'Loss G/D từ history',nearest_neighbors:'Generated / nearest train / nearest test'};
    put('a2-result-gallery',Object.entries(captions).map(([key,label])=>`<h4>${label}</h4><div class="ui-pair">${experiments.map(e=>figure(e,key,label)).join('')}</div>${key==='nearest_neighbors'?'<p class="a2-caption">Mỗi hàng: generated / nearest trong fixed 5.000 train / nearest test, đo khoảng cách Inception. Đây là chẩn đoán, không chứng nhận không ghi nhớ.</p>':''}`).join(''));bindFigures();
    const e=a.eda;if(e){
      put('a2-data-summary',table('Kiểm tra ảnh trước split',['Đã kiểm tra','Bản trùng loại bỏ','Ảnh hỏng','Unique','Train','Validation','Test'],[[e.inspected,e.duplicates,e.corrupt,e.unique,e.splits.train,e.splits.val,e.splits.test].map(v=>num(v,0))]));
      plot('a2-dimensions',[{name:'Chiều rộng',values:e.widths.counts},{name:'Chiều cao',values:e.heights.counts}],{title:'Số ảnh theo khoảng kích thước',x:e.widths.labels,unit:'Khoảng kích thước (pixel) · số ảnh'});
      plot('a2-ratios',[{name:'Số ảnh',values:e.aspect_ratios.counts}],{title:'Số ảnh theo tỷ lệ khung hình',x:e.aspect_ratios.labels,unit:'Khoảng rộng/cao · số ảnh'});
    }else put('a2-data-summary','<p>Chưa có manifest EDA đã kiểm chứng.</p>');
    const c=valid[0]?.config;if(c){
      put('a2-training-config',table('Cấu hình thực tế của export',['Optimizer / LR','Batch / updates','Latent / width','EMA / AMP','Checkpoint / workers'],[[`Adam / ${c.model.lr}`,`${c.effective_batch} / ${c.training.max_steps}`,`${c.model.latent} / ${c.model.width}`,`${c.model.ema} / ${c.training.amp?'bật':'tắt'}`,`${c.training.save_every} updates / ${c.training.workers}`]]));
      text('a2-generator-parameters',`Tham số generator: ${num(valid[0].metrics.parameters,0)}. Không phải tổng tham số G+D.`);
    }
    for(const [key,id] of [['g_loss','a2-g-history'],['d_loss','a2-d-history']])plot(id,valid.map(e=>({name:e.name,values:e.history.map(r=>r[key]),steps:e.history.map(r=>r.step)})),{title:key},true);
    const p=valid[0]?.metric_protocol;
    text('a2-metric-protocol',p?`Inception-2048 (${p.extractor}), torch-fidelity ${p.torch_fidelity}; RGB uint8, FP32; FID float64. KID ${p.evaluation.kid_subsets} subsets × ${p.evaluation.kid_subset_size}, scale gốc; IS${p.evaluation.is_splits} splits; manifold k=${p.evaluation.knn_k}. Không augmentation khi đánh giá.`:'Chờ metric protocol hợp lệ.');
    put('a2-provenance',`<p>Manifest SHA256: <code>${esc(a.manifest_sha256||'Chưa có')}</code></p><p>Worker SHA256: <code>${esc(a.worker_sha256||'Chưa có')}</code></p><p>Metric protocol: <code>${esc(a.metric_protocol_hash||'Chưa có')}</code></p>`+valid.map(e=>`<p>${esc(e.name)} · ${esc(e.history_points)} history rows · checkpoint SHA256 <code>${esc(e.metrics.checkpoint_sha256)}</code></p>`).join(''));
    const repo=state.siteConfig?.repository;const base=repo?`${repo.repo_url}/blob/${repo.branch}/`:'';
    put('a2-source-links',base?`Nguồn dữ liệu báo cáo: <a href="${esc(base+'artifacts/A2/full/dcgan_baseline/metrics.json')}">baseline metrics</a> · <a href="${esc(base+'artifacts/A2/full/dcgan_diffaugment/metrics.json')}">DiffAugment metrics</a> · <a href="${esc(base+'artifacts/A2/full/dcgan_baseline/history.csv')}">baseline history</a> · <a href="${esc(base+'artifacts/A2/full/dcgan_diffaugment/history.csv')}">DiffAugment history</a>. Full histories được giữ trong bundle nhẹ.`:'');
  };
})();
