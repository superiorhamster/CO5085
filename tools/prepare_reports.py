"""Prepare report tables/figures from measured artifacts; stdlib, no ML imports."""
from __future__ import annotations
import base64
import csv
import hashlib
import json
import re
from pathlib import Path
import shutil
from collections import Counter
from web_artifacts import validate_bundle, A1_IDS, MAIN, A2_IDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/latex/generated'
INPUTS: dict[str,str] = {}
MEASUREMENTS: dict[str,dict] = {}
NEEDED_IMAGES = set(re.findall(r'\{([^{}]+\.png)\}', '\n'.join(p.read_text() for p in (ROOT/'reports/latex').glob('*/main.tex'))))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def track(p):
    p=Path(p); INPUTS[p.relative_to(ROOT).as_posix()]=sha(p); return p

def read(p): return json.loads(track(p).read_text())
def rows(p):
    with track(p).open() as handle: return list(csv.DictReader(handle))
def esc(s):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in str(s))
def write(name, s): (OUT/name).write_text(s,encoding='utf-8')
def table(name, headers, data, caption, label, spec=None):
    spec=spec or ('l'+'r'*(len(headers)-1))
    s='\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{tab:'+label+'}\n'
    s+='\\begin{lrbox}{\\reporttable}\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
    s+='\n'.join(' & '.join(map(str,row))+r' \\' for row in data)
    s+='\n\\bottomrule\n\\end{tabular}\\end{lrbox}\n\\ifdim\\wd\\reporttable>\\linewidth\\resizebox{\\linewidth}{!}{\\usebox{\\reporttable}}\\else\\usebox{\\reporttable}\\fi\n\\end{table}\n'
    write(name+'.tex',s)
def copy(p,name):
    if name in NEEDED_IMAGES: shutil.copy2(track(p),OUT/name)
def extract_image(notebook,cell,index,name):
    d=read(ROOT/notebook); b=d['cells'][cell]['outputs'][index]['data']['image/png']
    (OUT/name).write_bytes(base64.b64decode(''.join(b)))
def pct(x): return f'{100*x:.2f}'
def f(x,n=3): return f'{x:.{n}f}'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for stale in OUT.glob('*.png'):
        if stale.name not in NEEDED_IMAGES: stale.unlink()
    validations={a:validate_bundle(ROOT/'artifacts'/a)['source_validation'] for a in ['A1','A2']}
    for a in ['A1','A2']: track(ROOT/'artifacts'/a/'export_manifest.json')
    for a,ids in [('E1',['softmax','mlp','cnn']),('E2',[f'{b}_{t}' for b in ['scratch','torch'] for t in ['patch','row','cnn']]),('E3',['lstm_row','lstm_patch','gru_row','gru_patch'])]:
        result=[];cost=[]
        for id in ids:
            p=ROOT/'artifacts'/a/'full'/id; m=read(p/'metrics.json'); c=read(p/'config.json');read(p/'environment.json')
            assert m['run_mode']==c['run_mode']=='full' and c['epochs']==20 and c['seed']==42
            assert c['split_hash']=='bbadd2fc92455ce7f7ca22890516a1f02c47a663679ec246709d7cf0cfdcb075'
            result.append([esc(id),pct(m['val_accuracy']),pct(m['test_accuracy']),f(m['test_loss']),str(m['best_epoch']+1),f"{m['parameters']:,}"])
            cost.append([esc(id),f(m['peak_gpu_memory_mb'],2),f(m['weight_size_mb'],3),f(m['latency_ms_per_image'],4)])
            MEASUREMENTS[a+'/'+id]=m
            for image in ['curves.png','confusion_matrix.png','failures.png']:
                if (p/image).exists():copy(p/image,a+'_'+id+'_'+image)
            if (p/'history.csv').exists():
                h=rows(p/'history.csv');assert len(h)==20
        table(a+'_results',['ID','Val (\\%)','Test (\\%)','Test loss','Best$^*$','Tham số'],result,a+': kết quả đo; best epoch đổi từ chỉ số 0 sang thứ tự 1--20.','results'+a)
        table(a+'_cost',['ID','VRAM train (MiB)','Weights (MiB)','ms/ảnh (B32)'],cost,a+': chi phí đo trên Tesla T4. Ký hiệu MB trong JSON được tính theo $2^{20}$ byte.','cost'+a)
    for a,notebook in [('E1','E1/e1_fashion_mnist_baselines.ipynb'),('E2','E2/e2_attention_tokenization.ipynb'),('E3','E3/e3_sequence_models.ipynb')]: read(ROOT/notebook)
    extract_image('E1/e1_fashion_mnist_baselines.ipynb',15,0,'fashion_samples.png')
    data=[];cost=[];bench=[]
    for id in A1_IDS:
        p=ROOT/'artifacts/A1/full'/id; m=read(p/'metrics.json');b=read(p/'benchmark.json');c=read(p/'config.json');h=rows(p/'history.csv')
        assert len(h)==25==m['epochs_completed']==c['recipe']['epochs']
        assert m['evaluated_sizes']=={'train':45000,'val':5000,'test':10000}
        best=min(h,key=lambda r:(-float(r['val_accuracy']),float(r['val_loss'])))
        assert m['best_epoch']==int(best['epoch'])+1 and 1<=m['best_epoch']<=25
        MEASUREMENTS['A1/'+id]=m
        if id in MAIN:
            short=id.replace('resnet18','R18').replace('deit_tiny','DeiT')
            data.append([esc(short),pct(m['train_accuracy']),pct(m['val_accuracy']),pct(m['test_accuracy']),pct(m['macro_f1']),str(m['best_epoch'])])
            cost.append([esc(short),f(m['parameters']/1e6,3),f(m['trainable_parameters']/1e6,3),f(m['training_seconds']/60,2),f(m['peak_training_memory_mb'],1)])
            for batch in ['1','32']:
                r=b['measurements'][batch]
                bench.append([esc(short),batch,f(r['latency_ms_per_image']),f(r['images_per_second'],1),f(r['peak_allocated_memory_mb'],1)])
        for image in ['curves.png','confusion_matrix.png','failures.png']:copy(p/image,'A1_'+id+'_'+image)
    table('A1_results',['ID','Train\\%','Val\\%','Test\\%','F1\\%','Best'],data,'Tám cấu hình chính A1; R18 = ResNet-18, DeiT = DeiT-Tiny; F1 là macro-F1 test.','A1results')
    table('A1_cost',['ID','Tổng (M)','Train (M)','Phút train','VRAM (MiB)'],cost,'Chi phí training khi hai queue chạy đồng thời; thời gian loại validation/checkpoint I/O.','A1cost')
    table('A1_benchmark',['ID','Batch','ms/ảnh','ảnh/s','MiB'],bench,'Benchmark FP32 tuần tự trên GPU0 Tesla T4, warmup 10, đo 50 lần.','A1bench')
    ab=[]
    for fam in ['resnet18','deit_tiny']:
        a=MEASUREMENTS['A1/'+fam+'_scratch'];n=MEASUREMENTS['A1/'+fam+'_scratch_noaug']
        for name,m in [('noaug',n),('aug',a)]:ab.append([esc(fam+' '+name),pct(m['val_accuracy']),pct(m['test_accuracy']),pct(m['macro_f1']),pct(m['generalization_gap'])])
        ab.append([esc(fam+' delta'),f(100*(a['val_accuracy']-n['val_accuracy']),2),f(100*(a['test_accuracy']-n['test_accuracy']),2),f(100*(a['macro_f1']-n['macro_f1']),2),f(100*(a['generalization_gap']-n['generalization_gap']),2)])
    table('A1_ablation',['Scratch','Val\\%','Test\\%','F1\\%','Gap (pp)'],ab,'Đối chứng augmentation. Delta = augmented trừ noaug, tính bằng điểm phần trăm; gap = train trừ validation.','A1ablation')
    for filename in ['eda_distribution.png','eda_all_classes.png','eda_color.png','eda_superclasses.png','eda_preprocessing.png','augmentation_delta.png','comparison.png']:copy(ROOT/'artifacts/A1'/filename,'A1_'+filename)
    for filename in ['manifest.json','eda_statistics.json','duplicate_audit.json','study_identity.json','augmentation_ablation.csv']:
        track(ROOT/'artifacts/A1'/filename)
    errors=[];classrows=[]
    for fam in ['resnet18','deit_tiny']:
        p=ROOT/'artifacts/A1/full'/(fam+'_full')
        for r in read(p/'top_confusions.json')[:6]: errors.append([esc(fam),esc(r['true']),esc(r['predicted']),r['count']])
        for r in sorted(rows(p/'class_metrics.csv'),key=lambda r:float(r['f1']))[:5]:classrows.append([esc(fam),esc(r['class_name']),pct(float(r['precision'])),pct(float(r['recall'])),pct(float(r['f1']))])
    table('A1_confusions',['Model','Nhãn thật','Dự đoán','Số ảnh'],errors,'Sáu cặp nhầm có hướng đứng đầu mỗi model full, trên test.','A1errors')
    table('A1_classes',['Model','Lớp','Precision\\%','Recall\\%','F1\\%'],classrows,'Năm lớp có F1 test thấp nhất mỗi model full; support mỗi lớp là 100.','A1classes')
    for nb in ['A1/a1_cifar100_resnet18_vs_deit.ipynb','A1/resnet18-vs-deit-tiny-on-cifar100.ipynb']: track(ROOT/nb)
    quality=[];cost=[];bench=[]
    for id in A2_IDS:
        p=ROOT/'artifacts/A2/full'/id;m=read(p/'metrics.json');b=read(p/'benchmark.json');c=read(p/'config.json');h=rows(p/'history.csv')
        assert m['steps']==20000 and len(h)==20000 and m['generated_samples']==5000
        MEASUREMENTS['A2/'+id]=m
        short='Baseline' if id=='dcgan_baseline' else '+ DiffAugment'
        quality.append([short,f(m['fid'],2),f(m['kid_mean'],5),f(m['kid_std'],5),f(m['inception_score_mean'],3),f(m['inception_score_split_std'],3)])
        cost.append([short,pct(m['generative_precision']),pct(m['generative_recall']),f(m['training_seconds'],2),f(m['peak_training_gpu_memory_mb'],2)])
        for n in ['1','64']:
            r=b[n];bench.append([short,n,f(r['latency_ms_per_image']),f(r['images_per_second'],1),f(r['peak_gpu_memory_mb'],2)])
        for image in ['curves.png','samples.png','interpolation.png','nearest_neighbors.png']:copy(p/image,'A2_'+id+'_'+image)
        for filename in ['metric_protocol.json','environment.json','kid_subsets.json']:track(p/filename)
    table('A2_quality',['Biến thể','FID$\\downarrow$','KID$\\downarrow$','KID std','IS$\\uparrow$','IS std'],quality,'Đánh giá final EMA; KID ở thang gốc (không nhân 100 hay 1000).','A2quality')
    table('A2_cost',['Biến thể','Precision\\%','Recall\\%','Train (s)','VRAM (MiB)'],cost,'Độ phủ trong không gian Inception và chi phí training.','A2cost')
    table('A2_bench',['Biến thể','Batch','ms/ảnh','ảnh/s','MiB'],bench,'Benchmark generator EMA FP32, 30 lần đo trên Tesla T4.','A2bench')
    m=read(ROOT/'artifacts/A2/manifest.json');counts=Counter(r['split'] for r in m['records'])
    assert counts=={'train':34522,'val':2000,'test':5000}
    assert len(m['duplicates'])==22043 and not m['corrupt']
    sizes=Counter((r['width'],r['height']) for r in m['records'])
    table('A2_eda',['Thuộc tính','Giá trị'],[['Tệp đã kiểm tra','63,565'],['Trùng hash pixel đã loại','22,043'],['Tệp lỗi','0'],['Ảnh duy nhất','41,522'],['Train / val / test','34,522 / 2,000 / 5,000'],['Chiều rộng min--max',f"{min(r['width'] for r in m['records'])}--{max(r['width'] for r in m['records'])}"],['Chiều cao min--max',f"{min(r['height'] for r in m['records'])}--{max(r['height'] for r in m['records'])}"]],'EDA từ manifest: dedup theo SHA-256 (shape + pixel RGB) trước khi split, không phải kiểm tra near-duplicate.','A2eda','lr')
    extract_image('A2/dcgan-vs-augmenteddcgan-dl-assignment-2.ipynb',22,1,'anime_train_grid.png')
    track(ROOT/'A2/a2_dcgan_diffaugment.ipynb')
    for name in ['a1_cifar100_train.py','a1_cifar100_resnet.py','a1_cifar100_deit.py','a1_cifar100_data.py','a2_dcgan_runtime.py','build_notebooks.py']:
        if (ROOT/'tools'/name).exists():track(ROOT/'tools'/name)
    for name in ['e1_e3_co5085.md','a1_co5085.md','a2_co5085.md','HCMUT_RP_template/SoftEng_252/main.tex','HCMUT_RP_template/SoftEng_252/hcmut.png']:track(ROOT/name)
    record={'schema':1,'inputs':dict(sorted(INPUTS.items())),'source_validation':validations,'measurements':MEASUREMENTS,'note':'A1/A2 bundle hashes rechecked; checkpoint validation is recorded from original packaging, not rechecked from absent checkpoint bytes.'}
    write('provenance.json',json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(f'Prepared {len(list(OUT.iterdir()))} report inputs from {len(INPUTS)} sources; no training.')

if __name__=='__main__': prepare()
