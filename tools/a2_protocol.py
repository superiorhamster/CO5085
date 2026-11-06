"""Lightweight website contract: no GPU dependencies or invented results."""
import json
import math
from pathlib import Path

PROTOCOL='dcgan_diffaugment_v1'
VARIANTS=('dcgan_baseline','dcgan_diffaugment')
REQUIRED=('fid','kid_mean','kid_std','inception_score_mean','inception_score_split_std','generative_precision','generative_recall','training_seconds','updates_per_second','real_images_per_second','peak_host_memory_mb','peak_training_gpu_memory_mb')
SPEC={
 'title':'A2 — DCGAN baseline và DCGAN + DiffAugment',
 'notebook':'A2/a2_dcgan_diffaugment.ipynb',
 'paper':{'title':'Differentiable Augmentation for Data-Efficient GAN Training','authors':'Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, Song Han','venue':'NeurIPS 2020','arxiv_url':'https://arxiv.org/abs/2006.10738','pdf_url':'https://arxiv.org/pdf/2006.10738'},
 'dataset':'Anime Face Dataset — RGB 64×64, dedup trước split',
 'split':'Toàn bộ train hợp lệ; validation 2.000 / test 5.000; seed42',
 'optimizer':'Adam 2e-4, betas (0.5,0.999), batch64,20.000 updates,final EMA0.999',
 'experiments':[{'id':v,'name':'DCGAN baseline' if v==VARIANTS[0] else 'DCGAN + DiffAugment','policy':'none' if v==VARIANTS[0] else 'color,translation,cutout','preview':{}} for v in VARIANTS]}

def valid_run(path,variant):
    try:
        def read(name):return json.loads((Path(path)/name).read_text())
        m,c,t,s,b=map(read,['metrics.json','config.json','trained.json','evaluation_status.json','benchmark.json'])
        bs=read('benchmark_status.json')
        if c.get('protocol')!=PROTOCOL or c.get('experiment_id')!=variant or c.get('assignment')!='A2' or c.get('seed')!=42:return False
        if c.get('run_mode')!='full' or m.get('run_mode')!='full' or m.get('steps')!=20000 or m.get('generated_samples')!=5000 or not m.get('completed_training'):return False
        e=c.get('evaluation',{}); training=c.get('training',{})
        if training.get('max_steps')!=20000 or c.get('effective_batch')!=64 or any(e.get(k)!=v for k,v in {'samples':5000,'kid_subsets':100,'kid_subset_size':1000,'is_splits':10,'knn_k':3}.items()):return False
        if not m.get('fingerprint') or not m.get('metric_protocol_hash') or not(m['fingerprint']==t.get('fingerprint')==s.get('fingerprint')) or s.get('state')!='complete':return False
        if bs.get('state')!='complete' or bs.get('fingerprint')!=m['fingerprint'] or not m.get('checkpoint_sha256'):return False
        if any(not isinstance(m.get(k),(int,float)) or not math.isfinite(m[k]) for k in REQUIRED):return False
        if any(str(n) not in b or b[str(n)].get('precision')!='FP32' or not b[str(n)].get('device') or any(not isinstance(b[str(n)].get(k),(int,float)) or not math.isfinite(b[str(n)][k]) or b[str(n)][k]<=0 for k in ('images_per_second','latency_ms_per_image')) for n in [1,64]):return False
        if any(b[str(n)].get('fingerprint')!=m['fingerprint'] or b[str(n)].get('checkpoint_sha256')!=m['checkpoint_sha256'] for n in [1,64]):return False
        if any(not 0<=m[k]<=1 for k in ('generative_precision','generative_recall')):return False
        protocol=read('metric_protocol.json')
        import hashlib
        if hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()!=m['metric_protocol_hash']:return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False
