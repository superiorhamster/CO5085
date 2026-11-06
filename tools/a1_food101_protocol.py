"""Food-101 publishing contract, deliberately independent of ML dependencies."""
import csv
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = 'food101_resnet18_swin_v1'
NOTEBOOK = 'A1/a1_food101_resnet18_vs_swin.ipynb'
MODELS = {'resnet18': 'resnet18.a1_in1k', 'swin_tiny': 'swin_tiny_patch4_window7_224.ms_in1k'}
STRATEGIES = ('scratch', 'head_only', 'partial', 'full')
EXPERIMENT_IDS = tuple(f'{f}_{s}' for f in MODELS for s in STRATEGIES)
SIZES = {'train': 68175, 'val': 7575, 'test': 25250}
RECIPE = dict(epochs=20, image_size=224, microbatch=32, accumulation=2,
              effective_batch=64, warmup_epochs=2, scratch_lr=3e-4,
              scratch_weight_decay=.05, backbone_lr=1e-4, head_lr=1e-3,
              pretrained_weight_decay=.01, grad_clip=1., dropout=0., drop_path=0.,
              train_crop_scale=[.5, 1.], eval_resize=256, interpolation='bicubic',
              mean=[.485,.456,.406], std=[.229,.224,.225], loss='cross_entropy')
REQUIRED = ('val_accuracy','test_accuracy','macro_f1','parameters','trainable_parameters',
            'training_seconds','epoch_seconds','peak_gpu_memory_mb','val_loss','test_loss')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fingerprint(config):
    return digest({k: config[k] for k in ('protocol','experiment_id','model_name','strategy',
                  'seed','run_mode','manifest_hash','split_hash','recipe','versions','synthetic','pretrained_sha256')})


def study_identity(config):
    """Runs compared together must share data, split, recipe and library versions."""
    return digest({k: config[k] for k in ('protocol','seed','manifest_hash','split_hash','recipe','versions','synthetic')})


def valid_run(path, experiment_id):
    """Only a complete, internally consistent official 20-epoch run is publishable."""
    try:
        path = Path(path)
        def read(name): return json.loads((path / (name + '.json')).read_text())
        c,m,t,e,b = map(read, ('config','metrics','trained','evaluation_status','benchmark'))
        bs = read('benchmark_status')
        family = next(f for f in MODELS if experiment_id.startswith(f + '_'))
        strategy = experiment_id[len(family)+1:]
        if experiment_id not in EXPERIMENT_IDS or c.get('experiment_id') != experiment_id: return False
        if c.get('assignment') != 'A1' or c.get('protocol') != PROTOCOL or c.get('run_mode') != 'full': return False
        if c.get('synthetic') or c.get('seed') != 42 or c.get('recipe') != RECIPE: return False
        if c.get('model_name') != MODELS[family] or c.get('strategy') != strategy: return False
        if c.get('versions',{}).get('timm') != '1.0.22' or c.get('sizes') != SIZES: return False
        fp = fingerprint(c)
        if any(x.get('fingerprint') != fp for x in (c,m,t,e,bs)): return False
        if t.get('epochs_completed') != 20 or m.get('epochs_completed') != 20: return False
        if m.get('run_mode') != 'full' or e.get('state') != 'complete' or bs.get('state') != 'complete': return False
        if not m.get('checkpoint_sha256') or e.get('checkpoint_sha256') != m['checkpoint_sha256']: return False
        if any(not isinstance(m.get(k),(int,float)) or isinstance(m[k],bool) or not math.isfinite(m[k]) for k in REQUIRED): return False
        if any(not 0 <= m[k] <= 1 for k in ('val_accuracy','test_accuracy','macro_f1')): return False
        if not 0 < m['trainable_parameters'] <= m['parameters']: return False
        if any(m[k] < 0 for k in ('training_seconds','epoch_seconds','peak_gpu_memory_mb','val_loss','test_loss')): return False
        if m.get('evaluated_sizes') != {'val':7575,'test':25250}: return False
        if not 0 <= m.get('best_epoch',-1) < 20: return False
        with (path/'history.csv').open() as handle: rows=list(csv.DictReader(handle))
        if [int(r['epoch']) for r in rows] != list(range(20)): return False
        if any(not math.isfinite(float(r[k])) for r in rows for k in ('train_loss','val_loss','train_accuracy','val_accuracy','epoch_seconds')): return False
        for n in (1,32):
            row=b[str(n)]
            if row.get('fingerprint') != fp or row.get('checkpoint_sha256') != m['checkpoint_sha256']: return False
            if row.get('precision') != 'FP32' or row.get('physical_gpu') != 0 or 'T4' not in row.get('device',''): return False
            if row.get('warmup') != 10 or row.get('repeats') != 50: return False
            if any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) or row[k]<=0 for k in ('latency_ms_per_image','images_per_second')): return False
        if b['1']['device'] != b['32']['device']: return False
        for name in ('curves.png','confusion_matrix.png','failures.png','confusion_matrix.json','failures.json','class_metrics.json','run_manifest.json'):
            if not (path/name).is_file(): return False
        manifest=read('run_manifest')
        if digest(manifest) != c['manifest_hash'] or manifest.get('split_hash') != c['split_hash']: return False
        return True
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


LABELS={'scratch':'Scratch','head_only':'Head-only','partial':'Partial','full':'Full fine-tune'}
SPEC = dict(title='A1 — ResNet-18 và Swin-Tiny trên Food-101', notebook=NOTEBOOK,
            protocol=PROTOCOL, dataset='Food-101 · 101 lớp · ảnh món ăn thực tế',
            split='68.175 train / 7.575 validation / 25.250 test · seed42',
            optimizer='20 epoch · AdamW · effective batch64 · cosine + warmup2',
            invariants='Freeze weights/buffers; ResNet frozen BatchNorm stays eval.',
            experiments=[dict(id=f'{f}_{s}',family=f,model_id=MODELS[f],strategy=s,
                              strategy_label=LABELS[s],preview={},
                              unfrozen_layers=('Toàn bộ mạng' if s in ('scratch','full') else
                                ('fc' if f=='resnet18' else 'head.fc') if s=='head_only' else
                                ('layer4 + fc' if f=='resnet18' else 'layers.3 + norm + head')))
                         for f in MODELS for s in STRATEGIES])
