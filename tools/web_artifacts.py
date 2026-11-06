"""Checkpoint-free website bundles, verified against original experiment exports.

Pack: python tools/web_artifacts.py pack
Check: python tools/web_artifacts.py check --artifacts-dir artifacts
Only stdlib; never imports ML runtimes or unpickles a checkpoint.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'cifar100_resnet18_deit_v1'
FAMILIES = ('resnet18', 'deit_tiny')
STRATEGIES = ('scratch', 'head_only', 'partial', 'full')
MAIN = [f'{f}_{s}' for f in FAMILIES for s in STRATEGIES]
CONTROLS = [f'{f}_scratch_noaug' for f in FAMILIES]
A1_IDS = MAIN + CONTROLS
A2_IDS = ('dcgan_baseline', 'dcgan_diffaugment')
RUN_FILES = ('metrics.json', 'config.json', 'environment.json', 'history.csv', 'benchmark.json',
             'trained.json', 'status.json', 'curves.png', 'confusion_matrix.png', 'failures.png',
             'confusion_matrix.json', 'class_metrics.csv', 'top_confusions.json', 'failures.json',
             'metric_protocol.json', 'evaluation_status.json', 'benchmark_status.json', 'kid_subsets.json',
             'samples.png', 'interpolation.png', 'nearest_neighbors.png')
A1_FILES = ('manifest.json', 'study_identity.json', 'completion.json', 'main_summary.csv',
            'augmentation_ablation.csv', 'augmentation_delta.png', 'comparison.png', 'duplicate_audit.json',
            'eda_all_classes.png', 'eda_color.png', 'eda_counts.csv', 'eda_decisions.md',
            'eda_distribution.png', 'eda_preprocessing.png', 'eda_statistics.json', 'eda_superclasses.png', 'report.md')
PRIVATE = {'root','data_root','output_root','cache_root','resume_from','pretrained_dir','checkpoint_input','input_root'}


def read(path): return json.loads(Path(path).read_text())
def digest(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()
def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def require(condition, reason):
    if not condition: raise ValueError(reason)
def rows(path):
    with Path(path).open() as f:
        result = []
        for row in csv.DictReader(f):
            for key,value in row.items():
                try: row[key] = float(value) if any(x in value.lower() for x in ('.','e')) else int(value)
                except (ValueError, AttributeError): pass
            result.append(row)
        return result

def public(value):
    if isinstance(value, dict): return {k: public(v) for k,v in value.items() if k not in PRIVATE}
    if isinstance(value, list): return [public(v) for v in value]
    if isinstance(value, str) and value.startswith(('/kaggle/', '/home/', '/tmp/', '/content/')): return '[local path omitted]'
    return value


def int_array(archive, name):
    data = archive.read(name+'.npy'); require(data[:6] == b'\x93NUMPY', 'Invalid NPY')
    start = 10 if data[6] == 1 else 12
    length = struct.unpack('<H' if start == 10 else '<I', data[8:start])[0]
    meta = ast.literal_eval(data[start:start+length].decode())
    require(meta['descr'] in ('<i8','<i4') and len(meta['shape']) == 1, 'Expected integer vector')
    return struct.unpack('<'+('q' if meta['descr']=='<i8' else 'i')*meta['shape'][0], data[start+length:])


def validate_a1_metadata(root):
    manifest = read(root/'manifest.json'); study = digest(read(root/'study_identity.json'))
    require(manifest['dataset']=='CIFAR-100' and manifest['sizes']==dict(train=45000,val=5000,test=10000), 'Wrong A1 dataset/sizes')
    require(len(manifest['classes'])==100, 'Expected100 classes')
    require(not set(manifest['indices']['train']) & set(manifest['indices']['val']), 'Split overlap')
    budgets = set()
    for e in A1_IDS:
        d=root/'full'/e
        c,m,t,b,s=(read(d/f'{name}.json') for name in ('config','metrics','trained','benchmark','status'))
        require(c['protocol']==PROTOCOL and c['run_mode']==m['run_mode']=='full', f'{e}: wrong protocol/mode')
        require(c['experiment_id']==e and m['experiment_id']==e, f'{e}: wrong ID')
        require(c['study_fingerprint']==study and c['manifest_hash']==digest(manifest), f'{e}: mixed study')
        require(c['fingerprint']==digest({k:v for k,v in c.items() if k!='fingerprint'}), f'{e}: invalid config fingerprint')
        require(all(o['fingerprint']==c['fingerprint'] for o in (m,t,b,s)), f'{e}: fingerprint mismatch')
        epochs=c['recipe']['epochs'];budgets.add(epochs)
        require(epochs==m['epochs_completed']==t['epochs_completed'] and epochs>0, f'{e}: incomplete budget')
        require(s['state']=='complete' and m['evaluated_sizes']==manifest['sizes'], f'{e}: incomplete evaluation')
        require(m['checkpoint_sha256']==b['checkpoint_sha256']==t['checkpoint_sha256'], f'{e}: checkpoint records differ')
        hist=rows(d/'history.csv')
        require([r['epoch'] for r in hist]==list(range(epochs)), f'{e}: incomplete history')
        require(all(r['optimizer_steps']>0 for r in hist), f'{e}: epoch without updates')
        for key in ('train_loss','val_loss','train_accuracy','val_accuracy','training_seconds'):
            require(all(math.isfinite(r[key]) for r in hist), f'{e}: nonfinite history')
        for key in ('train_accuracy','val_accuracy','test_accuracy','macro_f1'):
            require(isinstance(m[key],(int,float)) and math.isfinite(m[key]) and 0<=m[key]<=1, f'{e}: invalid {key}')
        for n in (1,32):
            row=b['measurements'][str(n)]
            require(row['device']=='cuda:0' and row['precision']=='fp32' and row['batch_size']==n and row['warmup']==10 and row['repeats']==50, f'{e}: wrong benchmark')
            require(all(math.isfinite(row[k]) and row[k]>0 for k in ('latency_ms_per_image','images_per_second','peak_allocated_memory_mb')), f'{e}: invalid benchmark values')
        require(b['measurements']['1']['device_name']==b['measurements']['32']['device_name'], f'{e}: benchmark devices differ')
        for name in ('curves.png','confusion_matrix.png','failures.png','confusion_matrix.json','class_metrics.csv','top_confusions.json','failures.json'):
            require((d/name).is_file(), f'{e}: missing {name}')
        require(len(rows(d/'class_metrics.csv'))==100, f'{e}: incomplete class metrics')
    for delta in rows(root/'augmentation_ablation.csv'):
        family=delta['family']
        augmented=read(root/'full'/(family+'_scratch')/'metrics.json')
        baseline=read(root/'full'/(family+'_scratch_noaug')/'metrics.json')
        for key in ('val_accuracy','test_accuracy','macro_f1','generalization_gap'):
            require(math.isclose(delta['delta_'+key],augmented[key]-baseline[key],abs_tol=1e-12),'Incorrect augmentation delta')
    require(len(budgets)==1,'A1 budgets differ')
    return dict(study_fingerprint=study, epochs=budgets.pop(), manifest_hash=digest(manifest))


def validate_original_a1(source):
    # Same logical layout as the packaged bundle, without changing originals.
    with tempfile.TemporaryDirectory() as temp:
        proxy=Path(temp)
        for name in A1_FILES:
            if (source/name).is_file(): (proxy/name).symlink_to((source/name).resolve())
        (proxy/'full').symlink_to(source.resolve(),target_is_directory=True)
        identity=validate_a1_metadata(proxy)
    for e in A1_IDS:
        d=source/e;m=read(d/'metrics.json');c=read(d/'config.json');manifest=read(source/'manifest.json')
        require(sha(d/'best.pt')==m['checkpoint_sha256'],f'{e}: checkpoint bytes mismatch')
        for split in ('train','val','test'):
            with zipfile.ZipFile(d/f'{split}_predictions.npz') as z:
                y,p,indices=(int_array(z,key) for key in ('y_true','y_pred','indices'))
            require(len(y)==len(p)==manifest['sizes'][split] and list(indices)==manifest['indices'][split], f'{e}: bad prediction sizes/order')
            cm=[[0]*100 for _ in range(100)]
            for a,b in zip(y,p):
                require(0<=a<100 and 0<=b<100, f'{e}: bad prediction label');cm[a][b]+=1
            acc=sum(cm[k][k] for k in range(100))/len(y)
            f1=sum(2*cm[k][k]/den if (den:=sum(cm[k])+sum(row[k] for row in cm)) else 0 for k in range(100))/100
            require(math.isclose(acc,m['scores'][split]['accuracy'],abs_tol=1e-12) and math.isclose(f1,m['scores'][split]['macro_f1'],abs_tol=1e-12), f'{e}: predictions disagree with metrics')
            if split=='test':require(cm==read(d/'confusion_matrix.json')['counts'],f'{e}: confusion mismatch')
    return identity


def validate_bundle(root):
    root=Path(root);export=read(root/'export_manifest.json')
    require(export['schema']==1 and export['source_validation']['passed'] is True,'Missing source validation')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='export_manifest.json'}
    require(actual==set(export['files']), 'Bundle inventory differs')
    for name,expected in export['files'].items():
        p=root/name
        require(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'Unsafe bundle path')
        require(p.suffix in ('.json','.csv','.png','.md'),f'Unexpected public file: {name}')
        require(sha(p)==expected,f'Bundle hash mismatch: {name}')
        if p.suffix=='.png':
            blob=p.read_bytes();require(blob[:8]==b'\x89PNG\r\n\x1a\n',f'Invalid PNG: {name}')
            pos=8
            while pos<len(blob):
                length=struct.unpack('>I',blob[pos:pos+4])[0]
                chunk=blob[pos+4:pos+8+length]
                crc=struct.unpack('>I',blob[pos+8+length:pos+12+length])[0]
                require(zlib.crc32(chunk)&0xffffffff==crc,f'Corrupt PNG: {name}')
                pos+=12+length
    if export['assignment']=='A1':
        info=validate_a1_metadata(root)
        require(info==export['identity'],'A1 export identity differs')
    elif export['assignment']=='A2':
        from a2_web_export import inspect_export, EXECUTED
        require(inspect_export(root,ROOT/EXECUTED)==export['identity'],'A2 provenance differs')
        for e in A2_IDS:
            for name in ('curves.png','samples.png','interpolation.png','nearest_neighbors.png','kid_subsets.json'):
                require((root/'full'/e/name).is_file(),f'A2 {e}: missing {name}')
    else: raise ValueError('Unknown assignment')
    return export


def pack(a1, a2, destination):
    from a2_web_export import inspect_export, EXECUTED
    a1,a2,destination=Path(a1),Path(a2),Path(destination)
    identity1=validate_original_a1(a1)
    identity2=inspect_export(a2,ROOT/EXECUTED)
    for e in A2_IDS:
        d=a2/'full'/e
        require(sha(d/'last.pt')==read(d/'metrics.json')['checkpoint_sha256'],f'A2 {e}: checkpoint bytes mismatch')
    destination.mkdir(parents=True,exist_ok=True)
    # Stage and verify both packages before replacing either canonical bundle.
    with tempfile.TemporaryDirectory(dir=destination,prefix='.web-stage-') as temp:
        stage=Path(temp)
        for assignment,source,ids,identity in [('A1',a1,A1_IDS,identity1),('A2',a2,A2_IDS,identity2)]:
            target=stage/assignment;target.mkdir()
            names=A1_FILES if assignment=='A1' else ('manifest.json',)
            for name in names:
                if (source/name).is_file():shutil.copy2(source/name,target/name)
            for e in ids:
                src=source/e if assignment=='A1' else source/'full'/e
                dst=target/'full'/e;dst.mkdir(parents=True)
                for name in RUN_FILES:
                    if (src/name).is_file():
                        if assignment=='A2' and name in ('config.json','environment.json'):
                            write(dst/name,public(read(src/name)))
                        else:shutil.copy2(src/name,dst/name)
            # CIFAR configs remain intact for full-config fingerprint verification.
            # A2 omits local paths outside its provenance fingerprint inputs.
            inventory={p.relative_to(target).as_posix():sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
            write(target/'export_manifest.json',dict(schema=1,assignment=assignment,identity=identity,
                source_validation=dict(passed=True,checkpoint_hashes_verified=True,
                   predictions_verified=assignment=='A1',note='Verified at packaging from original export; checkpoints/predictions intentionally omitted. Bundle hashes are rechecked on import.'),files=inventory))
            validate_bundle(target)
        for assignment in ('A1','A2'):
            target=destination/assignment
            # Never erase non-bundle outputs accidentally.
            require(not target.exists() or (target/'export_manifest.json').is_file(),f'{target}: existing non-bundle folder; choose an empty destination')
        for assignment in ('A1','A2'):
            target=destination/assignment
            if target.exists():shutil.rmtree(target)
            shutil.move(str(stage/assignment),target)
    for assignment in ('A1','A2'):print('PACKED',destination/assignment)


def a1_snapshot(root):
    root=Path(root);export=validate_bundle(root);manifest=read(root/'manifest.json')
    experiments=[];controls=[]
    for e in A1_IDS:
        d=root/'full'/e;c=read(d/'config.json');m=read(d/'metrics.json');bench=read(d/'benchmark.json')['measurements']
        metrics={**m, 'epoch_seconds':m['training_seconds']/m['epochs_completed'],
                 'peak_gpu_memory_mb':m['peak_training_memory_mb'], 'latency_ms_per_image':bench['32']['latency_ms_per_image'],
                 'val_loss':m['scores']['val']['loss'],'test_loss':m['scores']['test']['loss']}
        item=dict(id=e,family=c['family'],strategy=c['strategy'],has_full_metrics=True,status='complete',
                  metrics=metrics,config=public(c),environment=public(read(d/'environment.json')),history=rows(d/'history.csv'),
                  benchmark=read(d/'benchmark.json'),class_metrics=rows(d/'class_metrics.csv'),
                  confusion={**read(d/'confusion_matrix.json'),'top_confusions':read(d/'top_confusions.json')},
                  failures=read(d/'failures.json'),images={name:f'assets/artifacts/A1/{e}/{name}.png' for name in ('curves','confusion_matrix','failures')})
        (experiments if e in MAIN else controls).append(item)
    eda={name.removesuffix('.png'):f'assets/artifacts/A1/eda/{name}' for name in A1_FILES if name.startswith('eda_') and name.endswith('.png')}
    return dict(title='A1 — ResNet-18 và DeiT-Tiny trên CIFAR-100',protocol=PROTOCOL,
                notebook='A1/a1_cifar100_resnet18_vs_deit.ipynb',executed_notebook='A1/resnet18-vs-deit-tiny-on-cifar100.ipynb',
                dataset='CIFAR-100 · 100 lớp · ảnh RGB32×32',split='45.000 train /5.000 validation /10.000 official test; seed42',
                epochs=export['identity']['epochs'],seed=42,expected_count=8,completed_full_count=8,all_full_completed=True,
                experiments=experiments,augmentation=dict(completed_count=2,controls=controls,deltas=rows(root/'augmentation_ablation.csv'),
                image='assets/artifacts/A1/eda/augmentation_delta.png'),
                eda=dict(images=eda,statistics=read(root/'eda_statistics.json'),counts=rows(root/'eda_counts.csv'),
                         duplicates=read(root/'duplicate_audit.json'),decisions=(root/'eda_decisions.md').read_text()),
                provenance=export['source_validation'],study_fingerprint=export['identity']['study_fingerprint'])


def sync_bundles(artifacts_root, docs):
    """Import prevalidated bundles while preserving other published assignments."""
    artifacts_root,docs=Path(artifacts_root),Path(docs)
    bundles={a:validate_bundle(artifacts_root/a) for a in ('A1','A2') if (artifacts_root/a/'export_manifest.json').exists()}
    require(bool(bundles),'No website bundles found')
    previous=read(docs/'data/experiments_manifest.json');result=json.loads(json.dumps(previous))
    # Construct everything before writing site files.
    if 'A1' in bundles:result['assignments']['A1']=a1_snapshot(artifacts_root/'A1')
    if 'A2' in bundles:
        from a2_protocol import SPEC
        from a2_web_export import public_history
        root=artifacts_root/'A2';experiments=[]
        for spec in SPEC['experiments']:
            d=root/'full'/spec['id'];old=next((e for e in previous['assignments']['A2']['experiments'] if e['id']==spec['id']),{})
            metrics=read(d/'metrics.json')
            if old.get('has_full_metrics'):require(old['metrics']==metrics,'Refusing to replace existing A2 metric values')
            hist=rows(d/'history.csv')
            experiments.append(dict(**spec,has_full_metrics=True,metrics=metrics,config=public(read(d/'config.json')),
                environment=public(read(d/'environment.json')),history=public_history(hist),history_points=len(hist),
                history_aggregation='Mean over consecutive 50-update windows',benchmark=read(d/'benchmark.json'),
                metric_protocol=read(d/'metric_protocol.json'),images={key:f'assets/artifacts/A2/{spec["id"]}/{key}.png' for key in ('curves','samples','interpolation','nearest_neighbors')}))
        result['assignments']['A2']={**{k:v for k,v in SPEC.items() if k!='experiments'},**bundles['A2']['identity'],
            'expected_count':2,'completed_full_count':2,'all_full_completed':True,'experiments':experiments}
    for assignment in bundles:
        target=docs/'assets/artifacts'/assignment
        if target.exists():shutil.rmtree(target)
        target.mkdir(parents=True)
        root=artifacts_root/assignment
        for p in sorted((root/'full').glob('*/*.png')):
            dst=target/p.parent.name/p.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(p,dst)
        if assignment=='A1':
            folder=target/'eda';folder.mkdir()
            for p in root.glob('*.png'):shutil.copy2(p,folder/p.name)
    result['total_completed_full_experiments']=sum(a['completed_full_count'] for a in result['assignments'].values())
    if previous.get('web_bundle_schema') != 1 or result['assignments'] != previous['assignments']:
        import datetime
        result['generated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['web_bundle_schema']=1
    result['artifact_source']='Validated lightweight A1/A2 bundles; original checkpoint validation recorded at packaging'
    result['conventions']['cifar100_split']='45,000 train /5,000 val /10,000 official test, seed42'
    result['conventions'].pop('food101_split',None)
    # PDF availability follows the validated build record, independent of A1 import.
    from report_publication import NAMES, report_status
    for filename in NAMES:
        result.setdefault('reports', {})[filename] = report_status(docs.parent, filename)
        if result['reports'][filename]['available']:
            (docs/'reports').mkdir(exist_ok=True)
            shutil.copy2(docs.parent/'reports'/filename, docs/'reports'/filename)
    write(docs/'data/experiments_manifest.json',result)
    cfg=read(docs/'data/site_config.json')
    (docs/'assets/js/fallback_data.js').write_text('// Generated by tools/sync_site_data.py\nwindow.FALLBACK_SITE_CONFIG = '+json.dumps(cfg,ensure_ascii=False)+';\nwindow.FALLBACK_EXPERIMENTS_MANIFEST = '+json.dumps(result,ensure_ascii=False)+';\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('pack','check'))
    parser.add_argument('--a1',type=Path,default=ROOT/'A1/A1_full');parser.add_argument('--a2',type=Path,default=ROOT/'A2/A2_full')
    parser.add_argument('--artifacts-dir',type=Path,default=ROOT/'artifacts');args=parser.parse_args()
    if args.action=='pack':pack(args.a1,args.a2,args.artifacts_dir)
    else:
        for a in ('A1','A2'):validate_bundle(args.artifacts_dir/a);print('VALID',a)


if __name__=='__main__':main()
