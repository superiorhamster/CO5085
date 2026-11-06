"""Read-only verification and public summaries for an executed A2 export."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from a2_protocol import VARIANTS, valid_run

EXECUTED = 'A2/dcgan-vs-augmenteddcgan-dl-assignment-2.ipynb'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def worker_hash(notebook):
    nb=json.loads(Path(notebook).read_text())
    if any(o.get('output_type')=='error' for c in nb['cells'] for o in c.get('outputs',[])):
        raise ValueError('Executed A2 notebook contains error outputs')
    fragments=[]
    for cell in nb['cells']:
        source=''.join(cell['source'])
        if cell['cell_type']=='code' and source.startswith('%%writefile'):
            if cell.get('execution_count') is None: raise ValueError('A2 worker fragment was not executed')
            body=source.split('\n',1)[1]
            fragments.append(body if body.endswith('\n') else body+'\n')
    if len(fragments)!=6: raise ValueError('Expected six executed A2 worker fragments')
    return hashlib.sha256('\n'.join(fragments).encode()).hexdigest()

def bins(values,edges):
    counts=[0]*(len(edges)-1)
    for value in values:
        for i in range(len(counts)):
            if edges[i]<=value<edges[i+1]: counts[i]+=1;break
    labels=[f'{edges[i]:g}–{edges[i+1]:g}' for i in range(len(counts))]
    return {'labels':labels,'counts':counts}

def inspect_export(root,notebook):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());records=m['records']
    if digest({'records':records,'preprocessing':'rgb_fit_bilinear64_v1'})!=m['hash']:
        raise ValueError('A2 manifest hash mismatch')
    if len({r['sha256'] for r in records})!=len(records): raise ValueError('Duplicates cross A2 splits')
    splits=dict(Counter(r['split'] for r in records))
    if set(splits)!={'train','val','test'} or splits['val']!=2000 or splits['test']!=5000: raise ValueError('Unexpected A2 splits')
    source=worker_hash(notebook);protocols=set();settings=[]
    for name in VARIANTS:
        run=root/'full'/name
        if not valid_run(run,name): raise ValueError(f'Invalid full A2 run: {name}')
        protocol=json.loads((run/'metric_protocol.json').read_text())
        c=json.loads((run/'config.json').read_text());metrics=json.loads((run/'metrics.json').read_text())
        expected=digest({'protocol':c['protocol'],'variant':name,'model':c['model'],'training':c['training'],
          'augmentation':c['augmentation'],'batch':c['effective_batch'],'seed':c['seed'],'manifest':m['hash'],
          'mode':c['run_mode'],'source':source})
        if metrics['fingerprint']!=expected: raise ValueError(f'A2 config/source fingerprint mismatch: {name}')
        settings.append({k:c[k] for k in ('seed','model','training','effective_batch','augmentation')})
        if protocol.get('source_sha256')!=source or protocol.get('manifest')!=m['hash']:
            raise ValueError(f'A2 notebook/manifest provenance mismatch: {name}')
        protocols.add(digest(protocol))
        with (run/'history.csv').open() as f: rows=list(csv.DictReader(f))
        if [int(row['step']) for row in rows]!=list(range(1,20001)): raise ValueError(f'Incomplete A2 history: {name}')
        if any(not math.isfinite(float(row[k])) for row in rows for k in ('g_loss','d_loss')): raise ValueError('Nonfinite A2 loss history')
    if len(protocols)!=1: raise ValueError('A2 metric protocols differ')
    if settings[0]!=settings[1]: raise ValueError('A2 controlled training settings differ')
    widths=[r['width'] for r in records];heights=[r['height'] for r in records]
    edges=[0,64,96,128,192,256,max(512,max(widths+heights)+1)]
    ratios=[w/h for w,h in zip(widths,heights)]
    return {'executed_notebook':EXECUTED,'worker_sha256':source,'manifest_sha256':m['hash'],
        'metric_protocol_hash':next(iter(protocols)),
        'eda':{'unique':len(records),'duplicates':len(m['duplicates']),'corrupt':len(m['corrupt']),
          'inspected':len(records)+len(m['duplicates'])+len(m['corrupt']),'splits':splits,
          'widths':bins(widths,edges),'heights':bins(heights,edges),
          'aspect_ratios':bins(ratios,[0,.75,.9,1.01,1.1,1.5,max(2,max(ratios)+.01)])}}

def public_history(rows):
    """Mean losses in consecutive 50-update windows, retain exact final update."""
    result=[]
    for start in range(0,len(rows),50):
        part=rows[start:start+50]
        result.append({'step':int(part[-1]['step']),'g_loss':sum(float(r['g_loss']) for r in part)/len(part),
                       'd_loss':sum(float(r['d_loss']) for r in part)/len(part)})
    return result
