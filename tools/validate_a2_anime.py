#!/usr/bin/env python3
"""Dependency-free validation of the anime notebook and its generated worker."""
import ast
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def validate():
    payload=json.loads((ROOT/'A2/a2_dcgan_diffaugment.ipynb').read_text())
    fragments=[]; sections=[]
    for i,cell in enumerate(payload['cells']):
        source=''.join(cell['source'])
        if cell['cell_type']=='markdown':
            sections.append(source)
            continue
        assert i and payload['cells'][i-1]['cell_type']=='markdown'
        description=''.join(payload['cells'][i-1]['source'])
        assert all(term in description for term in ('Purpose:', 'Inputs:', 'Outputs:'))
        if source.startswith('%%writefile'):
            source=source.split('\n',1)[1]; fragments.append(source)
        ast.parse(source,filename=f'A2:cell-{i+1}')
        assert not cell['outputs'] and cell['execution_count'] is None
    assert len(fragments)==6
    worker='\n'.join(fragments); ast.parse(worker)
    assert 'diffusers' not in worker and 'DDPMScheduler' not in worker
    author=worker[worker.index('# Differentiable Augmentation for Data-Efficient GAN Training'):worker.index('# ---- Controlled experiment')].rstrip()+'\n'
    assert hashlib.sha256(author.encode()).hexdigest()=='d61fed24900fea3be17e3cd10d8880f895905732fd53f1f9b3013673b36fa0e2', 'Author DiffAugment source was modified'
    combined=worker+'\n'.join(sections)+'\n'.join(''.join(c['source']) for c in payload['cells'])
    for term in ('CUDA_VISIBLE_DEVICES','heartbeat_seconds','fingerprint','restore_rng','DiffAugment','binary_cross_entropy_with_logits'):
        assert term in combined, term
    for section in range(1,8): assert f'## {section}.' in '\n'.join(sections)
    print(f'OK anime A2: {len(payload["cells"])} cells; 6 readable worker fragments; 7 sections; Python syntax valid')

if __name__=='__main__': validate()
