"""Dependency-free checks for publishing new A2 artifacts without losing E1–A1."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
import a2_protocol as p
import sync_site_data as sync

class ContractTests(unittest.TestCase):
    def test_nested_paths_are_not_published(self):
        config={'seed':42,'data':{'root':'/home/user/images','cache_root':'/tmp/cache','val_size':2000},'resume_from':{'dcgan_baseline':'/home/user/last.pt'}}
        self.assertEqual(sync.public_config(config),{'seed':42,'data':{'val_size':2000}})

    def fixture(self,root,variant):
        run=root/'A2'/p.PROTOCOL/'full'/variant;run.mkdir(parents=True)
        config={'assignment':'A2','experiment_id':variant,'protocol':p.PROTOCOL,'seed':42,'run_mode':'full','effective_batch':64,'training':{'max_steps':20000},'evaluation':{'samples':5000,'kid_subsets':100,'kid_subset_size':1000,'is_splits':10,'knn_k':3}}
        protocol={'extractor':'fixture','manifest':'fixture','evaluation':config['evaluation']}
        metrics={k:1. for k in p.REQUIRED};metrics.update(run_mode='full',steps=20000,generated_samples=5000,completed_training=True,fingerprint='fixture',checkpoint_sha256='fixture_checkpoint',metric_protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest())
        benchmark={str(n):{'fingerprint':'fixture','checkpoint_sha256':'fixture_checkpoint','precision':'FP32','device':'Tesla T4','images_per_second':100.,'latency_ms_per_image':10.,'peak_gpu_memory_mb':100.} for n in [1,64]}
        for name,data in {'metrics':metrics,'config':config,'trained':{'fingerprint':'fixture'},'evaluation_status':{'state':'complete','fingerprint':'fixture'},'benchmark_status':{'state':'complete','fingerprint':'fixture'},'benchmark':benchmark,'metric_protocol':protocol}.items():
            (run/(name+'.json')).write_text(json.dumps(data))
        return run

    def test_incomplete_smoke_wrong_protocol_and_nan_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run=self.fixture(Path(temp),p.VARIANTS[0]);self.assertTrue(p.valid_run(run,p.VARIANTS[0]))
            original=json.loads((run/'metrics.json').read_text())
            for key,value in [('fid',float('nan')),('run_mode','smoke'),('steps',100),('generated_samples',16),('metric_protocol_hash','wrong')]:
                data={**original,key:value};(run/'metrics.json').write_text(json.dumps(data))
                self.assertFalse(p.valid_run(run,p.VARIANTS[0]),key)
            (run/'metrics.json').write_text(json.dumps(original));(run/'evaluation_status.json').write_text('{"state":"failed"}')
            self.assertFalse(p.valid_run(run,p.VARIANTS[0]))

    def test_a2_only_export_preserves_published_other_assignments(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'docs'/'data';data.mkdir(parents=True)
            before=json.loads((sync.ROOT/'docs/data/experiments_manifest.json').read_text())
            before['total_completed_full_experiments']-=before['assignments']['A2']['completed_full_count']
            before['assignments']['A2']['completed_full_count']=0
            before['assignments']['A2']['all_full_completed']=False
            for exp in before['assignments']['A2']['experiments']:
                exp.update(has_full_metrics=False,metrics=None,config=None,benchmark=None)
            (data/'experiments_manifest.json').write_text(json.dumps(before))
            for variant in p.VARIANTS:self.fixture(root/'input',variant)
            with patch.object(sync,'DATA_OUT_DIR',data),patch.object(sync,'DOCS_DIR',root/'docs'),patch.object(sync,'ARTIFACTS_OUT_DIR',root/'docs/assets/artifacts'),patch.object(sync,'ensure_reports',return_value=before['reports']):
                result=sync.sync_experiment_artifacts(root/'input')
            for name in ['E1','E2','E3','A1']:
                self.assertEqual(result['assignments'][name]['experiments'],before['assignments'][name]['experiments'])
            self.assertEqual(result['assignments']['A2']['completed_full_count'],2)
            self.assertEqual(result['total_completed_full_experiments'],before['total_completed_full_experiments']+2)
            import shutil
            shutil.rmtree(root/'input/A2'/p.PROTOCOL/'full'/p.VARIANTS[1])
            (data/'experiments_manifest.json').write_text(json.dumps(before))
            with patch.object(sync,'DATA_OUT_DIR',data),patch.object(sync,'DOCS_DIR',root/'docs'),patch.object(sync,'ARTIFACTS_OUT_DIR',root/'docs/assets/artifacts'),patch.object(sync,'ensure_reports',return_value=before['reports']):
                partial=sync.sync_experiment_artifacts(root/'input')
            self.assertEqual(partial['assignments']['A2']['completed_full_count'],1)

if __name__=='__main__':unittest.main()
