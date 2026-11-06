"""Exercise real export provenance without running notebook code or training."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
import a2_web_export as export
import sync_site_data as sync

ROOT=Path(__file__).resolve().parents[1]

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'export'
        shutil.copytree(ROOT/'artifacts/A2',self.root,ignore=shutil.ignore_patterns('*.png','*.log','*.jsonl','report.md'))
        self.notebook=ROOT/export.EXECUTED

    def test_real_eda_history_and_worker_hash(self):
        summary=export.inspect_export(self.root,self.notebook)
        self.assertEqual(summary['eda']['inspected'],63565)
        self.assertEqual(summary['eda']['splits'],{'train':34522,'val':2000,'test':5000})
        for key in ['widths','heights','aspect_ratios']:self.assertEqual(sum(summary['eda'][key]['counts']),41522)
        for name in ['dcgan_baseline','dcgan_diffaugment']:
            rows=sync.read_csv_rows(self.root/'full'/name/'history.csv');points=export.public_history(rows)
            self.assertEqual((len(points),points[-1]['step']),(400,20000))
            self.assertAlmostEqual(points[0]['g_loss'],sum(float(r['g_loss']) for r in rows[:50])/50)

    def test_rejects_missing_invalid_and_incompatible_runs(self):
        run=self.root/'full/dcgan_diffaugment'
        cases=[('metrics.json','fid',float('nan')),('metrics.json','steps',2),('config.json','seed',7),('evaluation_status.json','state','failed'),('benchmark_status.json','state','failed'),('metric_protocol.json','source_sha256','wrong')]
        for file,key,value in cases:
            p=run/file;original=p.read_text();data=json.loads(original);data[key]=value;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError,msg=f'{file}:{key}'):export.inspect_export(self.root,self.notebook)
            p.write_text(original)
        p=run/'benchmark.json';original=p.read_text();p.unlink()
        with self.assertRaises(ValueError):export.inspect_export(self.root,self.notebook)
        p.write_text(original)
        p=run/'metric_protocol.json';original=p.read_text();data=json.loads(original);data['evaluation']['knn_k']=4;p.write_text(json.dumps(data))
        mp=run/'metrics.json';metrics=json.loads(mp.read_text());metrics['metric_protocol_hash']=export.digest(data);mp.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError,'protocols differ'):export.inspect_export(self.root,self.notebook)

    def test_explicit_import_preserves_other_assignments(self):
        before=json.loads((ROOT/'docs/data/experiments_manifest.json').read_text());docs=Path(self.tmp.name)/'docs';data=docs/'data';data.mkdir(parents=True)
        (data/'experiments_manifest.json').write_text(json.dumps(before))
        with patch.object(sync,'DATA_OUT_DIR',data),patch.object(sync,'DOCS_DIR',docs),patch.object(sync,'ARTIFACTS_OUT_DIR',docs/'assets/artifacts'),patch.object(sync,'ensure_reports',return_value=before['reports']):
            actual=sync.sync_experiment_artifacts(Path(self.tmp.name)/'absent',a2_export=self.root)
        for name in ['E1','E2','E3','A1']:self.assertEqual(before['assignments'][name],actual['assignments'][name])
        self.assertEqual(actual['assignments']['A2']['completed_full_count'],2)
        self.assertEqual(actual['total_completed_full_experiments'],before['total_completed_full_experiments'])

if __name__=='__main__':unittest.main()
