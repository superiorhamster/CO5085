"""Dependency-free regression tests for Git-ready bundles and site imports."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import web_artifacts as w


class WebBundleTests(unittest.TestCase):
    def test_committed_bundles_and_no_heavy_files(self):
        for assignment in ('A1','A2'):
            root=w.ROOT/'artifacts'/assignment
            record=w.validate_bundle(root)
            self.assertTrue(record['source_validation']['checkpoint_hashes_verified'])
            self.assertTrue(all(Path(name).suffix in ('.json','.csv','.png','.md') for name in record['files']))
        snapshot=w.a1_snapshot(w.ROOT/'artifacts/A1')
        self.assertEqual(snapshot['completed_full_count'],8)
        self.assertEqual(snapshot['augmentation']['completed_count'],2)
        self.assertEqual(snapshot['epochs'],25)
        self.assertEqual(snapshot['eda']['duplicates']['cross_split_groups'],12)
        self.assertAlmostEqual(snapshot['augmentation']['deltas'][0]['delta_test_accuracy'],.0717)

    def test_changed_missing_smoke_and_mixed_exports_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'A1';shutil.copytree(w.ROOT/'artifacts/A1',root)
            path=root/'full/resnet18_full/metrics.json';original=path.read_bytes()
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):w.validate_bundle(root)
            path.write_bytes(original)
            (root/'full/resnet18_full/failures.png').unlink()
            with self.assertRaisesRegex(ValueError,'inventory'):w.validate_bundle(root)
        for key,value in [('run_mode','smoke'),('study_fingerprint','other-study')]:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'A1';shutil.copytree(w.ROOT/'artifacts/A1',root)
                path=root/'full/resnet18_full/config.json';c=w.read(path);c[key]=value
                c['fingerprint']=w.digest({k:v for k,v in c.items() if k!='fingerprint'});w.write(path,c)
                export=w.read(root/'export_manifest.json');export['files'][path.relative_to(root).as_posix()]=w.sha(path)
                w.write(root/'export_manifest.json',export)
                with self.assertRaises(ValueError):w.validate_bundle(root)

    def test_import_is_idempotent_preserves_e1_e3_a2_and_no_local_paths(self):
        previous=w.read(w.ROOT/'docs/data/experiments_manifest.json')
        with tempfile.TemporaryDirectory() as temp:
            docs=Path(temp)/'docs';(docs/'data').mkdir(parents=True);(docs/'assets/js').mkdir(parents=True)
            for name in ('experiments_manifest.json','site_config.json'):shutil.copy2(w.ROOT/'docs/data'/name,docs/'data'/name)
            legacy=w.read(docs/'data/experiments_manifest.json')
            for assignment in legacy['assignments'].values():
                for experiment in assignment['experiments']:
                    experiment['preview']={'test_accuracy':.99999}
            w.write(docs/'data/experiments_manifest.json',legacy)
            first=w.sync_bundles(w.ROOT/'artifacts',docs)
            second=w.sync_bundles(w.ROOT/'artifacts',docs)
            self.assertEqual(first,second)
            for assignment in first['assignments'].values():
                for experiment in assignment['experiments']:
                    self.assertNotIn('preview',experiment)
            for assignment in ('E1','E2','E3'):self.assertEqual(first['assignments'][assignment],previous['assignments'][assignment])
            self.assertEqual([e['metrics'] for e in first['assignments']['A2']['experiments']],
                             [e['metrics'] for e in previous['assignments']['A2']['experiments']])
            self.assertEqual(first['total_completed_full_experiments'],23)
            for marker in ('/kaggle/','/home/','/tmp/'):
                self.assertNotIn(marker,json.dumps(first))
            self.assertTrue((docs/'assets/artifacts/A1/deit_tiny_full/confusion_matrix.png').exists())
            self.assertFalse(first['reports']['assignment1.pdf']['available'])

    def test_invalid_bundle_does_not_touch_site(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shutil.copytree(w.ROOT/'artifacts/A1',root/'artifacts/A1')
            (root/'artifacts/A1/full/deit_tiny_full/config.json').write_text('{}')
            docs=root/'docs';(docs/'data').mkdir(parents=True)
            target=docs/'data/experiments_manifest.json';target.write_text('unchanged')
            with self.assertRaises(ValueError):w.sync_bundles(root/'artifacts',docs)
            self.assertEqual(target.read_text(),'unchanged')


if __name__=='__main__':unittest.main()
