"""Publication regressions and numerical traceability; no TeX/ML runtime needed."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import report_publication as pub
from prepare_reports import esc, pct, f

ROOT=Path(__file__).resolve().parents[1]

class ReportsTests(unittest.TestCase):
    def test_published_pdfs_and_recorded_sources(self):
        result=pub.check_publication(ROOT)
        for name in pub.NAMES:
            self.assertTrue(result[name]['available'])
            self.assertFalse(result[name]['is_placeholder'])
            self.assertGreaterEqual(result[name]['pages'],15)
            self.assertLessEqual(result[name]['pages'],25)

    def test_missing_modified_pdf_and_stale_source_fail_closed(self):
        record=json.loads((ROOT/'reports/report_manifest.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertFalse(pub.report_status(root,'assignment1.pdf')['available'])
            for path in record['reports']['assignment1.pdf']['sources']:
                destination=root/path;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/path,destination)
            (root/'reports').mkdir(exist_ok=True)
            shutil.copy2(ROOT/'reports/report_manifest.json',root/'reports/report_manifest.json')
            pdf=root/'reports/assignment1.pdf';shutil.copy2(ROOT/'reports/assignment1.pdf',pdf)
            self.assertTrue(pub.report_status(root,'assignment1.pdf')['available'])
            original=pdf.read_bytes();pdf.write_bytes(original+b'changed')
            self.assertFalse(pub.report_status(root,'assignment1.pdf')['available'])
            pdf.write_bytes(original)
            (root/'reports/latex/assignment1/main.tex').write_text('stale source')
            self.assertFalse(pub.report_status(root,'assignment1.pdf')['available'])

    def test_all_metrics_tables_match_artifacts_and_budget(self):
        generated=ROOT/'reports/latex/generated'
        provenance=json.loads((generated/'provenance.json').read_text())
        self.assertEqual(len(provenance['measurements']),25) # 23 main +2 controls
        for key,record in provenance['measurements'].items():
            a,id=key.split('/');m=json.loads((ROOT/'artifacts'/a/'full'/id/'metrics.json').read_text())
            self.assertEqual(record,m,key)
            if a in ['E1','E2','E3']:
                text=(generated/(a+'_results.tex')).read_text()
                row=[esc(id),pct(m['val_accuracy']),pct(m['test_accuracy']),f(m['test_loss']),str(m['best_epoch']+1),f"{m['parameters']:,}"]
                self.assertIn(' & '.join(row),text,key)
            elif a=='A1' and not id.endswith('noaug'):
                text=(generated/'A1_results.tex').read_text()
                short=id.replace('resnet18','R18').replace('deit_tiny','DeiT')
                row=[esc(short),pct(m['train_accuracy']),pct(m['val_accuracy']),pct(m['test_accuracy']),pct(m['macro_f1']),str(m['best_epoch'])]
                self.assertIn(' & '.join(row),text,key)
                self.assertEqual(m['epochs_completed'],25)
                import csv
                with (ROOT/'artifacts'/a/'full'/id/'history.csv').open() as handle:
                    history=list(csv.DictReader(handle))
                best=min(history,key=lambda r:(-float(r['val_accuracy']),float(r['val_loss'])))
                self.assertEqual(m['best_epoch'],int(best['epoch'])+1)
                self.assertIn(m['best_epoch'],range(1,26))
            elif a=='A2':
                self.assertEqual(m['steps'],20000)
                text=(generated/'A2_quality.tex').read_text()
                row=['Baseline' if id=='dcgan_baseline' else '+ DiffAugment',f(m['fid'],2),f(m['kid_mean'],5),f(m['kid_std'],5),f(m['inception_score_mean']),f(m['inception_score_split_std'])]
                self.assertIn(' & '.join(row),text,key)
        delta=(generated/'A1_ablation.tex').read_text()
        self.assertIn('7.17',delta);self.assertIn('7.09',delta)

    def test_cost_benchmark_and_confusion_consistency(self):
        import math
        generated=ROOT/'reports/latex/generated'
        for assignment in ['E1','E2','E3','A1','A2']:
            for run in (ROOT/'artifacts'/assignment/'full').iterdir():
                if not run.is_dir():continue
                m=json.loads((run/'metrics.json').read_text())
                if assignment in ['E1','E2','E3']:
                    expected=[esc(run.name),f(m['peak_gpu_memory_mb'],2),f(m['weight_size_mb'],3),f(m['latency_ms_per_image'],4)]
                    self.assertIn(' & '.join(expected),(generated/(assignment+'_cost.tex')).read_text())
                elif assignment=='A1':
                    cm=json.loads((run/'confusion_matrix.json').read_text())['counts']
                    accuracy=sum(cm[i][i] for i in range(100))/10000
                    macro=sum(2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) for i in range(100))/100
                    self.assertAlmostEqual(accuracy,m['test_accuracy'],places=12)
                    self.assertAlmostEqual(macro,m['macro_f1'],places=12)
                    if run.name.endswith('noaug'):continue
                    name=esc(run.name.replace('resnet18','R18').replace('deit_tiny','DeiT'))
                    row=[name,f(m['parameters']/1e6,3),f(m['trainable_parameters']/1e6,3),f(m['training_seconds']/60,2),f(m['peak_training_memory_mb'],1)]
                    self.assertIn(' & '.join(row),(generated/'A1_cost.tex').read_text())
                    bench=json.loads((run/'benchmark.json').read_text())['measurements']
                    for n,r in bench.items():
                        row=[name,n,f(r['latency_ms_per_image']),f(r['images_per_second'],1),f(r['peak_allocated_memory_mb'],1)]
                        self.assertIn(' & '.join(row),(generated/'A1_benchmark.tex').read_text())
                        self.assertTrue(math.isclose(1000/r['latency_ms_per_image'],r['images_per_second'],rel_tol=1e-10))
                else:
                    name='Baseline' if run.name=='dcgan_baseline' else '+ DiffAugment'
                    row=[name,pct(m['generative_precision']),pct(m['generative_recall']),f(m['training_seconds'],2),f(m['peak_training_gpu_memory_mb'],2)]
                    self.assertIn(' & '.join(row),(generated/'A2_cost.tex').read_text())
                    for n,r in json.loads((run/'benchmark.json').read_text()).items():
                        row=[name,n,f(r['latency_ms_per_image']),f(r['images_per_second'],1),f(r['peak_gpu_memory_mb'],2)]
                        self.assertIn(' & '.join(row),(generated/'A2_bench.tex').read_text())

    def test_no_heavy_materials_in_latex_source(self):
        forbidden={'.pt','.pth','.npz','.npy','.safetensors','.ckpt'}
        for path in (ROOT/'reports/latex').rglob('*'):
            self.assertNotIn(path.suffix,forbidden)
        metadata=(ROOT/'reports/latex/shared/metadata.tex').read_text()
        self.assertNotIn('Chưa cung cấp',metadata)
        self.assertIn('Nguyễn Phan Tuấn Duy - 2310491',metadata)
        self.assertIn('Phạm Duy Anh - 2310139',metadata)
        self.assertIn('Phạm Duy Anh',metadata)

if __name__=='__main__':unittest.main()
