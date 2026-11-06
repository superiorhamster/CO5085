"""Compile evidence-based Vietnamese reports with pdfLaTeX + latexmk.

All: python tools/build_reports.py
One: python tools/build_reports.py --report assignment1
Check committed PDFs without TeX: python tools/build_reports.py --check
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
from report_publication import NAMES, sha, check_publication

ROOT=Path(__file__).resolve().parents[1]
LATEX=ROOT/'reports/latex'

def command(args, **kwargs):
    return subprocess.run(args,check=True,text=True,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,**kwargs).stdout

def normalize(text): return unicodedata.normalize('NFC',text)

def compile_report(name):
    directory=LATEX/name
    build=LATEX/'.build'/name;build.mkdir(parents=True,exist_ok=True)
    # BibTeX resolves explicit ../shared paths relative to its aux directory.
    bibdir=build.parent/'shared';bibdir.mkdir(exist_ok=True)
    shutil.copy2(LATEX/'shared/references.bib',bibdir/'references.bib')
    env=os.environ.copy();env['max_print_line']='1000'
    env['BIBINPUTS']=str(directory)+os.pathsep
    result=subprocess.run(['latexmk','-g','-pdf','-interaction=nonstopmode','-halt-on-error',
        '-file-line-error',f'-outdir={build}','main.tex'],cwd=directory,env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace")
    (build/'build.log').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{name}: LaTeX failed; see {build}/build.log\n'+result.stdout[-2500:])
    log=(build/'main.log').read_text(errors='replace')
    bad=re.findall(r'LaTeX Warning: (?:Reference .* undefined|Citation .* undefined|There were undefined references)|Overfull \\[hv]box|Missing character:',log)
    if bad: raise ValueError(f'{name}: unresolved citations/references, missing glyphs or overflow: {bad}')
    pdf=build/'main.pdf';info=command(['pdfinfo',str(pdf)])
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
    text=normalize(command(['pdftotext','-layout',str(pdf),'-']))
    for marker in ['Phạm Duy Anh','2310139','Nguyễn Phan Tuấn Duy','2310491','CO5085','2026','Tài liệu','AI']:
        if marker not in text:raise ValueError(f'{name}: missing searchable text {marker!r}')
    for marker in ['[Project Name]','[Project Title]','[Advisor Name]','[Course / Report Type]','CNPM','??','PLACEHOLDER','Fashion-MNIS T']:
        if marker in text: raise ValueError(f'{name}: template placeholder/unresolved reference {marker!r}')
    expected={'exercises':['91,95','89,87','54.000','20 epoch'],
              'assignment1':['CIFAR-100','82,59','25 epoch','7,17','7,09'],
              'assignment2':['DiffAugment','41,84','5,84','20.000']}[name]
    for marker in expected:
        if marker not in text:raise ValueError(f'{name}: missing required result {marker!r}')
    (build/'text.txt').write_text(text)
    # Actual file dependencies tracked by pdfTeX; only repository report sources.
    sources={}
    for line in (build/'main.fls').read_text().splitlines():
        if line.startswith('INPUT '):
            path=Path(line[6:]);path=path if path.is_absolute() else directory/path
            path=path.resolve()
            if path.is_relative_to(LATEX) and path.suffix in ('.tex','.bib','.png','.json') and '.build' not in path.parts:
                sources[path.relative_to(ROOT).as_posix()]=sha(path)
    groups={'exercises':('E1','E2','E3'),'assignment1':('A1',),'assignment2':('A2',)}[name]
    provenance=json.loads((LATEX/'generated/provenance.json').read_text())
    for path,digest in provenance['inputs'].items():
        if any(path.startswith('artifacts/'+g+'/') for g in groups):
            # Optional local E1/E3 CSVs are not in the lightweight Git bundle.
            if path.endswith('/history.csv') and not path.startswith(('artifacts/A1/','artifacts/A2/')):
                continue
            sources[path]=digest
    sources['reports/latex/shared/references.bib']=sha(LATEX/'shared/references.bib')
    sources['reports/latex/generated/provenance.json']=sha(LATEX/'generated/provenance.json')
    for tool in ['prepare_reports.py','build_reports.py','report_publication.py']:
        sources['tools/'+tool]=sha(ROOT/'tools'/tool)
    return pdf, dict(sha256=sha(pdf),pages=pages,content_checked=True,text_searchable=True,
                    unresolved_references=0,overfull_boxes=0,sources=dict(sorted(sources.items())))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',choices=[n[:-4] for n in NAMES])
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        print(json.dumps(check_publication(ROOT),ensure_ascii=False,indent=2));return
    if not shutil.which('latexmk'):raise SystemExit('Install TeX Live + latexmk; see reports/latex/README.md')
    command([os.sys.executable,str(ROOT/'tools/prepare_reports.py')])
    selected=[args.report] if args.report else [n[:-4] for n in NAMES]
    staged={name:compile_report(name) for name in selected}
    path=ROOT/'reports/report_manifest.json'
    manifest=json.loads(path.read_text()) if path.exists() else {'schema':1,'reports':{}}
    manifest['built_at']=dt.datetime.now(dt.timezone.utc).isoformat()
    manifest['engine']='pdfLaTeX + latexmk'
    manifest['source_validation_note']='Bundle hashes checked now; original checkpoint/prediction validation is recorded by export manifests.'
    for name,(pdf,entry) in staged.items():
        filename=name+'.pdf'
        shutil.copy2(pdf,ROOT/'reports'/filename)
        (ROOT/'docs/reports').mkdir(parents=True,exist_ok=True)
        shutil.copy2(pdf,ROOT/'docs/reports'/filename)
        manifest['reports'][filename]=entry
        print(f'{filename}: {entry["pages"]} pages, searchable text, no unresolved references/overflow')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    if not args.report:check_publication(ROOT)

if __name__=='__main__':main()
