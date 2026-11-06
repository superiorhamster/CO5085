"""Create three independent LaTeX folders and ZIPs for Prism/Overleaf.

python tools/package_overleaf.py
python tools/package_overleaf.py --report assignment1
Existing edited projects are protected; --force explicitly replaces them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from report_publication import check_publication

ROOT=Path(__file__).resolve().parents[1]
PROJECTS=('exercises','assignment1','assignment2')
TITLES={'exercises':'CO5085 — E1–E3', 'assignment1':'CO5085 — A1', 'assignment2':'CO5085 — A2'}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def project_files(name: str, entry: dict) -> dict[str,bytes]:
    files={}
    for source in entry['sources']:
        path=ROOT/source
        if not path.is_relative_to(ROOT/'reports/latex') or path.suffix not in ('.tex','.bib','.png'):
            continue
        relative=path.relative_to(ROOT/'reports/latex')
        if relative.parts[0]==name:
            if relative.name!='main.tex':raise ValueError('Unexpected report entry point')
            relative=Path('main.tex')
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe project path')
        content=path.read_bytes()
        if path.suffix=='.tex':
            content=content.decode('utf-8').replace('../shared/','shared/').replace('../generated/','generated/').encode('utf-8')
        files[relative.as_posix()]=content
    assert all(p in files for p in ['main.tex','shared/preamble.tex','shared/metadata.tex','shared/references.bib','shared/hcmut.png'])
    files['latexmkrc']=b"$pdf_mode = 1;\n@default_files = ('main.tex');\n"
    files['README.md']=f'''# {TITLES[name]}

This folder is a complete, independent LaTeX project. Upload the entire ZIP or
folder as one project. Do not upload main.tex alone.

- Main document: `main.tex`
- Compiler: **pdfLaTeX**, with BibTeX for references
- Suggested TeX Live: 2025 or newer (the reference report was built with 2026)
- Edit author/student/course details: `shared/metadata.tex`
- Edit the report: `main.tex`
- Style/logo/bibliography: `shared/`
- Included tables and figures: `generated/`

No other report folder, repository checkout, Python/ML environment, dataset,
checkpoint, or absolute filesystem path is needed to compile this project.
The original report is {entry['pages']} pages; this archive keeps its LaTeX,
bibliography and image bytes, changing only paths to be project-local.

## Local build

Install TeX Live with vntex, latexmk, latex-extra/xurl and the standard LaTeX fonts.
From this folder run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Import

In Overleaf, create a new project from the ZIP and select `main.tex` / pdfLaTeX.
Prism supports folder/ZIP import where the service remains available. OpenAI has
announced Prism retirement; use Overleaf or local LaTeX if Prism access is unavailable.
Official status: https://help.openai.com/en/articles/20001615-prism-to-codex-migration-guide

Authors: **Phạm Duy Anh - 2310139** and **Nguyễn Phan Tuấn Duy - 2310491**.
Review the report and AI assistance disclosure
before submission. Local edits are independent of the canonical report sources.
Running the packaging command again refuses to overwrite edited project folders
unless `--force` is explicitly supplied.
'''.encode('utf-8')
    manifest={'schema':1,'project':name,'main':'main.tex','compiler':'pdflatex',
              'reference_pdf_sha256':entry['sha256'],'reference_pages':entry['pages'],
              'files':{p:digest(content) for p,content in sorted(files.items())}}
    files['project_manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    return files


def ensure_replaceable(folder: Path, force: bool):
    if not folder.exists() or force:return
    try:
        record=json.loads((folder/'project_manifest.json').read_text())
        inventory={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        assert inventory==set(record['files'])|{'project_manifest.json'}
        for name,expected in record['files'].items():
            path=folder/name
            assert path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink()
            assert digest(path.read_bytes())==expected
    except (OSError,KeyError,ValueError,AssertionError):
        raise SystemExit(f'{folder}: contains edits or extra files. Back up your edits; use --force to replace it.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',choices=PROJECTS)
    parser.add_argument('--force',action='store_true',help='Replace locally edited generated project folders')
    args=parser.parse_args()
    check_publication(ROOT)
    manifest=json.loads((ROOT/'reports/report_manifest.json').read_text())
    names=[args.report] if args.report else list(PROJECTS)
    destination=ROOT/'reports/projects';destination.mkdir(exist_ok=True)
    # Check every target before modifying any project.
    for name in names:ensure_replaceable(destination/name,args.force)
    for name in names:
        files=project_files(name,manifest['reports'][name+'.pdf'])
        folder=destination/name
        if folder.exists():shutil.rmtree(folder)
        folder.mkdir()
        for relative,content in files.items():
            path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        archive=destination/(name+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
            for relative,content in sorted(files.items()):
                info=zipfile.ZipInfo(relative,date_time=(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                output.writestr(info,content)
        print(f'{folder.relative_to(ROOT)}/  ->  {archive.relative_to(ROOT)}')

if __name__=='__main__':main()
