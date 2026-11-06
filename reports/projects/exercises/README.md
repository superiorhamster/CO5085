# CO5085 — E1–E3

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
The original report is 20 pages; this archive keeps its LaTeX,
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
