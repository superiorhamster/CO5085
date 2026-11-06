# Independent CO5085 LaTeX projects

Each folder and ZIP is a separate, complete project:

| Report | Editable folder | Upload archive |
| --- | --- | --- |
| E1–E3 | [exercises/](exercises/) | [exercises.zip](exercises.zip) |
| A1 | [assignment1/](assignment1/) | [assignment1.zip](assignment1.zip) |
| A2 | [assignment2/](assignment2/) | [assignment2.zip](assignment2.zip) |

Upload one archive per project. Select **main.tex** and **pdfLaTeX**. Each archive
includes its own style, logo, metadata, bibliography, tables and figures; no
other folder or repository files are needed. The two authors are configured in `shared/metadata.tex`: **Phạm Duy Anh - 2310139**
and **Nguyễn Phan Tuấn Duy - 2310491**.

Regenerate from the verified canonical reports with:

```bash
python tools/package_overleaf.py
```

Local edits in an unpacked project are protected from replacement. `--force`
replaces them only when explicitly requested. ZIPs are reproducible generated
outputs; these three upload archives and their project folders are retained for Git.

`validation.json` records the packaging checks. All LaTeX/bibliography/image
inputs match the independent archives compiled during the previous report task;
this task rechecked local dependencies and hashes without claiming a fresh build.

Prism's [import guide](https://help.openai.com/en/articles/20001050-troubleshooting-prism)
accepts folders/ZIPs. OpenAI has announced [Prism retirement](https://help.openai.com/en/articles/20001615-prism-to-codex-migration-guide);
Overleaf or local LaTeX can use the same source packages.
