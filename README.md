# CO5085 — GitHub Pages and coursework evidence

The repository contains the static coursework website, its validated publication
inputs, and the notebooks/report sources linked as evidence. GitHub Pages deploys
**only `docs/`**. Training exports, checkpoints, development-only tools and build
caches are kept outside this deployment repository.

## Repository layout

| Path | Purpose |
| --- | --- |
| `docs/` | Published HTML, CSS/JS, data, figures and three PDF reports |
| `.github/workflows/deploy-pages.yml` | Validate the committed bundle, then deploy `docs/` |
| `index.html` | Root redirect into the published site |
| `E1/`, `E2/`, `E3/`, `A1/`, `A2/` | Linked standalone notebooks and executed A1/A2 evidence |
| `artifacts/` | Lightweight measured results, histories, EDA and provenance |
| `reports/latex/` | Canonical LaTeX sources, generated tables/figures and review record |
| `reports/projects/` | Independent editable E1–E3, A1 and A2 LaTeX projects |
| `reports/*.pdf`, `reports/report_manifest.json` | Verified report originals and their build records |
| `tools/` | Deployment validation, artifact import and report publication utilities |
| `plans/`, `*_co5085.md` | Current protocols and assignment requirements |
| `HCMUT_RP_template/SoftEng_252/` | Original template source/logo required by report preparation |

Current published results: **23/23 main experiments + 2 A1 controls**. A1 uses
CIFAR-100, ResNet-18/DeiT-Tiny, 25 epochs. A2 uses DCGAN baseline/DiffAugment,
20,000 updates. The E1–E3/A1/A2 PDFs contain 20/22/20 pages respectively.

## Validate before deployment

Python 3.11+ and Node.js 18+ are sufficient. These checks use no ML libraries,
training data, checkpoints, GPU or LaTeX installation:

```bash
python tools/validate_notebooks.py
python tools/validate_site.py
python tools/web_artifacts.py check --artifacts-dir artifacts
python -m unittest tools.test_reports tools.test_web_artifacts tools.test_a2_site_contract tools.test_a2_web_export
node tools/test_a1_dashboard.cjs
node tools/test_a2_dashboard.cjs
```

The workflow runs these checks before uploading `docs/`. It does not train,
rebuild notebooks/PDFs or replace the committed website data on the CI runner.
Configure GitHub **Settings → Pages → Source → GitHub Actions** when deploying.
No commit, push or deployment is performed by the local validation commands.

Preview locally:

```bash
python -m http.server 8000 --directory docs
```

The JSON fallback also supports opening the generated site through `file://`.

## Update measured results

```bash
python tools/sync_site_data.py --artifacts-dir artifacts
python tools/validate_site.py
```

Keep the updated manifest, fallback JS and copied figures together. The importer
rejects incomplete/mixed A1/A2 studies and preserves E1–E3 and existing A2 values.
Source checkpoint/prediction verification is recorded by the export manifests;
CI checks the lightweight files and their hashes, not absent checkpoint bytes.
See [artifact provenance](artifacts/README.md).

Raw A1/A2 exports were moved outside the repository during cleanup. Packaging a
future export is optional and takes explicit external input paths:

```bash
python tools/web_artifacts.py pack --a1 /path/to/A1_full --a2 /path/to/A2_full --artifacts-dir artifacts
```

## Notebooks and reports

The retained notebooks are self-contained and can run on Kaggle/Colab without
the archived notebook generators or training-module source files. The executed
A1/A2 notebooks remain unchanged as provenance. `requirements-notebooks.txt` is
for notebook execution only and is not installed for website deployment.

- [Report sources, build and review instructions](reports/latex/README.md)
- [Independent LaTeX projects for upload](reports/projects/README.md)

```bash
python tools/build_reports.py --check   # validate existing PDFs without TeX
python tools/build_reports.py           # rebuild with TeX Live + latexmk/Poppler
python tools/package_overleaf.py         # create independent projects and ZIPs
```

After rebuilding PDFs, run the website sync. MSSV, the official group ID and
video URLs remain incomplete; there is no evidence of lecturer approval for the
A1 dataset change. Check submission readiness separately:

```bash
python tools/validate_site.py --strict-release
```

## Cleanup boundary

Heavy original exports, checkpoints/predictions, duplicate archives, LaTeX
compiler output, old Food-101/Pet tooling and historical design/validation notes
were moved to a dated sibling backup. The backup includes checksums, pre-edit
copies and restoration instructions. Git history was not rewritten.

The small `a1_food101_protocol.py` module is retained because the existing
artifact importer imports its legacy compatibility contract. Its training
runtime, notebook builder and dedicated legacy CI tests are archived; the active
A1 website and linked notebooks use CIFAR-100 only.
