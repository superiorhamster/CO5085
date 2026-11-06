# CO5085 deployment repository

The active repository is a static GitHub Pages site with linked coursework
sources and measured evidence. See README.md for the retained directory layout.
GitHub Actions publishes only docs/. Do not train, commit, push or deploy unless
requested. Do not restore archived development files just to run an obsolete command.

## Preserve

- Keep existing uncommitted work. Do not reset/clean/stage unrelated changes.
- A1 is CIFAR-100 / ResNet-18 / DeiT-Tiny: 25 epochs, 8 main +2 controls.
- A2 is DCGAN baseline / DiffAugment: 20,000 updates, 2 variants.
- Main completion is23/23. Do not mix historical Food-101/Pet/DDPM results.
- Notebooks are standalone; clean and executed A1/A2 notebooks are distinct evidence.
- Raw exports, checkpoints, predictions and caches belong outside the repository.
- Preserve verified metrics, bundle inventories, PDF bytes and build provenance.
- Reports/independent projects disclose AI assistance and missing student details.

## Required lightweight checks

```bash
python tools/validate_notebooks.py
python tools/validate_site.py
python tools/web_artifacts.py check --artifacts-dir artifacts
python -m unittest tools.test_reports tools.test_web_artifacts tools.test_a2_site_contract tools.test_a2_web_export
node tools/test_a1_dashboard.cjs
node tools/test_a2_dashboard.cjs
```

These require Python and Node, not an ML/LaTeX runtime. PDF rebuilding and import
commands are documented in README.md and reports/latex/README.md. Sync only
validated artifacts; do not fabricate placeholders or measurements.

## Cleanup history

On2026-11-06, heavy and unused local development material was moved into a dated
sibling cleanup backup, with per-file SHA-256 checks and pre-edit snapshots.
The backup is outside this repository and has MANIFEST.json / RESTORE.md.
Do not overwrite newer work when selectively restoring a file.
Managed .git/.agents/.codex/.aws directories and Git history were left untouched.

The importer still needs a1_food101_protocol.py as a lightweight compatibility
module; its training tools and dedicated CI tests are not deployment dependencies.
MSSV, official group ID and video URLs remain missing. A1 dataset-change approval
is not evidenced. These gaps must not be marked complete by a cleanup or PDF build.
