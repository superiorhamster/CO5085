# Git-ready experiment materials

A1 and A2 follow the E1–E3 layout: `artifacts/<assignment>/full/<experiment>/`.
Assignment-level dataset manifests, EDA and summaries sit beside `full/`.

- A1: CIFAR-100, ResNet-18/DeiT-Tiny,25 epochs,8 main configurations plus2 noaug controls.
- A2: the existing DCGAN baseline/DiffAugment study,20.000 updates, unchanged metric values.

## Rebuild and import

```bash
python tools/web_artifacts.py pack --a1 /path/to/A1_full --a2 /path/to/A2_full --artifacts-dir artifacts
python tools/web_artifacts.py check --artifacts-dir artifacts
python tools/sync_site_data.py --artifacts-dir artifacts
```

The original local exports were archived outside the repository during cleanup.
Packaging a future export takes explicit external paths and needs the original exports and executed A2 notebook for source provenance.
Bundle checking and website import need only these lightweight bundles and the committed
executed A2 notebook; all commands use Python stdlib. The generated `docs/` bundle works
without raw exports or checkpoints, including its existing JSON fallback for `file://`.

Each assignment's `export_manifest.json` inventories every packaged file with SHA-256.
Checkpoint hashes (A1 best.pt, A2 last.pt) were checked against original bytes at packaging.
A1 accuracy/F1/confusion were recomputed from original predictions. Those source checks
are recorded evidence, not claims of rechecking bytes absent from this bundle.

The bundles contain JSON/CSV/PNG/Markdown only: full histories, metrics, configs, benchmark
protocols, diagnostic details and figures. A2 local config paths are omitted; source fingerprint
inputs and metric values are preserved. No `.pt`, `.npz`, datasets, raw-image caches, training
logs or heartbeat files belong here. Original exports and heavy E1–E3 training outputs are archived outside the repository.

Website figures are copied to `docs/assets/artifacts/A1|A2/`; summarized results and histories
are generated in `docs/data/experiments_manifest.json` and its JS fallback. Only `docs/`
is deployed by GitHub Pages. Checkpoints are not needed for readers of the website.

Git-ready changes include `artifacts/A1/`, `artifacts/A2/`, this README, generated `docs/`,
the importer/packaging/test tools, workflow and documentation changes. The linked clean and
executed notebooks must also be committed if not already tracked. No commit/push/deploy is
performed by these commands. Final report PDFs are retained in `reports/` and mirrored under `docs/reports/`;
their LaTeX/build provenance is verified separately.
