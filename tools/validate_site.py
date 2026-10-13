#!/usr/bin/env python3
"""Validate the deployable GitHub Pages bundle and optional release readiness."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

EXPECTED_PAGES = {
    "index.html": ["CO5085", "E1", "A1", "A2", "timeline-container"],
    "exercises.html": ["e1-section", "e2-section", "e3-section", "ScratchMSA", "StratifiedShuffleSplit"],
    "a1.html": ["milestone-a1-1", "milestone-a1-2", "freeze-inspector", "resnet18", "deit_tiny"],
    "a2.html": ["paper-analysis", "schedule-math", "pipeline-results", "Nichol", "squaredcos_cap_v2"],
}
EXPECTED_COUNTS = {"E1": 3, "E2": 6, "E3": 4, "A1": 8, "A2": 2}
PLACEHOLDER_MARKERS = ("2XXXXXX", "2YYYYYY", "student1@", "student2@", "AllInOneAssignment")


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.repo_paths: list[str] = []
        self.colab_paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        for key in ("href", "src"):
            value = attr_map.get(key)
            if value:
                self.links.append(value)
        if attr_map.get("data-repo-path"):
            self.repo_paths.append(attr_map["data-repo-path"] or "")
        if attr_map.get("data-colab-notebook"):
            self.colab_paths.append(attr_map["data-colab-notebook"] or "")


def parse_page(html_path: Path) -> LinkCollector:
    parser = LinkCollector()
    parser.feed(html_path.read_text(encoding="utf-8"))
    return parser


def verify_links(html_path: Path, base_dir: Path) -> LinkCollector:
    parser = parse_page(html_path)
    for raw_link in parser.links:
        if raw_link.startswith(("mailto:", "#", "data:")):
            continue
        if raw_link.startswith(("http://", "https://")):
            parsed = urlparse(raw_link)
            assert parsed.scheme == "https" and parsed.netloc, (
                f"External URL must be absolute HTTPS in {html_path.relative_to(ROOT)}: {raw_link}"
            )
            continue
        clean_path = raw_link.split("#", 1)[0].split("?", 1)[0]
        if not clean_path:
            continue
        target = (base_dir / clean_path).resolve()
        assert target.exists(), (
            f"Broken local link {raw_link!r} in {html_path.relative_to(ROOT)} -> {target}"
        )
    for relative in parser.repo_paths + parser.colab_paths:
        assert (ROOT / relative).is_file(), (
            f"Repository/Colab target does not exist: {relative!r} in {html_path.relative_to(ROOT)}"
        )
    return parser


def validate_release_readiness(site_cfg: dict, manifest: dict) -> list[str]:
    issues: list[str] = []
    if site_cfg["team"].get("group_id") == "Group-CO5085-HK261":
        issues.append("team.group_id still uses the default value")
    for index, member in enumerate(site_cfg["team"].get("members", []), start=1):
        if not member.get("student_id"):
            issues.append(f"team member {index} is missing student_id")

    media = site_cfg.get("media", {})
    for key in ("a1_demo_video_url", "a2_demo_video_url", "a2_presentation_video_url"):
        if not media.get(key):
            issues.append(f"media.{key} is empty")

    completed = manifest.get("total_completed_full_experiments", 0)
    expected = manifest.get("total_expected_experiments", 23)
    if completed != expected:
        issues.append(f"only {completed}/{expected} full-run experiments are published")

    for name, report in manifest.get("reports", {}).items():
        if report.get("is_placeholder"):
            issues.append(f"reports/{name} is still a placeholder PDF")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help="Fail when submission metadata, full-run results, videos, or final reports are incomplete.",
    )
    args = parser.parse_args()

    assert (DOCS / ".nojekyll").exists(), "Missing docs/.nojekyll"
    assert (ROOT / "index.html").exists(), "Missing root index.html redirect"
    workflow_path = ROOT / ".github" / "workflows" / "deploy-pages.yml"
    assert workflow_path.exists(), "Missing GitHub Actions workflow"

    site_cfg_path = DOCS / "data" / "site_config.json"
    manifest_path = DOCS / "data" / "experiments_manifest.json"
    site_cfg = json.loads(site_cfg_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "course" in site_cfg and "team" in site_cfg and "timeline" in site_cfg
    assert site_cfg["team"].get("members"), "At least one team member is required"
    config_text = site_cfg_path.read_text(encoding="utf-8")
    for marker in PLACEHOLDER_MARKERS:
        assert marker not in config_text, f"Placeholder marker remains in site_config.json: {marker}"

    repository = site_cfg["repository"]
    repo_url = repository["repo_url"].rstrip("/")
    branch = repository["branch"]
    assert repo_url.startswith("https://github.com/") and repo_url.count("/") == 4, "Invalid GitHub repo_url"
    repo_slug = repo_url.removeprefix("https://github.com/")
    expected_colab = f"https://colab.research.google.com/github/{repo_slug}/blob/{branch}"
    assert repository["colab_base_url"].rstrip("/") == expected_colab, "colab_base_url does not match repo_url/branch"
    for relative in repository.get("paths", {}).values():
        assert (ROOT / relative).exists(), f"Configured repository path does not exist: {relative}"

    assert manifest["total_expected_experiments"] == sum(EXPECTED_COUNTS.values()) == 23
    assert 0 <= manifest["total_completed_full_experiments"] <= 23
    assert not Path(manifest.get("artifacts_root", ".")).is_absolute(), "Manifest exposes an absolute artifacts_root"
    manifest_text = json.dumps(manifest)
    assert "/home/" not in manifest_text and "C:\\\\Users\\\\" not in manifest_text, (
        "Manifest exposes a local workstation path"
    )
    for assignment, expected_n in EXPECTED_COUNTS.items():
        assignment_data = manifest["assignments"][assignment]
        assert assignment_data["expected_count"] == expected_n
        assert len(assignment_data["experiments"]) == expected_n
        completed_count = sum(bool(exp["has_full_metrics"]) for exp in assignment_data["experiments"])
        assert assignment_data["completed_full_count"] == completed_count

    for pdf_name in ("exercises.pdf", "assignment1.pdf", "assignment2.pdf"):
        source = ROOT / "reports" / pdf_name
        published = DOCS / "reports" / pdf_name
        for pdf_path in (source, published):
            assert pdf_path.exists(), f"Missing PDF {pdf_path.relative_to(ROOT)}"
            assert pdf_path.read_bytes()[:8].startswith(b"%PDF-1."), (
                f"Invalid PDF header in {pdf_path.relative_to(ROOT)}"
            )
        assert source.read_bytes() == published.read_bytes(), f"Published report is stale: {pdf_name}"

    for page_name, required_tokens in EXPECTED_PAGES.items():
        page_path = DOCS / page_name
        content = page_path.read_text(encoding="utf-8")
        for token in required_tokens:
            assert token in content, f"Missing required content token {token!r} in {page_name}"
        assert "anhpd/AllInOneAssignment" not in content, f"Stale repository URL in {page_name}"
        verify_links(page_path, DOCS)
        print(f"OK docs/{page_name} (links, targets & required sections verified)")

    verify_links(ROOT / "index.html", ROOT)
    site_js = (DOCS / "assets" / "js" / "site.js").read_text(encoding="utf-8")
    assert "${cfg.repository.repo_url}/blob/${cfg.repository.branch}/${subpath}" in site_js
    assert "${cfg.repository.repo_url}/tree/${cfg.repository.branch}/${subpath}" not in site_js

    workflow = workflow_path.read_text(encoding="utf-8")
    assert "actions/upload-pages-artifact@v4" in workflow, "Pages artifact action must use v4"
    assert "tools/sync_site_data.py" not in workflow, "CI must not overwrite committed site data from an empty runner"

    release_issues = validate_release_readiness(site_cfg, manifest)
    if release_issues:
        print("Release-readiness issues:")
        for issue in release_issues:
            print(f"  - {issue}")
    if args.strict_release and release_issues:
        raise SystemExit("Strict release validation failed")

    print("GitHub Pages bundle is deployable; local data contracts and links are valid.")


if __name__ == "__main__":
    main()
