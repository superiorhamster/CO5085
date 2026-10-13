#!/usr/bin/env python3
"""Validate CO5085 GitHub Pages structure, JSON data contracts, internal links, and PDF reports."""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

EXPECTED_PAGES = {
    "index.html": ["CO5085", "E1", "A1", "A2", "timeline-container"],
    "exercises.html": ["e1-section", "e2-section", "e3-section", "ScratchMSA", "StratifiedShuffleSplit"],
    "a1.html": ["milestone-a1-1", "milestone-a1-2", "freeze-inspector", "resnet18", "deit_tiny"],
    "a2.html": ["paper-analysis", "schedule-math", "pipeline-results", "Nichol", "squaredcos_cap_v2"],
}

EXPECTED_Counts = {"E1": 3, "E2": 6, "E3": 4, "A1": 8, "A2": 2}


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        for key in ("href", "src"):
            val = attr_map.get(key)
            if val:
                self.links.append(val)


def verify_local_links(html_path: Path, base_dir: Path) -> None:
    parser = LinkCollector()
    parser.feed(html_path.read_text(encoding="utf-8"))
    for raw_link in parser.links:
        if raw_link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean_path = raw_link.split("#", 1)[0].split("?", 1)[0]
        if not clean_path:
            continue
        target = (base_dir / clean_path).resolve()
        assert target.exists(), f"Broken local link {raw_link!r} in {html_path.relative_to(ROOT)} -> {target}"


def main() -> None:
    assert (DOCS / ".nojekyll").exists(), "Missing docs/.nojekyll"
    assert (ROOT / "index.html").exists(), "Missing root index.html redirect"
    assert (ROOT / ".github" / "workflows" / "deploy-pages.yml").exists(), "Missing GitHub Actions workflow"

    # 1. Verify JSON configurations
    site_cfg_path = DOCS / "data" / "site_config.json"
    manifest_path = DOCS / "data" / "experiments_manifest.json"
    assert site_cfg_path.exists(), "Missing docs/data/site_config.json"
    assert manifest_path.exists(), "Missing docs/data/experiments_manifest.json"

    site_cfg = json.loads(site_cfg_path.read_text(encoding="utf-8"))
    assert "course" in site_cfg and "team" in site_cfg and "timeline" in site_cfg

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_expected_experiments"] == 23
    for assignment, expected_n in EXPECTED_Counts.items():
        actual_n = len(manifest["assignments"][assignment]["experiments"])
        assert actual_n == expected_n, f"{assignment}: expected {expected_n}, got {actual_n}"

    # 2. Verify PDF reports in both reports/ and docs/reports/
    for pdf_name in ("exercises.pdf", "assignment1.pdf", "assignment2.pdf"):
        for folder in (ROOT / "reports", DOCS / "reports"):
            pdf_path = folder / pdf_name
            assert pdf_path.exists(), f"Missing PDF {pdf_path.relative_to(ROOT)}"
            header = pdf_path.read_bytes()[:8]
            assert header.startswith(b"%PDF-1."), f"Invalid PDF header in {pdf_path.relative_to(ROOT)}"

    # 3. Verify HTML pages and internal links
    for page_name, required_tokens in EXPECTED_PAGES.items():
        page_path = DOCS / page_name
        assert page_path.exists(), f"Missing page {page_path.relative_to(ROOT)}"
        content = page_path.read_text(encoding="utf-8")
        for token in required_tokens:
            assert token in content, f"Missing required content token {token!r} in {page_name}"
        verify_local_links(page_path, DOCS)
        print(f"OK docs/{page_name} (links & required sections verified)")

    verify_local_links(ROOT / "index.html", ROOT)
    print("All GitHub Pages files, data contracts, PDF reports, and local links are valid.")


if __name__ == "__main__":
    main()

