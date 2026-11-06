"""Verify built report publication using hashes and recorded content checks (stdlib)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

NAMES = ('exercises.pdf', 'assignment1.pdf', 'assignment2.pdf')

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def report_status(root: Path, filename: str) -> dict:
    """Missing/unverified PDF is unavailable, regardless of its byte size.

    No TeX/Poppler/ML runtime is required during a website import. This verifies
    the published bytes and their build record; content checks ran at build time.
    """
    source = root / 'reports' / filename
    result = dict(path='reports/'+filename, available=False, is_placeholder=True,
                  size_kb=round(source.stat().st_size/1024, 2) if source.is_file() else 0)
    try:
        manifest = json.loads((root/'reports/report_manifest.json').read_text())
        entry = manifest['reports'][filename]
        assert manifest['schema'] == 1 and entry['content_checked'] is True
        assert entry['pages'] >= 3 and entry['unresolved_references'] == 0
        assert entry['overfull_boxes'] == 0 and entry['text_searchable'] is True
        assert source.read_bytes().startswith(b'%PDF-') and sha(source)==entry['sha256']
        assert entry['sources']
        for name, digest in entry['sources'].items():
            path = root/name
            assert path.resolve().is_relative_to(root.resolve()) and path.is_file()
            assert sha(path)==digest
        result.update(available=True, is_placeholder=False, sha256=entry['sha256'],
                      pages=entry['pages'], validation='built-content-and-hashes',
                      updated_at=manifest['built_at'])
    except (OSError, KeyError, ValueError, AssertionError, TypeError):
        result['validation']='missing-or-stale-build-record'
    return result

def check_publication(root: Path) -> dict:
    statuses={n:report_status(root,n) for n in NAMES}
    for name, status in statuses.items():
        if not status['available']: raise ValueError(f'{name}: missing/stale validated report')
        copy=root/'docs/reports'/name
        if not copy.is_file() or sha(copy)!=status['sha256']:
            raise ValueError(f'{name}: public PDF differs')
    return statuses
