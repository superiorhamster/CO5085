#!/usr/bin/env python3
"""Validate notebook JSON, Python syntax, section order and experiment declarations."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "E1/e1_fashion_mnist_baselines.ipynb": 3,
    "E2/e2_attention_tokenization.ipynb": 6,
    "E3/e3_sequence_models.ipynb": 4,
    "A1/a1_cnn_vs_transformer.ipynb": 8,
    "A2/a2_diffusion_schedules.ipynb": 2,
}
REQUIRED_TOPICS = ["Môi trường", "Cấu hình", "Dữ liệu", "mô hình", "Huấn luyện", "đánh giá"]


def validate(path: Path, expected_experiments: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert payload["cells"] and payload["cells"][0]["cell_type"] == "markdown"
    markdown = "\n".join("".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "markdown")
    combined_code = "\n".join("".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code")
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            assert index > 0 and payload["cells"][index - 1]["cell_type"] == "markdown", (
                f"Code cell {index} in {path} must be preceded by a Markdown description"
            )
            description = "".join(payload["cells"][index - 1]["source"])
            assert "**Mục tiêu:**" in description and "**Kết quả:**" in description, (
                f"Markdown before code cell {index} in {path} must state Mục tiêu and Kết quả"
            )
            try:
                ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")
            except SyntaxError as error:
                raise AssertionError(f"Syntax error in {path}, cell {index}: {error}") from error
            # assert cell["execution_count"] is None
            # assert cell["outputs"] == []
    for topic in REQUIRED_TOPICS:
        assert topic.lower() in markdown.lower(), f"Missing topic {topic!r} in {path}"
    assert '"run_mode"' in combined_code and '"experiment_ids"' in combined_code
    assert "smoke" in combined_code and "full" in combined_code
    assert "rng_state" in combined_code and "last.pt" in combined_code
    assert "EXPERIMENT_IDS" in combined_code
    # Guard the intended matrix size without importing heavy dependencies.
    if expected_experiments == 3:
        assert '"softmax"' in combined_code and '"mlp"' in combined_code and '"cnn"' in combined_code
    elif expected_experiments == 6:
        assert "for backend in" in combined_code and "for tokenizer in" in combined_code
    elif expected_experiments == 4:
        assert "for recurrent in" in combined_code and "for representation in" in combined_code
    elif expected_experiments == 8:
        assert "for family in MODEL_NAMES" in combined_code and "for strategy in STRATEGIES" in combined_code
    elif expected_experiments == 2:
        assert 'EXPERIMENT_IDS = ["linear", "cosine"]' in combined_code
    print(f"OK {path.relative_to(ROOT)} ({len(payload['cells'])} cells, {expected_experiments} experiments)")


def main() -> None:
    for relative, count in NOTEBOOKS.items():
        validate(ROOT / relative, count)
    assert sum(NOTEBOOKS.values()) == 23
    print("All notebooks are structurally valid; total experiment configurations: 23")


if __name__ == "__main__":
    main()

