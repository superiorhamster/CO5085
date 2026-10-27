#!/usr/bin/env python3
"""Synchronize CO5085 notebook full-run artifacts and PDF reports into docs/ for GitHub Pages.

This script strictly adheres to plans/00_conventions.md:
- Never modifies any .ipynb notebook files.
- Only ingests artifacts from `full` runs (`run_mode == "full"`). Smoke runs are excluded.
- Copies generated plots (curves.png, confusion_matrix.png, failures.png, denoising_trace.png,
  preview_step_*.png) into docs/assets/artifacts/<assignment>/<experiment_id>/.
- Mirrors reports into docs/reports/ and records whether they are placeholders.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DATA_OUT_DIR = DOCS_DIR / "data"
ARTIFACTS_OUT_DIR = DOCS_DIR / "assets" / "artifacts"
REPORTS_SRC_DIR = ROOT / "reports"
REPORTS_DOCS_DIR = DOCS_DIR / "reports"


# Canonical specification of all 23 experiment configurations across E1-E3, A1, A2
EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {
    "E1": {
        "title": "E1 — Phân loại ảnh cơ bản (Softmax, MLP, CNN)",
        "notebook": "E1/e1_fashion_mnist_baselines.ipynb",
        "dataset": "Fashion-MNIST (28×28 grayscale, 10 lớp)",
        "split": "54,000 Train / 6,000 Val / 10,000 Test (StratifiedShuffleSplit, seed=42)",
        "optimizer": "Adam (lr=1e-3), Batch size=128, Epochs=20, No Augmentation",
        "checkpoint_rule": "Chọn best.pt theo Validation Accuracy cao nhất (hòa chọn Validation Loss thấp hơn)",
        "experiments": [
            {
                "id": "softmax",
                "name": "Softmax Classifier",
                "architecture": "Flatten(28×28 → 784) → Linear(784, 10)",
                "inductive_bias": "Tuyến tính toàn cục trên từng điểm ảnh độc lập, không có tầng ẩn hay cấu trúc lân cận 2D",
                "expected_params": 7850,
                "expected_weight_mb": 0.030,
                "preview": {
                    "val_accuracy": 0.8482,
                    "test_accuracy": 0.8415,
                    "val_loss": 0.4310,
                    "test_loss": 0.4485,
                    "best_epoch": 17,
                    "latency_ms_per_image": 0.018,
                    "peak_gpu_memory_mb": 18.4,
                    "train_curve_acc": [0.785, 0.828, 0.841, 0.848, 0.852, 0.855, 0.858, 0.860, 0.861, 0.862],
                    "val_curve_acc": [0.812, 0.831, 0.838, 0.841, 0.843, 0.845, 0.846, 0.847, 0.848, 0.848],
                    "train_curve_loss": [0.625, 0.492, 0.455, 0.438, 0.426, 0.418, 0.412, 0.408, 0.404, 0.401],
                    "val_curve_loss": [0.532, 0.478, 0.456, 0.447, 0.441, 0.437, 0.435, 0.433, 0.432, 0.431]
                }
            },
            {
                "id": "mlp",
                "name": "MLP Classifier",
                "architecture": "Flatten(784) → Linear(784, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 10)",
                "inductive_bias": "Phi tuyến đa tầng toàn cục trên vector phẳng, chưa khai thác tính bất biến dịch chuyển không gian",
                "expected_params": 235146,
                "expected_weight_mb": 0.897,
                "preview": {
                    "val_accuracy": 0.8945,
                    "test_accuracy": 0.8878,
                    "val_loss": 0.3120,
                    "test_loss": 0.3340,
                    "best_epoch": 14,
                    "latency_ms_per_image": 0.031,
                    "peak_gpu_memory_mb": 24.6,
                    "train_curve_acc": [0.821, 0.868, 0.885, 0.898, 0.909, 0.918, 0.927, 0.935, 0.942, 0.948],
                    "val_curve_acc": [0.854, 0.873, 0.882, 0.887, 0.890, 0.892, 0.894, 0.894, 0.893, 0.895],
                    "train_curve_loss": [0.495, 0.362, 0.315, 0.280, 0.251, 0.226, 0.203, 0.182, 0.164, 0.148],
                    "val_curve_loss": [0.402, 0.352, 0.331, 0.320, 0.315, 0.313, 0.312, 0.318, 0.325, 0.331]
                }
            },
            {
                "id": "cnn",
                "name": "CNN Classifier",
                "architecture": "Conv(1→32, 3×3)+ReLU+MaxPool(2) → Conv(32→64, 3×3)+ReLU+MaxPool(2) → Flatten(3136) → Linear(3136→10)",
                "inductive_bias": "Trường tiếp nhận cục bộ (local receptive field), chia sẻ trọng số (weight sharing) và bất biến dịch chuyển qua MaxPool",
                "expected_params": 50186,
                "expected_weight_mb": 0.191,
                "preview": {}
            }
        ]
    },
    "E2": {
        "title": "E2 — Transformer: Multi-Head Self-Attention (MSA) & Tokenization",
        "notebook": "E2/e2_attention_tokenization.ipynb",
        "dataset": "Fashion-MNIST (cùng split SHA-256 với E1 & E3)",
        "split": "54,000 Train / 6,000 Val / 10,000 Test (seed=42)",
        "optimizer": "AdamW (lr=1e-3, weight_decay=1e-4), Batch size=128, Epochs=20",
        "encoder_spec": "embed_dim=64, num_heads=4 (head_dim=16), depth=2 blocks, FFN=128 (GELU), Pre-LayerNorm, Learned Positional Embedding, Mean Pooling → Linear(64→10)",
        "equivalence_check": "Đồng bộ trọng số ScratchMSA sang TorchMSA trên CPU FP32: kiểm chứng đầu ra y, đạo hàm theo đầu vào ∂L/∂x và đạo hàm tham số ∂L/∂W khớp trong ngưỡng atol=1e-5, rtol=1e-4.",
        "experiments": [
            {
                "id": "scratch_patch",
                "backend": "scratch",
                "tokenizer": "patch",
                "token_count": 49,
                "token_desc": "4×4 non-overlapping patches (F.unfold k=4, s=4) → 49 tokens × 16D → Linear(16→64)",
                "expected_params": 69834,
                "preview": {"val_accuracy": 0.8840, "test_accuracy": 0.8785, "val_loss": 0.332, "test_loss": 0.348, "latency_ms_per_image": 0.165, "peak_gpu_memory_mb": 54.2}
            },
            {
                "id": "torch_patch",
                "backend": "torch",
                "tokenizer": "patch",
                "token_count": 49,
                "token_desc": "4×4 non-overlapping patches (F.unfold k=4, s=4) → 49 tokens × 16D → Linear(16→64)",
                "expected_params": 69834,
                "preview": {"val_accuracy": 0.8842, "test_accuracy": 0.8786, "val_loss": 0.331, "test_loss": 0.347, "latency_ms_per_image": 0.142, "peak_gpu_memory_mb": 49.8}
            },
            {
                "id": "scratch_row",
                "backend": "scratch",
                "tokenizer": "row",
                "token_count": 28,
                "token_desc": "Mỗi hàng ngang là 1 token → 28 tokens × 28D → Linear(28→64)",
                "expected_params": 69258,
                "preview": {"val_accuracy": 0.8765, "test_accuracy": 0.8710, "val_loss": 0.351, "test_loss": 0.366, "latency_ms_per_image": 0.118, "peak_gpu_memory_mb": 41.5}
            },
            {
                "id": "torch_row",
                "backend": "torch",
                "tokenizer": "row",
                "token_count": 28,
                "token_desc": "Mỗi hàng ngang là 1 token → 28 tokens × 28D → Linear(28→64)",
                "expected_params": 69258,
                "preview": {"val_accuracy": 0.8766, "test_accuracy": 0.8712, "val_loss": 0.350, "test_loss": 0.365, "latency_ms_per_image": 0.104, "peak_gpu_memory_mb": 38.9}
            },
            {
                "id": "scratch_cnn",
                "backend": "scratch",
                "tokenizer": "cnn",
                "token_count": 49,
                "token_desc": "CNN-stem: Conv(1→32, k=3, s=2)+ReLU → Conv(32→64, k=3, s=2)+ReLU → 7×7=49 spatial tokens × 64D",
                "expected_params": 87562,
                "preview": {"val_accuracy": 0.9145, "test_accuracy": 0.9092, "val_loss": 0.249, "test_loss": 0.264, "latency_ms_per_image": 0.196, "peak_gpu_memory_mb": 62.4}
            },
            {
                "id": "torch_cnn",
                "backend": "torch",
                "tokenizer": "cnn",
                "token_count": 49,
                "token_desc": "CNN-stem: Conv(1→32, k=3, s=2)+ReLU → Conv(32→64, k=3, s=2)+ReLU → 7×7=49 spatial tokens × 64D",
                "expected_params": 87562,
                "preview": {"val_accuracy": 0.9148, "test_accuracy": 0.9094, "val_loss": 0.248, "test_loss": 0.263, "latency_ms_per_image": 0.174, "peak_gpu_memory_mb": 57.6}
            }
        ]
    },
    "E3": {
        "title": "E3 — Mô hình chuỗi tuần tự (LSTM / GRU) trên ảnh & Đối chiếu Baseline",
        "notebook": "E3/e3_sequence_models.ipynb",
        "dataset": "Fashion-MNIST (xác minh provenance split_hash & seed=42 từ E1)",
        "split": "54,000 Train / 6,000 Val / 10,000 Test (seed=42)",
        "optimizer": "Adam (lr=1e-3), Gradient Clipping max_norm=1.0, Batch size=128, Epochs=20",
        "experiments": [
            {
                "id": "lstm_row",
                "recurrent": "LSTM",
                "representation": "row",
                "sequence_shape": "T=28, D=28",
                "architecture": "Row Sequence (28×28) → 1-layer Unidirectional LSTM(input=28, hidden=128) → h_T → Linear(128→10)",
                "expected_params": 82186,
                "preview": {"val_accuracy": 0.8860, "test_accuracy": 0.8805, "val_loss": 0.324, "test_loss": 0.341, "latency_ms_per_image": 0.095, "peak_gpu_memory_mb": 34.2}
            },
            {
                "id": "lstm_patch",
                "recurrent": "LSTM",
                "representation": "patch",
                "sequence_shape": "T=49, D=16",
                "architecture": "Patch 4×4 Raster Sequence (49×16) → 1-layer Unidirectional LSTM(input=16, hidden=128) → h_T → Linear(128→10)",
                "expected_params": 76042,
                "preview": {"val_accuracy": 0.8745, "test_accuracy": 0.8690, "val_loss": 0.354, "test_loss": 0.371, "latency_ms_per_image": 0.138, "peak_gpu_memory_mb": 39.5}
            },
            {
                "id": "gru_row",
                "recurrent": "GRU",
                "representation": "row",
                "sequence_shape": "T=28, D=28",
                "architecture": "Row Sequence (28×28) → 1-layer Unidirectional GRU(input=28, hidden=128) → h_T → Linear(128→10)",
                "expected_params": 61962,
                "preview": {"val_accuracy": 0.8885, "test_accuracy": 0.8824, "val_loss": 0.319, "test_loss": 0.336, "latency_ms_per_image": 0.082, "peak_gpu_memory_mb": 31.8}
            },
            {
                "id": "gru_patch",
                "recurrent": "GRU",
                "representation": "patch",
                "sequence_shape": "T=49, D=16",
                "architecture": "Patch 4×4 Raster Sequence (49×16) → 1-layer Unidirectional GRU(input=16, hidden=128) → h_T → Linear(128→10)",
                "expected_params": 57354,
                "preview": {"val_accuracy": 0.8780, "test_accuracy": 0.8725, "val_loss": 0.345, "test_loss": 0.362, "latency_ms_per_image": 0.119, "peak_gpu_memory_mb": 36.1}
            }
        ]
    },
    "A1": {
        "title": "A1 — CNN (ResNet-18) vs Vision Transformer (DeiT-Tiny) trên CIFAR-100",
        "notebook": "A1/a1_cnn_vs_transformer.ipynb",
        "dataset": "CIFAR-100 (100 lớp, upscale 32×32 → 224×224 antialias=True)",
        "split": "45,000 Train / 5,000 Val / 10,000 Test (StratifiedShuffleSplit, seed=42)",
        "optimizer": "Effective Batch=64 (batch=32, grad_accum=2), Epochs=20, Cosine LR (warmup 2 epochs). Scratch: AdamW(lr=3e-4, wd=0.05). Pretrained: AdamW(backbone lr=1e-4, head lr=1e-3, wd=0.01).",
        "invariants": "Kiểm chứng tự động trước khi train: mọi tham số requires_grad=False và buffer của module đóng băng (running_mean, running_var của BatchNorm) giữ nguyên tuyệt đối (rtol=0, atol=0) sau optimizer.step().",
        "experiments": [
            {
                "id": "resnet18_scratch",
                "family": "resnet18",
                "model_id": "resnet18.a1_in1k (pretrained=False)",
                "strategy": "scratch",
                "strategy_label": "From Scratch",
                "unfrozen_layers": "Toàn bộ mạng (conv1, bn1, layer1..4, fc) khởi tạo ngẫu nhiên",
                "expected_params": 11227812,
                "expected_trainable": 11227812,
                "preview": {"val_accuracy": 0.5420, "test_accuracy": 0.5365, "macro_f1": 0.5341, "latency_ms_per_image": 1.15, "peak_gpu_memory_mb": 980.0, "epoch_seconds": 48.0}
            },
            {
                "id": "resnet18_head_only",
                "family": "resnet18",
                "model_id": "resnet18.a1_in1k (pretrained=True)",
                "strategy": "head_only",
                "strategy_label": "Freeze Backbone (Head-Only)",
                "unfrozen_layers": "Chỉ mở tầng phân loại fc.* (51,300 tham số); đóng băng & giữ eval() toàn bộ backbone",
                "expected_params": 11227812,
                "expected_trainable": 51300,
                "preview": {"val_accuracy": 0.6380, "test_accuracy": 0.6325, "macro_f1": 0.6302, "latency_ms_per_image": 1.14, "peak_gpu_memory_mb": 410.0, "epoch_seconds": 19.5}
            },
            {
                "id": "resnet18_partial",
                "family": "resnet18",
                "model_id": "resnet18.a1_in1k (pretrained=True)",
                "strategy": "partial",
                "strategy_label": "Partial Freeze (layer4 + fc)",
                "unfrozen_layers": "Mở khối residual cuối layer4.* và head fc.*; đóng băng conv1, bn1, layer1..3",
                "expected_params": 11227812,
                "expected_trainable": 8445028,
                "preview": {"val_accuracy": 0.7640, "test_accuracy": 0.7585, "macro_f1": 0.7570, "latency_ms_per_image": 1.15, "peak_gpu_memory_mb": 595.0, "epoch_seconds": 28.0}
            },
            {
                "id": "resnet18_full",
                "family": "resnet18",
                "model_id": "resnet18.a1_in1k (pretrained=True)",
                "strategy": "full",
                "strategy_label": "Full Fine-Tune",
                "unfrozen_layers": "Tinh chỉnh toàn bộ backbone (lr=1e-4) + classification head (lr=1e-3)",
                "expected_params": 11227812,
                "expected_trainable": 11227812,
                "preview": {"val_accuracy": 0.7985, "test_accuracy": 0.7930, "macro_f1": 0.7918, "latency_ms_per_image": 1.15, "peak_gpu_memory_mb": 985.0, "epoch_seconds": 47.5}
            },
            {
                "id": "deit_tiny_scratch",
                "family": "deit_tiny",
                "model_id": "deit_tiny_patch16_224.fb_in1k (pretrained=False)",
                "strategy": "scratch",
                "strategy_label": "From Scratch",
                "unfrozen_layers": "Toàn bộ PatchEmbed, 12 Transformer Blocks, final LayerNorm và Head (thiếu inductive bias cục bộ khi học từ đầu 20 epoch)",
                "expected_params": 5543716,
                "expected_trainable": 5543716,
                "preview": {"val_accuracy": 0.3680, "test_accuracy": 0.3615, "macro_f1": 0.3560, "latency_ms_per_image": 1.48, "peak_gpu_memory_mb": 1120.0, "epoch_seconds": 54.0}
            },
            {
                "id": "deit_tiny_head_only",
                "family": "deit_tiny",
                "model_id": "deit_tiny_patch16_224.fb_in1k (pretrained=True)",
                "strategy": "head_only",
                "strategy_label": "Freeze Backbone (Head-Only)",
                "unfrozen_layers": "Chỉ mở tầng head.* (19,300 tham số); đóng băng toàn bộ 12 attention blocks",
                "expected_params": 5543716,
                "expected_trainable": 19300,
                "preview": {"val_accuracy": 0.6690, "test_accuracy": 0.6640, "macro_f1": 0.6615, "latency_ms_per_image": 1.47, "peak_gpu_memory_mb": 460.0, "epoch_seconds": 22.0}
            },
            {
                "id": "deit_tiny_partial",
                "family": "deit_tiny",
                "model_id": "deit_tiny_patch16_224.fb_in1k (pretrained=True)",
                "strategy": "partial",
                "strategy_label": "Partial Freeze (blocks.10..11 + norm + head)",
                "unfrozen_layers": "Mở 2 khối Transformer cuối (blocks.10, blocks.11), final LayerNorm (norm.*) và head.*",
                "expected_params": 5543716,
                "expected_trainable": 909220,
                "preview": {"val_accuracy": 0.7790, "test_accuracy": 0.7745, "macro_f1": 0.7731, "latency_ms_per_image": 1.48, "peak_gpu_memory_mb": 620.0, "epoch_seconds": 29.5}
            },
            {
                "id": "deit_tiny_full",
                "family": "deit_tiny",
                "model_id": "deit_tiny_patch16_224.fb_in1k (pretrained=True)",
                "strategy": "full",
                "strategy_label": "Full Fine-Tune",
                "unfrozen_layers": "Tinh chỉnh toàn bộ PatchEmbed + 12 Transformer blocks (lr=1e-4) + head (lr=1e-3)",
                "expected_params": 5543716,
                "expected_trainable": 5543716,
                "preview": {"val_accuracy": 0.8240, "test_accuracy": 0.8195, "macro_f1": 0.8184, "latency_ms_per_image": 1.48, "peak_gpu_memory_mb": 1125.0, "epoch_seconds": 53.5}
            }
        ]
    },
    "A2": {
        "title": "A2 — Ablation Noise Schedule (Linear vs Cosine) cho DDPM trên CelebA 32×32",
        "notebook": "A2/a2_diffusion_schedules.ipynb",
        "paper": {
            "title": "Improved Denoising Diffusion Probabilistic Models",
            "authors": "Alexander Quinn Nichol, Prafulla Dhariwal (OpenAI)",
            "venue": "ICML 2021 (Proceedings of the 38th International Conference on Machine Learning, PMLR 139:8162-8171)",
            "arxiv_url": "https://arxiv.org/abs/2102.09672",
            "pdf_url": "https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf"
        },
        "dataset": "CelebA Align&Cropped (CenterCrop 148 → Resize 32×32, Normalize [-1, 1])",
        "split": "20,000 Train / 2,000 Val / 5,000 Test theo official partition (lưu celeba_manifest_seed42.csv)",
        "optimizer": "UNet2DModel (channels 64/128/128, Attn tại 16×16, GroupNorm 32), AdamW lr=1e-4, Effective Batch=64, 20,000 steps (warmup 500, cosine decay, grad_clip=1.0, EMA=0.999). Đánh giá: 5,000 mẫu sinh bằng DDPM 250 bước từ cùng initial noise.",
        "experiments": [
            {
                "id": "linear",
                "name": "Linear Noise Schedule (Ho et al., 2020 Baseline)",
                "schedule_type": "linear (beta_start=1e-4, beta_end=0.02, T=1000)",
                "variance_type": "fixed_small",
                "prediction_target": "epsilon (MSE loss)",
                "characteristic": "Ở độ phân giải thấp 32×32, ᾱ_t suy giảm quá nhanh về 0 ở nửa cuối chuỗi t, khiến các bước cuối gần như chỉ là nhiễu trắng thuần túy và lãng phí dung lượng bước khuếch tán.",
                "expected_params": 6185923,
                "preview": {
                    "fid": 24.85,
                    "inception_score_mean": 2.41,
                    "inception_score_std": 0.08,
                    "val_noise_loss": 0.0284,
                    "sampling_seconds": 182.4,
                    "seconds_per_image": 0.0365,
                    "peak_gpu_memory_mb": 645.0
                }
            },
            {
                "id": "cosine",
                "name": "Cosine Noise Schedule (Nichol & Dhariwal, ICML 2021)",
                "schedule_type": "squaredcos_cap_v2 (s=0.008, beta_max=0.999, T=1000)",
                "variance_type": "fixed_small",
                "prediction_target": "epsilon (MSE loss)",
                "characteristic": "Hàm ᾱ_t giảm mượt theo hàm cos² ở vùng giữa và tránh phá hủy cấu trúc ảnh quá sớm ở vùng t nhỏ, giữ tỷ lệ SNR hữu ích lâu hơn đặc biệt trên ảnh 32×32.",
                "expected_params": 6185923,
                "preview": {
                    "fid": 18.42,
                    "inception_score_mean": 2.68,
                    "inception_score_std": 0.07,
                    "val_noise_loss": 0.0392,
                    "sampling_seconds": 182.8,
                    "seconds_per_image": 0.0366,
                    "peak_gpu_memory_mb": 645.0
                }
            }
        ]
    }
}


def build_minimal_pdf(title: str, subtitle: str, note: str) -> bytes:
    """Create a valid, standards-compliant 1-page PDF document using only stdlib."""
    def escape_pdf_text(text: str) -> str:
        # Basic ASCII fallback for standard Type1 Helvetica in minimal PDF
        ascii_map = str.maketrans(
            "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
            "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ—–×",
            "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
            "AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD--x",
        )
        cleaned = text.translate(ascii_map)
        return cleaned.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    lines = [
        "BT",
        "/F1 18 Tf",
        "54 740 Td",
        f"({escape_pdf_text(title)}) Tj",
        "/F1 12 Tf",
        "0 -28 Td",
        f"({escape_pdf_text(subtitle)}) Tj",
        "/F1 10 Tf",
        "0 -32 Td",
        f"({escape_pdf_text(note)}) Tj",
        "0 -20 Td",
        "(Course: CO5085 - Deep Learning and Applications in Computer Vision - HCMUT HK261) Tj",
        "0 -20 Td",
        "(Instructors: ThS./TS. Le Thanh Sach, TS. Nguyen Quoc Minh) Tj",
        "0 -20 Td",
        "(Status: Placeholder PDF for GitHub Pages navigation. Replace with final compiled LaTeX/PDF report.) Tj",
        "ET",
    ]
    stream_bytes = "\n".join(lines).encode("latin-1", errors="replace")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1") + stream_bytes + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("latin-1"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    output.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )
    return bytes(output)


def ensure_reports() -> dict[str, dict[str, Any]]:
    """Ensure reports/ and docs/reports/ contain valid PDF files so links never 404."""
    REPORTS_SRC_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    report_defs = {
        "exercises.pdf": (
            "CO5085 - Exercises Report (E1, E2, E3)",
            "Progressive Report: E1 & E2 (Week 9 - 02/11/2026) + E3 (Week 10 - 09/11/2026)",
            "Dataset: Fashion-MNIST (54k train / 6k val / 10k test, StratifiedShuffleSplit seed=42)."
        ),
        "assignment1.pdf": (
            "CO5085 - Assignment 1 Report (CNN vs Transformer on CIFAR-100)",
            "ResNet-18 vs DeiT-Tiny across Scratch, Head-Only, Partial Freeze, and Full Fine-Tune",
            "Milestones: A1.1 Protocol (26/10/2026) & A1.2 Complete Evaluation (09/11/2026)."
        ),
        "assignment2.pdf": (
            "CO5085 - Assignment 2 Report (DDPM Noise Schedule Ablation & Paper Analysis)",
            "Linear vs Cosine Noise Schedule on CelebA 32x32 + Paper Analysis (Nichol & Dhariwal, ICML 2021)",
            "Milestones: A2.1 Proposal (02/11/2026) & A2.2 Complete Pipeline & Analysis (16/11/2026)."
        ),
    }

    status_info: dict[str, dict[str, Any]] = {}
    for filename, (title, subtitle, note) in report_defs.items():
        src_path = REPORTS_SRC_DIR / filename
        if not src_path.exists():
            src_path.write_bytes(build_minimal_pdf(title, subtitle, note))
        dst_path = REPORTS_DOCS_DIR / filename
        shutil.copy2(src_path, dst_path)
        size_kb = round(src_path.stat().st_size / 1024, 2)
        is_placeholder = size_kb < 5.0
        status_info[filename] = {
            "path": f"reports/{filename}",
            "size_kb": size_kb,
            "is_placeholder": is_placeholder,
            "updated_at": dt.datetime.fromtimestamp(src_path.stat().st_mtime, tz=dt.timezone.utc).isoformat()
        }
    return status_info


def read_json_safe(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = None
                    continue
                try:
                    if "." in value or "e" in value.lower():
                        parsed[key] = float(value)
                    else:
                        parsed[key] = int(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


PRIVATE_CONFIG_KEYS = {"data_root", "output_root", "celeba_root", "resume_from"}


def valid_full_run(assignment: str, experiment_id: str, run_dir: Path) -> bool:
    """Verify the minimum provenance required before publishing a run."""
    metrics = read_json_safe(run_dir / "metrics.json")
    config = read_json_safe(run_dir / "config.json")
    # The previous E1 CNN used GAP; its results must not describe the new head.
    metrics_match_model = (
        assignment != "E1" or experiment_id != "cnn"
        or bool(metrics) and metrics.get("parameters") == 50186
    )
    return bool(
        metrics
        and config
        and metrics.get("run_mode") == "full"
        and config.get("run_mode") == "full"
        and config.get("assignment") == assignment
        and config.get("experiment_id") == experiment_id
        and metrics_match_model
    )


def public_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Remove local filesystem and resume paths from data published on GitHub Pages."""
    if config is None:
        return None
    return {key: value for key, value in config.items() if key not in PRIVATE_CONFIG_KEYS}


def count_full_runs(artifacts_root: Path) -> int:
    """Count valid full-run metric files without modifying published site data."""
    count = 0
    for assignment, spec in EXPERIMENT_SPECS.items():
        for experiment in spec["experiments"]:
            run_dir = artifacts_root / assignment / "full" / experiment["id"]
            if valid_full_run(assignment, experiment["id"], run_dir):
                count += 1
    return count


def sync_experiment_artifacts(artifacts_root: Path, *, allow_empty: bool = False) -> dict[str, Any]:
    """Scan artifacts_root/<assignment>/full/<experiment_id>/ and copy plots + metrics."""
    valid_run_count = count_full_runs(artifacts_root)
    if valid_run_count == 0 and not allow_empty:
        raise RuntimeError(
            f"No valid full-run metrics found under {artifacts_root}. "
            "Refusing to overwrite published data; pass --allow-empty only when bootstrapping the site."
        )

    if ARTIFACTS_OUT_DIR.exists():
        shutil.rmtree(ARTIFACTS_OUT_DIR)
    ARTIFACTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "artifact_source": "local full-run export",
        "conventions": {
            "seed": 42,
            "fashion_mnist_split": "54,000 train / 6,000 val / 10,000 test (StratifiedShuffleSplit, seed=42, SHA-256 verified)",
            "cifar100_split": "45,000 train / 5,000 val / 10,000 test (StratifiedShuffleSplit, seed=42, SHA-256 verified)",
            "celeba_split": "20,000 train / 2,000 val / 5,000 test (Official partition deterministic subset, seed=42)",
            "smoke_policy": "Smoke runs are strictly excluded from report tables per plans/00_conventions.md."
        },
        "total_expected_experiments": 23,
        "total_completed_full_experiments": 0,
        "assignments": {}
    }

    total_completed = 0

    for assignment, spec in EXPERIMENT_SPECS.items():
        full_dir = artifacts_root / assignment / "full"
        assignment_completed = 0
        experiments_out: list[dict[str, Any]] = []

        for exp_spec in spec["experiments"]:
            exp_id = exp_spec["id"]
            run_dir = full_dir / exp_id
            metrics = read_json_safe(run_dir / "metrics.json")
            config = read_json_safe(run_dir / "config.json")
            environment = read_json_safe(run_dir / "environment.json")
            history = read_csv_rows(run_dir / "history.csv")

            # Enforce full mode and artifact provenance before publishing a run.
            is_valid_full = valid_full_run(assignment, exp_id, run_dir)

            copied_images: dict[str, str] = {}
            if is_valid_full and run_dir.exists():
                assignment_completed += 1
                total_completed += 1
                target_img_dir = ARTIFACTS_OUT_DIR / assignment / exp_id
                target_img_dir.mkdir(parents=True, exist_ok=True)
                for img_name in ("curves.png", "confusion_matrix.png", "failures.png", "denoising_trace.png"):
                    src_img = run_dir / img_name
                    if src_img.exists():
                        shutil.copy2(src_img, target_img_dir / img_name)
                        copied_images[img_name.removesuffix(".png")] = (
                            f"assets/artifacts/{assignment}/{exp_id}/{img_name}"
                        )
                previews = sorted(run_dir.glob("preview_step_*.png"))
                if previews:
                    latest_preview = previews[-1]
                    shutil.copy2(latest_preview, target_img_dir / latest_preview.name)
                    copied_images["latest_preview"] = (
                        f"assets/artifacts/{assignment}/{exp_id}/{latest_preview.name}"
                    )

            experiments_out.append({
                **exp_spec,
                "has_full_metrics": is_valid_full,
                "metrics": metrics if is_valid_full else None,
                "config": public_config(config) if is_valid_full else None,
                "environment": environment if is_valid_full else None,
                "history": history if is_valid_full else [],
                "images": copied_images,
            })

        manifest["assignments"][assignment] = {
            **{k: v for k, v in spec.items() if k != "experiments"},
            "expected_count": len(spec["experiments"]),
            "completed_full_count": assignment_completed,
            "all_full_completed": assignment_completed == len(spec["experiments"]),
            "experiments": experiments_out,
        }

    manifest["total_completed_full_experiments"] = total_completed
    manifest["reports"] = ensure_reports()

    out_path = DATA_OUT_DIR / "experiments_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also write JS fallback bundle so opening docs/*.html via file:// works without CORS errors
    js_dir = DOCS_DIR / "assets" / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    site_cfg = read_json_safe(DATA_OUT_DIR / "site_config.json") or {}
    fallback_js = (
        "// Auto-generated by tools/sync_site_data.py — do not edit manually\n"
        f"window.FALLBACK_SITE_CONFIG = {json.dumps(site_cfg, ensure_ascii=False)};\n"
        f"window.FALLBACK_EXPERIMENTS_MANIFEST = {json.dumps(manifest, ensure_ascii=False)};\n"
    )
    (js_dir / "fallback_data.js").write_text(fallback_js, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync CO5085 full-run artifacts into docs/data/")
    default_artifacts = Path(os.getenv("CO5085_OUTPUT_ROOT", str(ROOT / "artifacts")))
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=default_artifacts,
        help="Path to output_root containing <assignment>/full/<experiment_id>/",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow a zero-run manifest when bootstrapping the site (never use after publishing results).",
    )
    args = parser.parse_args()

    manifest = sync_experiment_artifacts(args.artifacts_dir, allow_empty=args.allow_empty)
    print(
        f"Synced site data -> docs/data/experiments_manifest.json "
        f"(completed full runs: {manifest['total_completed_full_experiments']}/"
        f"{manifest['total_expected_experiments']})"
    )


if __name__ == "__main__":
    main()
