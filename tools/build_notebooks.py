#!/usr/bin/env python3
"""Generate the five self-contained CO5085 notebooks with only stdlib."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def describe_cell(title: str, objective: str, result: str, note: str | None = None) -> dict:
    """Create the short Markdown guide placed immediately before a code cell."""
    lines = [f"### {title}", "", f"**Mục tiêu:** {objective}", "", f"**Kết quả:** {result}"]
    if note:
        lines.extend(["", f"**Lưu ý:** {note}"])
    return md("\n".join(lines))


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


INSTALL_BASE = r'''
import importlib.util, subprocess, sys

required = {
    "torch": "torch>=2.2,<3",
    "torchvision": "torchvision>=0.17,<1",
    "numpy": "numpy>=1.26,<3",
    "pandas": "pandas>=2.1,<4",
    "matplotlib": "matplotlib>=3.8,<4",
    "seaborn": "seaborn>=0.13,<1",
    "sklearn": "scikit-learn>=1.4,<2",
    "tqdm": "tqdm>=4.66,<5",
}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
print("Dependencies ready")
'''


COMMON_IMPORTS = r'''
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
print("device:", DEVICE)
'''


COMMON_CONFIG = r'''
def default_root(kind):
    env_name = "CO5085_DATA_ROOT" if kind == "data" else "CO5085_OUTPUT_ROOT"
    if os.getenv(env_name):
        return Path(os.environ[env_name])
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / ("data" if kind == "data" else "artifacts")
    if Path("/content").exists():
        return Path("/content") / ("data" if kind == "data" else "artifacts")
    return Path("./data" if kind == "data" else "./artifacts")


CONFIG = {
    "data_root": str(default_root("data")),
    "output_root": str(default_root("output")),
    "run_mode": os.getenv("CO5085_RUN_MODE", "smoke"),
    "experiment_ids": [],  # Empty means all experiments for this notebook.
    "seed": 42,
    "batch_size": 128,
    "epochs": 20,
    "resume_from": None,  # None, "last", or a checkpoint path.
    "evaluate_only": os.getenv("CO5085_EVALUATE_ONLY", "0") == "1",
    "num_workers": 2,
}
assert CONFIG["run_mode"] in {"smoke", "full"}
if CONFIG["run_mode"] == "smoke":
    CONFIG["epochs"] = 1
    CONFIG["num_workers"] = 0
print(json.dumps(CONFIG, indent=2))
'''


COMMON_UTILS = r'''
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def environment_info():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    return {"parameters": total, "trainable_parameters": trainable, "weight_size_mb": size_mb}


def split_digest(train_idx, val_idx):
    digest = hashlib.sha256()
    digest.update(np.asarray(train_idx, dtype=np.int64).tobytes())
    digest.update(np.asarray(val_idx, dtype=np.int64).tobytes())
    return digest.hexdigest()


def fashion_split(dataset, data_root, seed=42):
    split_path = Path(data_root) / "fashion_mnist_split_seed42.npz"
    labels = np.asarray(dataset.targets)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=6000, random_state=seed)
    expected_train, expected_val = next(splitter.split(np.zeros(len(labels)), labels))
    if split_path.exists():
        saved = np.load(split_path)
        train_idx, val_idx = saved["train_idx"], saved["val_idx"]
        if not (np.array_equal(train_idx, expected_train) and np.array_equal(val_idx, expected_val)):
            raise RuntimeError(f"Split file does not match seed={seed}: {split_path}")
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(split_path, train_idx=expected_train, val_idx=expected_val)
        train_idx, val_idx = expected_train, expected_val
    assert len(train_idx) == 54000 and len(val_idx) == 6000
    assert not set(train_idx).intersection(set(val_idx))
    return train_idx, val_idx, split_digest(train_idx, val_idx)


def classification_epoch(model, loader, criterion, optimizer=None, scaler=None,
                         max_batches=None, grad_clip=None, grad_accum=1, frozen_eval=None):
    training = optimizer is not None
    model.train(training)
    if training and frozen_eval is not None:
        frozen_eval(model)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss = total_correct = total_items = 0
    accumulation_count = 0
    start = time.perf_counter()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(images)
                loss = criterion(logits, labels)
                backward_loss = loss / grad_accum
            if training:
                scaler.scale(backward_loss).backward()
                accumulation_count += 1
                should_step = accumulation_count == grad_accum
                if should_step:
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_count = 0
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_items += labels.size(0)
        if training and accumulation_count:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - start
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items,
            "items": total_items, "seconds": elapsed}


@torch.no_grad()
def predict_classifier(model, loader, max_batches=None):
    model.eval()
    labels_all, predictions_all, probabilities_all = [], [], []
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        logits = model(images.to(DEVICE))
        probabilities = logits.softmax(1).cpu()
        labels_all.append(labels.cpu())
        predictions_all.append(probabilities.argmax(1))
        probabilities_all.append(probabilities)
    return (torch.cat(labels_all).numpy(), torch.cat(predictions_all).numpy(),
            torch.cat(probabilities_all).numpy())


def resolve_resume(run_dir, resume_from):
    if resume_from is None:
        return None
    if resume_from == "last":
        candidate = Path(run_dir) / "last.pt"
    else:
        candidate = Path(resume_from)
    return candidate if candidate.exists() else None


def fit_classifier(model, loaders, optimizer, scheduler, run_dir, run_config,
                   grad_clip=None, grad_accum=1, frozen_eval=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dump(run_config, run_dir / "config.json")
    json_dump(environment_info(), run_dir / "environment.json")
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    start_epoch, best_accuracy, best_loss, history = 0, -1.0, float("inf"), []
    if run_config.get("evaluate_only"):
        evaluation_path = run_dir / "best.pt"
        if not evaluation_path.exists():
            raise FileNotFoundError(f"evaluate_only requires {evaluation_path}")
        checkpoint = torch.load(evaluation_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        start_epoch = run_config["epochs"]
        best_accuracy, best_loss = checkpoint["best_accuracy"], checkpoint["best_loss"]
        history_path = run_dir / "history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
    else:
        resume_path = resolve_resume(run_dir, run_config.get("resume_from"))
        checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False) if resume_path is not None else None
    if not run_config.get("evaluate_only") and checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy, best_loss = checkpoint["best_accuracy"], checkpoint["best_loss"]
        history_path = run_dir / "history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, run_config["epochs"]):
        train_stats = classification_epoch(
            model, loaders["train"], criterion, optimizer, scaler,
            max_batches=run_config.get("max_train_batches"), grad_clip=grad_clip,
            grad_accum=grad_accum, frozen_eval=frozen_eval,
        )
        val_stats = classification_epoch(
            model, loaders["val"], criterion,
            max_batches=run_config.get("max_eval_batches"),
        )
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"], "train_accuracy": train_stats["accuracy"],
            "val_loss": val_stats["loss"], "val_accuracy": val_stats["accuracy"],
            "epoch_seconds": train_stats["seconds"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        improved = (val_stats["accuracy"] > best_accuracy or
                    (val_stats["accuracy"] == best_accuracy and val_stats["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss = val_stats["accuracy"], val_stats["loss"]
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(), "epoch": epoch,
            "best_accuracy": best_accuracy, "best_loss": best_loss,
            "rng_state": rng_state(), "config": run_config,
        }
        atomic_torch_save(state, run_dir / "last.pt")
        if improved:
            atomic_torch_save(state, run_dir / "best.pt")
        print(f"epoch={epoch + 1}/{run_config['epochs']} train={train_stats['accuracy']:.4f} val={val_stats['accuracy']:.4f}")

    best = torch.load(run_dir / "best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model"])
    val_stats = classification_epoch(model, loaders["val"], criterion,
                                     max_batches=run_config.get("max_eval_batches"))
    test_stats = classification_epoch(model, loaders["test"], criterion,
                                      max_batches=run_config.get("max_eval_batches"))
    metrics = {
        "run_mode": run_config["run_mode"], "best_epoch": best["epoch"],
        "val_loss": val_stats["loss"], "val_accuracy": val_stats["accuracy"],
        "test_loss": test_stats["loss"], "test_accuracy": test_stats["accuracy"],
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 2**20
                               if torch.cuda.is_available() else 0.0),
        **count_parameters(model),
    }
    y_true, y_pred, probabilities = predict_classifier(
        model, loaders["test"], max_batches=run_config.get("max_eval_batches"))
    np.savez_compressed(run_dir / "predictions.npz", y_true=y_true, y_pred=y_pred,
                        probabilities=probabilities)
    json_dump(metrics, run_dir / "metrics.json")
    return metrics, pd.DataFrame(history), (y_true, y_pred, probabilities)


@torch.no_grad()
def benchmark_model(model, sample, repeats=30, warmup=5):
    model.eval()
    sample = sample.to(DEVICE)
    for _ in range(warmup):
        model(sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        model(sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "latency_ms_per_image": 1000 * elapsed / (repeats * len(sample)),
        "benchmark_batch_size": len(sample),
        "benchmark_peak_memory_mb": (torch.cuda.max_memory_allocated() / 2**20
                                     if torch.cuda.is_available() else 0.0),
    }


def plot_history(history, path):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label="train")
    axes[0].plot(frame.epoch, frame.val_loss, label="validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[1].plot(frame.epoch, frame.train_accuracy, label="train")
    axes[1].plot(frame.epoch, frame.val_accuracy, label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch")
    for axis in axes:
        axis.legend(); axis.grid(alpha=.2)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def plot_confusion(y_true, y_pred, class_names, path, normalize=True):
    matrix = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    size = max(7, min(20, len(class_names) * .22))
    figure, axis = plt.subplots(figsize=(size, size))
    sns.heatmap(matrix, cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                annot=len(class_names) <= 10, fmt=".2f" if normalize else "d", ax=axis)
    axis.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def completed_metrics(run_dir, run_mode, expected_parameters=None):
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    if expected_parameters is not None and metrics.get("parameters") != expected_parameters:
        if CONFIG["evaluate_only"] or CONFIG["resume_from"] is not None:
            raise ValueError(
                "CNN architecture changed: start a fresh run with resume_from=None and "
                "evaluate_only=False, or select the output_root of a matching run."
            )
        print("Ignoring metrics from a different CNN architecture:", path)
        return None
    return metrics if metrics.get("run_mode") == run_mode else None


seed_everything(CONFIG["seed"])
Path(CONFIG["data_root"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["output_root"]).mkdir(parents=True, exist_ok=True)
'''


FASHION_DATA = r'''
raw_train = datasets.FashionMNIST(CONFIG["data_root"], train=True, download=True)
mean = (raw_train.data.float() / 255.0).mean().item()
std = (raw_train.data.float() / 255.0).std().item()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
train_full = datasets.FashionMNIST(CONFIG["data_root"], train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(CONFIG["data_root"], train=False, transform=transform, download=True)
train_idx, val_idx, SPLIT_HASH = fashion_split(raw_train, CONFIG["data_root"], CONFIG["seed"])
train_dataset, val_dataset = Subset(train_full, train_idx), Subset(train_full, val_idx)

loaders = {
    "train": DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                        num_workers=CONFIG["num_workers"], pin_memory=AMP_ENABLED),
    "val": DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                      num_workers=CONFIG["num_workers"], pin_memory=AMP_ENABLED),
    "test": DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                       num_workers=CONFIG["num_workers"], pin_memory=AMP_ENABLED),
}
CLASS_NAMES = list(raw_train.classes)
print({"mean": mean, "std": std, "split_hash": SPLIT_HASH,
       "sizes": [len(train_dataset), len(val_dataset), len(test_dataset)]})
'''


def write_notebook(relative_path: str, cells: list[dict]) -> None:
    destination = ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", destination.relative_to(ROOT))


def build_e1() -> None:
    models = r'''
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, images):
        return self.linear(images.flatten(1))


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.network(images)


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(64 * 7 * 7, 10)

    def forward(self, images):
        return self.head(self.features(images).flatten(1))


MODEL_FACTORIES = {"softmax": SoftmaxClassifier, "mlp": MLPClassifier, "cnn": CNNClassifier}
assert sum(parameter.numel() for parameter in CNNClassifier().parameters()) == 50186
assert sum(parameter.numel() for parameter in CNNClassifier().parameters()) < sum(
    parameter.numel() for parameter in MLPClassifier().parameters())
EXPERIMENT_IDS = list(MODEL_FACTORIES)
selected = CONFIG["experiment_ids"] or EXPERIMENT_IDS
unknown = set(selected) - set(EXPERIMENT_IDS)
if unknown:
    raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")
print("Selected:", selected)
'''
    sanity = r'''
def overfit_one_batch(factory, steps=30):
    seed_everything(CONFIG["seed"])
    model = factory().to(DEVICE)
    images, labels = next(iter(loaders["train"]))
    images, labels = images[:32].to(DEVICE), labels[:32].to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        if initial is None:
            initial = loss.item()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert loss.item() < initial, (initial, loss.item())
    return initial, loss.item()


for experiment_id in selected:
    print(experiment_id, "one-batch sanity:", overfit_one_batch(
        MODEL_FACTORIES[experiment_id], 10 if CONFIG["run_mode"] == "smoke" else 30))
'''
    run = r'''
summary = []
OUTPUT_BASE = Path(CONFIG["output_root"]) / "E1" / CONFIG["run_mode"]
for experiment_id in selected:
    run_dir = OUTPUT_BASE / experiment_id
    existing = completed_metrics(
        run_dir, CONFIG["run_mode"], expected_parameters=50186 if experiment_id == "cnn" else None)
    if existing is not None and not CONFIG["evaluate_only"]:
        print("skip completed:", experiment_id)
        summary.append({"experiment_id": experiment_id, **existing})
        continue
    seed_everything(CONFIG["seed"])
    model = MODEL_FACTORIES[experiment_id]().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    run_config = {
        **CONFIG, "assignment": "E1", "experiment_id": experiment_id,
        "split_hash": SPLIT_HASH, "normalization": {"mean": mean, "std": std},
        "max_train_batches": 4 if CONFIG["run_mode"] == "smoke" else None,
        "max_eval_batches": 2 if CONFIG["run_mode"] == "smoke" else None,
    }
    metrics, history, predictions = fit_classifier(
        model, loaders, optimizer, None, run_dir, run_config)
    sample_images, _ = next(iter(loaders["test"]))
    metrics.update(benchmark_model(model, sample_images[:32], repeats=5 if CONFIG["run_mode"] == "smoke" else 30))
    json_dump(metrics, run_dir / "metrics.json")
    plot_history(history, run_dir / "curves.png")
    plot_confusion(predictions[0], predictions[1], CLASS_NAMES, run_dir / "confusion_matrix.png")
    summary.append({"experiment_id": experiment_id, **metrics})

summary_frame = pd.DataFrame(summary).sort_values("experiment_id")
summary_frame.to_csv(OUTPUT_BASE / "summary.csv", index=False)
display(summary_frame)
'''
    errors = r'''
def plot_failures(dataset, predictions_path, path, count=16):
    saved = np.load(predictions_path)
    y_true, y_pred, probabilities = saved["y_true"], saved["y_pred"], saved["probabilities"]
    wrong = np.flatnonzero(y_true != y_pred)
    confidence = probabilities[np.arange(len(y_pred)), y_pred]
    chosen = wrong[np.argsort(confidence[wrong])[-count:][::-1]]
    rows = max(1, math.ceil(len(chosen) / 4))
    figure, axes = plt.subplots(rows, 4, figsize=(10, 2.6 * rows))
    for axis, index in zip(np.asarray(axes).ravel(), chosen):
        image, _ = dataset[index]
        axis.imshow(image.squeeze().numpy() * std + mean, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"true={CLASS_NAMES[y_true[index]]}\npred={CLASS_NAMES[y_pred[index]]} ({confidence[index]:.2f})")
        axis.axis("off")
    for axis in np.asarray(axes).ravel()[len(chosen):]:
        axis.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


for experiment_id in selected:
    run_dir = OUTPUT_BASE / experiment_id
    if (run_dir / "predictions.npz").exists():
        plot_failures(test_dataset, run_dir / "predictions.npz", run_dir / "failures.png")
print("Artifacts:", OUTPUT_BASE)
'''
    cells = [
        md('''# E1 — Softmax, MLP và CNN trên Fashion-MNIST

**Câu hỏi nghiên cứu:** tăng độ phức tạp từ mô hình tuyến tính đến CNN cải thiện accuracy và thay đổi loại lỗi như thế nào?

Notebook dùng cùng split 54k/6k/10k với E2–E3, tự viết training loop PyTorch và chỉ đưa kết quả `full` vào báo cáo.'''),
        md("## 1. Môi trường"),
        describe_cell(
            "1.1. Cài đặt thư viện",
            "Kiểm tra các dependency nền tảng và chỉ cài những gói còn thiếu trong runtime.",
            "Môi trường có đủ PyTorch, torchvision, NumPy, pandas, scikit-learn và các thư viện vẽ biểu đồ.",
            "Kaggle hoặc Colab có thể yêu cầu khởi động lại kernel nếu pip thay đổi package đã được nạp.",
        ), code(INSTALL_BASE),
        describe_cell(
            "1.2. Import và chọn thiết bị",
            "Nạp thư viện dùng chung, chọn CPU/GPU và xác định có bật mixed precision hay không.",
            "Tạo `DEVICE`, `AMP_ENABLED` và các namespace được dùng trong toàn notebook.",
        ), code(COMMON_IMPORTS),
        md("## 2. Cấu hình"),
        describe_cell(
            "2.1. Khai báo cấu hình chạy",
            "Thiết lập đường dẫn dữ liệu/artifact, seed, batch size, số epoch, chế độ `smoke` hoặc `full` và danh sách experiment.",
            "Tạo dictionary `CONFIG`; ở chế độ `smoke`, notebook tự giảm ngân sách để kiểm tra pipeline.",
        ), code(COMMON_CONFIG),
        describe_cell(
            "2.2. Khởi tạo tiện ích thực nghiệm",
            "Định nghĩa các hàm seed, hash split, checkpoint nguyên tử, resume, training loop, metric và biểu đồ dùng chung.",
            "Cung cấp toàn bộ hàm hạ tầng cho các cell dữ liệu, huấn luyện và đánh giá phía sau.",
            "Cell này chỉ định nghĩa hàm, chưa bắt đầu huấn luyện.",
        ), code(COMMON_UTILS),
        md("## 3. Dữ liệu, split và EDA"),
        describe_cell(
            "3.1. Chuẩn bị Fashion-MNIST",
            "Tải dữ liệu, tạo split phân tầng 54.000/6.000/10.000, tính thống kê chuẩn hóa từ train và dựng DataLoader.",
            "Tạo `train_dataset`, `val_dataset`, `test_dataset`, `loaders`, `CLASS_NAMES` và `SPLIT_HASH`.",
            "Nếu file split đã tồn tại nhưng khác seed 42, notebook dừng để bảo vệ tính tái lập.",
        ), code(FASHION_DATA),
        describe_cell(
            "3.2. Quan sát mẫu dữ liệu",
            "Hiển thị mười ảnh train sau khi hoàn nguyên chuẩn hóa để kiểm tra nhãn và chất lượng đầu vào.",
            "Sinh một lưới ảnh Fashion-MNIST phục vụ EDA; cell không thay đổi dataset.",
        ), code('''figure, axes = plt.subplots(2, 5, figsize=(11, 5))
for axis, (image, label) in zip(axes.ravel(), [train_full[i] for i in range(10)]):
    axis.imshow(image.squeeze() * std + mean, cmap="gray", vmin=0, vmax=1)
    axis.set_title(CLASS_NAMES[label]); axis.axis("off")
plt.tight_layout(); plt.show()'''),
        md("## 4. Mô hình"),
        describe_cell(
            "4.1. Định nghĩa ba baseline",
            "Cài đặt Softmax tuyến tính, MLP hai tầng ẩn và CNN hai block tích chập + Flatten 3136 + Linear với cùng đầu ra 10 logits.",
            "Tạo `MODEL_FACTORIES`, ba experiment ID và danh sách `selected` lấy từ cấu hình.",
            "CNN có 50.186 tham số so với 235.146 của MLP (ít hơn 78,7%). Head tuyến tính giữ thông tin vị trí trên feature map 7×7; accuracy cần đo bằng full run mới. Checkpoint CNN GAP cũ không tương thích.",
        ), code(models),
        md("## 5. Kiểm tra nhanh: overfit một batch và gradient hữu hạn"),
        describe_cell(
            "5.1. Chạy sanity check",
            "Huấn luyện từng mô hình được chọn trên một mini-batch nhỏ để xác nhận loss giảm và mọi gradient đều hữu hạn.",
            "In loss đầu/cuối; assertion sẽ dừng notebook nếu pipeline forward/backward có lỗi.",
        ), code(sanity),
        md("## 6. Huấn luyện và chọn checkpoint"),
        describe_cell(
            "6.1. Chạy ma trận E1",
            "Huấn luyện các experiment đã chọn, resume khi cần, đánh giá checkpoint tốt nhất và đo latency/bộ nhớ.",
            "Ghi config, history, checkpoint, predictions, metrics, learning curves, confusion matrix và `summary.csv`.",
            "Run đã hoàn tất sẽ được bỏ qua; chỉ kết quả `full` nên dùng trong báo cáo chính.",
        ), code(run),
        md("## 7. Đánh giá, phân tích lỗi và xuất artifact"),
        describe_cell(
            "7.1. Trực quan hóa dự đoán sai",
            "Chọn các mẫu test bị phân loại sai có độ tin cậy cao nhất và hoàn nguyên ảnh để quan sát.",
            "Lưu `failures.png` trong thư mục từng experiment và in đường dẫn artifact gốc.",
        ), code(errors),
        md('''## 8. Câu hỏi phân tích

1. Khoảng cách train/validation của ba mô hình cho biết gì về underfitting và overfitting?
2. Accuracy tăng thêm có tương xứng với số tham số, latency và dung lượng model không?
3. Những cặp lớp nào bị nhầm nhiều nhất? Quan sát nào trong ảnh giải thích được lỗi?
4. CNN sửa được loại lỗi nào của MLP, và vẫn thất bại ở loại lỗi nào?

Viết kết luận từ artifact của run `full`; không suy diễn từ smoke run.'''),
    ]
    write_notebook("E1/e1_fashion_mnist_baselines.ipynb", cells)


def build_e2() -> None:
    models = r'''
class PatchTokenizer(nn.Module):
    token_count = 49
    def __init__(self, embed_dim=64):
        super().__init__(); self.projection = nn.Linear(16, embed_dim)
    def forward(self, images):
        patches = F.unfold(images, kernel_size=4, stride=4).transpose(1, 2)
        return self.projection(patches)


class RowTokenizer(nn.Module):
    token_count = 28
    def __init__(self, embed_dim=64):
        super().__init__(); self.projection = nn.Linear(28, embed_dim)
    def forward(self, images):
        return self.projection(images.squeeze(1))


class CNNTokenizer(nn.Module):
    token_count = 49
    def __init__(self, embed_dim=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, embed_dim, 3, stride=2, padding=1), nn.ReLU())
    def forward(self, images):
        return self.stem(images).flatten(2).transpose(1, 2)


class ScratchMSA(nn.Module):
    def __init__(self, embed_dim=64, heads=4, dropout=0.0):
        super().__init__()
        assert embed_dim % heads == 0
        self.heads, self.head_dim = heads, embed_dim // heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, inputs):
        batch, tokens, width = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = self.dropout(attention.softmax(-1))
        context = (attention @ v).transpose(1, 2).reshape(batch, tokens, width)
        return self.out(context)


class TorchMSA(nn.Module):
    def __init__(self, embed_dim=64, heads=4, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
    def forward(self, inputs):
        return self.mha(inputs, inputs, inputs, need_weights=False)[0]


class EncoderBlock(nn.Module):
    def __init__(self, backend, embed_dim=64, heads=4, ffn_dim=128):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.attn = ScratchMSA(embed_dim, heads) if backend == "scratch" else TorchMSA(embed_dim, heads)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, embed_dim))
    def forward(self, inputs):
        hidden = inputs + self.attn(self.norm1(inputs))
        return hidden + self.ffn(self.norm2(hidden))


TOKENIZERS = {"patch": PatchTokenizer, "row": RowTokenizer, "cnn": CNNTokenizer}


class ImageAttentionClassifier(nn.Module):
    def __init__(self, backend, tokenizer_name, embed_dim=64, heads=4, depth=2, ffn_dim=128):
        super().__init__()
        self.backend, self.tokenizer_name = backend, tokenizer_name
        self.tokenizer = TOKENIZERS[tokenizer_name](embed_dim)
        self.position = nn.Parameter(torch.zeros(1, self.tokenizer.token_count, embed_dim))
        nn.init.trunc_normal_(self.position, std=.02)
        self.blocks = nn.ModuleList([EncoderBlock(backend, embed_dim, heads, ffn_dim) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 10)
    def forward(self, images):
        tokens = self.tokenizer(images) + self.position
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens).mean(1))


def copy_attention(scratch, torch_msa):
    torch_msa.mha.in_proj_weight.data.copy_(scratch.qkv.weight.data)
    torch_msa.mha.in_proj_bias.data.copy_(scratch.qkv.bias.data)
    torch_msa.mha.out_proj.weight.data.copy_(scratch.out.weight.data)
    torch_msa.mha.out_proj.bias.data.copy_(scratch.out.bias.data)


def make_model(backend, tokenizer_name):
    seed_everything(CONFIG["seed"])
    reference = ImageAttentionClassifier("scratch", tokenizer_name)
    if backend == "scratch":
        return reference
    target = ImageAttentionClassifier("torch", tokenizer_name)
    target.tokenizer.load_state_dict(reference.tokenizer.state_dict())
    target.position.data.copy_(reference.position.data)
    target.norm.load_state_dict(reference.norm.state_dict())
    target.head.load_state_dict(reference.head.state_dict())
    for source_block, target_block in zip(reference.blocks, target.blocks):
        target_block.norm1.load_state_dict(source_block.norm1.state_dict())
        target_block.norm2.load_state_dict(source_block.norm2.state_dict())
        target_block.ffn.load_state_dict(source_block.ffn.state_dict())
        copy_attention(source_block.attn, target_block.attn)
    return target


EXPERIMENT_IDS = [f"{backend}_{tokenizer}" for backend in ("scratch", "torch")
                  for tokenizer in ("patch", "row", "cnn")]
selected = CONFIG["experiment_ids"] or EXPERIMENT_IDS
unknown = set(selected) - set(EXPERIMENT_IDS)
if unknown: raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")
print("Selected:", selected)
'''
    equivalence = r'''
def verify_attention_equivalence():
    seed_everything(CONFIG["seed"])
    scratch, builtin = ScratchMSA().cpu(), TorchMSA().cpu()
    copy_attention(scratch, builtin)
    x1 = torch.randn(3, 7, 64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = scratch(x1), builtin(x2)
    torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-4)
    weights = torch.randn_like(y1)
    (y1 * weights).sum().backward(); (y2 * weights).sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(scratch.qkv.weight.grad, builtin.mha.in_proj_weight.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(scratch.qkv.bias.grad, builtin.mha.in_proj_bias.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(scratch.out.weight.grad, builtin.mha.out_proj.weight.grad, atol=1e-5, rtol=1e-4)
    print("MSA output, input gradients and parameter gradients match")


verify_attention_equivalence()
for tokenizer_name in TOKENIZERS:
    sample = torch.randn(2, 1, 28, 28)
    assert TOKENIZERS[tokenizer_name]()(sample).shape[1] == TOKENIZERS[tokenizer_name].token_count
print("Tokenizer shapes are valid")

probe = make_model("scratch", "patch").to(DEVICE)
images, labels = next(iter(loaders["train"])); images, labels = images[:32].to(DEVICE), labels[:32].to(DEVICE)
probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
probe_initial = None
for _ in range(10 if CONFIG["run_mode"] == "smoke" else 30):
    probe_optimizer.zero_grad(set_to_none=True)
    probe_loss = F.cross_entropy(probe(images), labels)
    if probe_initial is None: probe_initial = probe_loss.item()
    probe_loss.backward(); probe_optimizer.step()
assert torch.isfinite(probe_loss) and probe_loss.item() < probe_initial
print("E2 one-batch overfit:", probe_initial, "->", probe_loss.item())
del probe, probe_optimizer
if torch.cuda.is_available(): torch.cuda.empty_cache()
'''
    run = r'''
summary = []
OUTPUT_BASE = Path(CONFIG["output_root"]) / "E2" / CONFIG["run_mode"]
for experiment_id in selected:
    backend, tokenizer_name = experiment_id.split("_", 1)
    run_dir = OUTPUT_BASE / experiment_id
    existing = completed_metrics(run_dir, CONFIG["run_mode"])
    if existing is not None and not CONFIG["evaluate_only"]:
        summary.append({"experiment_id": experiment_id, **existing}); continue
    model = make_model(backend, tokenizer_name).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    run_config = {
        **CONFIG, "assignment": "E2", "experiment_id": experiment_id,
        "backend": backend, "tokenizer": tokenizer_name,
        "token_count": model.tokenizer.token_count, "split_hash": SPLIT_HASH,
        "max_train_batches": 4 if CONFIG["run_mode"] == "smoke" else None,
        "max_eval_batches": 2 if CONFIG["run_mode"] == "smoke" else None,
    }
    metrics, history, predictions = fit_classifier(model, loaders, optimizer, None, run_dir, run_config)
    sample_images, _ = next(iter(loaders["test"]))
    metrics.update(benchmark_model(model, sample_images[:32], repeats=5 if CONFIG["run_mode"] == "smoke" else 30))
    metrics["token_count"] = model.tokenizer.token_count
    json_dump(metrics, run_dir / "metrics.json")
    plot_history(history, run_dir / "curves.png")
    plot_confusion(predictions[0], predictions[1], CLASS_NAMES, run_dir / "confusion_matrix.png")
    summary.append({"experiment_id": experiment_id, **metrics})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

summary_frame = pd.DataFrame(summary).sort_values("experiment_id")
summary_frame.to_csv(OUTPUT_BASE / "summary.csv", index=False)
display(summary_frame)
'''
    cells = [
        md('''# E2 — Multi-head self-attention và tokenization

So sánh MSA viết từ các phép toán nền tảng với `nn.MultiheadAttention`, đồng thời khảo sát patch/row/CNN tokenization trên cùng Fashion-MNIST split.'''),
        md("## 1. Môi trường"),
        describe_cell(
            "1.1. Cài đặt thư viện",
            "Kiểm tra các dependency nền tảng và chỉ cài những gói còn thiếu trong runtime.",
            "Môi trường có đủ thư viện để xử lý dữ liệu, huấn luyện attention và trực quan hóa kết quả.",
        ), code(INSTALL_BASE),
        describe_cell(
            "1.2. Import và chọn thiết bị",
            "Nạp thư viện dùng chung, chọn CPU/GPU và cấu hình mixed precision theo thiết bị.",
            "Tạo `DEVICE`, `AMP_ENABLED` và các namespace được dùng ở các cell sau.",
        ), code(COMMON_IMPORTS),
        md("## 2. Cấu hình"),
        describe_cell(
            "2.1. Khai báo cấu hình chạy",
            "Thiết lập đường dẫn, seed, batch size, epoch, chế độ chạy và các experiment cần thực hiện.",
            "Tạo `CONFIG`; chế độ `smoke` tự giảm ngân sách để kiểm tra nhanh.",
        ), code(COMMON_CONFIG),
        describe_cell(
            "2.2. Khởi tạo tiện ích thực nghiệm",
            "Định nghĩa hàm tái lập, checkpoint, resume, training loop, benchmark và biểu đồ dùng chung.",
            "Cung cấp hạ tầng cho toàn bộ sáu cấu hình E2 mà chưa bắt đầu train.",
        ), code(COMMON_UTILS),
        md("## 3. Dữ liệu"),
        describe_cell(
            "3.1. Chuẩn bị Fashion-MNIST",
            "Dùng đúng split phân tầng và thống kê chuẩn hóa của E1/E3, sau đó dựng DataLoader cho train/validation/test.",
            "Tạo các dataset, `loaders`, `CLASS_NAMES` và `SPLIT_HASH` để đối chiếu giữa notebook.",
        ), code(FASHION_DATA),
        md("## 4. Tokenizer, attention và mô hình encoder"),
        describe_cell(
            "4.1. Xây dựng tokenizer và encoder",
            "Cài đặt patch, row, CNN-stem tokenizer; MSA tự viết; MSA PyTorch; encoder hai block và classifier dùng mean pooling.",
            "Tạo sáu experiment ID từ hai backend attention và ba tokenizer, cùng hàm đồng bộ khởi tạo giữa hai backend.",
        ), code(models),
        md("## 5. Kiểm chứng tương đương số học"),
        describe_cell(
            "5.1. So sánh MSA và kiểm tra pipeline",
            "Sao chép trọng số giữa hai MSA, so output/gradient trên CPU FP32, kiểm tra shape tokenizer và overfit một batch.",
            "Các assertion xác nhận tolerance `atol=1e-5`, `rtol=1e-4`, shape token đúng và loss có giảm.",
        ), code(equivalence),
        md("## 6. Huấn luyện và đánh giá ma trận 2 × 3"),
        describe_cell(
            "6.1. Chạy sáu cấu hình E2",
            "Huấn luyện từng cặp backend–tokenizer với cùng seed và ngân sách, sau đó đo accuracy, latency, memory và số token.",
            "Lưu checkpoint, metrics, predictions, curves, confusion matrix và bảng tổng hợp `summary.csv`.",
            "Run đã hoàn tất được bỏ qua trừ khi bật chế độ đánh giá lại checkpoint.",
        ), code(run),
        md('''## 7. Câu hỏi phân tích

1. Sau khi đồng bộ trọng số, sai số output/gradient có nằm trong tolerance đã công bố không?
2. Token count liên hệ ra sao với thời gian, bộ nhớ và accuracy?
3. CNN-stem có lợi thế gì và đã thêm inductive bias/tham số nào vào phép so sánh?
4. Backend scratch và PyTorch có accuracy tương đương khi khởi tạo tương ứng không? Nếu không, kiểm tra lại randomness và kernel trước khi kết luận.'''),
    ]
    write_notebook("E2/e2_attention_tokenization.ipynb", cells)


def build_e3() -> None:
    models = r'''
class SequenceClassifier(nn.Module):
    def __init__(self, recurrent, representation, hidden_size=128):
        super().__init__()
        self.representation = representation
        input_size = 28 if representation == "row" else 16
        recurrent_class = nn.LSTM if recurrent == "lstm" else nn.GRU
        self.recurrent = recurrent_class(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 10)
    def as_sequence(self, images):
        if self.representation == "row":
            return images.squeeze(1)
        return F.unfold(images, kernel_size=4, stride=4).transpose(1, 2)
    def forward(self, images):
        sequence = self.as_sequence(images)
        output, state = self.recurrent(sequence)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.head(hidden[-1])


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, images): return self.network(images)


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Linear(64 * 7 * 7, 10)
    def forward(self, images): return self.head(self.features(images).flatten(1))


assert sum(parameter.numel() for parameter in CNNClassifier().parameters()) == 50186
assert sum(parameter.numel() for parameter in CNNClassifier().parameters()) < sum(
    parameter.numel() for parameter in MLPClassifier().parameters())
EXPERIMENT_IDS = [f"{recurrent}_{representation}" for recurrent in ("lstm", "gru")
                  for representation in ("row", "patch")]
selected = CONFIG["experiment_ids"] or EXPERIMENT_IDS
unknown = set(selected) - set(EXPERIMENT_IDS)
if unknown: raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")

for representation, expected in (("row", (2, 28, 28)), ("patch", (2, 49, 16))):
    model = SequenceClassifier("lstm", representation)
    assert model.as_sequence(torch.randn(2, 1, 28, 28)).shape == expected
print("Sequence shapes and IDs are valid:", selected)

probe = SequenceClassifier("gru", "row").to(DEVICE)
images, labels = next(iter(loaders["train"])); images, labels = images[:32].to(DEVICE), labels[:32].to(DEVICE)
probe_optimizer = torch.optim.Adam(probe.parameters(), lr=3e-3)
probe_initial = None
for _ in range(10 if CONFIG["run_mode"] == "smoke" else 30):
    probe_optimizer.zero_grad(set_to_none=True)
    probe_loss = F.cross_entropy(probe(images), labels)
    if probe_initial is None: probe_initial = probe_loss.item()
    probe_loss.backward(); nn.utils.clip_grad_norm_(probe.parameters(), 1.0); probe_optimizer.step()
assert torch.isfinite(probe_loss) and probe_loss.item() < probe_initial
print("E3 one-batch overfit:", probe_initial, "->", probe_loss.item())
del probe, probe_optimizer
if torch.cuda.is_available(): torch.cuda.empty_cache()
'''
    baseline = r'''
import shutil

# Tự động đồng bộ E1 baseline artifacts từ repo khi chạy trên Google Colab
if Path("/content").exists() and not (Path(CONFIG["output_root"]) / "E1" / "full").exists():
    repo_dir = Path("/content/CO5085")
    if not repo_dir.exists():
        print("Đang clone repo CO5085 để lấy artifacts E1...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/superiorhamster/CO5085.git", str(repo_dir)], check=False)
    src_artifacts = repo_dir / "artifacts" / "E1" / "full"
    if src_artifacts.exists():
        shutil.copytree(src_artifacts, Path(CONFIG["output_root"]) / "E1" / "full", dirs_exist_ok=True)
        print("Đã nạp sẵn artifacts E1/full từ repo về:", Path(CONFIG["output_root"]) / "E1" / "full")


def load_e1_baseline(experiment_id):
    run_dir = Path(CONFIG["output_root"]) / "E1" / "full" / experiment_id
    config_path, metrics_path = run_dir / "config.json", run_dir / "metrics.json"
    if not (config_path.exists() and metrics_path.exists()):
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if config.get("split_hash") != SPLIT_HASH or config.get("seed") != CONFIG["seed"]:
        raise RuntimeError(f"E1 artifact mismatch: {run_dir}")
    expected_parameters = 50186 if experiment_id == "cnn" else 235146
    if metrics.get("parameters") != expected_parameters:
        print(f"Stale E1 baseline ignored (parameters={metrics.get('parameters')}, expected={expected_parameters}): {run_dir}")
        return None
    if config.get("experiment_id") != experiment_id or metrics.get("run_mode") != "full":
        raise RuntimeError(f"Invalid E1 artifact: {run_dir}")
    return {"experiment_id": f"e1_{experiment_id}", "source": "E1/full", **metrics}


baseline_results = [result for name in ("mlp", "cnn") if (result := load_e1_baseline(name)) is not None]
missing_baselines = [name for name in ("mlp", "cnn") if not any(row["experiment_id"] == f"e1_{name}" for row in baseline_results)]
print("Imported baselines:", [row["experiment_id"] for row in baseline_results])
print("Missing baselines (the next cell can reproduce them):", missing_baselines)
'''
    run = r'''
summary = list(baseline_results)
OUTPUT_BASE = Path(CONFIG["output_root"]) / "E3" / CONFIG["run_mode"]

# In smoke mode, reproduce missing baselines to validate the E1 comparison path.
# For a report table, run E1/full first or change this notebook to full.
baseline_factories = {"mlp": MLPClassifier, "cnn": CNNClassifier}
for baseline_name in missing_baselines:
    run_dir = OUTPUT_BASE / f"baseline_{baseline_name}"
    seed_everything(CONFIG["seed"])
    model = baseline_factories[baseline_name]().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    baseline_config = {
        **CONFIG, "assignment": "E3", "experiment_id": f"baseline_{baseline_name}",
        "split_hash": SPLIT_HASH, "max_train_batches": 4 if CONFIG["run_mode"] == "smoke" else None,
        "max_eval_batches": 2 if CONFIG["run_mode"] == "smoke" else None,
    }
    metrics, history, predictions = fit_classifier(model, loaders, optimizer, None, run_dir, baseline_config)
    plot_history(history, run_dir / "curves.png")
    summary.append({"experiment_id": f"baseline_{baseline_name}", "source": "reproduced", **metrics})

for experiment_id in selected:
    recurrent, representation = experiment_id.split("_", 1)
    run_dir = OUTPUT_BASE / experiment_id
    existing = completed_metrics(run_dir, CONFIG["run_mode"])
    if existing is not None and not CONFIG["evaluate_only"]:
        summary.append({"experiment_id": experiment_id, "source": "E3", **existing}); continue
    seed_everything(CONFIG["seed"])
    model = SequenceClassifier(recurrent, representation).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    run_config = {
        **CONFIG, "assignment": "E3", "experiment_id": experiment_id,
        "recurrent": recurrent, "representation": representation, "split_hash": SPLIT_HASH,
        "sequence_length": 28 if representation == "row" else 49,
        "input_size": 28 if representation == "row" else 16,
        "max_train_batches": 4 if CONFIG["run_mode"] == "smoke" else None,
        "max_eval_batches": 2 if CONFIG["run_mode"] == "smoke" else None,
    }
    metrics, history, predictions = fit_classifier(
        model, loaders, optimizer, None, run_dir, run_config, grad_clip=1.0)
    sample_images, _ = next(iter(loaders["test"]))
    metrics.update(benchmark_model(model, sample_images[:32], repeats=5 if CONFIG["run_mode"] == "smoke" else 30))
    json_dump(metrics, run_dir / "metrics.json")
    plot_history(history, run_dir / "curves.png")
    plot_confusion(predictions[0], predictions[1], CLASS_NAMES, run_dir / "confusion_matrix.png")
    summary.append({"experiment_id": experiment_id, "source": "E3", **metrics})

summary_frame = pd.DataFrame(summary).sort_values("experiment_id")
summary_frame.to_csv(OUTPUT_BASE / "summary.csv", index=False)
display(summary_frame)
'''
    cells = [
        md('''# E3 — LSTM và GRU trên chuỗi ảnh

Ảnh Fashion-MNIST được xem như chuỗi hàng hoặc chuỗi patch. Bốn mô hình recurrent được so sánh với MLP/CNN E1 trên đúng split.'''),
        md("## 1. Môi trường"),
        describe_cell(
            "1.1. Cài đặt thư viện",
            "Kiểm tra các dependency nền tảng và chỉ cài những gói còn thiếu trong runtime.",
            "Môi trường sẵn sàng cho xử lý Fashion-MNIST, huấn luyện recurrent model và xuất biểu đồ.",
        ), code(INSTALL_BASE),
        describe_cell(
            "1.2. Import và chọn thiết bị",
            "Nạp thư viện dùng chung, chọn CPU/GPU và bật mixed precision khi có CUDA.",
            "Tạo `DEVICE`, `AMP_ENABLED` và các namespace dùng trong notebook.",
        ), code(COMMON_IMPORTS),
        md("## 2. Cấu hình"),
        describe_cell(
            "2.1. Khai báo cấu hình chạy",
            "Thiết lập đường dẫn, seed, batch size, epoch, resume, chế độ `smoke`/`full` và experiment ID.",
            "Tạo dictionary `CONFIG` điều khiển toàn bộ pipeline E3.",
        ), code(COMMON_CONFIG),
        describe_cell(
            "2.2. Khởi tạo tiện ích thực nghiệm",
            "Định nghĩa các hàm seed, split hash, checkpoint, training loop, metric, benchmark và trực quan hóa.",
            "Cung cấp hạ tầng dùng chung cho baseline và recurrent model.",
        ), code(COMMON_UTILS),
        md("## 3. Dữ liệu và split dùng chung"),
        describe_cell(
            "3.1. Chuẩn bị Fashion-MNIST",
            "Tạo lại đúng split 54.000/6.000/10.000 và normalization đã dùng trong E1/E2.",
            "Tạo dataset, DataLoader, tên lớp và `SPLIT_HASH` để kiểm tra provenance của baseline.",
        ), code(FASHION_DATA),
        md("## 4. Biểu diễn chuỗi và mô hình"),
        describe_cell(
            "4.1. Xây dựng LSTM/GRU và baseline",
            "Biểu diễn ảnh theo 28 hàng hoặc 49 patch, cài đặt LSTM/GRU một chiều và định nghĩa lại MLP/CNN của E1, trong đó CNN dùng Flatten 3136 và Linear 3136→10.",
            "Tạo bốn experiment recurrent, kiểm tra shape chuỗi và xác nhận GRU có thể overfit một batch nhỏ.",
            "CNN baseline có 50.186 tham số, ít hơn MLP 78,7%. Artifact CNN GAP với 19.466 tham số sẽ bị bỏ qua và baseline được chạy lại.",
        ), code(models),
        md("## 5. Nhập baseline E1 có kiểm tra provenance"),
        describe_cell(
            "5.1. Đọc kết quả MLP/CNN từ E1",
            "Tìm artifact `E1/full` (tự động clone và đồng bộ từ GitHub repo trên Colab nếu chưa có), kiểm tra split hash, seed, experiment ID và `run_mode` trước khi tái sử dụng metric.",
            "Tạo `baseline_results` và `missing_baselines`; baseline thiếu sẽ được chạy lại ở cell kế tiếp.",
        ), code(baseline),
        md("## 6. Huấn luyện và đánh giá"),
        describe_cell(
            "6.1. Chạy baseline thiếu và bốn mô hình chuỗi",
            "Tái tạo baseline khi cần, huấn luyện LSTM/GRU với gradient clipping và đánh giá trên cùng split.",
            "Lưu checkpoint, metrics, curves, confusion matrix và `summary.csv` kết hợp E1–E3.",
            "Bảng báo cáo chính nên dùng baseline E1/full thay vì baseline smoke được tái tạo tại đây.",
        ), code(run),
        md('''## 7. Câu hỏi phân tích

1. Row và patch representation thay đổi độ dài chuỗi, input dimension và latency ra sao?
2. LSTM/GRU nào đạt trade-off tốt hơn giữa accuracy và số tham số?
3. Thứ tự quét raster tạo giả định nào về quan hệ không gian?
4. Những lỗi nào CNN xử lý tốt hơn recurrent model, và quan sát này có phù hợp với inductive bias của convolution không?'''),
    ]
    write_notebook("E3/e3_sequence_models.ipynb", cells)


def build_a1() -> None:
    install = INSTALL_BASE + r'''
if importlib.util.find_spec("timm") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0,<2"])
'''
    imports = COMMON_IMPORTS + "\nimport timm\n"
    config = COMMON_CONFIG + r'''
CONFIG.update({
    "batch_size": 32, "effective_batch_size": 64,
    "gradient_accumulation": 2, "epochs": 1 if CONFIG["run_mode"] == "smoke" else 20,
})
'''
    data = r'''
raw_train = datasets.CIFAR100(CONFIG["data_root"], train=True, download=True)
labels = np.asarray(raw_train.targets)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=CONFIG["seed"])
train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
SPLIT_HASH = split_digest(train_idx, val_idx)
split_path = Path(CONFIG["data_root"]) / "cifar100_split_seed42.npz"
if split_path.exists():
    saved = np.load(split_path)
    if not (np.array_equal(saved["train_idx"], train_idx) and np.array_equal(saved["val_idx"], val_idx)):
        raise RuntimeError("Existing CIFAR-100 split does not match seed 42")
else:
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx)

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
train_aug = datasets.CIFAR100(CONFIG["data_root"], train=True, transform=train_transform, download=True)
train_eval = datasets.CIFAR100(CONFIG["data_root"], train=True, transform=eval_transform, download=True)
test_dataset = datasets.CIFAR100(CONFIG["data_root"], train=False, transform=eval_transform, download=True)
train_dataset, val_dataset = Subset(train_aug, train_idx), Subset(train_eval, val_idx)
loaders = {
    "train": DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                        num_workers=CONFIG["num_workers"], pin_memory=AMP_ENABLED),
    "val": DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                      num_workers=CONFIG["num_workers"], pin_memory=AMP_ENABLED),
    "test": DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                       num_workers=CONFIG["num_workers"], pin_memory=AMP_ENABLED),
}
CLASS_NAMES = list(raw_train.classes)
assert len(train_dataset) == 45000 and len(val_dataset) == 5000 and len(test_dataset) == 10000
assert not set(train_idx).intersection(set(val_idx))
print({"sizes": [len(train_dataset), len(val_dataset), len(test_dataset)], "split_hash": SPLIT_HASH})
'''
    models = r'''
MODEL_NAMES = {"resnet18": "resnet18.a1_in1k", "deit_tiny": "deit_tiny_patch16_224.fb_in1k"}
STRATEGIES = ("scratch", "head_only", "partial", "full")
EXPERIMENT_IDS = [f"{family}_{strategy}" for family in MODEL_NAMES for strategy in STRATEGIES]
selected = CONFIG["experiment_ids"] or EXPERIMENT_IDS
unknown = set(selected) - set(EXPERIMENT_IDS)
if unknown: raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")


def is_head_parameter(name):
    return name.startswith(("fc.", "head.", "head_dist."))


def configure_trainable(model, family, strategy):
    for parameter in model.parameters(): parameter.requires_grad = strategy in {"scratch", "full"}
    if strategy == "head_only":
        for name, parameter in model.named_parameters(): parameter.requires_grad = is_head_parameter(name)
    elif strategy == "partial":
        for name, parameter in model.named_parameters():
            if family == "resnet18":
                parameter.requires_grad = name.startswith("layer4.") or is_head_parameter(name)
            else:
                parameter.requires_grad = (name.startswith(("blocks.10.", "blocks.11.", "norm."))
                                              or is_head_parameter(name))
    assert any(p.requires_grad for p in model.parameters())
    return model


def build_model(family, strategy):
    seed_everything(CONFIG["seed"])
    model = timm.create_model(MODEL_NAMES[family], pretrained=strategy != "scratch", num_classes=100)
    return configure_trainable(model, family, strategy)


def enforce_frozen_eval(model):
    for module in model.modules():
        parameters = list(module.parameters(recurse=True))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()


def make_optimizer(model, strategy):
    if strategy == "scratch":
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.05)
    head = [p for name, p in model.named_parameters() if p.requires_grad and is_head_parameter(name)]
    backbone = [p for name, p in model.named_parameters() if p.requires_grad and not is_head_parameter(name)]
    groups = []
    if backbone: groups.append({"params": backbone, "lr": 1e-4})
    if head: groups.append({"params": head, "lr": 1e-3})
    return torch.optim.AdamW(groups, weight_decay=.01)


def make_scheduler(optimizer, epochs, warmup=2):
    def factor(epoch):
        if epoch < warmup: return (epoch + 1) / warmup
        progress = (epoch - warmup + 1) / max(1, epochs - warmup)
        return .5 * (1 + math.cos(math.pi * min(progress, 1)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


print("Selected:", selected)
'''
    invariant = r'''
def verify_freeze_invariant(model, strategy, batch):
    if strategy in {"scratch", "full"}: return
    initial = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    frozen_buffer_names = set()
    for module_name, module in model.named_modules():
        parameters = list(module.parameters(recurse=True))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            prefix = f"{module_name}." if module_name else ""
            for buffer_name, _ in module.named_buffers(recurse=False):
                frozen_buffer_names.add(prefix + buffer_name)
    optimizer = make_optimizer(model, strategy)
    model.train(); enforce_frozen_eval(model)
    images, labels = batch[0][:2].to(DEVICE), batch[1][:2].to(DEVICE)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(images), labels).backward(); optimizer.step()
    current = model.state_dict()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter.detach().cpu(), initial[name], rtol=0, atol=0)
    for name in frozen_buffer_names:
        torch.testing.assert_close(current[name].detach().cpu(), initial[name], rtol=0, atol=0)
    model.load_state_dict(initial)
    print("freeze invariant passed:", strategy)
'''
    run = r'''
def plot_cifar_failures(dataset, predictions, path, count=16):
    y_true, y_pred, probabilities = predictions
    wrong = np.flatnonzero(y_true != y_pred)
    confidence = probabilities[np.arange(len(y_pred)), y_pred]
    chosen = wrong[np.argsort(confidence[wrong])[-count:][::-1]]
    figure, axes = plt.subplots(4, 4, figsize=(12, 12))
    mean_tensor = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std_tensor = torch.tensor(IMAGENET_STD)[:, None, None]
    for axis, index in zip(axes.ravel(), chosen):
        image, _ = dataset[index]
        image = (image * std_tensor + mean_tensor).clamp(0, 1)
        axis.imshow(image.permute(1, 2, 0))
        axis.set_title(f"T: {CLASS_NAMES[y_true[index]]}\nP: {CLASS_NAMES[y_pred[index]]} ({confidence[index]:.2f})", fontsize=8)
        axis.axis("off")
    for axis in axes.ravel()[len(chosen):]: axis.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


summary = []
OUTPUT_BASE = Path(CONFIG["output_root"]) / "A1" / CONFIG["run_mode"]
first_batch = next(iter(loaders["train"]))
for experiment_id in selected:
    family = "resnet18" if experiment_id.startswith("resnet18_") else "deit_tiny"
    strategy = experiment_id[len(family) + 1:]
    run_dir = OUTPUT_BASE / experiment_id
    existing = completed_metrics(run_dir, CONFIG["run_mode"])
    if existing is not None and not CONFIG["evaluate_only"]:
        summary.append({"experiment_id": experiment_id, **existing}); continue
    model = build_model(family, strategy).to(DEVICE)
    verify_freeze_invariant(model, strategy, first_batch)
    optimizer = make_optimizer(model, strategy)
    scheduler = make_scheduler(optimizer, CONFIG["epochs"])
    run_config = {
        **CONFIG, "assignment": "A1", "experiment_id": experiment_id,
        "family": family, "strategy": strategy, "split_hash": SPLIT_HASH,
        "model_name": MODEL_NAMES[family], "image_size": 224,
        "max_train_batches": 2 if CONFIG["run_mode"] == "smoke" else None,
        "max_eval_batches": 1 if CONFIG["run_mode"] == "smoke" else None,
    }
    metrics, history, predictions = fit_classifier(
        model, loaders, optimizer, scheduler, run_dir, run_config,
        grad_accum=CONFIG["gradient_accumulation"], frozen_eval=enforce_frozen_eval)
    metrics["macro_f1"] = f1_score(predictions[0], predictions[1], average="macro")
    sample_images, _ = first_batch
    metrics.update(benchmark_model(model, sample_images[:8], repeats=3 if CONFIG["run_mode"] == "smoke" else 20))
    json_dump(metrics, run_dir / "metrics.json")
    plot_history(history, run_dir / "curves.png")
    plot_confusion(predictions[0], predictions[1], CLASS_NAMES, run_dir / "confusion_matrix.png")
    plot_cifar_failures(test_dataset, predictions, run_dir / "failures.png")
    summary.append({"experiment_id": experiment_id, **metrics})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

summary_frame = pd.DataFrame(summary).sort_values("experiment_id")
summary_frame.to_csv(OUTPUT_BASE / "summary.csv", index=False)
display(summary_frame)
'''
    cells = [
        md('''# A1 — CNN và Transformer trên CIFAR-100

So sánh ResNet-18 với DeiT-Tiny trong bốn chế độ: scratch, head-only, partial freeze và full fine-tune. Kết quả là so sánh dưới ngân sách 20 epoch, không phải tuyên bố scratch đã hội tụ.'''),
        md("## 1. Môi trường"),
        describe_cell(
            "1.1. Cài đặt thư viện",
            "Kiểm tra dependency chung và cài thêm `timm` nếu runtime chưa có.",
            "Môi trường có đủ package để tải ResNet-18/DeiT-Tiny và chạy pipeline CIFAR-100.",
        ), code(install),
        describe_cell(
            "1.2. Import và chọn thiết bị",
            "Nạp thư viện dùng chung cùng `timm`, chọn CPU/GPU và cấu hình mixed precision.",
            "Tạo `DEVICE`, `AMP_ENABLED` và các namespace cần cho A1.",
        ), code(imports),
        md("## 2. Cấu hình"),
        describe_cell(
            "2.1. Khai báo cấu hình A1",
            "Thiết lập đường dẫn, seed, chế độ chạy và effective batch 64 thông qua gradient accumulation.",
            "Tạo `CONFIG` với microbatch 32, accumulation 2 và ngân sách 1 hoặc 20 epoch.",
        ), code(config),
        describe_cell(
            "2.2. Khởi tạo tiện ích thực nghiệm",
            "Định nghĩa hàm tái lập, checkpoint, resume, training loop, metric, benchmark và biểu đồ dùng chung.",
            "Cung cấp hạ tầng train/evaluate cho cả ResNet-18 và DeiT-Tiny.",
        ), code(COMMON_UTILS),
        md("## 3. Dữ liệu, augmentation và protocol"),
        describe_cell(
            "3.1. Chuẩn bị CIFAR-100",
            "Tạo split phân tầng 45.000/5.000/10.000, augmentation train, resize 224×224 và ImageNet normalization.",
            "Tạo dataset, DataLoader, `CLASS_NAMES` và `SPLIT_HASH`; validation/test dùng transform xác định.",
            "Ảnh 32×32 được upscale lên 224×224 để hai kiến trúc nhận cùng đầu vào.",
        ), code(data),
        md('''### Nguồn và phạm vi sử dụng

- CIFAR-100: trang chính thức của Alex Krizhevsky/University of Toronto. Ghi citation dataset trong báo cáo; không tự gán license nếu nguồn không công bố rõ.
- Checkpoint pretrained do `timm` phân phối. Ghi chính xác model ID, phiên bản timm và model card vào artifact.'''),
        md("## 4. Mô hình và freeze strategies"),
        describe_cell(
            "4.1. Xây dựng mô hình và chiến lược fine-tune",
            "Khởi tạo ResNet-18/DeiT-Tiny từ scratch hoặc pretrained, cấu hình head-only, partial và full fine-tune.",
            "Tạo tám experiment ID, nhóm tham số optimizer, learning rate riêng cho head/backbone và cosine scheduler.",
        ), code(models),
        md("## 5. Kiểm tra frozen parameters và BatchNorm statistics"),
        describe_cell(
            "5.1. Xác minh bất biến khi đóng băng",
            "Chạy một optimizer step thử và so sánh tuyệt đối tham số cùng buffer của các module frozen.",
            "Assertion bảo đảm trọng số frozen và BatchNorm running statistics không đổi trước khi train dài.",
        ), code(invariant),
        md("## 6. Huấn luyện và đánh giá ma trận 8 thí nghiệm"),
        describe_cell(
            "6.1. Chạy ma trận A1",
            "Huấn luyện tám cấu hình độc lập, đánh giá accuracy/macro-F1, benchmark chi phí và phân tích failure case.",
            "Lưu checkpoint, history, predictions, metrics, curves, confusion matrix, failures và `summary.csv`.",
            "Mỗi chế độ pretrained khởi tạo lại từ checkpoint gốc; run hoàn tất được bỏ qua an toàn.",
        ), code(run),
        md('''## 7. Phân tích cho A1.1/A1.2

1. So sánh độ lợi của pretraining trong từng họ kiến trúc.
2. Head-only, partial và full fine-tune đánh đổi accuracy, trainable parameters, thời gian và memory như thế nào?
3. Liệt kê các cặp lớp nhầm nhiều nhất và failure cases đại diện.
4. Nêu hạn chế của việc upscale ảnh 32×32 lên 224×224 và sự khác biệt recipe pretraining.
5. Không so latency nếu các run không cùng GPU/precision/batch size.'''),
    ]
    write_notebook("A1/a1_cnn_vs_transformer.ipynb", cells)


def build_a2() -> None:
    install = INSTALL_BASE + r'''
extra = {"diffusers": "diffusers>=0.30,<1", "accelerate": "accelerate>=0.33,<2",
         "torchmetrics": "torchmetrics[image]>=1.4,<2", "torch_fidelity": "torch-fidelity>=0.3,<1"}
missing_extra = [package for module, package in extra.items() if importlib.util.find_spec(module) is None]
if missing_extra:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_extra])
'''
    imports = COMMON_IMPORTS + r'''
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import Sampler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.utils import make_grid
'''
    config = COMMON_CONFIG + r'''
CONFIG.update({
    "celeba_root": os.getenv("CO5085_CELEBA_ROOT", str(Path(CONFIG["data_root"]) / "celeba")),
    "batch_size": 32, "effective_batch_size": 64, "gradient_accumulation": 2,
    "max_steps": 2 if CONFIG["run_mode"] == "smoke" else 20000,
    "save_every": 1 if CONFIG["run_mode"] == "smoke" else 1000,
    "preview_inference_steps": 10 if CONFIG["run_mode"] == "smoke" else 50,
    "eval_inference_steps": 10 if CONFIG["run_mode"] == "smoke" else 250,
    "eval_samples": 64 if CONFIG["run_mode"] == "smoke" else 5000,
})
CONFIG["num_workers"] = 0  # Required for exact sampler resume.
'''
    data = r'''
CELEBA_ROOT = Path(CONFIG["celeba_root"])
IMAGE_ROOT = CELEBA_ROOT / "img_align_celeba"
PARTITION_FILE = CELEBA_ROOT / "list_eval_partition.txt"
if not IMAGE_ROOT.exists() or not PARTITION_FILE.exists():
    raise FileNotFoundError(
        "Mount CelebA Align&Cropped, including list_eval_partition.txt, then set "
        "CO5085_CELEBA_ROOT to that directory. Dataset images must not be committed to the repo."
    )

partition_frame = pd.read_csv(PARTITION_FILE, sep=r"\s+", names=["filename", "partition"])
requested = {"train": (0, 20000), "val": (1, 2000), "test": (2, 5000)}
manifest_rows = []
for offset, (split_name, (partition_id, count)) in enumerate(requested.items()):
    candidates = partition_frame.loc[partition_frame.partition == partition_id, "filename"].to_numpy()
    rng = np.random.default_rng(CONFIG["seed"] + offset)
    chosen = candidates[rng.permutation(len(candidates))[:count]]
    manifest_rows.extend({"filename": name, "split": split_name} for name in chosen)
manifest = pd.DataFrame(manifest_rows)
assert len(manifest) == 27000 and manifest.filename.is_unique
MANIFEST_HASH = hashlib.sha256("\n".join(manifest.filename).encode()).hexdigest()
manifest_path = Path(CONFIG["output_root"]) / "A2" / "celeba_manifest_seed42.csv"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
if manifest_path.exists():
    saved_manifest = pd.read_csv(manifest_path)
    if not saved_manifest.equals(manifest):
        raise RuntimeError("Existing CelebA manifest does not match deterministic selection")
else:
    manifest.to_csv(manifest_path, index=False)

train_transform = transforms.Compose([
    transforms.CenterCrop(148), transforms.Resize((32, 32), antialias=True),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
])
eval_transform = transforms.Compose([
    transforms.CenterCrop(148), transforms.Resize((32, 32), antialias=True),
    transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
])


class CelebASubset(Dataset):
    def __init__(self, frame, image_root, transform):
        self.filenames = frame.filename.tolist(); self.image_root = Path(image_root); self.transform = transform
    def __len__(self): return len(self.filenames)
    def __getitem__(self, index):
        with Image.open(self.image_root / self.filenames[index]) as image:
            return self.transform(image.convert("RGB"))


split_frames = {name: manifest[manifest.split == name].reset_index(drop=True) for name in requested}
train_dataset = CelebASubset(split_frames["train"], IMAGE_ROOT, train_transform)
val_dataset = CelebASubset(split_frames["val"], IMAGE_ROOT, eval_transform)
test_dataset = CelebASubset(split_frames["test"], IMAGE_ROOT, eval_transform)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
print({"sizes": [len(train_dataset), len(val_dataset), len(test_dataset)], "manifest_hash": MANIFEST_HASH})
'''
    model_code = r'''
class StatefulRandomSampler(Sampler):
    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = None
        self.position = 0
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        if self.permutation is None or self.position >= len(self.permutation):
            self.permutation = torch.randperm(len(self.data_source), generator=self.generator).tolist()
            self.position = 0
        while self.position < len(self.permutation):
            index = self.permutation[self.position]
            self.position += 1
            yield index
    def state_dict(self):
        return {"permutation": self.permutation, "position": self.position,
                "generator_state": self.generator.get_state()}
    def load_state_dict(self, state):
        self.permutation = state["permutation"]
        self.position = state["position"]
        self.generator.set_state(state["generator_state"])


class EMA:
    def __init__(self, model, decay=.999):
        self.decay = decay
        self.shadow = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        self.backup = None
    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)
    def state_dict(self): return {"decay": self.decay, "shadow": self.shadow}
    def load_state_dict(self, state): self.decay, self.shadow = state["decay"], state["shadow"]
    @torch.no_grad()
    def apply(self, model):
        self.backup = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        for name, parameter in model.named_parameters(): parameter.copy_(self.shadow[name])
    @torch.no_grad()
    def restore(self, model):
        for name, parameter in model.named_parameters(): parameter.copy_(self.backup[name])
        self.backup = None


def build_unet():
    seed_everything(CONFIG["seed"])
    return UNet2DModel(
        sample_size=32, in_channels=3, out_channels=3, layers_per_block=2,
        block_out_channels=(64, 128, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
    )


def build_noise_scheduler(schedule):
    arguments = dict(num_train_timesteps=1000, variance_type="fixed_small",
                     prediction_type="epsilon", clip_sample=True)
    if schedule == "linear":
        return DDPMScheduler(beta_schedule="linear", beta_start=1e-4, beta_end=.02, **arguments)
    return DDPMScheduler(beta_schedule="squaredcos_cap_v2", **arguments)


EXPERIMENT_IDS = ["linear", "cosine"]
selected = CONFIG["experiment_ids"] or EXPERIMENT_IDS
unknown = set(selected) - set(EXPERIMENT_IDS)
if unknown: raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")

reference_states = []
for schedule_name in EXPERIMENT_IDS:
    scheduler = build_noise_scheduler(schedule_name)
    assert torch.isfinite(scheduler.betas).all() and (scheduler.betas > 0).all() and (scheduler.betas < 1).all()
    model = build_unet()
    reference_states.append({name: value.clone() for name, value in model.state_dict().items()})
    clean, noise, timesteps = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.tensor([0, 999])
    noisy = scheduler.add_noise(clean, noise, timesteps)
    prediction = model(noisy, timesteps).sample
    reverse = scheduler.step(prediction, 999, noisy).prev_sample
    assert prediction.shape == clean.shape and torch.isfinite(noisy).all() and torch.isfinite(reverse).all()
for name in reference_states[0]:
    torch.testing.assert_close(reference_states[0][name], reference_states[1][name], rtol=0, atol=0)
print("U-Net initialization, add-noise and reverse-step checks passed")
del model, reference_states
'''
    training = r'''
@torch.no_grad()
def validation_noise_loss(model, scheduler, loader, max_batches=20):
    model.eval(); losses = []
    generator = torch.Generator(device=DEVICE.type).manual_seed(CONFIG["seed"] + 999)
    for batch_index, clean in enumerate(loader):
        if batch_index >= max_batches: break
        clean = clean.to(DEVICE)
        noise = torch.randn(clean.shape, generator=generator, device=DEVICE)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (len(clean),),
                                  generator=generator, device=DEVICE)
        noisy = scheduler.add_noise(clean, noise, timesteps)
        losses.append(F.mse_loss(model(noisy, timesteps).sample, noise).item())
    return float(np.mean(losses))


@torch.no_grad()
def sample_images(model, scheduler, initial_noise, inference_steps, return_trace=False):
    sampler = DDPMScheduler.from_config(scheduler.config)
    sampler.set_timesteps(inference_steps, device=DEVICE)
    images, trace = initial_noise.to(DEVICE), []
    interval = max(1, len(sampler.timesteps) // 8)
    for index, timestep in enumerate(sampler.timesteps):
        predicted_noise = model(images, timestep).sample
        images = sampler.step(predicted_noise, timestep, images).prev_sample
        if return_trace and (index % interval == 0 or index == len(sampler.timesteps) - 1):
            trace.append(images.detach().cpu())
    return images.clamp(-1, 1), trace


def save_preview(model, ema, scheduler, run_dir, step):
    ema.apply(model); model.eval()
    generator = torch.Generator().manual_seed(CONFIG["seed"] + 123)
    initial = torch.randn(16, 3, 32, 32, generator=generator)
    images, _ = sample_images(model, scheduler, initial, CONFIG["preview_inference_steps"])
    ema.restore(model)
    grid = make_grid((images.cpu() + 1) / 2, nrow=4)
    figure, axis = plt.subplots(figsize=(6, 6)); axis.imshow(grid.permute(1, 2, 0)); axis.axis("off")
    figure.tight_layout(); figure.savefig(Path(run_dir) / f"preview_step_{step:06d}.png", dpi=160); plt.close(figure)


def train_diffusion(experiment_id):
    run_dir = Path(CONFIG["output_root"]) / "A2" / CONFIG["run_mode"] / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**CONFIG, "assignment": "A2", "experiment_id": experiment_id,
                  "noise_schedule": experiment_id, "manifest_hash": MANIFEST_HASH,
                  "train_samples": len(train_dataset), "timesteps": 1000,
                  "prediction_target": "epsilon", "variance_type": "fixed_small"}
    json_dump(run_config, run_dir / "config.json"); json_dump(environment_info(), run_dir / "environment.json")
    model = build_unet().to(DEVICE)
    scheduler = build_noise_scheduler(experiment_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 500) if step < 500 else
        .5 * (1 + math.cos(math.pi * (step - 500) / max(1, CONFIG["max_steps"] - 500))))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    ema = EMA(model, .999)
    sampler = StatefulRandomSampler(train_dataset, CONFIG["seed"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler,
                              num_workers=0, drop_last=True, pin_memory=AMP_ENABLED)
    global_step, history = 0, []
    if CONFIG["evaluate_only"]:
        resume_path = run_dir / "last.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"evaluate_only requires {resume_path}")
    else:
        resume_path = resolve_resume(run_dir, CONFIG["resume_from"])
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["scaler"])
        ema.load_state_dict(checkpoint["ema"]); sampler.load_state_dict(checkpoint["sampler"])
        restore_rng_state(checkpoint["rng_state"]); global_step = checkpoint["global_step"]
        history_path = run_dir / "history.csv"
        if history_path.exists(): history = pd.read_csv(history_path).to_dict("records")
    if CONFIG["evaluate_only"]:
        return model, ema, scheduler, run_dir, history
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True); accumulation = 0
    progress = tqdm(total=CONFIG["max_steps"], initial=global_step, desc=experiment_id)
    while global_step < CONFIG["max_steps"]:
        for clean in train_loader:
            clean = clean.to(DEVICE)
            noise = torch.randn_like(clean)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (len(clean),), device=DEVICE)
            noisy = scheduler.add_noise(clean, noise, timesteps)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                prediction = model(noisy, timesteps).sample
                loss = F.mse_loss(prediction, noise) / CONFIG["gradient_accumulation"]
            scaler.scale(loss).backward(); accumulation += 1
            if accumulation < CONFIG["gradient_accumulation"]: continue
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step(); ema.update(model); accumulation = 0; global_step += 1
            progress.update(1)
            history.append({"global_step": global_step,
                            "train_loss": loss.item() * CONFIG["gradient_accumulation"],
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            if global_step % CONFIG["save_every"] == 0 or global_step == CONFIG["max_steps"]:
                ema.apply(model)
                val_loss = validation_noise_loss(model, scheduler, val_loader,
                                                 2 if CONFIG["run_mode"] == "smoke" else 20)
                ema.restore(model); history[-1]["val_noise_loss"] = val_loss
                pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
                checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": lr_scheduler.state_dict(), "scaler": scaler.state_dict(),
                              "ema": ema.state_dict(), "sampler": sampler.state_dict(),
                              "global_step": global_step, "rng_state": rng_state(), "config": run_config}
                atomic_torch_save(checkpoint, run_dir / f"step_{global_step:06d}.pt")
                atomic_torch_save(checkpoint, run_dir / "last.pt")
                save_preview(model, ema, scheduler, run_dir, global_step)
            if global_step >= CONFIG["max_steps"]: break
    progress.close()
    return model, ema, scheduler, run_dir, history
'''
    evaluation = r'''
def real_fid_state(loader, sample_count, cache_path):
    cache_path = Path(cache_path)
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    metric.persistent(True)
    if cache_path.exists():
        metric.load_state_dict(torch.load(cache_path, map_location=DEVICE, weights_only=True))
        return metric.state_dict()
    seen = 0
    for real in tqdm(loader, desc="cache real FID"):
        real = ((real.to(DEVICE) + 1) / 2).clamp(0, 1)
        take = min(len(real), sample_count - seen)
        metric.update(real[:take], real=True); seen += take
        if seen >= sample_count: break
    atomic_torch_save(metric.state_dict(), cache_path)
    return metric.state_dict()


@torch.no_grad()
def evaluate_generation(model, ema, scheduler, run_dir):
    eval_count = CONFIG["eval_samples"]
    cache = (Path(CONFIG["output_root"]) / "A2" / "cache" /
             f"fid_real_32px_{eval_count}_{MANIFEST_HASH[:12]}.pt")
    state = real_fid_state(test_loader, eval_count, cache)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    fid.persistent(True); fid.load_state_dict(state)
    inception = InceptionScore(normalize=True).to(DEVICE)
    ema.apply(model); model.eval()
    generated, generator = 0, torch.Generator().manual_seed(CONFIG["seed"] + 2026)
    start = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    while generated < eval_count:
        count = min(CONFIG["batch_size"], eval_count - generated)
        initial = torch.randn(count, 3, 32, 32, generator=generator)
        images, _ = sample_images(model, scheduler, initial, CONFIG["eval_inference_steps"])
        normalized = ((images + 1) / 2).clamp(0, 1)
        fid.update(normalized, real=False); inception.update(normalized)
        generated += count
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fid_value = fid.compute().item(); is_mean, is_std = inception.compute()
    trace_initial = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(CONFIG["seed"] + 7))
    _, trace = sample_images(model, scheduler, trace_initial, CONFIG["eval_inference_steps"], True)
    ema.restore(model)
    figure, axes = plt.subplots(1, len(trace), figsize=(2 * len(trace), 2))
    for axis, image in zip(np.asarray(axes).ravel(), trace):
        axis.imshow(((image[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0)); axis.axis("off")
    figure.tight_layout(); figure.savefig(Path(run_dir) / "denoising_trace.png", dpi=160); plt.close(figure)
    return {"run_mode": CONFIG["run_mode"], "fid": fid_value,
            "inception_score_mean": is_mean.item(), "inception_score_std": is_std.item(),
            "generated_samples": eval_count, "inference_steps": CONFIG["eval_inference_steps"],
            "sampling_seconds": elapsed, "seconds_per_image": elapsed / eval_count,
            "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0),
            **count_parameters(model)}


summary = []
for experiment_id in selected:
    run_dir = Path(CONFIG["output_root"]) / "A2" / CONFIG["run_mode"] / experiment_id
    existing = completed_metrics(run_dir, CONFIG["run_mode"])
    if existing is not None and not CONFIG["evaluate_only"]:
        summary.append({"experiment_id": experiment_id, **existing})
        continue
    model, ema, scheduler, run_dir, history = train_diffusion(experiment_id)
    metrics = evaluate_generation(model, ema, scheduler, run_dir)
    json_dump(metrics, run_dir / "metrics.json")
    summary.append({"experiment_id": experiment_id, **metrics})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

OUTPUT_BASE = Path(CONFIG["output_root"]) / "A2" / CONFIG["run_mode"]
summary_frame = pd.DataFrame(summary).sort_values("experiment_id")
summary_frame.to_csv(OUTPUT_BASE / "summary.csv", index=False)
display(summary_frame)
'''
    cells = [
        md('''# A2 — Linear vs cosine noise schedule cho DDPM

**Đề tài:** ảnh hưởng của noise schedule tới mô hình sinh khuôn mặt không điều kiện ở 32×32.

Paper chính: *Improved Denoising Diffusion Probabilistic Models* — Nichol & Dhariwal, ICML 2021. Thực nghiệm này chỉ tái hiện ablation linear/cosine schedule; không tái hiện learned variance hoặc toàn bộ kết quả paper.'''),
        md("## 1. Môi trường"),
        describe_cell(
            "1.1. Cài đặt thư viện",
            "Kiểm tra dependency chung và cài thêm Diffusers, Accelerate, TorchMetrics cùng torch-fidelity khi còn thiếu.",
            "Môi trường có đủ package để train U-Net, lấy mẫu DDPM và tính FID/IS.",
            "Việc cài package có thể cần khởi động lại kernel trên một số runtime.",
        ), code(install),
        describe_cell(
            "1.2. Import và chọn thiết bị",
            "Nạp thư viện dùng chung cùng Diffusers, PIL, sampler, FID, IS và tiện ích tạo lưới ảnh.",
            "Tạo `DEVICE`, `AMP_ENABLED` và các namespace cần cho pipeline diffusion.",
        ), code(imports),
        md("## 2. Cấu hình"),
        describe_cell(
            "2.1. Khai báo cấu hình A2",
            "Thiết lập CelebA root, effective batch, số optimizer step, chu kỳ checkpoint và ngân sách sampling/evaluation.",
            "Tạo `CONFIG`; chế độ `smoke` dùng 2 step và 64 mẫu để kiểm tra nhanh toàn pipeline.",
            "`num_workers=0` được giữ cố định để resume sampler chính xác theo step.",
        ), code(config),
        describe_cell(
            "2.2. Khởi tạo tiện ích thực nghiệm",
            "Định nghĩa hàm seed, RNG state, checkpoint nguyên tử, environment log và các tiện ích artifact dùng chung.",
            "Cung cấp nền tảng lưu/khôi phục đầy đủ trạng thái huấn luyện A2.",
        ), code(COMMON_UTILS),
        md('''## 3. Dữ liệu CelebA và điều khoản sử dụng

CelebA chỉ được dùng cho nghiên cứu phi thương mại. Không commit hoặc phân phối lại ảnh. Mount bản Align&Cropped cùng `list_eval_partition.txt`, sau đó đặt `CO5085_CELEBA_ROOT`. Notebook lưu manifest filename để tái tạo subset.'''),
        describe_cell(
            "3.1. Tạo subset CelebA xác định",
            "Đọc official partition, lấy cố định 20.000/2.000/5.000 ảnh, lưu manifest và dựng pipeline crop–resize–normalize.",
            "Tạo `train_dataset`, `val_dataset`, `test_dataset`, loader đánh giá và `MANIFEST_HASH`.",
            "Cell dừng nếu thiếu ảnh hoặc `list_eval_partition.txt`; ảnh gốc không được ghi vào repository.",
        ), code(data),
        md("## 4. Stateful sampler, EMA, mô hình U-Net và noise scheduler"),
        describe_cell(
            "4.1. Xây dựng thành phần diffusion",
            "Cài đặt sampler có thể resume, EMA, U-Net 32×32 và hai DDPM scheduler linear/cosine.",
            "Tạo hai experiment ID và kiểm tra beta hợp lệ, add-noise/reverse-step hữu hạn cùng khởi tạo U-Net giống nhau.",
        ), code(model_code),
        md("## 5. Vòng lặp huấn luyện và checkpoint theo optimizer step"),
        describe_cell(
            "5.1. Định nghĩa train và sampling",
            "Cài đặt validation noise loss, reverse sampling, preview cố định và vòng lặp train epsilon prediction với AMP, EMA, clipping và resume.",
            "Mỗi mốc lưu history, preview, checkpoint theo step và `last.pt` chứa optimizer, scheduler, scaler, EMA, sampler cùng RNG state.",
        ), code(training),
        md("## 6. Đánh giá FID, Inception Score và denoising trace"),
        describe_cell(
            "6.1. Đánh giá hai noise schedule",
            "Cache feature ảnh thật, sinh cùng số mẫu cho mỗi schedule, tính FID/IS và ghi chuỗi khử nhiễu.",
            "Lưu `metrics.json`, `denoising_trace.png` và `summary.csv` kèm thời gian sampling, memory và protocol đánh giá.",
            "FID chỉ được so sánh giữa hai run dùng cùng cache, sampler, số mẫu và preprocessing.",
        ), code(evaluation),
        md('''## 7. Khung phân tích paper

1. **Phương pháp:** giải thích forward noising, epsilon prediction loss, reverse process và vai trò của beta schedule.
2. **Thực nghiệm paper:** dataset, baseline, likelihood/sample-quality metrics và phát hiện liên quan cosine schedule/learned variance.
3. **Hạn chế:** chi phí sampling, độ nhạy compute/model size và giới hạn của FID/IS.
4. **Đối chiếu:** notebook giữ U-Net, seed, data và budget cố định; chỉ thay schedule. Nêu rõ độ phân giải, subset, fixed variance và số mẫu khác paper.

Không so trực tiếp FID với bảng paper. Loss dự đoán noise là diagnostic, không thay thế metric chất lượng sinh ảnh.'''),
    ]
    write_notebook("A2/a2_diffusion_schedules.ipynb", cells)


def main() -> None:
    build_e1()
    build_e2()
    build_e3()
    build_a1()
    build_a2()


if __name__ == "__main__":
    main()

