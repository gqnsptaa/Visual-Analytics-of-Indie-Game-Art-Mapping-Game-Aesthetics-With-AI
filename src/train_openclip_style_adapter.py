#!/usr/bin/env python3
"""Train a style classifier on top of frozen OpenCLIP embeddings.

This script provides a practical adapter-based fine-tuning workflow:
- OpenCLIP image encoder is kept frozen for stability/speed.
- A bottleneck adapter + classifier head is trained on labeled style data.
- Supports train/val CSV splits, class imbalance weighting, and checkpoint export.

Expected CSV format:
    path,label
    GameA/img1.jpg,cinematic low-key lighting
    GameB/img2.png,flat geometric vector design

`path` can be absolute or relative to --image-root.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, Dataset, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover
    missing = exc.name or "unknown"
    raise SystemExit(f"[ERROR] Missing dependency '{missing}'. Run: pip install -r requirements.txt") from exc


class TrainingError(RuntimeError):
    """Controlled, user-facing training error."""


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: str


class StyleCsvDataset(Dataset):
    """Loads images from CSV rows and applies OpenCLIP preprocessing."""

    def __init__(self, records: list[SampleRecord], preprocess) -> None:
        self.records = records
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        rec = self.records[index]
        try:
            with Image.open(rec.path) as image:
                tensor = self.preprocess(image.convert("RGB"))
        except Exception:
            # Skip unreadable images without killing the full run.
            return None

        return {
            "image": tensor,
            "label": rec.label,
            "path": str(rec.path),
        }


class BottleneckAdapter(nn.Module):
    """Small residual adapter on top of CLIP embeddings."""

    def __init__(self, dim: int, rank: int, dropout: float, scale: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("adapter rank must be > 0")
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        self.act = nn.GELU()

        # Start close to identity so early optimization is stable.
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.up(h)
        return x + self.scale * h


class StyleAdapterHead(nn.Module):
    """Adapter + linear classifier for style labels."""

    def __init__(self, embedding_dim: int, num_classes: int, rank: int, dropout: float, scale: float) -> None:
        super().__init__()
        self.adapter = BottleneckAdapter(embedding_dim, rank=rank, dropout=dropout, scale=scale)
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        x = self.norm(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train style adapter head on frozen OpenCLIP embeddings.")
    parser.add_argument("--train-csv", type=Path, required=True, help="CSV with columns: path,label")
    parser.add_argument("--val-csv", type=Path, default=None, help="Optional validation CSV. If missing, split train.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional base dir for relative image paths.")
    parser.add_argument("--output-dir", type=Path, default=Path("training_outputs/style_adapter"))
    parser.add_argument("--model-name", type=str, default="ViT-B/32", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", type=str, default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-embeddings", action="store_true", help="Cache train/val embeddings in output dir.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if device_arg == "mps":
        mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not mps_ok:
            raise TrainingError("MPS requested but unavailable.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def read_records(csv_path: Path, image_root: Path | None, max_samples: int) -> list[SampleRecord]:
    if not csv_path.exists():
        raise TrainingError(f"CSV not found: {csv_path}")

    records: list[SampleRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = {h.strip().lower() for h in (reader.fieldnames or [])}
        if "path" not in headers or "label" not in headers:
            raise TrainingError(f"CSV must include 'path' and 'label' columns: {csv_path}")

        for row in reader:
            raw_path = str(row.get("path", "")).strip()
            label = str(row.get("label", "")).strip()
            if not raw_path or not label:
                continue
            path = Path(raw_path)
            if not path.is_absolute() and image_root is not None:
                path = image_root / path
            records.append(SampleRecord(path=path, label=label))
            if max_samples > 0 and len(records) >= max_samples:
                break

    if not records:
        raise TrainingError(f"No valid rows found in CSV: {csv_path}")

    existing = [r for r in records if r.path.exists() and r.path.is_file()]
    if not existing:
        raise TrainingError(f"No image files exist from CSV paths: {csv_path}")
    return existing


def split_train_val(records: list[SampleRecord], val_ratio: float, seed: int) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not (0.0 < val_ratio < 0.95):
        raise TrainingError("val-ratio must be in (0, 0.95)")
    if len(records) < 8:
        raise TrainingError("Need at least 8 records to auto-split train/val.")

    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)
    val_count = max(1, int(len(items) * val_ratio))
    val_records = items[:val_count]
    train_records = items[val_count:]
    if len(train_records) < 2:
        raise TrainingError("Training split is too small; reduce val-ratio or add more samples.")
    return train_records, val_records


def collate_skip_invalid(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    valid = [b for b in batch if b is not None]
    if not valid:
        return None
    images = torch.stack([v["image"] for v in valid], dim=0)
    labels = [v["label"] for v in valid]
    paths = [v["path"] for v in valid]
    return {"images": images, "labels": labels, "paths": paths}


def load_openclip(model_name: str, pretrained: str, device: torch.device):
    try:
        import open_clip  # type: ignore
    except Exception as exc:
        raise TrainingError(f"open_clip import failed: {exc}") from exc

    openclip_name = model_name.replace("/", "-")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(openclip_name, pretrained=pretrained)
    except Exception as exc:
        raise TrainingError(f"Failed to load OpenCLIP model '{model_name}' with pretrained='{pretrained}': {exc}") from exc

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess


def build_label_maps(records: list[SampleRecord]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({r.label for r in records})
    if len(labels) < 2:
        raise TrainingError("Need at least 2 unique labels for classification training.")
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def encode_embeddings(
    model,
    preprocess,
    records: list[SampleRecord],
    label_to_id: dict[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if batch_size <= 0:
        raise TrainingError("batch-size must be > 0")

    dataset = StyleCsvDataset(records, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_invalid,
    )

    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    skipped = 0
    seen = 0

    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"]
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = model.encode_image(images)
            else:
                emb = model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            emb = emb.detach().cpu().to(torch.float32)

            batch_targets = []
            for label in labels:
                if label not in label_to_id:
                    skipped += 1
                    continue
                batch_targets.append(label_to_id[label])

            if len(batch_targets) != emb.shape[0]:
                # Defensive path; keep only aligned subset if label mapping failed.
                keep_count = min(len(batch_targets), emb.shape[0])
                emb = emb[:keep_count]
                batch_targets = batch_targets[:keep_count]

            if keep_count := len(batch_targets):
                features.append(emb[:keep_count])
                targets.append(torch.tensor(batch_targets, dtype=torch.long))
                seen += keep_count

    if seen == 0 or not features:
        raise TrainingError("No valid embeddings were encoded. Check image files and labels.")

    stats = {
        "encoded": seen,
        "skipped": skipped,
        "total_input": len(records),
    }
    return torch.cat(features, dim=0), torch.cat(targets, dim=0), stats


def compute_class_weights(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=num_classes).to(torch.float32)
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return weights


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    total = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.inference_mode():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            pred = torch.argmax(logits, dim=1)

            n = y.shape[0]
            loss_sum += float(loss.item()) * n
            total += n
            all_true.extend(y.detach().cpu().tolist())
            all_pred.extend(pred.detach().cpu().tolist())

    if total == 0:
        return {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0}

    accuracy = float(np.mean(np.array(all_true) == np.array(all_pred)))
    macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    return {
        "loss": loss_sum / total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = choose_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_records(args.train_csv, args.image_root, args.max_samples)
    if args.val_csv is not None:
        val_records = read_records(args.val_csv, args.image_root, args.max_samples)
    else:
        train_records, val_records = split_train_val(train_records, args.val_ratio, args.seed)

    all_records = train_records + val_records
    label_to_id, id_to_label = build_label_maps(all_records)

    print(f"Device: {device.type}")
    print(f"Train samples: {len(train_records)} | Val samples: {len(val_records)} | Classes: {len(label_to_id)}")

    model, preprocess = load_openclip(args.model_name, args.pretrained, device)
    print(f"Loaded OpenCLIP model={args.model_name} pretrained={args.pretrained}")

    train_cache_path = output_dir / "train_embeddings.pt"
    val_cache_path = output_dir / "val_embeddings.pt"

    if args.cache_embeddings and train_cache_path.exists() and val_cache_path.exists():
        train_blob = torch.load(train_cache_path, map_location="cpu")
        val_blob = torch.load(val_cache_path, map_location="cpu")
        train_x, train_y = train_blob["x"], train_blob["y"]
        val_x, val_y = val_blob["x"], val_blob["y"]
        train_stats = train_blob.get("stats", {})
        val_stats = val_blob.get("stats", {})
    else:
        train_x, train_y, train_stats = encode_embeddings(
            model=model,
            preprocess=preprocess,
            records=train_records,
            label_to_id=label_to_id,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        val_x, val_y, val_stats = encode_embeddings(
            model=model,
            preprocess=preprocess,
            records=val_records,
            label_to_id=label_to_id,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if args.cache_embeddings:
            torch.save({"x": train_x, "y": train_y, "stats": train_stats}, train_cache_path)
            torch.save({"x": val_x, "y": val_y, "stats": val_stats}, val_cache_path)

    print(f"Encoded train: {train_stats} | val: {val_stats}")

    emb_dim = int(train_x.shape[1])
    num_classes = len(label_to_id)

    head = StyleAdapterHead(
        embedding_dim=emb_dim,
        num_classes=num_classes,
        rank=args.adapter_rank,
        dropout=args.adapter_dropout,
        scale=args.adapter_scale,
    ).to(device)

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    class_weights = compute_class_weights(train_y, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best = {
        "epoch": -1,
        "macro_f1": -1.0,
        "accuracy": 0.0,
        "loss": float("inf"),
    }
    best_path = output_dir / "best_style_adapter.pt"

    for epoch in range(1, args.epochs + 1):
        head.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = head(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = yb.shape[0]
            train_loss_sum += float(loss.item()) * n
            train_count += n

        metrics = evaluate(head, val_loader, device)
        train_loss = train_loss_sum / max(1, train_count)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best["macro_f1"]:
            best = {
                "epoch": epoch,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
            }
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "embedding_dim": emb_dim,
                    "num_classes": num_classes,
                    "adapter_rank": args.adapter_rank,
                    "adapter_dropout": args.adapter_dropout,
                    "adapter_scale": args.adapter_scale,
                    "model_name": args.model_name,
                    "pretrained": args.pretrained,
                    "best_metrics": best,
                },
                best_path,
            )

    summary = {
        "best": best,
        "output_checkpoint": str(best_path),
        "num_train": int(train_y.shape[0]),
        "num_val": int(val_y.shape[0]),
        "classes": id_to_label,
        "args": vars(args),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best metrics: {best}")


if __name__ == "__main__":
    try:
        main()
    except TrainingError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
