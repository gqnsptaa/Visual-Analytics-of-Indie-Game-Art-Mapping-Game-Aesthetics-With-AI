#!/usr/bin/env python3
"""Phase-3 advanced separability analysis for AAA vs Indie covers.

Features:
- Overlapping Coefficient (OVL) on multiple distributions.
- Adjusted Rand Index (ARI) from k-means clustering across PCA levels.
- 2D KDE heatmaps (AAA, Indie, and Indie-AAA difference) on PCA-2 space.
- PCA-level evaluation of separability metrics.
- Residual analysis for a logistic baseline.
- Pairwise cosine-similarity distribution analysis.
- Two prompt-style cosine-similarity comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, log_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] scikit-learn is required for phase-3 analysis. "
        "Install dependencies from requirements.txt."
    ) from exc

DATASET_ROOT = Path("/Users/gqnsptaa/Desktop/Codex_Project/indie_games_dataset")

COLOR_BY_GROUP = {
    "indie": "#10b981",
    "aaa": "#f97316",
}


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    label: str
    path: Path


class PipelineError(RuntimeError):
    """Controlled pipeline exception with user-facing messages."""


class ClipAdapter:
    """Unified interface for OpenAI CLIP and OpenCLIP backends."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess,
        tokenize_fn,
        backend_name: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.preprocess = preprocess
        self.tokenize_fn = tokenize_fn
        self.backend_name = backend_name
        self.device = device

    def encode_images(self, batch: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(batch)
        return image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_text_tokens(self, token_batch: torch.Tensor) -> torch.Tensor:
        text_features = self.model.encode_text(token_batch)
        return text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise PipelineError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not mps_ok:
            raise PipelineError("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def load_clip_adapter(backend: str, model_name: str, device: torch.device) -> ClipAdapter:
    errors: list[str] = []

    if backend in {"auto", "openai"}:
        try:
            import clip  # type: ignore

            model, preprocess = clip.load(model_name, device=device)
            model.eval()
            return ClipAdapter(
                model=model,
                preprocess=preprocess,
                tokenize_fn=clip.tokenize,
                backend_name="openai",
                device=device,
            )
        except Exception as exc:
            errors.append(f"openai backend failed: {exc}")
            if backend == "openai":
                raise

    if backend in {"auto", "open_clip"}:
        try:
            import open_clip  # type: ignore

            model_to_use = model_name.replace("/", "-")
            model, _, preprocess = open_clip.create_model_and_transforms(model_to_use, pretrained="openai")
            model = model.to(device)
            model.eval()
            tokenizer = open_clip.get_tokenizer(model_to_use)
            return ClipAdapter(
                model=model,
                preprocess=preprocess,
                tokenize_fn=tokenizer,
                backend_name="open_clip",
                device=device,
            )
        except Exception as exc:
            errors.append(f"open_clip backend failed: {exc}")

    raise PipelineError(
        "Unable to load CLIP backend. Install one of:\n"
        "1) OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git\n"
        "2) OpenCLIP: pip install open-clip-torch\n"
        f"Backend errors: {' | '.join(errors)}"
    )


def normalize_group_name(raw_value: str) -> str:
    value = raw_value.strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    return value or "unassigned"


def load_game_groups(path: Path) -> dict[str, str]:
    if not path.exists():
        raise PipelineError(f"Game groups file not found: {path}")

    mapping: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = {h.strip().lower() for h in (reader.fieldnames or [])}
            if "game" not in headers or "group" not in headers:
                raise PipelineError(f"Game groups file must include columns 'game' and 'group': {path}")
            for row in reader:
                game = str(row.get("game", "")).strip()
                group = normalize_group_name(str(row.get("group", "")))
                if game:
                    mapping[game] = group
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"Failed to read game groups file {path}: {exc}") from exc

    return mapping


def collect_image_records(root_dir: Path, valid_extensions: tuple[str, ...]) -> list[ImageRecord]:
    if not root_dir.exists():
        raise PipelineError(f"Dataset folder does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise PipelineError(f"Dataset path is not a directory: {root_dir}")

    records: list[ImageRecord] = []
    image_id = 0
    for game_dir in sorted(root_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        label = game_dir.name
        for img_path in sorted(game_dir.rglob("*")):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in valid_extensions:
                continue
            records.append(ImageRecord(image_id=image_id, label=label, path=img_path))
            image_id += 1
    return records


def normalize_extensions(raw: str) -> tuple[str, ...]:
    normalized = []
    for ext in raw.split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        normalized.append(ext if ext.startswith(".") else f".{ext}")
    if not normalized:
        raise PipelineError("No valid image extensions provided.")
    return tuple(normalized)


def encode_images(
    adapter: ClipAdapter,
    records: list[ImageRecord],
    batch_size: int,
    progress_every_batches: int,
) -> tuple[np.ndarray, list[ImageRecord], list[dict]]:
    if batch_size <= 0:
        raise PipelineError("batch-size must be > 0.")

    valid_records: list[ImageRecord] = []
    skipped: list[dict] = []
    features: list[np.ndarray] = []

    batch_tensors: list[torch.Tensor] = []
    batch_records: list[ImageRecord] = []
    processed_batches = 0
    progress_every_batches = max(1, int(progress_every_batches))

    def flush_batch() -> None:
        nonlocal processed_batches
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(adapter.device)
        with torch.no_grad():
            encoded = adapter.encode_images(batch)
        features.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
        valid_records.extend(batch_records)
        batch_tensors.clear()
        batch_records.clear()
        processed_batches += 1
        if processed_batches % progress_every_batches == 0:
            print(
                f"[CLIP] Encoded batches: {processed_batches} | valid images so far: {len(valid_records)}",
                flush=True,
            )

    for rec in records:
        try:
            with Image.open(rec.path) as img:
                rgb = img.convert("RGB")
            tensor = adapter.preprocess(rgb)
            batch_tensors.append(tensor)
            batch_records.append(rec)
        except Exception as exc:
            skipped.append(
                {
                    "image_id": rec.image_id,
                    "label": rec.label,
                    "path": str(rec.path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()
    if not features:
        raise PipelineError("No valid images were encoded. Check image files and extensions.")

    return np.vstack(features).astype(np.float32, copy=False), valid_records, skipped


def encode_text_prompts(
    adapter: ClipAdapter,
    prompts: list[str],
    batch_size: int = 128,
) -> np.ndarray:
    if not prompts:
        raise PipelineError("Prompt list is empty.")
    encoded: list[np.ndarray] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        tokens = adapter.tokenize_fn(chunk)
        if not isinstance(tokens, torch.Tensor):
            try:
                tokens = torch.as_tensor(tokens)
            except Exception as exc:
                raise PipelineError(f"Tokenizer returned unsupported object type: {type(tokens)}") from exc
        tokens = tokens.to(adapter.device)
        with torch.no_grad():
            emb = adapter.encode_text_tokens(tokens)
        encoded.append(emb.detach().cpu().numpy().astype(np.float32, copy=False))
    out = np.vstack(encoded).astype(np.float32, copy=False)
    if out.shape[0] == 0:
        raise PipelineError("Failed to encode prompts.")
    return out


def load_prompt_file(path: Path) -> list[str]:
    if not path.exists():
        raise PipelineError(f"Prompt file not found: {path}")
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    prompts: list[str] = []
    seen: set[str] = set()
    for line in raw_lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)
    if not prompts:
        raise PipelineError(f"No usable prompts found in file: {path}")
    return prompts


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = int(token)
        except ValueError as exc:
            raise PipelineError(f"Invalid integer token in list: '{token}'") from exc
        if val <= 0:
            raise PipelineError(f"Integer list values must be > 0, got {val}.")
        values.append(val)
    if not values:
        raise PipelineError("No valid integers provided.")
    return sorted(set(values))


def overlapping_coefficient(values_a: np.ndarray, values_b: np.ndarray, bins: int = 120) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, max(20, int(bins)) + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    ovl = float(np.sum(np.minimum(ha, hb) * widths))
    return max(0.0, min(1.0, ovl))


def compute_group_centroid_distances(
    embeddings: np.ndarray,
    groups: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if embeddings.ndim != 2:
        raise PipelineError("Embeddings must be a 2D matrix.")
    if len(groups) != embeddings.shape[0]:
        raise PipelineError("Group label length mismatch for centroid distance computation.")

    centroids: dict[str, np.ndarray] = {}
    for group in ("aaa", "indie"):
        idx = [i for i, g in enumerate(groups) if g == group]
        if not idx:
            raise PipelineError(f"No samples found for group '{group}' in centroid-distance calculation.")
        centroid = embeddings[idx].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroids[group] = centroid / max(1e-12, norm)

    distances = np.zeros((embeddings.shape[0],), dtype=np.float32)
    by_group: dict[str, np.ndarray] = {}
    for group in ("aaa", "indie"):
        idx = np.array([i for i, g in enumerate(groups) if g == group], dtype=np.int32)
        dots = np.sum(embeddings[idx] * centroids[group], axis=1)
        group_dist = (1.0 - dots).astype(np.float32)
        distances[idx] = group_dist
        by_group[group] = group_dist
    return distances, by_group


def summarize_dist(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def bernoulli_deviance_residual(y_true: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    log_term = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    dev = np.sqrt(np.maximum(0.0, -2.0 * log_term))
    sign = np.where(y - p >= 0.0, 1.0, -1.0)
    return (sign * dev).astype(np.float64)


def sample_within_pairwise_cosine(
    embeddings: np.ndarray,
    indices: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    n = idx.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=np.float32)

    max_pairs = max(1, int(max_pairs))
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs and n <= 1200:
        sims: list[float] = []
        for i in range(n):
            vi = embeddings[idx[i]]
            for j in range(i + 1, n):
                sims.append(float(np.dot(vi, embeddings[idx[j]])))
        return np.asarray(sims, dtype=np.float32)

    i = rng.choice(idx, size=max_pairs, replace=True)
    j = rng.choice(idx, size=max_pairs, replace=True)
    same = i == j
    while np.any(same):
        j[same] = rng.choice(idx, size=int(np.sum(same)), replace=True)
        same = i == j
    sims = np.sum(embeddings[i] * embeddings[j], axis=1)
    return sims.astype(np.float32)


def sample_cross_pairwise_cosine(
    embeddings: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ia = np.asarray(indices_a, dtype=np.int64)
    ib = np.asarray(indices_b, dtype=np.int64)
    if ia.size == 0 or ib.size == 0:
        return np.zeros((0,), dtype=np.float32)
    max_pairs = max(1, int(max_pairs))
    ai = rng.choice(ia, size=max_pairs, replace=True)
    bi = rng.choice(ib, size=max_pairs, replace=True)
    sims = np.sum(embeddings[ai] * embeddings[bi], axis=1)
    return sims.astype(np.float32)


def make_kde_heatmap_figure(
    coords_2d: np.ndarray,
    groups: list[str],
    title_prefix: str,
    grid_size: int,
) -> go.Figure:
    a_idx = np.array([i for i, g in enumerate(groups) if g == "aaa"], dtype=np.int32)
    i_idx = np.array([i for i, g in enumerate(groups) if g == "indie"], dtype=np.int32)
    if a_idx.size < 8 or i_idx.size < 8:
        raise PipelineError("Not enough points per group for KDE heatmap.")

    all_x = coords_2d[:, 0]
    all_y = coords_2d[:, 1]
    x_span = max(1e-6, float(np.max(all_x) - np.min(all_x)))
    y_span = max(1e-6, float(np.max(all_y) - np.min(all_y)))
    x_pad = 0.08 * x_span
    y_pad = 0.08 * y_span
    gx = np.linspace(float(np.min(all_x) - x_pad), float(np.max(all_x) + x_pad), max(60, int(grid_size)))
    gy = np.linspace(float(np.min(all_y) - y_pad), float(np.max(all_y) + y_pad), max(60, int(grid_size)))
    xx, yy = np.meshgrid(gx, gy)
    query = np.vstack([xx.ravel(), yy.ravel()])

    a_xy = coords_2d[a_idx]
    i_xy = coords_2d[i_idx]
    kde_a = gaussian_kde(np.vstack([a_xy[:, 0], a_xy[:, 1]]))
    kde_i = gaussian_kde(np.vstack([i_xy[:, 0], i_xy[:, 1]]))
    z_a = kde_a(query).reshape(xx.shape)
    z_i = kde_i(query).reshape(xx.shape)
    z_diff = z_i - z_a

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"{title_prefix} | AAA KDE", f"{title_prefix} | Indie KDE", f"{title_prefix} | Indie-AAA KDE"],
        horizontal_spacing=0.05,
    )

    fig.add_trace(
        go.Heatmap(x=gx, y=gy, z=z_a, colorscale="Oranges", showscale=False, name="AAA KDE"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(x=gx, y=gy, z=z_i, colorscale="Greens", showscale=False, name="Indie KDE"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            x=gx,
            y=gy,
            z=z_diff,
            colorscale="RdBu",
            zmid=0.0,
            showscale=False,
            name="Indie-AAA KDE",
        ),
        row=1,
        col=3,
    )
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="PCA-1", row=1, col=col)
        fig.update_yaxes(title_text="PCA-2", row=1, col=col)

    fig.update_layout(
        title=f"{title_prefix}: 2D KDE Heatmaps",
        template="plotly_white",
        height=520,
        width=1500,
        margin={"l": 40, "r": 20, "t": 70, "b": 60},
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run advanced AAA vs Indie separability analysis with overlap, ARI, KDE, PCA levels, residuals, and prompt-style similarity."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Dataset root (default: {DATASET_ROOT}).",
    )
    parser.add_argument(
        "--game-groups-file",
        type=Path,
        default=Path("src/game_groups.csv"),
        help="CSV file with columns game,group (aaa/indie).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("web/data/phase3_advanced"),
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Total balanced sample size (0 = maximum balanced available). Must be even.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.webp",
        help="Comma-separated valid image extensions.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP encoding batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
    )
    parser.add_argument(
        "--clip-backend",
        type=str,
        default="open_clip",
        choices=["auto", "openai", "open_clip"],
        help="CLIP backend to use.",
    )
    parser.add_argument("--model-name", type=str, default="ViT-B/32", help="CLIP model name.")
    parser.add_argument(
        "--pca-levels",
        type=str,
        default="2,5,10,25,50,100,200",
        help="Comma-separated PCA component counts to evaluate.",
    )
    parser.add_argument(
        "--ari-seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds for k-means ARI averaging.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio for supervised metrics.")
    parser.add_argument("--kde-grid-size", type=int, default=140, help="Grid size for KDE heatmaps.")
    parser.add_argument(
        "--max-pairs-per-bucket",
        type=int,
        default=200000,
        help="Max sampled pairs for cosine-similarity distributions per bucket.",
    )
    parser.add_argument(
        "--prompt-style-a-file",
        type=Path,
        default=Path("src/style_prompts_graphic_design.txt"),
        help="Prompt file for style A.",
    )
    parser.add_argument(
        "--prompt-style-b-file",
        type=Path,
        default=Path("src/affective_prompts_thesis.txt"),
        help="Prompt file for style B.",
    )
    parser.add_argument("--prompt-style-a-name", type=str, default="graphic_design", help="Label for prompt style A.")
    parser.add_argument("--prompt-style-b-name", type=str, default="affective", help="Label for prompt style B.")
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=5,
        help="Print CLIP encoding progress every N batches.",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.expanduser().resolve()
    groups_file = args.game_groups_file.expanduser().resolve()

    valid_ext = normalize_extensions(args.extensions)
    pca_levels_req = parse_int_list(args.pca_levels)
    ari_seeds = parse_int_list(args.ari_seeds)

    if not (0.05 <= float(args.test_size) <= 0.5):
        raise PipelineError("test-size must be between 0.05 and 0.5.")

    print("[Init] Loading records and game-group mapping...", flush=True)
    records = collect_image_records(dataset_root, valid_ext)
    group_map = load_game_groups(groups_file)

    group_records: dict[str, list[ImageRecord]] = {"aaa": [], "indie": []}
    for rec in records:
        group = group_map.get(rec.label, "unassigned")
        if group in group_records:
            group_records[group].append(rec)

    aaa_count = len(group_records["aaa"])
    indie_count = len(group_records["indie"])
    max_balanced = 2 * min(aaa_count, indie_count)
    if max_balanced < 20:
        raise PipelineError(
            f"Not enough balanced samples. AAA={aaa_count}, Indie={indie_count}, max_balanced={max_balanced}."
        )

    target_size = int(args.sample_size) if int(args.sample_size) > 0 else int(max_balanced)
    if target_size % 2 != 0:
        raise PipelineError(f"sample-size must be even for balanced sampling, got {target_size}.")
    if target_size > max_balanced:
        raise PipelineError(
            f"sample-size={target_size} exceeds max balanced size {max_balanced} "
            f"(AAA={aaa_count}, Indie={indie_count})."
        )
    per_group = target_size // 2

    selected_records: list[ImageRecord] = []
    for group in ("aaa", "indie"):
        rows = list(group_records[group])
        rng.shuffle(rows)
        selected_records.extend(rows[:per_group])
    rng.shuffle(selected_records)
    print(
        f"[Init] Selected balanced set: n={target_size} ({per_group} AAA + {per_group} Indie).",
        flush=True,
    )

    print("[Step 1/8] Loading CLIP backend...", flush=True)
    device = choose_device(args.device)
    adapter = load_clip_adapter(args.clip_backend, args.model_name, device)
    print(
        f"[Step 1/8] CLIP backend ready: backend={adapter.backend_name} model={args.model_name} device={device.type}",
        flush=True,
    )

    print("[Step 2/8] Encoding selected covers with CLIP...", flush=True)
    embeddings_raw, valid_records, skipped_images = encode_images(
        adapter,
        selected_records,
        batch_size=int(args.batch_size),
        progress_every_batches=int(args.progress_every_batches),
    )
    groups_raw = [group_map.get(rec.label, "unassigned") for rec in valid_records]
    valid_idx = {"aaa": [], "indie": []}
    for i, g in enumerate(groups_raw):
        if g in valid_idx:
            valid_idx[g].append(i)
    for group in ("aaa", "indie"):
        if len(valid_idx[group]) < per_group:
            raise PipelineError(
                f"After decode/filtering, group '{group}' has only {len(valid_idx[group])}, need {per_group}."
            )

    rng_subset = random.Random(args.seed + 101)
    for group in ("aaa", "indie"):
        rng_subset.shuffle(valid_idx[group])
    final_indices = valid_idx["aaa"][:per_group] + valid_idx["indie"][:per_group]
    rng_subset.shuffle(final_indices)
    final_idx_arr = np.asarray(final_indices, dtype=np.int32)

    embeddings = embeddings_raw[final_idx_arr]
    records_final = [valid_records[i] for i in final_indices]
    groups = [groups_raw[i] for i in final_indices]
    y = np.array([1 if g == "indie" else 0 for g in groups], dtype=np.int32)
    n_samples, n_features = embeddings.shape

    print("[Step 3/8] Computing centroid distances and overlap metrics...", flush=True)
    all_dist, by_group_dist = compute_group_centroid_distances(embeddings, groups)
    dist_summary = {
        "aaa": summarize_dist(by_group_dist["aaa"]),
        "indie": summarize_dist(by_group_dist["indie"]),
    }
    dist_summary["indie_vs_aaa_ratio"] = float(
        dist_summary["indie"]["mean"] / max(1e-12, dist_summary["aaa"]["mean"])
    )
    dist_summary["mean_gap_indie_minus_aaa"] = float(dist_summary["indie"]["mean"] - dist_summary["aaa"]["mean"])
    dist_summary["ovl_centroid_distance"] = overlapping_coefficient(by_group_dist["aaa"], by_group_dist["indie"])

    print("[Step 4/8] Evaluating PCA levels + ARI + supervised metrics...", flush=True)
    max_pca = int(min(n_samples - 1, n_features))
    pca_levels = [n for n in pca_levels_req if n <= max_pca]
    if not pca_levels:
        raise PipelineError(
            f"No requested PCA levels fit this data (max allowed={max_pca}, requested={pca_levels_req})."
        )
    pca_full = PCA(n_components=max_pca, random_state=args.seed)
    pca_full.fit(embeddings)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    pca2_coords = pca_full.transform(embeddings)[:, :2].astype(np.float32, copy=False)

    train_idx, test_idx = train_test_split(
        np.arange(n_samples, dtype=np.int32),
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y,
    )
    y_train = y[train_idx]
    y_test = y[test_idx]

    level_rows: list[dict] = []

    def evaluate_level(level_components: int | None) -> dict:
        level_name = "raw" if level_components is None else f"pca_{level_components}"
        if level_components is None:
            x_train = embeddings[train_idx]
            x_test = embeddings[test_idx]
            ari_input = embeddings
            cum_var = 1.0
            n_comp = n_features
            clf_pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
                ]
            )
        else:
            n_comp = int(level_components)
            ari_pca = PCA(n_components=n_comp, random_state=args.seed)
            ari_input = ari_pca.fit_transform(embeddings)
            cum_var = float(cumulative[n_comp - 1])
            clf_pipe = Pipeline(
                steps=[
                    ("pca", PCA(n_components=n_comp, random_state=args.seed)),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
                ]
            )
            x_train = embeddings[train_idx]
            x_test = embeddings[test_idx]

        ari_values: list[float] = []
        for kseed in ari_seeds:
            km = KMeans(n_clusters=2, n_init=20, random_state=int(kseed))
            labels = km.fit_predict(ari_input)
            ari_values.append(float(adjusted_rand_score(y, labels)))
        ari_mean = float(np.mean(ari_values))
        ari_std = float(np.std(ari_values))

        clf_pipe.fit(x_train, y_train)
        prob = clf_pipe.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(np.int32)
        auc = float(roc_auc_score(y_test, prob))
        acc = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred))
        ll = float(log_loss(y_test, np.clip(prob, 1e-9, 1 - 1e-9)))
        ovl_prob = overlapping_coefficient(prob[y_test == 0], prob[y_test == 1], bins=120)
        return {
            "level": level_name,
            "n_components": int(n_comp),
            "cumulative_explained_variance": float(cum_var),
            "ari_mean": ari_mean,
            "ari_std": ari_std,
            "logreg_auc": auc,
            "logreg_accuracy": acc,
            "logreg_f1": f1,
            "logreg_logloss": ll,
            "ovl_predicted_probability_by_group": float(ovl_prob),
        }

    level_rows.append(evaluate_level(None))
    for n_comp in pca_levels:
        level_rows.append(evaluate_level(int(n_comp)))

    level_rows_sorted = sorted(level_rows, key=lambda r: float(r["logreg_auc"]), reverse=True)
    best_level = level_rows_sorted[0]["level"]

    pca_csv = output_dir / "pca_level_metrics.csv"
    with pca_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "n_components",
                "cumulative_explained_variance",
                "ari_mean",
                "ari_std",
                "logreg_auc",
                "logreg_accuracy",
                "logreg_f1",
                "logreg_logloss",
                "ovl_predicted_probability_by_group",
            ],
        )
        writer.writeheader()
        writer.writerows(level_rows)

    pca_fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["LogReg AUC by Level", "KMeans ARI by Level", "Cumulative Explained Variance"],
        horizontal_spacing=0.08,
    )
    x_labels = [r["level"] for r in level_rows]
    pca_fig.add_trace(go.Scatter(x=x_labels, y=[r["logreg_auc"] for r in level_rows], mode="lines+markers"), row=1, col=1)
    pca_fig.add_trace(go.Scatter(x=x_labels, y=[r["ari_mean"] for r in level_rows], mode="lines+markers"), row=1, col=2)
    pca_fig.add_trace(
        go.Scatter(x=x_labels, y=[r["cumulative_explained_variance"] for r in level_rows], mode="lines+markers"),
        row=1,
        col=3,
    )
    for col in (1, 2, 3):
        pca_fig.update_xaxes(title_text="Representation", row=1, col=col)
    pca_fig.update_yaxes(title_text="AUC", row=1, col=1)
    pca_fig.update_yaxes(title_text="ARI", row=1, col=2)
    pca_fig.update_yaxes(title_text="Cum. Explained Variance", row=1, col=3)
    pca_fig.update_layout(
        title="PCA-Level Separability Metrics",
        template="plotly_white",
        height=520,
        width=1500,
        margin={"l": 40, "r": 20, "t": 70, "b": 80},
    )
    pca_html = output_dir / "pca_level_metrics.html"
    pca_fig.write_html(pca_html, include_plotlyjs="cdn")

    print("[Step 5/8] Building KDE heatmap analysis...", flush=True)
    kde_fig = make_kde_heatmap_figure(
        pca2_coords,
        groups,
        title_prefix="PCA-2",
        grid_size=int(args.kde_grid_size),
    )
    kde_html = output_dir / "kde_heatmap_pca2.html"
    kde_fig.write_html(kde_html, include_plotlyjs="cdn")

    print("[Step 6/8] Running residual analysis...", flush=True)
    if best_level == "raw":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
            ]
        )
    else:
        n_best = int(best_level.split("_", 1)[1])
        model = Pipeline(
            steps=[
                ("pca", PCA(n_components=n_best, random_state=args.seed)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
            ]
        )
    model.fit(embeddings[train_idx], y_train)
    p_test = model.predict_proba(embeddings[test_idx])[:, 1]
    pred_test = (p_test >= 0.5).astype(np.int32)
    raw_resid = y_test.astype(np.float64) - p_test
    dev_resid = bernoulli_deviance_residual(y_test, p_test)

    residual_rows: list[dict] = []
    for local_pos, idx in enumerate(test_idx.tolist()):
        rec = records_final[idx]
        residual_rows.append(
            {
                "index": int(idx),
                "image_id": int(rec.image_id),
                "game": rec.label,
                "path": str(rec.path),
                "group": groups[idx],
                "y_true_indie": int(y[idx]),
                "p_pred_indie": float(p_test[local_pos]),
                "pred_label": "indie" if int(pred_test[local_pos]) == 1 else "aaa",
                "is_error": int(pred_test[local_pos] != y_test[local_pos]),
                "residual_raw": float(raw_resid[local_pos]),
                "residual_deviance": float(dev_resid[local_pos]),
                "residual_abs_deviance": float(abs(dev_resid[local_pos])),
            }
        )
    residual_rows.sort(key=lambda r: -float(r["residual_abs_deviance"]))
    residual_csv = output_dir / "residual_analysis_rows.csv"
    with residual_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "image_id",
                "game",
                "path",
                "group",
                "y_true_indie",
                "p_pred_indie",
                "pred_label",
                "is_error",
                "residual_raw",
                "residual_deviance",
                "residual_abs_deviance",
            ],
        )
        writer.writeheader()
        writer.writerows(residual_rows)

    res_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Deviance Residual vs Predicted Probability", "Deviance Residual Distribution"],
        horizontal_spacing=0.12,
    )
    test_groups = [groups[idx] for idx in test_idx.tolist()]
    for grp in ("aaa", "indie"):
        m = np.array([g == grp for g in test_groups], dtype=bool)
        res_fig.add_trace(
            go.Scattergl(
                x=p_test[m],
                y=dev_resid[m],
                mode="markers",
                marker={"size": 6, "opacity": 0.55, "color": COLOR_BY_GROUP[grp]},
                name=f"{grp.upper()}",
                legendgroup=grp,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        res_fig.add_trace(
            go.Histogram(
                x=dev_resid[m],
                opacity=0.45,
                nbinsx=50,
                marker={"color": COLOR_BY_GROUP[grp]},
                name=f"{grp.upper()} residuals",
                legendgroup=grp,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    res_fig.update_xaxes(title_text="Predicted P(Indie)", row=1, col=1)
    res_fig.update_yaxes(title_text="Deviance Residual", row=1, col=1)
    res_fig.update_xaxes(title_text="Deviance Residual", row=1, col=2)
    res_fig.update_yaxes(title_text="Count", row=1, col=2)
    res_fig.update_layout(
        title=f"Residual Analysis (Best Level: {best_level})",
        template="plotly_white",
        barmode="overlay",
        height=520,
        width=1400,
        margin={"l": 40, "r": 20, "t": 70, "b": 80},
    )
    residual_html = output_dir / "residual_analysis.html"
    res_fig.write_html(residual_html, include_plotlyjs="cdn")

    print("[Step 7/8] Computing cosine-similarity distributions...", flush=True)
    idx_aaa = np.array([i for i, g in enumerate(groups) if g == "aaa"], dtype=np.int32)
    idx_indie = np.array([i for i, g in enumerate(groups) if g == "indie"], dtype=np.int32)
    sims_aaa = sample_within_pairwise_cosine(embeddings, idx_aaa, int(args.max_pairs_per_bucket), np_rng)
    sims_indie = sample_within_pairwise_cosine(embeddings, idx_indie, int(args.max_pairs_per_bucket), np_rng)
    sims_cross = sample_cross_pairwise_cosine(embeddings, idx_aaa, idx_indie, int(args.max_pairs_per_bucket), np_rng)

    cos_summary = {
        "aaa_within": summarize_dist(sims_aaa),
        "indie_within": summarize_dist(sims_indie),
        "cross_group": summarize_dist(sims_cross),
        "ovl_aaa_vs_indie_within": overlapping_coefficient(sims_aaa, sims_indie),
        "ovl_aaa_within_vs_cross": overlapping_coefficient(sims_aaa, sims_cross),
        "ovl_indie_within_vs_cross": overlapping_coefficient(sims_indie, sims_cross),
    }

    cos_fig = go.Figure()
    cos_fig.add_trace(
        go.Histogram(
            x=sims_aaa,
            name="AAA-AA",
            opacity=0.45,
            nbinsx=80,
            marker={"color": COLOR_BY_GROUP["aaa"]},
        )
    )
    cos_fig.add_trace(
        go.Histogram(
            x=sims_indie,
            name="Indie-Indie",
            opacity=0.45,
            nbinsx=80,
            marker={"color": COLOR_BY_GROUP["indie"]},
        )
    )
    cos_fig.add_trace(
        go.Histogram(
            x=sims_cross,
            name="AAA-Indie",
            opacity=0.4,
            nbinsx=80,
            marker={"color": "#6366f1"},
        )
    )
    cos_fig.update_layout(
        title="Pairwise Cosine Similarity Distributions",
        template="plotly_white",
        barmode="overlay",
        xaxis_title="Cosine Similarity",
        yaxis_title="Count",
        height=520,
        width=1100,
        margin={"l": 40, "r": 20, "t": 70, "b": 80},
    )
    cosine_html = output_dir / "cosine_similarity_distributions.html"
    cos_fig.write_html(cosine_html, include_plotlyjs="cdn")

    cosine_csv = output_dir / "cosine_similarity_samples.csv"
    with cosine_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "value"])
        writer.writeheader()
        for v in sims_aaa.tolist():
            writer.writerow({"bucket": "aaa_within", "value": float(v)})
        for v in sims_indie.tolist():
            writer.writerow({"bucket": "indie_within", "value": float(v)})
        for v in sims_cross.tolist():
            writer.writerow({"bucket": "cross_group", "value": float(v)})

    print("[Step 8/8] Computing two prompt-style cosine similarity metrics...", flush=True)
    prompts_a = load_prompt_file(args.prompt_style_a_file)
    prompts_b = load_prompt_file(args.prompt_style_b_file)
    text_emb_a = encode_text_prompts(adapter, prompts_a, batch_size=128)
    text_emb_b = encode_text_prompts(adapter, prompts_b, batch_size=128)

    sims_a = embeddings @ text_emb_a.T
    sims_b = embeddings @ text_emb_b.T
    style_a_mean = sims_a.mean(axis=1)
    style_a_max = sims_a.max(axis=1)
    style_b_mean = sims_b.mean(axis=1)
    style_b_max = sims_b.max(axis=1)
    delta_mean = style_a_mean - style_b_mean
    delta_max = style_a_max - style_b_max

    gmask_aaa = np.array([g == "aaa" for g in groups], dtype=bool)
    gmask_indie = ~gmask_aaa

    prompt_summary = {
        "style_a_name": args.prompt_style_a_name,
        "style_b_name": args.prompt_style_b_name,
        "style_a_prompt_count": len(prompts_a),
        "style_b_prompt_count": len(prompts_b),
        "style_a_mean_score": {
            "aaa": summarize_dist(style_a_mean[gmask_aaa]),
            "indie": summarize_dist(style_a_mean[gmask_indie]),
            "ovl_aaa_vs_indie": overlapping_coefficient(style_a_mean[gmask_aaa], style_a_mean[gmask_indie]),
        },
        "style_b_mean_score": {
            "aaa": summarize_dist(style_b_mean[gmask_aaa]),
            "indie": summarize_dist(style_b_mean[gmask_indie]),
            "ovl_aaa_vs_indie": overlapping_coefficient(style_b_mean[gmask_aaa], style_b_mean[gmask_indie]),
        },
        "delta_mean_styleA_minus_styleB": {
            "aaa": summarize_dist(delta_mean[gmask_aaa]),
            "indie": summarize_dist(delta_mean[gmask_indie]),
            "ovl_aaa_vs_indie": overlapping_coefficient(delta_mean[gmask_aaa], delta_mean[gmask_indie]),
        },
        "style_a_max_score": {
            "aaa": summarize_dist(style_a_max[gmask_aaa]),
            "indie": summarize_dist(style_a_max[gmask_indie]),
            "ovl_aaa_vs_indie": overlapping_coefficient(style_a_max[gmask_aaa], style_a_max[gmask_indie]),
        },
        "style_b_max_score": {
            "aaa": summarize_dist(style_b_max[gmask_aaa]),
            "indie": summarize_dist(style_b_max[gmask_indie]),
            "ovl_aaa_vs_indie": overlapping_coefficient(style_b_max[gmask_aaa], style_b_max[gmask_indie]),
        },
        "delta_max_styleA_minus_styleB": {
            "aaa": summarize_dist(delta_max[gmask_aaa]),
            "indie": summarize_dist(delta_max[gmask_indie]),
            "ovl_aaa_vs_indie": overlapping_coefficient(delta_max[gmask_aaa], delta_max[gmask_indie]),
        },
    }

    prompt_rows_csv = output_dir / "prompt_style_similarity_rows.csv"
    with prompt_rows_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "image_id",
                "game",
                "path",
                "group",
                "style_a_mean",
                "style_a_max",
                "style_b_mean",
                "style_b_max",
                "delta_mean_a_minus_b",
                "delta_max_a_minus_b",
            ],
        )
        writer.writeheader()
        for i, rec in enumerate(records_final):
            writer.writerow(
                {
                    "index": i,
                    "image_id": int(rec.image_id),
                    "game": rec.label,
                    "path": str(rec.path),
                    "group": groups[i],
                    "style_a_mean": float(style_a_mean[i]),
                    "style_a_max": float(style_a_max[i]),
                    "style_b_mean": float(style_b_mean[i]),
                    "style_b_max": float(style_b_max[i]),
                    "delta_mean_a_minus_b": float(delta_mean[i]),
                    "delta_max_a_minus_b": float(delta_max[i]),
                }
            )

    prompt_fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"{args.prompt_style_a_name} Mean Similarity by Group",
            f"{args.prompt_style_b_name} Mean Similarity by Group",
            f"Delta Mean ({args.prompt_style_a_name} - {args.prompt_style_b_name})",
        ],
        horizontal_spacing=0.08,
    )
    for grp in ("aaa", "indie"):
        mask = np.array([g == grp for g in groups], dtype=bool)
        prompt_fig.add_trace(
            go.Histogram(
                x=style_a_mean[mask],
                nbinsx=70,
                opacity=0.45,
                marker={"color": COLOR_BY_GROUP[grp]},
                name=f"{grp.upper()}",
                legendgroup=grp,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        prompt_fig.add_trace(
            go.Histogram(
                x=style_b_mean[mask],
                nbinsx=70,
                opacity=0.45,
                marker={"color": COLOR_BY_GROUP[grp]},
                name=f"{grp.upper()}",
                legendgroup=grp,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        prompt_fig.add_trace(
            go.Histogram(
                x=delta_mean[mask],
                nbinsx=70,
                opacity=0.45,
                marker={"color": COLOR_BY_GROUP[grp]},
                name=f"{grp.upper()}",
                legendgroup=grp,
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    for col in (1, 2, 3):
        prompt_fig.update_xaxes(title_text="Cosine Similarity", row=1, col=col)
        prompt_fig.update_yaxes(title_text="Count", row=1, col=col)
    prompt_fig.update_layout(
        title="Prompt-Style Cosine Similarity Distributions",
        template="plotly_white",
        barmode="overlay",
        height=520,
        width=1500,
        margin={"l": 40, "r": 20, "t": 70, "b": 80},
    )
    prompt_html = output_dir / "prompt_style_similarity_distributions.html"
    prompt_fig.write_html(prompt_html, include_plotlyjs="cdn")

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "game_groups_file": str(groups_file),
        "output_dir": str(output_dir),
        "clip_backend": adapter.backend_name,
        "model_name": args.model_name,
        "device": device.type,
        "sample_size": target_size,
        "selected_before_decode": len(selected_records),
        "valid_after_decode": len(valid_records),
        "final_balanced_after_decode": int(n_samples),
        "skipped_images_count": len(skipped_images),
        "pca_levels_requested": pca_levels_req,
        "pca_levels_evaluated": pca_levels,
        "best_level_by_auc": best_level,
        "centroid_distance_summary": dist_summary,
        "cosine_similarity_summary": cos_summary,
        "prompt_style_summary": prompt_summary,
        "outputs": {
            "pca_metrics_csv": str(pca_csv),
            "pca_metrics_html": str(pca_html),
            "kde_heatmap_html": str(kde_html),
            "residual_rows_csv": str(residual_csv),
            "residual_html": str(residual_html),
            "cosine_samples_csv": str(cosine_csv),
            "cosine_html": str(cosine_html),
            "prompt_rows_csv": str(prompt_rows_csv),
            "prompt_html": str(prompt_html),
        },
    }
    summary_json = output_dir / "phase3_advanced_report.json"
    summary_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {pca_csv}")
    print(f"Saved: {pca_html}")
    print(f"Saved: {kde_html}")
    print(f"Saved: {residual_csv}")
    print(f"Saved: {residual_html}")
    print(f"Saved: {cosine_csv}")
    print(f"Saved: {cosine_html}")
    print(f"Saved: {prompt_rows_csv}")
    print(f"Saved: {prompt_html}")
    print(f"Saved report: {summary_json}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except PipelineError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()

