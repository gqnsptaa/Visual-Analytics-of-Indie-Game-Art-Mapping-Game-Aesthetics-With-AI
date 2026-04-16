#!/usr/bin/env python3
"""Phase-2 overlap/distribution analysis for AAA vs Indie cover aesthetics.

This script focuses on local overlap and distribution spread (not strict separation):
- Balanced sample sizes: 500, 1000, 2000 covers (AAA + Indie).
- 2D UMAP and 2D densMAP projections.
- Overlaid density contours for both groups on each projection.
- Distance-to-centroid metric in CLIP embedding space + histograms.

Outputs are written as web-friendly HTML plots and CSV/JSON tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
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

DATASET_ROOT = Path("/Users/gqnsptaa/Desktop/Codex_Project/indie_games_dataset")

COLOR_BY_GROUP = {
    "indie": "#10b981",
    "aaa": "#f97316",
}


@dataclass(frozen=True)
class SampleSelection:
    size: int
    indices: np.ndarray


@dataclass(frozen=True)
class ImageRecord:
    """Metadata for one image sample."""

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


def pca_2d_fallback(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise PipelineError("Expected a 2D embedding matrix for PCA fallback.")
    x = x - np.mean(x, axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        if vh.shape[0] >= 2:
            coords = x @ vh[:2, :].T
        elif vh.shape[0] == 1:
            coords = np.concatenate([x @ vh[:1, :].T, np.zeros((x.shape[0], 1), dtype=np.float32)], axis=1)
        else:
            coords = np.zeros((x.shape[0], 2), dtype=np.float32)
    except Exception:
        coords = np.zeros((x.shape[0], 2), dtype=np.float32)
    return coords.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run overlap-focused AAA vs Indie analysis with UMAP + densMAP density contours "
            "for sample sizes 500/1000/2000."
        )
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
        default=Path("web/data/phase2_overlap"),
        help="Directory for generated HTML/CSV/JSON outputs.",
    )
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="500,1000,2000",
        help="Comma-separated total sample sizes (must be even, balanced by group).",
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
    parser.add_argument("--umap-n-neighbors", type=int, default=20, help="UMAP/densMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.2, help="UMAP/densMAP min_dist.")
    parser.add_argument("--densmap-lambda", type=float, default=2.0, help="densMAP lambda.")
    parser.add_argument("--densmap-frac", type=float, default=0.3, help="densMAP dens_frac.")
    parser.add_argument("--density-grid-size", type=int, default=120, help="Grid size for KDE contours.")
    parser.add_argument("--density-levels", type=int, default=8, help="Number of contour levels.")
    parser.add_argument(
        "--density-kde-bandwidth",
        type=float,
        default=1.0,
        help="KDE bandwidth scale for contour smoothing (higher = smoother contours).",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=5,
        help="Print CLIP encoding progress every N batches.",
    )
    return parser.parse_args()


def parse_sample_sizes(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            size = int(token)
        except ValueError:
            raise PipelineError(f"Invalid sample size: '{token}'. Expected integers like 500,1000,2000.")
        if size < 2:
            raise PipelineError(f"Sample size must be >= 2: {size}")
        if size % 2 != 0:
            raise PipelineError(f"Sample size must be even for balanced AAA/Indie sampling: {size}")
        values.append(size)
    if not values:
        raise PipelineError("No valid sample sizes provided.")
    return sorted(set(values))


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


def project_2d_umap(
    embeddings: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    densmap: bool,
    dens_lambda: float,
    dens_frac: float,
) -> tuple[np.ndarray, dict]:
    n_samples = int(embeddings.shape[0])
    if n_samples < 3:
        return pca_2d_fallback(embeddings), {
            "method": "pca_fallback",
            "reason": "too_few_samples",
        }

    safe_neighbors = min(max(2, int(n_neighbors)), max(2, n_samples - 1))
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=safe_neighbors,
            min_dist=float(min_dist),
            n_components=2,
            metric="cosine",
            random_state=seed,
            densmap=bool(densmap),
            dens_lambda=float(dens_lambda),
            dens_frac=float(dens_frac),
        )
        coords = reducer.fit_transform(embeddings).astype(np.float32)
        return coords, {
            "method": "densmap" if densmap else "umap",
            "n_neighbors": int(safe_neighbors),
            "min_dist": float(min_dist),
            "densmap": bool(densmap),
            "dens_lambda": float(dens_lambda),
            "dens_frac": float(dens_frac),
        }
    except Exception as exc:
        return pca_2d_fallback(embeddings), {
            "method": "pca_fallback",
            "reason": f"UMAP/densMAP failed: {exc}",
        }


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
        group_dist = (1.0 - dots).astype(np.float32)  # cosine distance to own-group centroid
        distances[idx] = group_dist
        by_group[group] = group_dist
    return distances, by_group


def summarize_distances(distances: dict[str, np.ndarray]) -> dict:
    summary: dict[str, dict] = {}
    for group, values in distances.items():
        v = np.asarray(values, dtype=np.float32)
        summary[group] = {
            "n": int(v.shape[0]),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "std": float(np.std(v)),
            "p90": float(np.quantile(v, 0.90)),
            "p95": float(np.quantile(v, 0.95)),
        }
    if "aaa" in summary and "indie" in summary:
        aaa_mean = summary["aaa"]["mean"]
        indie_mean = summary["indie"]["mean"]
        summary["indie_vs_aaa_mean_ratio"] = float(indie_mean / max(1e-12, aaa_mean))
        summary["mean_gap_indie_minus_aaa"] = float(indie_mean - aaa_mean)
    return summary


def make_projection_contour_figure(
    method_name: str,
    projections: dict[int, dict],
    *,
    density_grid_size: int,
    density_levels: int,
    density_kde_bandwidth: float,
) -> go.Figure:
    sizes = sorted(projections.keys())
    fig = make_subplots(
        rows=1,
        cols=len(sizes),
        subplot_titles=[f"{method_name} | n={size}" for size in sizes],
        horizontal_spacing=0.05,
    )

    for col, size in enumerate(sizes, start=1):
        payload = projections[size]
        coords = payload["coords"]
        groups = payload["groups"]

        for group in ("indie", "aaa"):
            mask = np.array([g == group for g in groups], dtype=bool)
            xy = coords[mask]
            color = COLOR_BY_GROUP[group]
            label = "Indie" if group == "indie" else "AAA"

            fig.add_trace(
                go.Scattergl(
                    x=xy[:, 0],
                    y=xy[:, 1],
                    mode="markers",
                    marker={"size": 4, "opacity": 0.45, "color": color},
                    name=f"{label} points",
                    legendgroup=group,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

            if xy.shape[0] < 8:
                continue

            x = np.asarray(xy[:, 0], dtype=np.float64)
            y = np.asarray(xy[:, 1], dtype=np.float64)
            x_span = max(1e-6, float(x.max() - x.min()))
            y_span = max(1e-6, float(y.max() - y.min()))
            x_pad = 0.08 * x_span
            y_pad = 0.08 * y_span
            gx = np.linspace(x.min() - x_pad, x.max() + x_pad, max(50, int(density_grid_size)))
            gy = np.linspace(y.min() - y_pad, y.max() + y_pad, max(50, int(density_grid_size)))
            xx, yy = np.meshgrid(gx, gy)

            try:
                kde = gaussian_kde(np.vstack([x, y]), bw_method=float(density_kde_bandwidth))
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            except Exception:
                # Numerical edge case (e.g., singular covariance); skip contour for this group/panel.
                continue

            positive = zz[zz > 0]
            if positive.size < 8:
                continue
            level_count = max(3, int(density_levels))
            # Keep a broad density span so contours preserve multimodal/local structure.
            level_low = float(np.quantile(positive, 0.08))
            level_high = float(np.quantile(positive, 0.995))
            if not np.isfinite(level_low) or not np.isfinite(level_high):
                continue
            if level_high <= level_low:
                level_low = float(np.min(positive))
                level_high = float(np.max(positive))
            if level_high <= level_low:
                continue
            level_step = float(max(1e-12, (level_high - level_low) / max(1, level_count - 1)))

            fig.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=zz,
                    contours={
                        "coloring": "none",
                        "showlabels": False,
                        "start": level_low,
                        "end": level_high,
                        "size": level_step,
                    },
                    line={"color": color, "width": 2.1},
                    opacity=0.92,
                    showscale=False,
                    name=f"{label} density",
                    legendgroup=group,
                    showlegend=(col == 1),
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )

        fig.update_xaxes(title_text=f"{method_name}-1", row=1, col=col, zeroline=False)
        fig.update_yaxes(title_text=f"{method_name}-2", row=1, col=col, zeroline=False)

    fig.update_layout(
        title=f"{method_name} Projections with Overlaid Group Density Contours",
        template="plotly_white",
        height=560,
        width=max(1000, 460 * len(sizes)),
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.18, "xanchor": "left", "x": 0.0},
        margin={"l": 40, "r": 20, "t": 70, "b": 90},
    )
    return fig


def make_centroid_distance_hist_figure(distance_rows: list[dict]) -> go.Figure:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in distance_rows:
        grouped[int(row["sample_size"])][str(row["group"])].append(float(row["distance_to_group_centroid"]))

    sizes = sorted(grouped.keys())
    fig = make_subplots(
        rows=1,
        cols=len(sizes),
        subplot_titles=[f"Centroid Distance | n={size}" for size in sizes],
        horizontal_spacing=0.06,
    )

    for col, size in enumerate(sizes, start=1):
        for group in ("indie", "aaa"):
            vals = grouped[size].get(group, [])
            label = "Indie" if group == "indie" else "AAA"
            fig.add_trace(
                go.Histogram(
                    x=vals,
                    nbinsx=45,
                    opacity=0.55,
                    marker={"color": COLOR_BY_GROUP[group]},
                    name=label,
                    legendgroup=group,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Cosine distance to own-group centroid", row=1, col=col)
        fig.update_yaxes(title_text="Count", row=1, col=col)

    fig.update_layout(
        title="Distance-to-Centroid Distribution by Group (Lower = More Concentrated)",
        template="plotly_white",
        barmode="overlay",
        height=520,
        width=max(1000, 460 * len(sizes)),
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.18, "xanchor": "left", "x": 0.0},
        margin={"l": 40, "r": 20, "t": 70, "b": 90},
    )
    return fig


def run() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    sample_sizes = parse_sample_sizes(args.sample_sizes)
    max_size = max(sample_sizes)
    per_group_needed = max_size // 2
    valid_ext = normalize_extensions(args.extensions)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.expanduser().resolve()
    groups_file = args.game_groups_file.expanduser().resolve()

    print("[Init] Loading records and game-group mapping...", flush=True)
    records = collect_image_records(dataset_root, valid_ext)
    group_map = load_game_groups(groups_file)

    group_records: dict[str, list] = {"aaa": [], "indie": []}
    for rec in records:
        group = group_map.get(rec.label, "unassigned")
        if group in group_records:
            group_records[group].append(rec)

    for group in ("aaa", "indie"):
        if len(group_records[group]) < per_group_needed:
            raise PipelineError(
                f"Not enough '{group}' images for requested sample size {max_size}. "
                f"Need {per_group_needed}, found {len(group_records[group])}."
            )

    selected_records = []
    for group in ("aaa", "indie"):
        rows = list(group_records[group])
        rng.shuffle(rows)
        selected_records.extend(rows[:per_group_needed])
    rng.shuffle(selected_records)

    print(
        f"[Init] Selected balanced CLIP encoding set: {len(selected_records)} "
        f"({per_group_needed} AAA + {per_group_needed} Indie).",
        flush=True,
    )

    print("[Step 1/5] Loading CLIP backend...", flush=True)
    device = choose_device(args.device)
    adapter = load_clip_adapter(args.clip_backend, args.model_name, device)
    setattr(adapter, "progress_every_batches", max(1, int(args.progress_every_batches)))
    print(
        f"[Step 1/5] CLIP backend ready: backend={adapter.backend_name} model={args.model_name} device={device.type}",
        flush=True,
    )

    print("[Step 2/5] Encoding selected covers with CLIP...", flush=True)
    embeddings, valid_records, skipped_images = encode_images(
        adapter,
        selected_records,
        args.batch_size,
        progress_every_batches=args.progress_every_batches,
    )
    groups = [group_map.get(rec.label, "unassigned") for rec in valid_records]

    valid_group_indices = {"aaa": [], "indie": []}
    for i, g in enumerate(groups):
        if g in valid_group_indices:
            valid_group_indices[g].append(i)

    for group in ("aaa", "indie"):
        if len(valid_group_indices[group]) < per_group_needed:
            raise PipelineError(
                f"After image-read filtering, not enough '{group}' samples remain for max size {max_size}. "
                f"Needed {per_group_needed}, got {len(valid_group_indices[group])}. "
                "Check unreadable images or reduce sample sizes."
            )

    # Deterministic order used for nested subsets: n=500 is subset of n=1000, etc.
    rng_subset = random.Random(args.seed + 101)
    for group in ("aaa", "indie"):
        rng_subset.shuffle(valid_group_indices[group])

    selections: list[SampleSelection] = []
    for size in sample_sizes:
        k = size // 2
        idx = valid_group_indices["aaa"][:k] + valid_group_indices["indie"][:k]
        rng_subset.shuffle(idx)
        selections.append(SampleSelection(size=size, indices=np.array(idx, dtype=np.int32)))

    projection_payload_umap: dict[int, dict] = {}
    projection_payload_densmap: dict[int, dict] = {}
    distance_rows: list[dict] = []
    summary_rows: list[dict] = []

    print("[Step 3/5] Computing UMAP + densMAP projections and centroid distances...", flush=True)
    for selection in selections:
        size = int(selection.size)
        idx = selection.indices
        sub_emb = embeddings[idx]
        sub_groups = [groups[i] for i in idx.tolist()]
        sub_labels = [valid_records[i].label for i in idx.tolist()]

        umap_coords, umap_meta = project_2d_umap(
            sub_emb,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
            densmap=False,
            dens_lambda=args.densmap_lambda,
            dens_frac=args.densmap_frac,
        )
        dens_coords, dens_meta = project_2d_umap(
            sub_emb,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
            densmap=True,
            dens_lambda=args.densmap_lambda,
            dens_frac=args.densmap_frac,
        )

        projection_payload_umap[size] = {
            "coords": umap_coords,
            "groups": sub_groups,
            "labels": sub_labels,
            "meta": umap_meta,
        }
        projection_payload_densmap[size] = {
            "coords": dens_coords,
            "groups": sub_groups,
            "labels": sub_labels,
            "meta": dens_meta,
        }

        _, by_group_dist = compute_group_centroid_distances(sub_emb, sub_groups)
        dist_summary = summarize_distances(by_group_dist)
        summary_rows.append(
            {
                "sample_size": size,
                "aaa_mean": dist_summary["aaa"]["mean"],
                "indie_mean": dist_summary["indie"]["mean"],
                "aaa_std": dist_summary["aaa"]["std"],
                "indie_std": dist_summary["indie"]["std"],
                "indie_vs_aaa_mean_ratio": dist_summary["indie_vs_aaa_mean_ratio"],
                "mean_gap_indie_minus_aaa": dist_summary["mean_gap_indie_minus_aaa"],
            }
        )

        group_distance_lookup = {"aaa": by_group_dist["aaa"], "indie": by_group_dist["indie"]}
        running_group_pos = {"aaa": 0, "indie": 0}
        for local_i, global_i in enumerate(idx.tolist()):
            g = sub_groups[local_i]
            pos = running_group_pos[g]
            dist_val = float(group_distance_lookup[g][pos])
            running_group_pos[g] += 1
            rec = valid_records[global_i]
            distance_rows.append(
                {
                    "sample_size": size,
                    "image_id": int(rec.image_id),
                    "game": rec.label,
                    "group": g,
                    "path": str(rec.path),
                    "distance_to_group_centroid": dist_val,
                }
            )

        print(
            f"[Step 3/5] n={size}: projections done | "
            f"AAA mean dist={summary_rows[-1]['aaa_mean']:.4f}, "
            f"Indie mean dist={summary_rows[-1]['indie_mean']:.4f}, "
            f"ratio={summary_rows[-1]['indie_vs_aaa_mean_ratio']:.4f}",
            flush=True,
        )

    print("[Step 4/5] Building visualization outputs...", flush=True)
    umap_fig = make_projection_contour_figure(
        "UMAP",
        projection_payload_umap,
        density_grid_size=args.density_grid_size,
        density_levels=args.density_levels,
        density_kde_bandwidth=args.density_kde_bandwidth,
    )
    dens_fig = make_projection_contour_figure(
        "densMAP",
        projection_payload_densmap,
        density_grid_size=args.density_grid_size,
        density_levels=args.density_levels,
        density_kde_bandwidth=args.density_kde_bandwidth,
    )
    hist_fig = make_centroid_distance_hist_figure(distance_rows)

    size_suffix = "_".join(str(s) for s in sample_sizes)
    umap_html = output_dir / f"umap_density_contours_{size_suffix}.html"
    dens_html = output_dir / f"densmap_density_contours_{size_suffix}.html"
    hist_html = output_dir / f"distance_to_centroid_hist_{size_suffix}.html"
    umap_fig.write_html(umap_html, include_plotlyjs="cdn")
    dens_fig.write_html(dens_html, include_plotlyjs="cdn")
    hist_fig.write_html(hist_html, include_plotlyjs="cdn")

    print("[Step 5/5] Writing metrics/tables...", flush=True)
    with (output_dir / "distance_to_centroid_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_size",
                "image_id",
                "game",
                "group",
                "path",
                "distance_to_group_centroid",
            ],
        )
        writer.writeheader()
        writer.writerows(distance_rows)

    with (output_dir / "distance_to_centroid_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_size",
                "aaa_mean",
                "indie_mean",
                "aaa_std",
                "indie_std",
                "indie_vs_aaa_mean_ratio",
                "mean_gap_indie_minus_aaa",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "game_groups_file": str(groups_file),
        "output_dir": str(output_dir),
        "clip_backend": adapter.backend_name,
        "model_name": args.model_name,
        "device": device.type,
        "sample_sizes": sample_sizes,
        "selected_records_before_decode": len(selected_records),
        "valid_records_after_decode": len(valid_records),
        "skipped_images": len(skipped_images),
        "params": {
            "umap_n_neighbors": int(args.umap_n_neighbors),
            "umap_min_dist": float(args.umap_min_dist),
            "densmap_lambda": float(args.densmap_lambda),
            "densmap_frac": float(args.densmap_frac),
            "density_grid_size": int(args.density_grid_size),
            "density_levels": int(args.density_levels),
            "density_kde_bandwidth": float(args.density_kde_bandwidth),
            "seed": int(args.seed),
        },
        "summary_rows": summary_rows,
        "outputs": {
            "umap_contour_html": str(umap_html),
            "densmap_contour_html": str(dens_html),
            "distance_hist_html": str(hist_html),
            "distance_rows_csv": str(output_dir / "distance_to_centroid_rows.csv"),
            "distance_summary_csv": str(output_dir / "distance_to_centroid_summary.csv"),
        },
    }
    (output_dir / "phase2_overlap_report.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved: {umap_html}", flush=True)
    print(f"Saved: {dens_html}", flush=True)
    print(f"Saved: {hist_html}", flush=True)
    print(f"Saved report: {output_dir / 'phase2_overlap_report.json'}", flush=True)
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except PipelineError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
