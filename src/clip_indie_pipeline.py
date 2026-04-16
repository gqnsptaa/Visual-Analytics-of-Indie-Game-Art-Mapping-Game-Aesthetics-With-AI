#!/usr/bin/env python3
"""CLIP-based indie screenshot analysis pipeline.

This script mirrors the notebook workflow and adds production-friendly behavior:
- Batched CLIP embedding extraction (GPU-aware)
- 3D t-SNE and 3D UMAP projections
- Game centroid similarity matrix
- KMeans clustering diagnostics
- Prompt-based style similarity scoring
- JSON/CSV exports for web visualizations

Default dataset source:
<repo_root>/indie_games_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

try:
    import numpy as np
    import torch
    from PIL import Image
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.datasets import make_swiss_roll
    from sklearn.decomposition import PCA
    from sklearn.manifold import LocallyLinearEmbedding, TSNE, trustworthiness
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import LocalOutlierFactor
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    missing_name = exc.name or "unknown"
    raise SystemExit(
        f"[ERROR] Missing dependency '{missing_name}'. Install dependencies with: pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "indie_games_dataset"

DEFAULT_STYLE_PROMPTS = [
    "hand-painted illustration style indie game",
    "mixed media game visuals experimental",
    "cel-shaded rendering stylized",
    "low poly aesthetic minimalist",
    "high fidelity 3D graphics AAA game",
    "photorealistic rendering cinematic",
    "abstract symbolic game art",
    "stylized rendering non-realistic",
    "chiaroscuro lighting dramatic shadows",
    "low key lighting cinematic",
    "soft diffuse lighting environment",
    "neon lighting cyberpunk style",
    "global illumination realistic lighting",
    "flat lighting minimal shadows",
    "isometric game view",
    "top-down perspective game scene",
    "side-scrolling composition",
    "symmetrical composition game scene",
    "dynamic action scene with motion blur",
    "negative space emphasis minimal composition",
    "minimalist composition clean layout",
    "dense cluttered environment high detail",
    "muted color palette",
    "vibrant saturated colors",
    "monochrome palette",
    "pastel color scheme",
    "hand-painted textures visible brush strokes",
    "grainy textured surfaces rough detail",
    "bold typography game UI",
    "minimal UI text design",
    "ornate fantasy typography",
    "experimental typography layout",
    "diegetic UI integrated into environment",
    "HUD overlay interface",
    "symbolic environmental storytelling",
    "metaphorical scene design",
    "abstract narrative visual elements",
]


@dataclass(frozen=True)
class ImageRecord:
    """Metadata for a single image sample."""

    image_id: int
    label: str
    path: Path


class PipelineError(RuntimeError):
    """Controlled pipeline exception with user-facing messages."""


class ClipAdapter:
    """Unified interface for OpenAI CLIP and OpenCLIP backends."""

    def __init__(
        self,
        model,
        preprocess: Callable,
        tokenize_fn: Callable[[list[str]], torch.Tensor],
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

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        text_tokens = self.tokenize_fn(prompts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class BottleneckAdapter(torch.nn.Module):
    """Residual bottleneck adapter used by style fine-tuning."""

    def __init__(self, dim: int, rank: int, dropout: float, scale: float) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.down = torch.nn.Linear(dim, rank)
        self.up = torch.nn.Linear(rank, dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = float(scale)
        self.act = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.up(h)
        return x + self.scale * h


class StyleAdapterHead(torch.nn.Module):
    """Adapter + classifier head for style logits."""

    def __init__(self, embedding_dim: int, num_classes: int, rank: int, dropout: float, scale: float) -> None:
        super().__init__()
        self.adapter = BottleneckAdapter(embedding_dim, rank=rank, dropout=dropout, scale=scale)
        self.norm = torch.nn.LayerNorm(embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        x = self.norm(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CLIP indie aesthetics analysis using a dataset folder "
            "(default set to the repository dataset root)."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Dataset root folder containing one subfolder per game (default: {DATASET_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("web/data"),
        help="Directory for JSON/CSV output files.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.webp",
        help="Comma-separated list of valid image extensions.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP encoding batch size.")
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=5,
        help="Print CLIP encoding progress every N batches (min 1).",
    )
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
        default="auto",
        choices=["auto", "openai", "open_clip"],
        help="CLIP backend to use.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-B/32",
        help="CLIP model name. OpenAI format for openai backend, OpenCLIP format for open_clip backend.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional max number of images to process (0 = no limit).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--tsne-perplexity", type=float, default=10.0, help="Target t-SNE perplexity.")
    parser.add_argument("--umap-n-neighbors", type=int, default=10, help="Target UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument(
        "--kmeans-k",
        type=int,
        default=0,
        help="KMeans cluster count (0 = number of unique labels).",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.14,
        help="DBSCAN epsilon radius over CLIP embeddings.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=4,
        help="DBSCAN min_samples value.",
    )
    parser.add_argument(
        "--dbscan-metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric used by DBSCAN.",
    )
    parser.add_argument(
        "--swiss-roll-samples",
        type=int,
        default=250,
        help="Number of synthetic swiss-roll points for projection sanity check (0 disables).",
    )
    parser.add_argument(
        "--swiss-roll-noise",
        type=float,
        default=0.03,
        help="Noise level for synthetic swiss-roll points.",
    )
    parser.add_argument(
        "--swiss-roll-lle-neighbors",
        type=int,
        default=12,
        help="Neighbor count for LLE swiss-roll reduction.",
    )
    parser.add_argument(
        "--style-prompts-file",
        type=Path,
        default=None,
        help="Optional text file with one style prompt per line.",
    )
    parser.add_argument(
        "--prompt-focus-file",
        type=Path,
        default=Path("src/style_prompts_graphic_design_focus.txt"),
        help=(
            "Optional prompt subset file for cleaner UI heatmaps. "
            "CLIP still scores full prompt list; this only filters displayed/exported prompt matrices."
        ),
    )
    parser.add_argument(
        "--game-groups-file",
        type=Path,
        default=Path("src/game_groups.csv"),
        help="Optional CSV mapping with columns: game,group (for AAA vs indie comparison).",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save raw CLIP embeddings as .npy for reproducibility.",
    )
    parser.add_argument(
        "--style-adapter-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint from train_openclip_style_adapter.py. If provided, prompt heatmap uses adapter scores.",
    )
    parser.add_argument(
        "--export-thumbnails",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export small thumbnails to output_dir/thumbs for image-based 2D visualizations.",
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=48,
        help="Thumbnail square size in pixels when --export-thumbnails is enabled.",
    )
    parser.add_argument(
        "--thumbnail-progress-every",
        type=int,
        default=50,
        help="Print thumbnail export progress every N images (min 1).",
    )
    parser.add_argument(
        "--thumbnail-jpeg-quality",
        type=int,
        default=82,
        help="JPEG quality for exported thumbnails (1-95). Lower is faster/smaller.",
    )
    parser.add_argument(
        "--thumbnail-optimize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable JPEG optimize pass for thumbnails (slower).",
    )
    parser.add_argument(
        "--debug-vector-preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print one image embedding vector and one prompt embedding vector preview.",
    )
    parser.add_argument(
        "--debug-vector-preview-dims",
        type=int,
        default=12,
        help="How many leading dimensions to print for --debug-vector-preview (1-128).",
    )
    return parser.parse_args()


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        except Exception as exc:  # pragma: no cover - backend dependent
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
        except Exception as exc:  # pragma: no cover - backend dependent
            errors.append(f"open_clip backend failed: {exc}")

    raise PipelineError(
        "Unable to load CLIP backend. Install one of: \n"
        "1) OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git\n"
        "2) OpenCLIP: pip install open-clip-torch\n"
        f"Backend errors: {' | '.join(errors)}"
    )


def parse_prompts_file_lines(path: Path) -> list[str]:
    """Read prompt files with optional section headers/comments."""
    prompts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Support human-friendly grouped prompt files.
        if line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        prompts.append(line)
    return prompts


def load_style_prompts(path: Path | None) -> list[str]:
    if path is None:
        source_prompts = list(DEFAULT_STYLE_PROMPTS)
    else:
        if not path.exists():
            raise PipelineError(f"Style prompts file not found: {path}")
        source_prompts = parse_prompts_file_lines(path)
        if not source_prompts:
            raise PipelineError(f"Style prompts file is empty: {path}")

    # Remove exact duplicates while preserving order to avoid repeated prompt weighting.
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in source_prompts:
        key = prompt.strip().lower()
        if not key or key in seen:
            continue
        prompts.append(prompt)
        seen.add(key)

    if not prompts:
        raise PipelineError("No valid style prompts after cleanup/deduplication.")
    return prompts


def load_prompt_focus(path: Path | None) -> list[str]:
    """Load optional prompt subset for cleaner UI heatmaps."""
    if path is None:
        return []
    if not path.exists():
        return []

    raw = parse_prompts_file_lines(path)
    focus: list[str] = []
    seen: set[str] = set()
    for prompt in raw:
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        focus.append(prompt)
    return focus


def resolve_prompt_focus_indices(all_prompts: list[str], focus_prompts: list[str]) -> tuple[list[int], dict]:
    """Resolve focus prompts against available prompts (case-insensitive)."""
    total = len(all_prompts)
    if total == 0:
        return [], {"enabled": False, "reason": "no_prompts", "total_prompts": 0}

    if not focus_prompts:
        return list(range(total)), {
            "enabled": False,
            "reason": "focus_file_empty_or_missing",
            "total_prompts": total,
            "focus_requested": 0,
            "selected_prompts": total,
            "missing_prompts": [],
        }

    index_by_key: dict[str, int] = {}
    for idx, prompt in enumerate(all_prompts):
        key = prompt.strip().lower()
        if key and key not in index_by_key:
            index_by_key[key] = idx

    selected_indices: list[int] = []
    missing: list[str] = []
    for prompt in focus_prompts:
        idx = index_by_key.get(prompt.strip().lower())
        if idx is None:
            missing.append(prompt)
            continue
        selected_indices.append(idx)

    if not selected_indices:
        return list(range(total)), {
            "enabled": False,
            "reason": "no_focus_matches_in_prompt_set",
            "total_prompts": total,
            "focus_requested": len(focus_prompts),
            "selected_prompts": total,
            "missing_prompts": missing,
        }

    return selected_indices, {
        "enabled": True,
        "reason": "focus_applied",
        "total_prompts": total,
        "focus_requested": len(focus_prompts),
        "selected_prompts": len(selected_indices),
        "missing_prompts": missing,
    }


def select_prompt_columns(
    all_prompts: list[str],
    matrix: np.ndarray,
    selected_indices: list[int],
) -> tuple[list[str], np.ndarray]:
    if not selected_indices:
        return [], np.zeros((matrix.shape[0], 0), dtype=np.float32)
    selected_prompts = [all_prompts[idx] for idx in selected_indices]
    selected_matrix = matrix[:, selected_indices].astype(np.float32, copy=False)
    return selected_prompts, selected_matrix


def normalize_group_name(raw_value: str) -> str:
    value = raw_value.strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    return value or "unassigned"


def load_game_groups(path: Path | None) -> dict[str, str]:
    """Load game->group mapping from CSV; returns empty mapping if file is missing."""
    if path is None:
        return {}
    if not path.exists():
        return {}

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
                if not game:
                    continue
                mapping[game] = group
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"Failed to read game groups file {path}: {exc}") from exc

    return mapping


def collect_image_records(root_dir: Path, valid_extensions: tuple[str, ...], max_images: int) -> list[ImageRecord]:
    if not root_dir.exists():
        raise PipelineError(
            f"Dataset folder does not exist: {root_dir}\n"
            "Create game folders there and put screenshots inside each game folder."
        )
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

            if max_images > 0 and len(records) >= max_images:
                return records

    return records


def encode_images(adapter: ClipAdapter, records: list[ImageRecord], batch_size: int) -> tuple[np.ndarray, list[ImageRecord], list[dict]]:
    if batch_size <= 0:
        raise PipelineError("batch-size must be > 0.")

    valid_records: list[ImageRecord] = []
    skipped: list[dict] = []
    batches: list[np.ndarray] = []

    image_buffer: list[torch.Tensor] = []
    rec_buffer: list[ImageRecord] = []

    use_amp = adapter.device.type == "cuda"
    scanned_count = 0
    batch_count = 0
    total_records = len(records)
    progress_every = max(1, int(getattr(adapter, "progress_every_batches", 5)))

    def flush_batch() -> None:
        nonlocal batch_count
        if not image_buffer:
            return

        batch_count += 1
        batch_tensor = torch.stack(image_buffer).to(adapter.device, non_blocking=True)
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = adapter.encode_images(batch_tensor)
            else:
                emb = adapter.encode_images(batch_tensor)

        batches.append(emb.detach().cpu().numpy().astype(np.float32))
        valid_records.extend(rec_buffer)
        if batch_count == 1 or batch_count % progress_every == 0:
            print(
                "[CLIP] Encoded batch "
                f"{batch_count} | batch_size={len(rec_buffer)} | "
                f"scanned={scanned_count}/{total_records} | "
                f"valid={len(valid_records)} | skipped={len(skipped)}",
                flush=True,
            )
        image_buffer.clear()
        rec_buffer.clear()

    for rec in records:
        scanned_count += 1
        try:
            with Image.open(rec.path) as img:
                tensor = adapter.preprocess(img.convert("RGB"))
        except Exception as exc:
            skipped.append(
                {
                    "image_id": rec.image_id,
                    "label": rec.label,
                    "path": str(rec.path),
                    "reason": str(exc),
                }
            )
            continue

        image_buffer.append(tensor)
        rec_buffer.append(rec)
        if len(image_buffer) >= batch_size:
            flush_batch()

    flush_batch()

    if not batches:
        raise PipelineError("No valid images were encoded. Check image files and supported extensions.")

    print(
        "[CLIP] Encoding complete | "
        f"batches={batch_count} | valid={len(valid_records)} | skipped={len(skipped)}",
        flush=True,
    )
    return np.vstack(batches), valid_records, skipped


def normalized_centroids(embeddings: np.ndarray, labels: list[str]) -> tuple[list[str], np.ndarray]:
    unique_labels = sorted(set(labels))
    centroids = []

    for label in unique_labels:
        idx = [i for i, v in enumerate(labels) if v == label]
        centroid = embeddings[idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids.append((centroid / norm) if norm > 0 else centroid)

    return unique_labels, np.vstack(centroids).astype(np.float32)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    # Matrix is expected to be row-normalized; dot product equals cosine similarity.
    return np.clip(matrix @ matrix.T, -1.0, 1.0).astype(np.float32)


def pca_analysis_3d(embeddings: np.ndarray, seed: int = 42) -> tuple[np.ndarray, dict]:
    n_samples, n_features = embeddings.shape
    n_comp = min(3, n_samples, n_features)
    if n_comp <= 0:
        raise PipelineError("Cannot run PCA fallback on empty embeddings.")

    reducer = PCA(n_components=n_comp, random_state=seed)
    reduced = reducer.fit_transform(embeddings).astype(np.float32)
    explained = [float(v) for v in reducer.explained_variance_ratio_.tolist()]
    explained_total = float(np.sum(reducer.explained_variance_ratio_))

    if n_comp < 3:
        pad = np.zeros((n_samples, 3 - n_comp), dtype=np.float32)
        reduced = np.concatenate([reduced, pad], axis=1)
        explained.extend([0.0] * (3 - n_comp))

    return reduced, {
        "method": "pca",
        "n_components_fitted": int(n_comp),
        "explained_variance_ratio": explained[:3],
        "explained_variance_total": explained_total,
    }


def pca_to_3d(embeddings: np.ndarray) -> np.ndarray:
    reduced, _ = pca_analysis_3d(embeddings, seed=42)
    return reduced


def adaptive_tsne_3d(embeddings: np.ndarray, perplexity: float, seed: int) -> tuple[np.ndarray, dict]:
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise PipelineError("Need at least 2 samples for t-SNE.")

    max_perplexity = max(1.0, float(n_samples - 1))
    safe_perplexity = min(max(1.0, perplexity), max_perplexity)

    # Fallback when t-SNE is unstable on tiny datasets.
    if n_samples < 5:
        return pca_to_3d(embeddings), {
            "method": "pca_fallback",
            "reason": "Too few samples for stable t-SNE",
            "perplexity": safe_perplexity,
        }

    tsne = TSNE(
        n_components=3,
        perplexity=safe_perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    try:
        return tsne.fit_transform(embeddings).astype(np.float32), {
            "method": "tsne",
            "perplexity": safe_perplexity,
        }
    except Exception as exc:
        return pca_to_3d(embeddings), {
            "method": "pca_fallback",
            "reason": f"t-SNE failed: {exc}",
            "perplexity": safe_perplexity,
        }


def adaptive_umap_3d(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise PipelineError("Need at least 2 samples for UMAP.")

    safe_neighbors = min(max(2, n_neighbors), max(2, n_samples - 1))

    # Fallback when UMAP is unstable on tiny datasets.
    if n_samples < 5:
        return pca_to_3d(embeddings), {
            "method": "pca_fallback",
            "reason": "Too few samples for stable UMAP",
            "n_neighbors": safe_neighbors,
            "min_dist": min_dist,
        }

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=safe_neighbors,
            min_dist=min_dist,
            n_components=3,
            random_state=seed,
            metric="cosine",
        )
        return reducer.fit_transform(embeddings).astype(np.float32), {
            "method": "umap",
            "n_neighbors": safe_neighbors,
            "min_dist": min_dist,
        }
    except Exception as exc:
        return pca_to_3d(embeddings), {
            "method": "pca_fallback",
            "reason": f"UMAP failed: {exc}",
            "n_neighbors": safe_neighbors,
            "min_dist": min_dist,
        }


def safe_kmeans(embeddings: np.ndarray, labels: list[str], desired_k: int, seed: int) -> tuple[np.ndarray, dict]:
    n_samples = embeddings.shape[0]
    unique_label_count = len(set(labels))
    k = desired_k if desired_k > 0 else unique_label_count
    k = max(1, min(k, n_samples))

    if k == 1:
        cluster_ids = np.zeros(n_samples, dtype=np.int32)
        return cluster_ids, {"k": 1, "note": "Only one cluster possible for current sample count."}

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings).astype(np.int32)
    return cluster_ids, {"k": k}


def safe_dbscan(
    embeddings: np.ndarray,
    eps: float,
    min_samples: int,
    metric: str,
) -> tuple[np.ndarray, dict]:
    n_samples = int(embeddings.shape[0])
    if n_samples <= 0:
        raise PipelineError("Cannot run DBSCAN on empty embeddings.")
    if eps <= 0:
        raise PipelineError("dbscan-eps must be > 0.")
    if min_samples < 1:
        raise PipelineError("dbscan-min-samples must be >= 1.")

    try:
        model = DBSCAN(
            eps=float(eps),
            min_samples=int(min_samples),
            metric=metric,
        )
        cluster_ids = model.fit_predict(embeddings).astype(np.int32)
    except Exception as exc:
        cluster_ids = np.full((n_samples,), -1, dtype=np.int32)
        return cluster_ids, {
            "method": "dbscan",
            "metric": metric,
            "eps": float(eps),
            "min_samples": int(min_samples),
            "status": "failed",
            "reason": str(exc),
            "num_clusters": 0,
            "noise_count": int(n_samples),
            "noise_fraction": 1.0,
        }

    unique = set(int(v) for v in cluster_ids.tolist())
    n_noise = int(np.sum(cluster_ids == -1))
    n_clusters = len([v for v in unique if v != -1])
    return cluster_ids, {
        "method": "dbscan",
        "metric": metric,
        "eps": float(eps),
        "min_samples": int(min_samples),
        "status": "ok",
        "num_clusters": int(n_clusters),
        "noise_count": int(n_noise),
        "noise_fraction": float(n_noise / max(1, n_samples)),
    }


def build_swiss_roll_demo(
    n_samples: int,
    noise: float,
    seed: int,
    lle_neighbors: int,
) -> dict:
    if n_samples <= 0:
        return {
            "enabled": False,
            "reason": "disabled",
            "samples": [],
        }
    if n_samples < 20:
        raise PipelineError("swiss-roll-samples must be at least 20 when enabled.")
    if noise < 0:
        raise PipelineError("swiss-roll-noise must be >= 0.")
    if lle_neighbors < 2:
        raise PipelineError("swiss-roll-lle-neighbors must be >= 2.")

    raw_points, t_values = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    raw_points = raw_points.astype(np.float32)
    t_values = t_values.astype(np.float32)

    # Match the benchmark workflow: compare nonlinear LLE vs linear PCA in 2D.
    pca = PCA(n_components=2, random_state=seed)
    pca_2d = pca.fit_transform(raw_points).astype(np.float32)
    pca_error = float(max(0.0, 1.0 - float(np.sum(pca.explained_variance_ratio_))))

    safe_neighbors = max(2, min(int(lle_neighbors), int(n_samples - 1)))
    lle_error = None
    lle_status = "ok"
    lle_reason = ""
    try:
        lle = LocallyLinearEmbedding(
            n_neighbors=safe_neighbors,
            n_components=2,
            method="standard",
            eigen_solver="auto",
            random_state=seed,
        )
        lle_2d = lle.fit_transform(raw_points).astype(np.float32)
        lle_error = float(getattr(lle, "reconstruction_error_", np.nan))
    except Exception as exc:
        # Keep demo available even if LLE numerics fail for a parameter setting.
        lle_2d = pca_2d.copy()
        lle_status = "failed_fallback_to_pca"
        lle_reason = str(exc)
        lle_error = None

    t_min = float(np.min(t_values))
    t_max = float(np.max(t_values))
    denom = max(1e-12, t_max - t_min)
    t_norm = ((t_values - t_min) / denom).astype(np.float32)

    samples = []
    for idx in range(int(n_samples)):
        samples.append(
            {
                "id": int(idx),
                "color_value": float(t_norm[idx]),
                "original": [float(v) for v in raw_points[idx].tolist()],
                "lle_2d": [float(v) for v in lle_2d[idx].tolist()],
                "pca_2d": [float(v) for v in pca_2d[idx].tolist()],
            }
        )

    return {
        "enabled": True,
        "meta": {
            "n_samples": int(n_samples),
            "noise": float(noise),
            "seed": int(seed),
            "source": "sklearn.datasets.make_swiss_roll",
            "lle": {
                "status": lle_status,
                "reason": lle_reason,
                "n_neighbors": int(safe_neighbors),
                "n_components": 2,
                "reconstruction_error": lle_error,
            },
            "pca": {
                "n_components": 2,
                "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_.tolist()],
                "error": pca_error,
            },
        },
        "samples": samples,
    }


def crosstab_counts(labels: list[str], clusters: np.ndarray) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for label, cluster in zip(labels, clusters):
        table[label][str(int(cluster))] += 1
    return {label: dict(counts) for label, counts in sorted(table.items())}


def knn_indices(matrix: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Return nearest-neighbor indices (excluding self) for each point."""
    if matrix.shape[0] < 2:
        return np.zeros((matrix.shape[0], 0), dtype=np.int32)
    safe_k = max(1, min(int(k), int(matrix.shape[0] - 1)))
    distances = pairwise_distances(matrix, metric=metric)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1)[:, :safe_k].astype(np.int32)


def neighborhood_overlap(high_dim: np.ndarray, low_dim: np.ndarray, k: int, high_metric: str = "cosine") -> float:
    """Average k-NN overlap between high-dimensional and projected spaces."""
    if high_dim.shape[0] < 2:
        return 0.0
    high_nn = knn_indices(high_dim, k=k, metric=high_metric)
    low_nn = knn_indices(low_dim, k=k, metric="euclidean")
    if high_nn.shape[1] == 0 or low_nn.shape[1] == 0:
        return 0.0

    safe_k = int(min(high_nn.shape[1], low_nn.shape[1]))
    overlaps = []
    for i in range(high_nn.shape[0]):
        hs = set(int(v) for v in high_nn[i, :safe_k].tolist())
        ls = set(int(v) for v in low_nn[i, :safe_k].tolist())
        overlaps.append(len(hs & ls) / max(1, safe_k))
    return float(np.mean(overlaps))


def projection_quality_report(
    embeddings: np.ndarray,
    projections: dict[str, np.ndarray],
    n_neighbors: int,
) -> dict:
    """Compute embedding quality metrics to compare DR methods."""
    n_samples = int(embeddings.shape[0])
    if n_samples < 3:
        return {
            "enabled": False,
            "reason": "too_few_samples",
            "n_neighbors": int(max(1, n_neighbors)),
            "methods": {},
        }

    safe_k = max(2, min(int(n_neighbors), n_samples - 1))
    methods: dict[str, dict] = {}
    for name, coords in projections.items():
        if not isinstance(coords, np.ndarray) or coords.ndim != 2 or coords.shape[0] != n_samples:
            methods[name] = {
                "status": "invalid_projection",
            }
            continue
        try:
            trust = float(
                trustworthiness(
                    embeddings,
                    coords,
                    n_neighbors=safe_k,
                    metric="cosine",
                )
            )
        except Exception as exc:
            methods[name] = {
                "status": "failed",
                "reason": str(exc),
            }
            continue

        # Continuity proxy: reverse trustworthiness direction (approximation).
        try:
            continuity_proxy = float(
                trustworthiness(
                    coords,
                    embeddings,
                    n_neighbors=safe_k,
                    metric="euclidean",
                )
            )
        except Exception:
            continuity_proxy = float("nan")

        overlap = neighborhood_overlap(
            high_dim=embeddings,
            low_dim=coords,
            k=safe_k,
            high_metric="cosine",
        )

        methods[name] = {
            "status": "ok",
            "trustworthiness": trust,
            "continuity_proxy": continuity_proxy if np.isfinite(continuity_proxy) else None,
            "knn_overlap": float(overlap),
            "n_neighbors": int(safe_k),
        }

    return {
        "enabled": True,
        "n_neighbors": int(safe_k),
        "methods": methods,
    }


def detect_outliers(
    embeddings: np.ndarray,
    valid_records: list[ImageRecord],
    labels: list[str],
    groups: list[str],
    dbscan_ids: np.ndarray,
    max_rows: int = 80,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Detect outliers using LOF score and DBSCAN noise labels."""
    n_samples = int(embeddings.shape[0])
    if n_samples == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.bool_),
            [],
            {"enabled": False, "reason": "no_samples"},
        )
    if n_samples < 5:
        scores = np.zeros((n_samples,), dtype=np.float32)
        flags = (dbscan_ids == -1) if dbscan_ids.size == n_samples else np.zeros((n_samples,), dtype=np.bool_)
        return (
            scores,
            flags.astype(np.bool_),
            [],
            {
                "enabled": False,
                "reason": "too_few_samples_for_lof",
                "num_flagged": int(np.sum(flags)),
            },
        )

    lof_neighbors = max(5, min(20, n_samples - 1))
    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, metric="cosine")
    lof.fit_predict(embeddings)
    scores = (-lof.negative_outlier_factor_).astype(np.float32)

    dbscan_noise = (dbscan_ids == -1) if dbscan_ids.size == n_samples else np.zeros((n_samples,), dtype=np.bool_)
    score_threshold = float(np.quantile(scores, 0.95))
    score_flag = scores >= score_threshold
    flags = np.logical_or(dbscan_noise, score_flag)

    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n_samples)
    percentiles = ranks.astype(np.float32) / max(1, n_samples - 1)

    top_n = max(1, min(int(max_rows), n_samples))
    top_idx = np.argsort(scores)[::-1][:top_n]
    outlier_rows: list[dict] = []
    for rank_pos, idx in enumerate(top_idx, start=1):
        rec = valid_records[idx]
        outlier_rows.append(
            {
                "rank": int(rank_pos),
                "image_id": int(rec.image_id),
                "label": labels[idx],
                "group": groups[idx],
                "path": str(rec.path),
                "outlier_score": float(scores[idx]),
                "outlier_percentile": float(percentiles[idx]),
                "dbscan_noise": bool(dbscan_noise[idx]),
                "flagged": bool(flags[idx]),
            }
        )

    meta = {
        "enabled": True,
        "method": "local_outlier_factor_plus_dbscan_noise",
        "lof_neighbors": int(lof_neighbors),
        "score_threshold_q95": float(score_threshold),
        "num_flagged": int(np.sum(flags)),
        "num_dbscan_noise": int(np.sum(dbscan_noise)),
    }
    return scores, flags.astype(np.bool_), outlier_rows, meta


def average_scores_by_label(sample_scores: np.ndarray, labels: list[str]) -> tuple[list[str], np.ndarray]:
    label_values = sorted(set(labels))
    if not label_values:
        raise PipelineError("Cannot aggregate scores by label: label list is empty.")

    output = np.zeros((len(label_values), sample_scores.shape[1]), dtype=np.float32)
    for i, label in enumerate(label_values):
        idx = [j for j, value in enumerate(labels) if value == label]
        output[i] = sample_scores[idx].mean(axis=0)
    return label_values, output


def prompt_similarity_by_game(
    embeddings: np.ndarray,
    labels: list[str],
    text_features: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    sample_prompt_similarity = embeddings @ text_features.T
    games, game_prompt = average_scores_by_label(sample_prompt_similarity, labels)
    return games, game_prompt, sample_prompt_similarity


def run_style_adapter_inference(
    embeddings: np.ndarray,
    labels: list[str],
    checkpoint_path: Path,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[dict], dict]:
    """Run style adapter checkpoint on CLIP embeddings."""
    if not checkpoint_path.exists():
        raise PipelineError(f"Style adapter checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise PipelineError(f"Failed to load style adapter checkpoint: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise PipelineError("Invalid style adapter checkpoint format: expected a dict payload.")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise PipelineError("Style adapter checkpoint missing 'state_dict'.")

    embedding_dim = int(embeddings.shape[1])
    ckpt_embedding_dim = int(checkpoint.get("embedding_dim", embedding_dim))
    if ckpt_embedding_dim != embedding_dim:
        raise PipelineError(
            "Style adapter embedding_dim mismatch: "
            f"checkpoint={ckpt_embedding_dim}, current={embedding_dim}. "
            "Use checkpoint trained on the same CLIP model."
        )

    num_classes = int(checkpoint.get("num_classes", 0))
    if num_classes <= 1:
        raise PipelineError("Style adapter checkpoint has invalid num_classes.")

    rank = int(checkpoint.get("adapter_rank", 64))
    dropout = float(checkpoint.get("adapter_dropout", 0.1))
    scale = float(checkpoint.get("adapter_scale", 1.0))

    head = StyleAdapterHead(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        rank=rank,
        dropout=dropout,
        scale=scale,
    )
    try:
        head.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise PipelineError(f"Style adapter state_dict is incompatible: {exc}") from exc

    id_to_label_raw = checkpoint.get("id_to_label")
    if not isinstance(id_to_label_raw, dict):
        raise PipelineError("Style adapter checkpoint missing 'id_to_label' mapping.")

    class_labels: list[str] = []
    for class_id in range(num_classes):
        label = None
        if class_id in id_to_label_raw:
            label = id_to_label_raw[class_id]
        elif str(class_id) in id_to_label_raw:
            label = id_to_label_raw[str(class_id)]
        if not isinstance(label, str) or not label.strip():
            raise PipelineError(f"Style adapter checkpoint has invalid class label for id={class_id}.")
        class_labels.append(label)

    features = torch.from_numpy(embeddings).to(torch.float32)
    with torch.inference_mode():
        logits = head(features)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

    pred_ids = np.argmax(probs, axis=1)
    pred_conf = probs[np.arange(probs.shape[0]), pred_ids]
    sample_predictions = [
        {
            "style_pred_label": class_labels[int(pred_ids[i])],
            "style_pred_confidence": float(pred_conf[i]),
        }
        for i in range(probs.shape[0])
    ]

    games, game_scores = average_scores_by_label(probs, labels)

    meta = {
        "checkpoint": str(checkpoint_path),
        "num_classes": num_classes,
        "model_name": checkpoint.get("model_name"),
        "pretrained": checkpoint.get("pretrained"),
        "best_metrics": checkpoint.get("best_metrics"),
    }
    return games, game_scores, class_labels, probs, sample_predictions, meta


def export_sample_thumbnails(
    records: list[ImageRecord],
    output_dir: Path,
    thumbnail_size: int,
    progress_every: int,
    jpeg_quality: int,
    optimize_jpeg: bool,
) -> dict[int, str]:
    """Export thumbnails and return mapping image_id -> relative web path."""
    if thumbnail_size < 16 or thumbnail_size > 512:
        raise PipelineError("thumbnail-size must be between 16 and 512.")
    if jpeg_quality < 1 or jpeg_quality > 95:
        raise PipelineError("thumbnail-jpeg-quality must be between 1 and 95.")

    thumbs_dir = output_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[int, str] = {}
    progress_every = max(1, int(progress_every))
    total = len(records)
    exported = 0
    failed = 0

    # Keep center composition consistent by padding each image into a square canvas.
    for idx, rec in enumerate(records, start=1):
        out_path = thumbs_dir / f"{rec.image_id}.jpg"
        try:
            with Image.open(rec.path) as img:
                image = img.convert("RGB")
                image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)

                canvas = Image.new("RGB", (thumbnail_size, thumbnail_size), color=(243, 244, 246))
                offset = (
                    (thumbnail_size - image.width) // 2,
                    (thumbnail_size - image.height) // 2,
                )
                canvas.paste(image, offset)
                canvas.save(out_path, format="JPEG", quality=jpeg_quality, optimize=optimize_jpeg)
            mapping[rec.image_id] = f"data/thumbs/{rec.image_id}.jpg"
            exported += 1
        except Exception:
            # Thumbnail export is optional; skip failures without aborting analysis.
            failed += 1
            continue
        if idx == 1 or idx % progress_every == 0 or idx == total:
            print(
                "[Thumbs] Export progress "
                f"{idx}/{total} | exported={exported} | failed={failed}",
                flush=True,
            )

    return mapping


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text atomically to avoid readers seeing partial JSON."""
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(content, encoding=encoding)
    tmp_path.replace(path)


def format_vector_preview(vector: np.ndarray, dims: int) -> str:
    safe_dims = max(1, min(int(dims), int(vector.shape[0])))
    values = ", ".join(f"{float(v):+.5f}" for v in vector[:safe_dims])
    return f"[{values}]"


def log_vector_preview(
    valid_records: list[ImageRecord],
    embeddings: np.ndarray,
    prompts: list[str],
    text_features: np.ndarray,
    preview_dims: int,
) -> None:
    """Print one sample image/prompt embedding and similarity diagnostics."""
    if embeddings.shape[0] == 0 or text_features.shape[0] == 0:
        print("[Debug] Vector preview skipped: embeddings/prompts are empty.", flush=True)
        return

    sample_idx = 0
    prompt_idx = 0
    sample = valid_records[sample_idx]
    image_vec = embeddings[sample_idx]
    prompt_vec = text_features[prompt_idx]
    sims = image_vec @ text_features.T
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    print("[Debug] --- CLIP Vector Preview ---", flush=True)
    print(
        f"[Debug] Image sample: game='{sample.label}', file='{sample.path.name}', embedding_dim={image_vec.shape[0]}",
        flush=True,
    )
    print(
        f"[Debug] Image vector first {max(1, min(preview_dims, image_vec.shape[0]))} dims: "
        f"{format_vector_preview(image_vec, preview_dims)}",
        flush=True,
    )
    print(
        f"[Debug] Prompt sample: '{prompts[prompt_idx]}', embedding_dim={prompt_vec.shape[0]}",
        flush=True,
    )
    print(
        f"[Debug] Prompt vector first {max(1, min(preview_dims, prompt_vec.shape[0]))} dims: "
        f"{format_vector_preview(prompt_vec, preview_dims)}",
        flush=True,
    )
    print(
        f"[Debug] Similarity(image, prompt[0])={float(sims[prompt_idx]):+.5f} | "
        f"best_prompt='{prompts[best_idx]}' score={best_score:+.5f}",
        flush=True,
    )
    print("[Debug] ---------------------------", flush=True)


def run() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    set_seed(args.seed)
    if args.debug_vector_preview_dims < 1 or args.debug_vector_preview_dims > 128:
        raise PipelineError("debug-vector-preview-dims must be between 1 and 128.")
    if args.dbscan_eps <= 0:
        raise PipelineError("dbscan-eps must be > 0.")
    if args.dbscan_min_samples < 1:
        raise PipelineError("dbscan-min-samples must be >= 1.")
    if args.swiss_roll_samples < 0:
        raise PipelineError("swiss-roll-samples must be >= 0.")
    if args.swiss_roll_noise < 0:
        raise PipelineError("swiss-roll-noise must be >= 0.")

    def log(message: str) -> None:
        print(message, flush=True)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_ext = tuple(
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in args.extensions.split(",")
        if ext.strip()
    )
    if not valid_ext:
        raise PipelineError("No valid image extensions provided.")

    log("[Init] Selecting device and scanning dataset...")
    device = choose_device(args.device)
    records = collect_image_records(dataset_root, valid_ext, args.max_images)
    if not records:
        raise PipelineError(
            "No images found in dataset folder. Ensure structure is: "
            f"{dataset_root}/GameName/*.jpg"
        )

    log("[Init] Loading CLIP backend (this can take time on first run)...")
    adapter = load_clip_adapter(args.clip_backend, args.model_name, device)
    # Attach progress frequency as lightweight runtime metadata for encode_images.
    setattr(adapter, "progress_every_batches", max(1, args.progress_every_batches))
    log("[Init] Loading prompts and group mapping...")
    prompts = load_style_prompts(args.style_prompts_file)
    prompt_focus = load_prompt_focus(args.prompt_focus_file)
    game_group_map = load_game_groups(args.game_groups_file)

    log(f"Dataset source: local-only ({dataset_root})")
    log(f"Found {len(records)} image candidates across {len(set(r.label for r in records))} games.")
    log(f"Running CLIP on device={device.type} backend={adapter.backend_name} model={args.model_name}")

    log("[Step 1/7] Encoding images with CLIP...")
    embeddings, valid_records, skipped_images = encode_images(adapter, records, args.batch_size)
    labels = [r.label for r in valid_records]
    sample_groups = [game_group_map.get(label, "unassigned") for label in labels]
    group_counts: dict[str, int] = defaultdict(int)
    for group in sample_groups:
        group_counts[group] += 1

    if len(valid_records) < 2:
        raise PipelineError("Need at least 2 valid images after filtering/skips.")

    log("[Step 2/8] Computing 3D PCA, t-SNE, and UMAP coordinates...")
    pca_points, pca_meta = pca_analysis_3d(embeddings, seed=args.seed)
    tsne_points, tsne_meta = adaptive_tsne_3d(embeddings, args.tsne_perplexity, args.seed)
    umap_points, umap_meta = adaptive_umap_3d(embeddings, args.umap_n_neighbors, args.umap_min_dist, args.seed)

    log("[Step 3/7] Computing centroid similarity matrices...")
    unique_games, centroid_matrix = normalized_centroids(embeddings, labels)
    similarity_matrix = cosine_similarity_matrix(centroid_matrix)
    unique_groups, group_centroids = normalized_centroids(embeddings, sample_groups)
    group_similarity_matrix = cosine_similarity_matrix(group_centroids)

    log("[Step 4/8] Running KMeans and DBSCAN clustering...")
    cluster_ids, kmeans_meta = safe_kmeans(embeddings, labels, args.kmeans_k, args.seed)
    cluster_crosstab = crosstab_counts(labels, cluster_ids)
    dbscan_ids, dbscan_meta = safe_dbscan(
        embeddings=embeddings,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
        metric=args.dbscan_metric,
    )
    dbscan_crosstab = crosstab_counts(labels, dbscan_ids)

    log("[Step 4b/8] Evaluating projection quality and outliers...")
    quality_report = projection_quality_report(
        embeddings=embeddings,
        projections={
            "pca": pca_points,
            "umap": umap_points,
            "tsne": tsne_points,
        },
        n_neighbors=min(15, max(2, len(valid_records) - 1)),
    )
    outlier_scores, outlier_flags, outlier_rows, outlier_meta = detect_outliers(
        embeddings=embeddings,
        valid_records=valid_records,
        labels=labels,
        groups=sample_groups,
        dbscan_ids=dbscan_ids,
        max_rows=120,
    )

    log("[Step 5/8] Scoring prompts against CLIP embeddings...")
    with torch.inference_mode():
        text_features = adapter.encode_text(prompts).detach().cpu().numpy().astype(np.float32)
    if args.debug_vector_preview:
        log_vector_preview(
            valid_records=valid_records,
            embeddings=embeddings,
            prompts=prompts,
            text_features=text_features,
            preview_dims=args.debug_vector_preview_dims,
        )

    clip_prompt_games, clip_game_prompt_scores, clip_sample_prompt_scores = prompt_similarity_by_game(
        embeddings, labels, text_features
    )
    clip_prompt_groups, clip_group_prompt_scores = average_scores_by_label(clip_sample_prompt_scores, sample_groups)
    prompt_games = clip_prompt_games
    prompt_groups = clip_prompt_groups
    full_prompt_labels = list(prompts)
    full_game_prompt_scores = clip_game_prompt_scores
    full_group_prompt_scores = clip_group_prompt_scores
    prompt_source = "clip_text_prompts"
    style_adapter_meta: dict = {"enabled": False}
    style_sample_predictions = [
        {"style_pred_label": "", "style_pred_confidence": 0.0}
        for _ in range(len(valid_records))
    ]

    if args.style_adapter_checkpoint is not None:
        log("[Step 5b/8] Applying style adapter checkpoint...")
        (
            adapter_games,
            adapter_scores,
            adapter_labels,
            adapter_sample_scores,
            style_sample_predictions,
            style_meta,
        ) = run_style_adapter_inference(embeddings, labels, args.style_adapter_checkpoint)
        prompt_games = adapter_games
        full_prompt_labels = adapter_labels
        full_game_prompt_scores = adapter_scores
        prompt_groups, full_group_prompt_scores = average_scores_by_label(adapter_sample_scores, sample_groups)
        prompt_source = "style_adapter"
        style_adapter_meta = {
            "enabled": True,
            **style_meta,
        }
        log(f"Using style adapter checkpoint: {args.style_adapter_checkpoint}")

    selected_prompt_indices, prompt_focus_meta = resolve_prompt_focus_indices(full_prompt_labels, prompt_focus)
    prompt_labels, game_prompt_scores = select_prompt_columns(
        all_prompts=full_prompt_labels,
        matrix=full_game_prompt_scores,
        selected_indices=selected_prompt_indices,
    )
    _, group_prompt_scores = select_prompt_columns(
        all_prompts=full_prompt_labels,
        matrix=full_group_prompt_scores,
        selected_indices=selected_prompt_indices,
    )
    if prompt_focus_meta.get("enabled"):
        log(
            "[Step 5c/8] Applied prompt focus subset "
            f"{prompt_focus_meta.get('selected_prompts')}/{prompt_focus_meta.get('total_prompts')} prompts "
            f"from {args.prompt_focus_file}"
        )
    else:
        reason = str(prompt_focus_meta.get("reason", "unknown"))
        log(f"[Step 5c/8] Prompt focus not applied ({reason}); using all {len(full_prompt_labels)} prompts.")

    log("[Step 6/8] Generating Swiss-roll projection demo...")
    swiss_roll_demo = build_swiss_roll_demo(
        n_samples=args.swiss_roll_samples,
        noise=args.swiss_roll_noise,
        seed=args.seed,
        lle_neighbors=args.swiss_roll_lle_neighbors,
    )

    thumbnail_map: dict[int, str] = {}
    if args.export_thumbnails:
        log("[Step 7/8] Exporting thumbnails...")
        thumbnail_map = export_sample_thumbnails(
            records=valid_records,
            output_dir=output_dir,
            thumbnail_size=args.thumbnail_size,
            progress_every=args.thumbnail_progress_every,
            jpeg_quality=args.thumbnail_jpeg_quality,
            optimize_jpeg=args.thumbnail_optimize,
        )

    samples = []
    for i, rec in enumerate(valid_records):
        samples.append(
            {
                "image_id": rec.image_id,
                "label": rec.label,
                "group": sample_groups[i],
                "path": str(rec.path),
                "thumbnail": thumbnail_map.get(rec.image_id),
                "cluster_id": int(cluster_ids[i]),
                "dbscan_cluster_id": int(dbscan_ids[i]),
                "style_pred_label": style_sample_predictions[i]["style_pred_label"],
                "style_pred_confidence": style_sample_predictions[i]["style_pred_confidence"],
                "outlier_score": float(outlier_scores[i]),
                "outlier_flag": bool(outlier_flags[i]),
                "pca": [float(x) for x in pca_points[i].tolist()],
                "tsne": [float(x) for x in tsne_points[i].tolist()],
                "umap": [float(x) for x in umap_points[i].tolist()],
            }
        )

    results = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "root_dir": str(dataset_root),
            "device": device.type,
            "clip_backend": adapter.backend_name,
            "model_name": args.model_name,
            "extensions": list(valid_ext),
            "total_candidates": len(records),
            "valid_images": len(valid_records),
            "skipped_images": len(skipped_images),
            "num_games": len(set(labels)),
            "num_groups": len(set(sample_groups)),
            "group_counts": {
                group: int(count)
                for group, count in sorted(group_counts.items(), key=lambda item: item[0])
            },
            "data_source": {
                "source": "local_only",
                "dataset_root": str(dataset_root),
                "default_dataset_root": str(DATASET_ROOT),
                "game_groups_file": str(args.game_groups_file),
            },
        },
        "runtime_parameters": {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "tsne": tsne_meta,
            "umap": umap_meta,
            "pca": pca_meta,
            "kmeans": kmeans_meta,
            "dbscan": dbscan_meta,
            "projection_quality": quality_report,
            "outlier_detection": outlier_meta,
            "prompt_focus": {
                **prompt_focus_meta,
                "source_file": str(args.prompt_focus_file) if args.prompt_focus_file is not None else None,
                "full_prompt_count": len(full_prompt_labels),
                "display_prompt_count": len(prompt_labels),
            },
            "style_adapter": style_adapter_meta,
            "thumbnails": {
                "enabled": bool(args.export_thumbnails),
                "size": int(args.thumbnail_size),
                "exported_count": len(thumbnail_map),
            },
            "swiss_roll_demo": {
                "enabled": bool(swiss_roll_demo.get("enabled")),
                "n_samples": int(args.swiss_roll_samples),
                "noise": float(args.swiss_roll_noise),
                "lle_neighbors": int(args.swiss_roll_lle_neighbors),
            },
        },
        "samples": samples,
        "centroid_similarity": {
            "labels": unique_games,
            "matrix": [[float(v) for v in row] for row in similarity_matrix.tolist()],
        },
        "group_centroid_similarity": {
            "labels": unique_groups,
            "matrix": [[float(v) for v in row] for row in group_similarity_matrix.tolist()],
        },
        "clusters": {
            "k": int(kmeans_meta["k"]),
            "crosstab": cluster_crosstab,
            "kmeans": {
                "k": int(kmeans_meta["k"]),
                "crosstab": cluster_crosstab,
            },
            "dbscan": {
                "meta": dbscan_meta,
                "crosstab": dbscan_crosstab,
            },
        },
        "projection_quality": quality_report,
        "outliers": {
            "meta": outlier_meta,
            "rows": outlier_rows,
        },
        "prompt_similarity": {
            "games": prompt_games,
            "prompts": prompt_labels,
            "matrix": [[float(v) for v in row] for row in game_prompt_scores.tolist()],
            "source": prompt_source,
        },
        "prompt_similarity_full": {
            "games": prompt_games,
            "prompts": full_prompt_labels,
            "matrix": [[float(v) for v in row] for row in full_game_prompt_scores.tolist()],
            "source": prompt_source,
        },
        "clip_prompt_similarity": {
            "games": clip_prompt_games,
            "prompts": prompts,
            "matrix": [[float(v) for v in row] for row in clip_game_prompt_scores.tolist()],
            "source": "clip_text_prompts",
        },
        "prompt_similarity_by_group": {
            "groups": prompt_groups,
            "prompts": prompt_labels,
            "matrix": [[float(v) for v in row] for row in group_prompt_scores.tolist()],
            "source": prompt_source,
        },
        "prompt_similarity_by_group_full": {
            "groups": prompt_groups,
            "prompts": full_prompt_labels,
            "matrix": [[float(v) for v in row] for row in full_group_prompt_scores.tolist()],
            "source": prompt_source,
        },
        "clip_prompt_similarity_by_group": {
            "groups": clip_prompt_groups,
            "prompts": prompts,
            "matrix": [[float(v) for v in row] for row in clip_group_prompt_scores.tolist()],
            "source": "clip_text_prompts",
        },
        "swiss_roll_demo": swiss_roll_demo,
        "skipped_images": skipped_images,
    }

    json_path = output_dir / "analysis_results.json"
    log("[Step 8/8] Writing JSON/CSV outputs...")
    atomic_write_text(json_path, json.dumps(results, indent=2), encoding="utf-8")

    sample_rows = []
    for sample in samples:
        sample_rows.append(
            {
                "image_id": sample["image_id"],
                "game": sample["label"],
                "group": sample.get("group", "unassigned"),
                "path": sample["path"],
                "thumbnail": sample.get("thumbnail") or "",
                "cluster_id": sample["cluster_id"],
                "dbscan_cluster_id": sample.get("dbscan_cluster_id", -1),
                "style_pred_label": sample.get("style_pred_label", ""),
                "style_pred_confidence": sample.get("style_pred_confidence", 0.0),
                "outlier_score": sample.get("outlier_score", 0.0),
                "outlier_flag": sample.get("outlier_flag", False),
                "pca_x": sample["pca"][0],
                "pca_y": sample["pca"][1],
                "pca_z": sample["pca"][2],
                "tsne_x": sample["tsne"][0],
                "tsne_y": sample["tsne"][1],
                "tsne_z": sample["tsne"][2],
                "umap_x": sample["umap"][0],
                "umap_y": sample["umap"][1],
                "umap_z": sample["umap"][2],
            }
        )

    write_csv(
        output_dir / "sample_points.csv",
        sample_rows,
        [
            "image_id",
            "game",
            "group",
            "path",
            "thumbnail",
            "cluster_id",
            "dbscan_cluster_id",
            "style_pred_label",
            "style_pred_confidence",
            "outlier_score",
            "outlier_flag",
            "pca_x",
            "pca_y",
            "pca_z",
            "tsne_x",
            "tsne_y",
            "tsne_z",
            "umap_x",
            "umap_y",
            "umap_z",
        ],
    )

    cent_rows = []
    for i, g1 in enumerate(unique_games):
        for j, g2 in enumerate(unique_games):
            cent_rows.append({"game_a": g1, "game_b": g2, "cosine_similarity": float(similarity_matrix[i, j])})
    write_csv(output_dir / "centroid_similarity.csv", cent_rows, ["game_a", "game_b", "cosine_similarity"])

    prompt_rows = []
    for gi, game in enumerate(prompt_games):
        row = {"game": game}
        for pi, prompt in enumerate(prompt_labels):
            row[prompt] = float(game_prompt_scores[gi, pi])
        prompt_rows.append(row)
    write_csv(output_dir / "prompt_similarity_by_game.csv", prompt_rows, ["game", *prompt_labels])
    full_prompt_rows = []
    for gi, game in enumerate(prompt_games):
        row = {"game": game}
        for pi, prompt in enumerate(full_prompt_labels):
            row[prompt] = float(full_game_prompt_scores[gi, pi])
        full_prompt_rows.append(row)
    write_csv(output_dir / "prompt_similarity_by_game_full.csv", full_prompt_rows, ["game", *full_prompt_labels])

    prompt_group_rows = []
    for gi, group in enumerate(prompt_groups):
        row = {"group": group}
        for pi, prompt in enumerate(prompt_labels):
            row[prompt] = float(group_prompt_scores[gi, pi])
        prompt_group_rows.append(row)
    write_csv(output_dir / "prompt_similarity_by_group.csv", prompt_group_rows, ["group", *prompt_labels])
    full_prompt_group_rows = []
    for gi, group in enumerate(prompt_groups):
        row = {"group": group}
        for pi, prompt in enumerate(full_prompt_labels):
            row[prompt] = float(full_group_prompt_scores[gi, pi])
        full_prompt_group_rows.append(row)
    write_csv(
        output_dir / "prompt_similarity_by_group_full.csv",
        full_prompt_group_rows,
        ["group", *full_prompt_labels],
    )

    crosstab_rows = []
    for game, clusters in cluster_crosstab.items():
        for cluster_str, count in sorted(clusters.items(), key=lambda x: int(x[0])):
            crosstab_rows.append({"game": game, "cluster_id": cluster_str, "count": count})
    write_csv(output_dir / "cluster_crosstab.csv", crosstab_rows, ["game", "cluster_id", "count"])
    dbscan_crosstab_rows = []
    for game, clusters in dbscan_crosstab.items():
        for cluster_str, count in sorted(clusters.items(), key=lambda x: int(x[0])):
            dbscan_crosstab_rows.append({"game": game, "cluster_id": cluster_str, "count": count})
    write_csv(output_dir / "dbscan_cluster_crosstab.csv", dbscan_crosstab_rows, ["game", "cluster_id", "count"])
    write_csv(
        output_dir / "outlier_rows.csv",
        outlier_rows,
        ["rank", "image_id", "label", "group", "path", "outlier_score", "outlier_percentile", "dbscan_noise", "flagged"],
    )

    if args.save_embeddings:
        np.save(output_dir / "clip_embeddings.npy", embeddings)

    log(f"Saved JSON: {json_path}")
    log(f"Saved CSV files in: {output_dir}")
    if args.export_thumbnails:
        log(f"Exported {len(thumbnail_map)} thumbnails to: {output_dir / 'thumbs'}")
    if style_adapter_meta.get("enabled"):
        log(f"Style adapter source for prompt heatmap: {style_adapter_meta.get('checkpoint')}")
    log(
        "Prompt outputs: "
        f"display={len(prompt_labels)} prompts -> prompt_similarity_by_game.csv, "
        f"full={len(full_prompt_labels)} prompts -> prompt_similarity_by_game_full.csv"
    )
    if swiss_roll_demo.get("enabled"):
        log(
            "Swiss-roll demo: "
            f"{len(swiss_roll_demo.get('samples', []))} points "
            "(rendered in web UI under Swiss Roll Sanity Check)."
        )
    log(f"Processed {len(valid_records)} images (skipped {len(skipped_images)} unreadable files).")
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
