#!/usr/bin/env python3
"""CLIP-based indie screenshot analysis pipeline (local-only dataset).

This script mirrors the notebook workflow and adds production-friendly behavior:
- Batched CLIP embedding extraction (GPU-aware)
- 3D t-SNE and 3D UMAP projections
- Game centroid similarity matrix
- KMeans clustering diagnostics
- Prompt-based style similarity scoring
- JSON/CSV exports for web visualizations

Dataset source is fixed to:
/Users/gqnsptaa/Desktop/Codex_Project/indie_games_dataset
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
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    missing_name = exc.name or "unknown"
    raise SystemExit(
        f"[ERROR] Missing dependency '{missing_name}'. Install dependencies with: pip install -r requirements.txt"
    ) from exc


# Fixed local dataset root as requested.
DATASET_ROOT = Path("/Users/gqnsptaa/Desktop/Codex_Project/indie_games_dataset")

DEFAULT_STYLE_PROMPTS = [
    "cinematic low-key lighting",
    "high-key soft diffuse lighting",
    "volumetric god rays",
    "hyper-realistic physically based 3D render",
    "stylized cel-shaded render",
    "flat geometric vector design",
    "isometric composition",
    "epic wide-angle composition",
    "symmetrical centered composition",
    "asymmetrical dynamic composition",
    "strong negative space",
    "complementary color contrast",
    "muted desaturated color palette",
    "neon cyberpunk color palette",
    "grainy lo-fi texture",
    "halftone print texture",
    "gothic ornamental art direction",
    "minimal diegetic UI",
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
            "Run CLIP indie aesthetics analysis using local dataset folder "
            f"at: {DATASET_ROOT}"
        )
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
        "--style-prompts-file",
        type=Path,
        default=None,
        help="Optional text file with one style prompt per line.",
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


def load_style_prompts(path: Path | None) -> list[str]:
    if path is None:
        source_prompts = list(DEFAULT_STYLE_PROMPTS)
    else:
        if not path.exists():
            raise PipelineError(f"Style prompts file not found: {path}")
        source_prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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

    def flush_batch() -> None:
        if not image_buffer:
            return

        batch_tensor = torch.stack(image_buffer).to(adapter.device, non_blocking=True)
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = adapter.encode_images(batch_tensor)
            else:
                emb = adapter.encode_images(batch_tensor)

        batches.append(emb.detach().cpu().numpy().astype(np.float32))
        valid_records.extend(rec_buffer)
        image_buffer.clear()
        rec_buffer.clear()

    for rec in records:
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


def pca_to_3d(embeddings: np.ndarray) -> np.ndarray:
    n_samples, n_features = embeddings.shape
    n_comp = min(3, n_samples, n_features)
    if n_comp <= 0:
        raise PipelineError("Cannot run PCA fallback on empty embeddings.")

    reducer = PCA(n_components=n_comp, random_state=42)
    reduced = reducer.fit_transform(embeddings).astype(np.float32)

    if n_comp < 3:
        pad = np.zeros((n_samples, 3 - n_comp), dtype=np.float32)
        reduced = np.concatenate([reduced, pad], axis=1)

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


def crosstab_counts(labels: list[str], clusters: np.ndarray) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for label, cluster in zip(labels, clusters):
        table[label][str(int(cluster))] += 1
    return {label: dict(counts) for label, counts in sorted(table.items())}


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
        elif str(class_id) in i