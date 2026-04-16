#!/usr/bin/env python3
"""Thesis-oriented attribute analysis for Indie vs AAA artwork.

This script quantifies which visual attributes separate indie and AAA game artwork.
It combines:
1) Handcrafted image features (color, composition, texture, typography proxies)
2) CLIP affective prompt scores
3) A supervised classifier with permutation-based feature importance
4) Permutation-based feature statistics (with Benjamini-Hochberg correction)

Outputs are written as CSV/JSON for reporting and plotting.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

np = None
pd = None
torch = None
Image = None
permutation_importance = None
LogisticRegression = None
accuracy_score = None
balanced_accuracy_score = None
f1_score = None
roc_auc_score = None
train_test_split = None
Pipeline = None
StandardScaler = None

# Delay importing the main pipeline module to avoid long silent startup before logs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "indie_games_dataset"
PipelineError = RuntimeError
choose_device: Callable[[str], Any] | None = None
collect_image_records: Callable[..., Any] | None = None
encode_images: Callable[..., Any] | None = None
load_clip_adapter: Callable[..., Any] | None = None
load_game_groups: Callable[..., Any] | None = None
normalize_group_name: Callable[[str], str] | None = None


DEFAULT_AFFECTIVE_PROMPTS = [
    "cozy warm atmosphere",
    "calm serene atmosphere",
    "dark ominous atmosphere",
    "energetic intense action mood",
    "playful whimsical tone",
    "melancholic reflective mood",
    "epic heroic mood",
    "minimalist quiet mood",
    "gritty realistic mood",
    "dreamlike surreal mood",
]


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dependencies() -> None:
    global np, pd, torch, Image
    global permutation_importance, LogisticRegression, accuracy_score, balanced_accuracy_score
    global f1_score, roc_auc_score, train_test_split, Pipeline, StandardScaler

    if np is not None and pd is not None and torch is not None:
        return

    try:
        import numpy as _np
        import pandas as _pd
        import torch as _torch
        from PIL import Image as _Image
        from sklearn.inspection import permutation_importance as _permutation_importance
        from sklearn.linear_model import LogisticRegression as _LogisticRegression
        from sklearn.metrics import accuracy_score as _accuracy_score
        from sklearn.metrics import balanced_accuracy_score as _balanced_accuracy_score
        from sklearn.metrics import f1_score as _f1_score
        from sklearn.metrics import roc_auc_score as _roc_auc_score
        from sklearn.model_selection import train_test_split as _train_test_split
        from sklearn.pipeline import Pipeline as _Pipeline
        from sklearn.preprocessing import StandardScaler as _StandardScaler
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        missing_name = exc.name or "unknown"
        raise SystemExit(
            f"[ERROR] Missing dependency '{missing_name}'. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    np = _np
    pd = _pd
    torch = _torch
    Image = _Image
    permutation_importance = _permutation_importance
    LogisticRegression = _LogisticRegression
    accuracy_score = _accuracy_score
    balanced_accuracy_score = _balanced_accuracy_score
    f1_score = _f1_score
    roc_auc_score = _roc_auc_score
    train_test_split = _train_test_split
    Pipeline = _Pipeline
    StandardScaler = _StandardScaler


def ensure_pipeline_bindings() -> None:
    global DATASET_ROOT
    global PipelineError, choose_device, collect_image_records, encode_images
    global load_clip_adapter, load_game_groups, normalize_group_name

    if choose_device is not None:
        return

    try:
        from clip_indie_pipeline import (  # type: ignore
            DATASET_ROOT as imported_dataset_root,
            PipelineError as imported_pipeline_error,
            choose_device as imported_choose_device,
            collect_image_records as imported_collect_image_records,
            encode_images as imported_encode_images,
            load_clip_adapter as imported_load_clip_adapter,
            load_game_groups as imported_load_game_groups,
            normalize_group_name as imported_normalize_group_name,
        )
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "unknown"
        raise SystemExit(
            f"[ERROR] Missing dependency '{missing_name}'. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    DATASET_ROOT = imported_dataset_root
    PipelineError = imported_pipeline_error
    choose_device = imported_choose_device
    collect_image_records = imported_collect_image_records
    encode_images = imported_encode_images
    load_clip_adapter = imported_load_clip_adapter
    load_game_groups = imported_load_game_groups
    normalize_group_name = imported_normalize_group_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run thesis attribute analysis to distinguish indie vs AAA artwork."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("web/data/thesis"),
        help="Directory for thesis CSV/JSON outputs.",
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
        help="CLIP model name.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional max number of images to process (0 = no limit).",
    )
    parser.add_argument(
        "--game-groups-file",
        type=Path,
        default=Path("src/game_groups.csv"),
        help="CSV with columns game,group. Group should include indie/aaa.",
    )
    parser.add_argument(
        "--affective-prompts-file",
        type=Path,
        default=None,
        help="Optional text file with one affective prompt per line.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--image-max-side", type=int, default=384, help="Resize max side for handcrafted features.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split ratio for classifier.")
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=40,
        help="Permutation importance repeats on test set.",
    )
    parser.add_argument(
        "--stat-permutations",
        type=int,
        default=2000,
        help="Permutation count for feature-level p-values.",
    )
    parser.add_argument(
        "--use-game-aggregation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, aggregate features per game before modeling/stats.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_extensions(value: str) -> tuple[str, ...]:
    parsed = tuple(
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in value.split(",")
        if ext.strip()
    )
    if not parsed:
        raise PipelineError("No valid image extensions provided.")
    return parsed


def load_affective_prompts(path: Path | None) -> list[str]:
    if path is None:
        prompts = list(DEFAULT_AFFECTIVE_PROMPTS)
    else:
        if not path.exists():
            raise PipelineError(f"Affective prompts file not found: {path}")
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        key = prompt.strip().lower()
        if not key or key in seen:
            continue
        deduped.append(prompt)
        seen.add(key)
    if not deduped:
        raise PipelineError("No valid affective prompts found.")
    return deduped


def rgb_to_hsv_np(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB->HSV where rgb is float32 in [0, 1], shape (H, W, 3)."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc

    s = np.zeros_like(maxc, dtype=np.float32)
    nonzero = maxc > 1e-8
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    h = np.zeros_like(maxc, dtype=np.float32)
    nonzero_delta = delta > 1e-8
    mask_r = nonzero_delta & (maxc == r)
    mask_g = nonzero_delta & (maxc == g)
    mask_b = nonzero_delta & (maxc == b)
    h[mask_r] = ((g - b)[mask_r] / delta[mask_r]) % 6.0
    h[mask_g] = ((b - r)[mask_g] / delta[mask_g]) + 2.0
    h[mask_b] = ((r - g)[mask_b] / delta[mask_b]) + 4.0
    h = (h / 6.0) % 1.0
    return h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)


def shannon_entropy_from_hist(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def clip_percentile(value: float, low: float = -1e9, high: float = 1e9) -> float:
    return float(min(high, max(low, value)))


def extract_handcrafted_features(image_path: Path, image_max_side: int) -> dict[str, float]:
    """Extract interpretable low-level visual features from an image."""
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        if image_max_side > 0 and max(w, h) > image_max_side:
            scale = image_max_side / max(w, h)
            new_w = max(32, int(round(w * scale)))
            new_h = max(32, int(round(h * scale)))
            img_rgb = img_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)

    rgb = np.asarray(img_rgb, dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    h, w = gray.shape

    hue, sat, val = rgb_to_hsv_np(rgb)

    # Color usage
    lum_p95 = float(np.percentile(gray, 95))
    lum_p05 = float(np.percentile(gray, 5))
    lum_contrast = lum_p95 - lum_p05
    warm_mask = (((hue <= 1.0 / 6.0) | (hue >= 5.0 / 6.0)) & (sat > 0.2)).astype(np.float32)
    cool_mask = (((hue >= 0.45) & (hue <= 0.75)) & (sat > 0.2)).astype(np.float32)
    vivid_mask = ((sat > 0.55) & (val > 0.35)).astype(np.float32)

    # RGB histogram entropy as a palette-diversity proxy.
    bins = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    hist, _ = np.histogramdd(rgb.reshape(-1, 3), bins=(bins, bins, bins))
    palette_entropy = shannon_entropy_from_hist(hist.astype(np.float64))

    # Gradients and edge structure for composition/typography proxies.
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    grad_mag = np.hypot(gx, gy)
    grad_thr = float(np.percentile(grad_mag, 75))
    edge_mask = grad_mag > grad_thr

    # Composition proxies.
    saliency = grad_mag + 1e-8
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    sal_sum = float(np.sum(saliency))
    cx = float(np.sum((x_grid / max(1, (w - 1))) * saliency) / sal_sum)
    cy = float(np.sum((y_grid / max(1, (h - 1))) * saliency) / sal_sum)
    center_distance = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    x_norm = x_grid / max(1, (w - 1))
    y_norm = y_grid / max(1, (h - 1))
    thirds_band = 0.06
    thirds_mask = (
        (np.abs(x_norm - 1.0 / 3.0) < thirds_band)
        | (np.abs(x_norm - 2.0 / 3.0) < thirds_band)
        | (np.abs(y_norm - 1.0 / 3.0) < thirds_band)
        | (np.abs(y_norm - 2.0 / 3.0) < thirds_band)
    )
    thirds_saliency_ratio = float(np.sum(saliency[thirds_mask]) / sal_sum)

    half = w // 2
    if half >= 2:
        left = gray[:, :half]
        right = np.fliplr(gray[:, -half:])
        left_flat = left.reshape(-1)
        right_flat = right.reshape(-1)
        left_std = float(np.std(left_flat))
        right_std = float(np.std(right_flat))
        if left_std > 1e-8 and right_std > 1e-8:
            symmetry_lr = float(np.corrcoef(left_flat, right_flat)[0, 1])
        else:
            symmetry_lr = 0.0
    else:
        symmetry_lr = 0.0

    top_sal = float(np.sum(saliency[: h // 2]))
    bottom_sal = float(np.sum(saliency[h // 2 :]))
    top_heaviness = top_sal / max(1e-8, (top_sal + bottom_sal))
    negative_space_ratio = float(np.mean((grad_mag < np.percentile(grad_mag, 35)) & (sat < 0.2)))

    # Texture descriptors.
    gray_hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    gray_entropy = shannon_entropy_from_hist(gray_hist.astype(np.float64))
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    lap_var = float(np.var(lap))

    # Local contrast from cardinal neighbors.
    local_contrast = (
        np.abs(gray - np.roll(gray, 1, axis=0))
        + np.abs(gray - np.roll(gray, -1, axis=0))
        + np.abs(gray - np.roll(gray, 1, axis=1))
        + np.abs(gray - np.roll(gray, -1, axis=1))
    ) / 4.0
    texture_density = float(np.mean(local_contrast > np.percentile(local_contrast, 75)))

    # Frequency-domain high-frequency energy ratio.
    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(fft) ** 2
    yy, xx = np.mgrid[0:h, 0:w]
    cyi = (h - 1) / 2.0
    cxi = (w - 1) / 2.0
    dist = np.sqrt(((yy - cyi) / max(1.0, cyi)) ** 2 + ((xx - cxi) / max(1.0, cxi)) ** 2)
    high_freq_ratio = float(np.sum(power[dist > 0.45]) / max(1e-8, np.sum(power)))

    # Typography proxies (no OCR; these are structural approximations).
    edge_density = float(np.mean(edge_mask))
    horiz_edges = float(np.mean(np.abs(gx) > np.percentile(np.abs(gx), 75)))
    vert_edges = float(np.mean(np.abs(gy) > np.percentile(np.abs(gy), 75)))
    horiz_vert_ratio = horiz_edges / max(1e-8, vert_edges)
    text_like_density = float(np.mean((grad_mag > grad_thr) & (sat < 0.35) & (gray > 0.15) & (gray < 0.9)))

    features = {
        "color_sat_mean": float(np.mean(sat)),
        "color_sat_std": float(np.std(sat)),
        "color_val_mean": float(np.mean(val)),
        "color_val_std": float(np.std(val)),
        "color_luminance_contrast": clip_percentile(lum_contrast, 0.0, 1.0),
        "color_warm_ratio": float(np.mean(warm_mask)),
        "color_cool_ratio": float(np.mean(cool_mask)),
        "color_vivid_ratio": float(np.mean(vivid_mask)),
        "color_palette_entropy": max(0.0, palette_entropy),
        "composition_center_distance": clip_percentile(center_distance, 0.0, 1.0),
        "composition_rule_of_thirds_ratio": clip_percentile(thirds_saliency_ratio, 0.0, 1.0),
        "composition_symmetry_lr": clip_percentile(symmetry_lr, -1.0, 1.0),
        "composition_top_heaviness": clip_percentile(top_heaviness, 0.0, 1.0),
        "composition_negative_space_ratio": clip_percentile(negative_space_ratio, 0.0, 1.0),
        "texture_gray_entropy": max(0.0, gray_entropy),
        "texture_laplacian_var": max(0.0, lap_var),
        "texture_local_contrast_mean": float(np.mean(local_contrast)),
        "texture_texture_density": clip_percentile(texture_density, 0.0, 1.0),
        "texture_high_freq_ratio": clip_percentile(high_freq_ratio, 0.0, 1.0),
        "typography_edge_density": clip_percentile(edge_density, 0.0, 1.0),
        "typography_horiz_vert_ratio": max(0.0, horiz_vert_ratio),
        "typography_text_like_density": clip_percentile(text_like_density, 0.0, 1.0),
    }
    return features


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    diff = a[:, None] - b[None, :]
    gt = np.sum(diff > 0)
    lt = np.sum(diff < 0)
    return float((gt - lt) / (a.size * b.size))


def permutation_pvalue_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    permutations: int,
    seed: int,
) -> float:
    """Two-sided permutation p-value for mean difference."""
    if a.size == 0 or b.size == 0:
        return 1.0
    observed = abs(float(np.mean(a) - np.mean(b)))
    combined = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(max(1, permutations)):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        diff = abs(float(np.mean(perm_a) - np.mean(perm_b)))
        if diff >= observed:
            count += 1
    return float((count + 1) / (max(1, permutations) + 1))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.array(p_values, dtype=np.float64)[order]
    adjusted = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * m / rank)
        adjusted[i] = value
        prev = value
    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted
    return [float(v) for v in out]


def ensure_binary_groups(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["group"].isin(["indie", "aaa"])
    out = df.loc[keep].copy()
    if out.empty:
        raise PipelineError("No rows with group in {'indie', 'aaa'}. Check src/game_groups.csv.")
    return out


def aggregate_by_game(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(["game", "group"], as_index=False)[feature_names]
        .mean(numeric_only=True)
        .sort_values(["group", "game"])
        .reset_index(drop=True)
    )
    counts = df.groupby("game").size().rename("num_images").reset_index()
    return agg.merge(counts, on="game", how="left")


def compute_stats_table(df: pd.DataFrame, feature_names: list[str], permutations: int, seed: int) -> pd.DataFrame:
    indie = df[df["group"] == "indie"]
    aaa = df[df["group"] == "aaa"]
    rows: list[dict] = []
    for idx, feat in enumerate(feature_names):
        a = indie[feat].to_numpy(dtype=np.float64)
        b = aaa[feat].to_numpy(dtype=np.float64)
        p = permutation_pvalue_mean_diff(a, b, permutations=permutations, seed=seed + idx)
        delta = cliffs_delta(a, b)
        rows.append(
            {
                "feature": feat,
                "mean_indie": float(np.mean(a)) if a.size else 0.0,
                "mean_aaa": float(np.mean(b)) if b.size else 0.0,
                "mean_diff_aaa_minus_indie": float((np.mean(b) - np.mean(a))) if (a.size and b.size) else 0.0,
                "cliffs_delta": delta,
                "p_value_perm": p,
            }
        )
    pvals = [row["p_value_perm"] for row in rows]
    qvals = benjamini_hochberg(pvals)
    for row, q in zip(rows, qvals):
        row["q_value_bh"] = q
    return pd.DataFrame(rows).sort_values(["q_value_bh", "p_value_perm", "feature"]).reset_index(drop=True)


def train_and_explain(
    df: pd.DataFrame,
    feature_names: list[str],
    test_size: float,
    seed: int,
    importance_repeats: int,
) -> tuple[dict, pd.DataFrame]:
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = (df["group"] == "aaa").astype(int).to_numpy()

    class_counts = df["group"].value_counts().to_dict()
    if class_counts.get("indie", 0) < 2 or class_counts.get("aaa", 0) < 2:
        raise PipelineError("Need at least 2 indie and 2 AAA rows to train/test the classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
        ]
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }

    if len(np.unique(y_test)) > 1:
        prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, prob))
    else:
        metrics["roc_auc"] = None

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=max(5, importance_repeats),
        random_state=seed,
        scoring="balanced_accuracy",
    )
    clf = model.named_steps["clf"]
    coef = clf.coef_[0]
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
            "logreg_coef": coef,
            "logreg_abs_coef": np.abs(coef),
        }
    ).sort_values(["perm_importance_mean", "logreg_abs_coef"], ascending=False)

    return metrics, importance_df.reset_index(drop=True)


def run() -> int:
    args = parse_args()
    log("[Thesis] Importing dependencies...")
    ensure_dependencies()
    set_seed(args.seed)
    log("[Thesis] Starting thesis attribute analysis...")
    log("[Thesis] Loading pipeline helpers...")
    ensure_pipeline_bindings()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[Thesis] Output directory: {output_dir}")

    if (
        choose_device is None
        or load_game_groups is None
        or collect_image_records is None
        or load_clip_adapter is None
        or encode_images is None
        or normalize_group_name is None
    ):
        raise PipelineError("Pipeline helper initialization failed.")

    valid_ext = parse_extensions(args.extensions)
    device = choose_device(args.device)
    game_groups = load_game_groups(args.game_groups_file)
    prompts = load_affective_prompts(args.affective_prompts_file)
    log(
        f"[Thesis] Device={device.type} | backend={args.clip_backend} | model={args.model_name} | "
        f"prompts={len(prompts)}"
    )

    records = collect_image_records(DATASET_ROOT, valid_ext, args.max_images)
    if not records:
        raise PipelineError(f"No images found under dataset root: {DATASET_ROOT}")
    log(f"[Thesis] Found {len(records)} candidate images under: {DATASET_ROOT}")

    log("[Thesis] Loading CLIP model...")
    adapter = load_clip_adapter(args.clip_backend, args.model_name, device)
    log(f"[Thesis] CLIP backend ready: {adapter.backend_name}")
    log(f"[Thesis] Encoding images with batch_size={args.batch_size}...")
    embeddings, valid_records, skipped = encode_images(adapter, records, args.batch_size)
    if embeddings.shape[0] == 0:
        raise PipelineError("No valid CLIP embeddings generated.")
    log(
        f"[Thesis] Encoded {embeddings.shape[0]} images "
        f"(valid={len(valid_records)}, skipped_during_clip={len(skipped)})"
    )

    # Use only rows with explicit indie/aaa labels for thesis classification.
    rows: list[dict] = []
    kept_embeddings: list[np.ndarray] = []
    for i, rec in enumerate(valid_records):
        group = normalize_group_name(game_groups.get(rec.label, "unassigned"))
        if group not in {"indie", "aaa"}:
            continue
        rows.append(
            {
                "image_id": rec.image_id,
                "game": rec.label,
                "group": group,
                "path": str(rec.path),
            }
        )
        kept_embeddings.append(embeddings[i])

    if len(rows) < 8:
        raise PipelineError("Need at least 8 labeled indie/aaa samples to run thesis analysis.")
    log(f"[Thesis] Labeled indie/aaa rows retained: {len(rows)}")

    # Handcrafted attributes.
    handcrafted_failures: list[dict] = []
    handcrafted_rows: list[dict] = []
    valid_embedding_rows: list[np.ndarray] = []
    log("[Thesis] Extracting handcrafted features...")
    for i, row in enumerate(rows):
        path = Path(row["path"])
        try:
            feats = extract_handcrafted_features(path, args.image_max_side)
        except Exception as exc:
            handcrafted_failures.append({"path": str(path), "reason": str(exc)})
            continue
        merged = {**row, **feats}
        handcrafted_rows.append(merged)
        valid_embedding_rows.append(kept_embeddings[i])
        if (i + 1) % 25 == 0 or (i + 1) == len(rows):
            log(
                f"[Thesis] Handcrafted progress: {i + 1}/{len(rows)} "
                f"(failures={len(handcrafted_failures)})"
            )

    if len(handcrafted_rows) < 8:
        raise PipelineError(
            "Too many failures while extracting handcrafted features; fewer than 8 rows remained."
        )

    emb_matrix = np.vstack(valid_embedding_rows).astype(np.float32)
    log(f"[Thesis] Computing CLIP affective prompt scores for {len(prompts)} prompts...")
    with torch.inference_mode():
        text_features = adapter.encode_text(prompts).detach().cpu().numpy().astype(np.float32)
    prompt_scores = emb_matrix @ text_features.T

    for i, row in enumerate(handcrafted_rows):
        for j, prompt in enumerate(prompts):
            key = f"affect_{prompt.lower().strip().replace(' ', '_').replace('-', '_')}"
            row[key] = float(prompt_scores[i, j])

    df_samples = pd.DataFrame(handcrafted_rows)
    df_samples = ensure_binary_groups(df_samples)

    # Build deterministic feature list from category prefixes.
    feature_prefixes = ("color_", "composition_", "texture_", "typography_", "affect_")
    feature_names = sorted([col for col in df_samples.columns if col.startswith(feature_prefixes)])
    if len(feature_names) < 5:
        raise PipelineError("Feature extraction produced too few usable features.")

    if args.use_game_aggregation:
        df_model = aggregate_by_game(df_samples, feature_names)
        unit = "game"
    else:
        df_model = df_samples.copy()
        unit = "image"
    log(f"[Thesis] Modeling unit: {unit} | rows={df_model.shape[0]} | features={len(feature_names)}")

    df_model = ensure_binary_groups(df_model)

    log("[Thesis] Training classifier and computing permutation importance...")
    metrics, importance_df = train_and_explain(
        df=df_model,
        feature_names=feature_names,
        test_size=args.test_size,
        seed=args.seed,
        importance_repeats=args.importance_repeats,
    )
    log(f"[Thesis] Running feature permutation tests (n={args.stat_permutations})...")
    stats_df = compute_stats_table(
        df=df_model,
        feature_names=feature_names,
        permutations=args.stat_permutations,
        seed=args.seed,
    )

    # Group summaries.
    summary_df = (
        df_model.groupby("group", as_index=False)[feature_names]
        .mean(numeric_only=True)
        .sort_values("group")
        .reset_index(drop=True)
    )

    # Save outputs.
    df_samples.to_csv(output_dir / "attribute_features_per_image.csv", index=False)
    df_model.to_csv(output_dir / "attribute_features_modeling_table.csv", index=False)
    summary_df.to_csv(output_dir / "attribute_feature_group_means.csv", index=False)
    importance_df.to_csv(output_dir / "attribute_feature_importance.csv", index=False)
    stats_df.to_csv(output_dir / "attribute_feature_stats.csv", index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(DATASET_ROOT),
        "device": device.type,
        "clip_backend": adapter.backend_name,
        "clip_model_name": args.model_name,
        "group_mapping_file": str(args.game_groups_file),
        "num_candidate_images": len(records),
        "num_valid_images_after_clip": len(valid_records),
        "num_skipped_images_during_clip": len(skipped),
        "num_skipped_images_handcrafted": len(handcrafted_failures),
        "num_rows_modeling": int(df_model.shape[0]),
        "unit_of_analysis": unit,
        "class_counts": {
            "indie": int(np.sum(df_model["group"] == "indie")),
            "aaa": int(np.sum(df_model["group"] == "aaa")),
        },
        "affective_prompts": prompts,
        "classifier_metrics": metrics,
        "top_features_by_permutation_importance": importance_df.head(15).to_dict(orient="records"),
        "top_features_by_q_value": stats_df.head(15).to_dict(orient="records"),
    }
    (output_dir / "attribute_analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if handcrafted_failures:
        pd.DataFrame(handcrafted_failures).to_csv(output_dir / "attribute_handcrafted_failures.csv", index=False)

    log(f"[Thesis] Saved thesis attribute outputs to: {output_dir}")
    log(
        f"Rows used for modeling ({unit}-level): {df_model.shape[0]} "
        f"[indie={report['class_counts']['indie']}, aaa={report['class_counts']['aaa']}]"
    )
    log(
        "Classifier metrics: "
        f"balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
        f"f1={metrics['f1']:.3f}, "
        f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'n/a'}"
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
