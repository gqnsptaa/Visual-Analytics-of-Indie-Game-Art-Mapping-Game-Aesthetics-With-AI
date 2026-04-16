#!/usr/bin/env python3
"""Local web server for CLIP explorer with one-click pipeline execution.

Serves the `web/` app and provides:
- GET /api/status
- POST /api/run-analysis
- GET /api/phase3-status
- POST /api/run-phase3
- GET /api/igdb-status
- GET /api/igdb-search-games?q=...&company=...
- GET /api/game-folders
- POST /api/fetch-igdb-covers
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
PIPELINE_PATH = PROJECT_ROOT / "src" / "clip_indie_pipeline.py"
IGDB_FETCH_PATH = PROJECT_ROOT / "src" / "fetch_igdb_covers.py"
PHASE3_PATH = PROJECT_ROOT / "src" / "phase3_advanced_separability_analysis.py"
OUTPUT_DIR = WEB_DIR / "data"
PHASE3_OUTPUT_DIR = WEB_DIR / "data" / "phase3_advanced"
DATASET_DIR = PROJECT_ROOT / "indie_games_dataset"
DEMO_DATASET_DIR = PROJECT_ROOT / "indie_games_dataset_demo_100x2"
DEFAULT_STYLE_ADAPTER_CKPT = PROJECT_ROOT / "training_outputs" / "style_adapter" / "best_style_adapter.pt"
DEFAULT_STYLE_PROMPTS_FILE = PROJECT_ROOT / "src" / "style_prompts_graphic_design_expanded.txt"
DEFAULT_PROMPT_FOCUS_FILE = PROJECT_ROOT / "src" / "style_prompts_graphic_design_focus.txt"
DEFAULT_IGDB_MAPPING_CSV = PROJECT_ROOT / "src" / "igdb_game_mappings.csv"
DEFAULT_IGDB_REPORT_JSON = WEB_DIR / "data" / "igdb_cover_fetch_report.json"
DEFAULT_GAME_GROUPS_CSV = PROJECT_ROOT / "src" / "game_groups.csv"
MAX_REQUEST_BYTES = 64 * 1024
MAX_IGDB_SEARCH_RESULTS = 500


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_run_params(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize analysis parameters from API payload."""
    data = payload or {}
    if not isinstance(data, dict):
        raise ValueError("Request payload must be a JSON object.")

    def as_int(key: str, default: int, min_value: int, max_value: int) -> int:
        value = data.get(key, default)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be an integer.") from None
        if parsed < min_value or parsed > max_value:
            raise ValueError(f"'{key}' must be between {min_value} and {max_value}.")
        return parsed

    def as_float(key: str, default: float, min_value: float, max_value: float) -> float:
        value = data.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number.") from None
        if parsed < min_value or parsed > max_value:
            raise ValueError(f"'{key}' must be between {min_value} and {max_value}.")
        return parsed

    def as_choice(key: str, default: str, choices: set[str]) -> str:
        value = str(data.get(key, default)).strip()
        if value not in choices:
            raise ValueError(f"'{key}' must be one of: {', '.join(sorted(choices))}.")
        return value

    normalized = {
        "umap_n_neighbors": as_int("umap_n_neighbors", default=10, min_value=2, max_value=200),
        "umap_min_dist": as_float("umap_min_dist", default=0.1, min_value=0.0, max_value=0.99),
        "tsne_perplexity": as_float("tsne_perplexity", default=10.0, min_value=2.0, max_value=200.0),
        "batch_size": as_int("batch_size", default=32, min_value=1, max_value=512),
        "max_images": as_int("max_images", default=0, min_value=0, max_value=1_000_000),
        "seed": as_int("seed", default=42, min_value=0, max_value=2_147_483_647),
        "device": as_choice("device", default="auto", choices={"auto", "cpu", "cuda", "mps"}),
        "clip_backend": as_choice("clip_backend", default="auto", choices={"auto", "openai", "open_clip"}),
        "dataset_mode": as_choice("dataset_mode", default="full", choices={"full", "demo"}),
    }

    model_name = str(data.get("model_name", "ViT-B/32")).strip()
    if len(model_name) == 0 or len(model_name) > 128:
        raise ValueError("'model_name' must be a non-empty string shorter than 129 characters.")
    normalized["model_name"] = model_name
    return normalized


def parse_phase3_run_params(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize phase-3 analysis parameters from API payload."""
    data = payload or {}
    if not isinstance(data, dict):
        raise ValueError("Request payload must be a JSON object.")

    def as_int(key: str, default: int, min_value: int, max_value: int) -> int:
        value = data.get(key, default)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be an integer.") from None
        if parsed < min_value or parsed > max_value:
            raise ValueError(f"'{key}' must be between {min_value} and {max_value}.")
        return parsed

    def as_choice(key: str, default: str, choices: set[str]) -> str:
        value = str(data.get(key, default)).strip()
        if value not in choices:
            raise ValueError(f"'{key}' must be one of: {', '.join(sorted(choices))}.")
        return value

    def as_csv_int_list(key: str, default: str) -> str:
        value = str(data.get(key, default)).strip()
        if not value:
            raise ValueError(f"'{key}' must be a non-empty comma-separated integer list.")
        parts = [token.strip() for token in value.split(",") if token.strip()]
        if not parts:
            raise ValueError(f"'{key}' must be a non-empty comma-separated integer list.")
        normalized_parts: list[str] = []
        for token in parts:
            try:
                parsed = int(token)
            except ValueError:
                raise ValueError(f"'{key}' must contain integers only.") from None
            if parsed <= 0:
                raise ValueError(f"'{key}' values must be > 0.")
            normalized_parts.append(str(parsed))
        return ",".join(normalized_parts)

    normalized = {
        "batch_size": as_int("batch_size", default=32, min_value=1, max_value=512),
        "sample_size": as_int("sample_size", default=0, min_value=0, max_value=20000),
        "max_pairs_per_bucket": as_int("max_pairs_per_bucket", default=200000, min_value=100, max_value=2_000_000),
        "device": as_choice("device", default="auto", choices={"auto", "cpu", "cuda", "mps"}),
        "clip_backend": as_choice("clip_backend", default="open_clip", choices={"auto", "openai", "open_clip"}),
        "dataset_mode": as_choice("dataset_mode", default="full", choices={"full", "demo"}),
        "pca_levels": as_csv_int_list("pca_levels", default="2,5,10,25,50,100,200"),
        "ari_seeds": as_csv_int_list("ari_seeds", default="42,43,44"),
    }

    model_name = str(data.get("model_name", "ViT-B/32")).strip()
    if len(model_name) == 0 or len(model_name) > 128:
        raise ValueError("'model_name' must be a non-empty string shorter than 129 characters.")
    normalized["model_name"] = model_name
    return normalized


def parse_igdb_fetch_params(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize IGDB fetch parameters from API payload."""
    data = payload or {}
    if not isinstance(data, dict):
        raise ValueError("Request payload must be a JSON object.")

    def as_int(key: str, default: int, min_value: int, max_value: int) -> int:
        value = data.get(key, default)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be an integer.") from None
        if parsed < min_value or parsed > max_value:
            raise ValueError(f"'{key}' must be between {min_value} and {max_value}.")
        return parsed

    def as_float(key: str, default: float, min_value: float, max_value: float) -> float:
        value = data.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number.") from None
        if parsed < min_value or parsed > max_value:
            raise ValueError(f"'{key}' must be between {min_value} and {max_value}.")
        return parsed

    def as_bool(key: str, default: bool) -> bool:
        value = data.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"'{key}' must be a boolean.")

    def as_choice(key: str, default: str, choices: set[str]) -> str:
        value = str(data.get(key, default)).strip()
        if value not in choices:
            raise ValueError(f"'{key}' must be one of: {', '.join(sorted(choices))}.")
        return value

    image_size = str(data.get("image_size", "cover_big")).strip()
    if not image_size or len(image_size) > 64:
        raise ValueError("'image_size' must be a non-empty string with max length 64.")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if any(ch not in allowed for ch in image_size):
        raise ValueError("'image_size' may only contain letters, numbers, and underscore.")

    seed_genre_name = str(data.get("seed_genre_name", "")).strip()
    seed_mode = as_choice("seed_mode", default="genre", choices={"genre", "popular_year_range", "id_list", "name_list"})
    seed_group_label = str(data.get("seed_group_label", "indie")).strip() or "indie"
    seed_exclude_genre_name = str(data.get("seed_exclude_genre_name", "Indie")).strip()
    if len(seed_group_label) > 48:
        raise ValueError("'seed_group_label' must be shorter than 49 characters.")

    seed_count = as_int("seed_count", default=0, min_value=0, max_value=1000)
    raw_seed_game_ids = data.get("seed_game_ids", [])
    if isinstance(raw_seed_game_ids, str):
        seed_game_ids = []
        for token in raw_seed_game_ids.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed = int(token)
            except ValueError:
                raise ValueError("'seed_game_ids' string must contain comma-separated integers.") from None
            if parsed > 0 and parsed not in seed_game_ids:
                seed_game_ids.append(parsed)
    elif isinstance(raw_seed_game_ids, list):
        seed_game_ids = []
        for item in raw_seed_game_ids:
            try:
                parsed = int(item)
            except (TypeError, ValueError):
                raise ValueError("'seed_game_ids' list must contain integers.") from None
            if parsed > 0 and parsed not in seed_game_ids:
                seed_game_ids.append(parsed)
    else:
        raise ValueError("'seed_game_ids' must be a list of ids or comma-separated string.")

    raw_seed_game_names = data.get("seed_game_names", [])
    if isinstance(raw_seed_game_names, str):
        normalized = raw_seed_game_names.replace("\r\n", "\n").replace("\r", "\n")
        if "\n" in normalized:
            tokens = normalized.split("\n")
        elif ";" in normalized:
            tokens = normalized.split(";")
        else:
            tokens = normalized.split(",")
        seed_game_names = []
        for token in tokens:
            name = token.strip()
            if not name:
                continue
            if name.startswith("#"):
                continue
            if name not in seed_game_names:
                seed_game_names.append(name)
    elif isinstance(raw_seed_game_names, list):
        seed_game_names = []
        for item in raw_seed_game_names:
            if not isinstance(item, str):
                raise ValueError("'seed_game_names' list must contain strings.")
            name = item.strip()
            if not name:
                continue
            if name.startswith("#"):
                continue
            if name not in seed_game_names:
                seed_game_names.append(name)
    else:
        raise ValueError("'seed_game_names' must be a list of names or separated string.")

    seed_year_start = as_int("seed_year_start", default=2010, min_value=1970, max_value=2100)
    seed_year_end = as_int("seed_year_end", default=2020, min_value=1970, max_value=2100)
    seed_min_total_rating_count = as_int("seed_min_total_rating_count", default=100, min_value=0, max_value=1_000_000)
    if seed_count > 0 and seed_mode == "genre" and not seed_genre_name:
        raise ValueError("'seed_genre_name' is required when 'seed_mode' is 'genre' and 'seed_count' > 0.")
    if seed_count > 0 and seed_mode == "popular_year_range" and seed_year_end < seed_year_start:
        raise ValueError("'seed_year_end' must be greater than or equal to 'seed_year_start'.")
    if seed_mode == "id_list" and len(seed_game_ids) == 0:
        raise ValueError("'seed_game_ids' is required when 'seed_mode' is 'id_list'.")
    if seed_mode == "name_list" and len(seed_game_names) == 0:
        raise ValueError("'seed_game_names' is required when 'seed_mode' is 'name_list'.")

    return {
        "dry_run": as_bool("dry_run", default=False),
        "max_games": as_int("max_games", default=0, min_value=0, max_value=2000),
        "image_size": image_size,
        "min_match_score": as_float("min_match_score", default=0.72, min_value=0.0, max_value=1.0),
        "min_token_overlap": as_float("min_token_overlap", default=0.34, min_value=0.0, max_value=1.0),
        "allow_low_confidence": as_bool("allow_low_confidence", default=False),
        "strict_match_mode": as_bool("strict_match_mode", default=True),
        "auto_clean_local_names": as_bool("auto_clean_local_names", default=True),
        "skip_if_any_image": as_bool("skip_if_any_image", default=True),
        "overwrite": as_bool("overwrite", default=False),
        "seed_mode": seed_mode,
        "seed_genre_name": seed_genre_name,
        "seed_game_ids": seed_game_ids,
        "seed_game_names": seed_game_names,
        "seed_count": seed_count,
        "seed_only": as_bool("seed_only", default=False),
        "seed_group_label": seed_group_label,
        "seed_update_groups": as_bool("seed_update_groups", default=True),
        "seed_write_mappings": as_bool("seed_write_mappings", default=True),
        "seed_year_start": seed_year_start,
        "seed_year_end": seed_year_end,
        "seed_min_total_rating_count": seed_min_total_rating_count,
        "seed_exclude_genre_name": seed_exclude_genre_name,
    }


def parse_igdb_search_params(query: dict[str, list[str]]) -> tuple[str, int, str]:
    raw_q = (query.get("q", [""])[0] or "").strip()
    raw_company = (query.get("company", [""])[0] or "").strip()
    if not raw_q and not raw_company:
        raise ValueError("Provide at least one search field: 'q' or 'company'.")
    if raw_q and len(raw_q) < 2:
        raise ValueError("'q' must be at least 2 characters when provided.")
    if len(raw_q) > 120:
        raise ValueError("'q' must be shorter than 121 characters.")
    if raw_company and len(raw_company) < 2:
        raise ValueError("'company' must be at least 2 characters when provided.")
    if len(raw_company) > 2000:
        raise ValueError("'company' must be shorter than 2001 characters.")
    raw_limit = (query.get("limit", ["20"])[0] or "20").strip()
    try:
        limit = int(raw_limit)
    except ValueError:
        raise ValueError("'limit' must be an integer.") from None
    if limit < 1 or limit > MAX_IGDB_SEARCH_RESULTS:
        raise ValueError(f"'limit' must be between 1 and {MAX_IGDB_SEARCH_RESULTS}.")
    return raw_q, limit, raw_company


class AnalysisRunner:
    """Stateful runner for a single in-flight analysis process."""

    def __init__(
        self,
        project_root: Path,
        pipeline_path: Path,
        output_dir: Path,
        full_dataset_dir: Path,
        demo_dataset_dir: Path,
    ) -> None:
        self.project_root = project_root
        self.pipeline_path = pipeline_path
        self.output_dir = output_dir
        self.full_dataset_dir = full_dataset_dir
        self.demo_dataset_dir = demo_dataset_dir

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_exit_code: int | None = None
        self._last_error: str | None = None
        self._last_started_at: str | None = None
        self._last_finished_at: str | None = None
        self._output_tail: deque[str] = deque(maxlen=200)
        self._last_params: dict[str, Any] = {}

    def start(self, run_params: dict[str, Any]) -> tuple[bool, str]:
        """Start analysis if no job is currently running."""
        with self._lock:
            if self._running:
                return False, "Analysis is already running."

            self._running = True
            self._last_exit_code = None
            self._last_error = None
            self._last_started_at = utc_now_iso()
            self._last_finished_at = None
            self._last_params = dict(run_params)
            self._output_tail.clear()

            self._thread = threading.Thread(target=self._run_pipeline, args=(dict(run_params),), daemon=True)
            self._thread.start()
            return True, "Analysis started."

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "last_exit_code": self._last_exit_code,
                "last_error": self._last_error,
                "last_started_at": self._last_started_at,
                "last_finished_at": self._last_finished_at,
                "last_params": dict(self._last_params),
                "output_tail": list(self._output_tail),
                "dataset_modes": {
                    "default": "full",
                    "full_root": str(self.full_dataset_dir),
                    "demo_root": str(self.demo_dataset_dir),
                    "demo_available": bool(self.demo_dataset_dir.exists() and self.demo_dataset_dir.is_dir()),
                },
            }

    def _resolve_dataset_root(self, dataset_mode: str) -> Path:
        mode = str(dataset_mode or "full").strip().lower()
        if mode == "demo":
            return self.demo_dataset_dir
        return self.full_dataset_dir

    def _append_output(self, line: str) -> None:
        with self._lock:
            self._output_tail.append(line.rstrip("\n"))

    def _finish(self, exit_code: int, error: str | None) -> None:
        with self._lock:
            self._running = False
            self._last_exit_code = exit_code
            self._last_error = error
            self._last_finished_at = utc_now_iso()

    def _run_pipeline(self, run_params: dict[str, Any]) -> None:
        dataset_mode = str(run_params.get("dataset_mode", "full"))
        dataset_root = self._resolve_dataset_root(dataset_mode)
        cmd = [
            sys.executable,
            str(self.pipeline_path),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(self.output_dir),
            "--tsne-perplexity",
            str(run_params["tsne_perplexity"]),
            "--umap-n-neighbors",
            str(run_params["umap_n_neighbors"]),
            "--umap-min-dist",
            str(run_params["umap_min_dist"]),
            "--batch-size",
            str(run_params["batch_size"]),
            "--max-images",
            str(run_params["max_images"]),
            "--seed",
            str(run_params["seed"]),
            "--device",
            str(run_params["device"]),
            "--clip-backend",
            str(run_params["clip_backend"]),
            "--model-name",
            str(run_params["model_name"]),
        ]
        if DEFAULT_STYLE_ADAPTER_CKPT.exists():
            cmd.extend(["--style-adapter-checkpoint", str(DEFAULT_STYLE_ADAPTER_CKPT)])
            self._append_output(f"[info] style adapter checkpoint: {DEFAULT_STYLE_ADAPTER_CKPT}\n")
        if DEFAULT_STYLE_PROMPTS_FILE.exists():
            cmd.extend(["--style-prompts-file", str(DEFAULT_STYLE_PROMPTS_FILE)])
            self._append_output(f"[info] style prompts file: {DEFAULT_STYLE_PROMPTS_FILE}\n")
        if DEFAULT_PROMPT_FOCUS_FILE.exists():
            cmd.extend(["--prompt-focus-file", str(DEFAULT_PROMPT_FOCUS_FILE)])
            self._append_output(f"[info] prompt focus file: {DEFAULT_PROMPT_FOCUS_FILE}\n")
        self._append_output(f"[info] dataset mode: {dataset_mode}\n")
        self._append_output(f"[info] dataset root: {dataset_root}\n")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                self._append_output(line)

            exit_code = process.wait()
            self._finish(exit_code=exit_code, error=None)
        except Exception as exc:  # pragma: no cover - runtime guard
            self._finish(exit_code=1, error=str(exc))


class Phase3Runner:
    """Stateful runner for a single in-flight phase-3 analysis process."""

    def __init__(
        self,
        project_root: Path,
        phase3_script_path: Path,
        output_dir: Path,
        full_dataset_dir: Path,
        demo_dataset_dir: Path,
    ) -> None:
        self.project_root = project_root
        self.phase3_script_path = phase3_script_path
        self.output_dir = output_dir
        self.full_dataset_dir = full_dataset_dir
        self.demo_dataset_dir = demo_dataset_dir

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_exit_code: int | None = None
        self._last_error: str | None = None
        self._last_started_at: str | None = None
        self._last_finished_at: str | None = None
        self._output_tail: deque[str] = deque(maxlen=260)
        self._last_params: dict[str, Any] = {}

    def start(self, run_params: dict[str, Any]) -> tuple[bool, str]:
        with self._lock:
            if self._running:
                return False, "Phase-3 analysis is already running."

            self._running = True
            self._last_exit_code = None
            self._last_error = None
            self._last_started_at = utc_now_iso()
            self._last_finished_at = None
            self._last_params = dict(run_params)
            self._output_tail.clear()

            self._thread = threading.Thread(target=self._run_phase3, args=(dict(run_params),), daemon=True)
            self._thread.start()
            return True, "Phase-3 analysis started."

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "last_exit_code": self._last_exit_code,
                "last_error": self._last_error,
                "last_started_at": self._last_started_at,
                "last_finished_at": self._last_finished_at,
                "last_params": dict(self._last_params),
                "output_tail": list(self._output_tail),
                "dataset_modes": {
                    "default": "full",
                    "full_root": str(self.full_dataset_dir),
                    "demo_root": str(self.demo_dataset_dir),
                    "demo_available": bool(self.demo_dataset_dir.exists() and self.demo_dataset_dir.is_dir()),
                },
            }

    def _resolve_dataset_root(self, dataset_mode: str) -> Path:
        mode = str(dataset_mode or "full").strip().lower()
        if mode == "demo":
            return self.demo_dataset_dir
        return self.full_dataset_dir

    def _append_output(self, line: str) -> None:
        with self._lock:
            self._output_tail.append(line.rstrip("\n"))

    def _finish(self, exit_code: int, error: str | None) -> None:
        with self._lock:
            self._running = False
            self._last_exit_code = exit_code
            self._last_error = error
            self._last_finished_at = utc_now_iso()

    def _run_phase3(self, run_params: dict[str, Any]) -> None:
        dataset_mode = str(run_params.get("dataset_mode", "full"))
        dataset_root = self._resolve_dataset_root(dataset_mode)
        cmd = [
            sys.executable,
            str(self.phase3_script_path),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(self.output_dir),
            "--sample-size",
            str(run_params["sample_size"]),
            "--batch-size",
            str(run_params["batch_size"]),
            "--device",
            str(run_params["device"]),
            "--clip-backend",
            str(run_params["clip_backend"]),
            "--model-name",
            str(run_params["model_name"]),
            "--pca-levels",
            str(run_params["pca_levels"]),
            "--ari-seeds",
            str(run_params["ari_seeds"]),
            "--max-pairs-per-bucket",
            str(run_params["max_pairs_per_bucket"]),
        ]

        self._append_output(f"[info] dataset mode: {dataset_mode}\n")
        self._append_output(f"[info] dataset root: {dataset_root}\n")
        self._append_output(f"[info] phase3 output dir: {self.output_dir}\n")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                self._append_output(line)

            exit_code = process.wait()
            self._finish(exit_code=exit_code, error=None)
        except Exception as exc:  # pragma: no cover - runtime guard
            self._finish(exit_code=1, error=str(exc))


class IgdbFetchRunner:
    """Stateful runner for a single in-flight IGDB cover fetch process."""

    def __init__(
        self,
        project_root: Path,
        fetch_script_path: Path,
        dataset_dir: Path,
        mapping_csv_path: Path,
        report_json_path: Path,
        groups_csv_path: Path,
    ) -> None:
        self.project_root = project_root
        self.fetch_script_path = fetch_script_path
        self.dataset_dir = dataset_dir
        self.mapping_csv_path = mapping_csv_path
        self.report_json_path = report_json_path
        self.groups_csv_path = groups_csv_path

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_exit_code: int | None = None
        self._last_error: str | None = None
        self._last_started_at: str | None = None
        self._last_finished_at: str | None = None
        self._output_tail: deque[str] = deque(maxlen=240)
        self._last_params: dict[str, Any] = {}

    def start(self, run_params: dict[str, Any]) -> tuple[bool, str]:
        with self._lock:
            if self._running:
                return False, "IGDB cover fetch is already running."

            self._running = True
            self._last_exit_code = None
            self._last_error = None
            self._last_started_at = utc_now_iso()
            self._last_finished_at = None
            self._last_params = dict(run_params)
            self._output_tail.clear()

            self._thread = threading.Thread(target=self._run_fetch, args=(dict(run_params),), daemon=True)
            self._thread.start()
            return True, "IGDB cover fetch started."

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "last_exit_code": self._last_exit_code,
                "last_error": self._last_error,
                "last_started_at": self._last_started_at,
                "last_finished_at": self._last_finished_at,
                "last_params": dict(self._last_params),
                "output_tail": list(self._output_tail),
            }

    def search_games(self, query: str, limit: int, company: str = "") -> dict[str, Any]:
        cmd = [
            sys.executable,
            str(self.fetch_script_path),
            "--search-limit",
            str(limit),
            "--request-timeout",
            "25",
            "--max-retries",
            "3",
        ]
        if query.strip():
            cmd.extend(["--search-query", query.strip()])
        if company.strip():
            cmd.extend(["--search-company", company.strip()])
        # Multi-studio lookups can take longer because each studio triggers additional IGDB queries.
        company_count = len([token for token in re.split(r"[,\n;]+", company) if token.strip()])
        computed_timeout = max(120, 45 * max(1, company_count) + int(limit * 1.5))
        search_timeout_seconds = min(900, computed_timeout)
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=search_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "IGDB search timed out. "
                f"Studios provided: {max(1, company_count)}. Requested limit: {limit}. "
                "Try fewer studios per search (for example 3-5), then run another search."
            ) from exc
        output = (result.stdout or "").strip()
        if result.returncode != 0:
            raise RuntimeError(output or f"search script exited with code {result.returncode}")
        if not output:
            return {"query": query, "count": 0, "results": []}
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return {"query": query, "count": 0, "results": []}
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid search response: {lines[-1]}") from exc

    def _append_output(self, line: str) -> None:
        with self._lock:
            self._output_tail.append(line.rstrip("\n"))

    def _finish(self, exit_code: int, error: str | None) -> None:
        with self._lock:
            self._running = False
            self._last_exit_code = exit_code
            self._last_error = error
            self._last_finished_at = utc_now_iso()

    def _run_fetch(self, run_params: dict[str, Any]) -> None:
        cmd = [
            sys.executable,
            str(self.fetch_script_path),
            "--dataset-root",
            str(self.dataset_dir),
            "--mapping-csv",
            str(self.mapping_csv_path),
            "--report-json",
            str(self.report_json_path),
            "--image-size",
            str(run_params["image_size"]),
            "--max-games",
            str(run_params["max_games"]),
            "--min-match-score",
            str(run_params["min_match_score"]),
            "--min-token-overlap",
            str(run_params["min_token_overlap"]),
            "--rate-limit-rps",
            "3.8",
            "--max-retries",
            "4",
            "--request-timeout",
            "25",
            "--dry-run" if run_params["dry_run"] else "--no-dry-run",
            "--allow-low-confidence" if run_params["allow_low_confidence"] else "--no-allow-low-confidence",
            "--strict-match-mode" if run_params["strict_match_mode"] else "--no-strict-match-mode",
            "--auto-clean-local-names" if run_params["auto_clean_local_names"] else "--no-auto-clean-local-names",
            "--skip-if-any-image" if run_params["skip_if_any_image"] else "--no-skip-if-any-image",
            "--overwrite" if run_params["overwrite"] else "--no-overwrite",
            "--seed-mode",
            str(run_params["seed_mode"]),
            "--seed-count",
            str(run_params["seed_count"]),
            "--seed-group-label",
            str(run_params["seed_group_label"]),
            "--seed-groups-file",
            str(self.groups_csv_path),
            "--seed-year-start",
            str(run_params["seed_year_start"]),
            "--seed-year-end",
            str(run_params["seed_year_end"]),
            "--seed-min-total-rating-count",
            str(run_params["seed_min_total_rating_count"]),
            "--seed-only" if run_params["seed_only"] else "--no-seed-only",
            "--seed-update-groups" if run_params["seed_update_groups"] else "--no-seed-update-groups",
            "--seed-write-mappings" if run_params["seed_write_mappings"] else "--no-seed-write-mappings",
        ]
        if run_params["seed_count"] > 0 and run_params["seed_mode"] == "genre":
            cmd.extend(["--seed-genre-name", str(run_params["seed_genre_name"])])
        if run_params["seed_game_ids"]:
            cmd.extend(["--seed-game-ids", ",".join(str(value) for value in run_params["seed_game_ids"])])
        if run_params["seed_game_names"]:
            cmd.extend(["--seed-game-names", "\n".join(str(value) for value in run_params["seed_game_names"])])
        if run_params["seed_exclude_genre_name"].strip():
            cmd.extend(["--seed-exclude-genre-name", str(run_params["seed_exclude_genre_name"])])

        self._append_output(f"[info] fetch script: {self.fetch_script_path}")
        self._append_output(f"[info] dataset root: {self.dataset_dir}")
        self._append_output(f"[info] mapping csv: {self.mapping_csv_path}")
        self._append_output(f"[info] report json: {self.report_json_path}")
        self._append_output(f"[info] groups csv: {self.groups_csv_path}")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                self._append_output(line)

            exit_code = process.wait()
            self._finish(exit_code=exit_code, error=None)
        except Exception as exc:  # pragma: no cover - runtime guard
            self._finish(exit_code=1, error=str(exc))


class AppHandler(SimpleHTTPRequestHandler):
    """Static file handler + tiny JSON API for analysis control."""

    runner: AnalysisRunner
    phase3_runner: Phase3Runner
    igdb_runner: IgdbFetchRunner

    def end_headers(self) -> None:  # noqa: D401
        """Disable browser caching for local dev to avoid stale JS/CSS/data."""
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, payload: dict[str, Any], status_code: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(self.runner.status())
            return
        if parsed.path == "/api/phase3-status":
            self._send_json(self.phase3_runner.status())
            return
        if parsed.path == "/api/igdb-status":
            self._send_json(self.igdb_runner.status())
            return
        if parsed.path == "/api/game-folders":
            try:
                folders = sorted(
                    [p.name for p in self.igdb_runner.dataset_dir.iterdir() if p.is_dir()],
                    key=lambda value: value.lower(),
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                self._send_json({"ok": False, "message": str(exc)}, status_code=500)
                return
            self._send_json({"ok": True, "folders": folders, "count": len(folders)})
            return
        if parsed.path == "/api/igdb-search-games":
            try:
                query, limit, company = parse_igdb_search_params(parse_qs(parsed.query, keep_blank_values=False))
            except ValueError as exc:
                self._send_json({"ok": False, "message": str(exc)}, status_code=400)
                return
            try:
                payload = self.igdb_runner.search_games(query=query, limit=limit, company=company)
            except Exception as exc:  # pragma: no cover - runtime guard
                self._send_json({"ok": False, "message": str(exc)}, status_code=500)
                return
            self._send_json({"ok": True, "query": query, "company": company, "data": payload})
            return

        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/run-analysis":
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            if content_length > MAX_REQUEST_BYTES:
                self._send_json(
                    {"ok": False, "message": f"Payload too large (>{MAX_REQUEST_BYTES} bytes)."},
                    status_code=413,
                )
                return

            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"ok": False, "message": "Invalid JSON payload."}, status_code=400)
                return

            try:
                run_params = parse_run_params(payload)
            except ValueError as exc:
                self._send_json({"ok": False, "message": str(exc)}, status_code=400)
                return

            if self.igdb_runner.status().get("running"):
                self._send_json(
                    {"ok": False, "message": "Cannot run analysis while IGDB cover fetch is running."},
                    status_code=409,
                )
                return
            if self.phase3_runner.status().get("running"):
                self._send_json(
                    {"ok": False, "message": "Cannot run analysis while phase-3 analysis is running."},
                    status_code=409,
                )
                return

            started, message = self.runner.start(run_params)
            status_code = 202 if started else 409
            self._send_json(
                {"ok": started, "message": message, "params": run_params},
                status_code=status_code,
            )
            return

        if parsed.path == "/api/run-phase3":
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            if content_length > MAX_REQUEST_BYTES:
                self._send_json(
                    {"ok": False, "message": f"Payload too large (>{MAX_REQUEST_BYTES} bytes)."},
                    status_code=413,
                )
                return

            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"ok": False, "message": "Invalid JSON payload."}, status_code=400)
                return

            try:
                run_params = parse_phase3_run_params(payload)
            except ValueError as exc:
                self._send_json({"ok": False, "message": str(exc)}, status_code=400)
                return

            if self.runner.status().get("running"):
                self._send_json(
                    {"ok": False, "message": "Cannot run phase-3 analysis while main analysis is running."},
                    status_code=409,
                )
                return
            if self.igdb_runner.status().get("running"):
                self._send_json(
                    {"ok": False, "message": "Cannot run phase-3 analysis while IGDB cover fetch is running."},
                    status_code=409,
                )
                return

            started, message = self.phase3_runner.start(run_params)
            status_code = 202 if started else 409
            self._send_json(
                {"ok": started, "message": message, "params": run_params},
                status_code=status_code,
            )
            return

        if parsed.path == "/api/fetch-igdb-covers":
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            if content_length > MAX_REQUEST_BYTES:
                self._send_json(
                    {"ok": False, "message": f"Payload too large (>{MAX_REQUEST_BYTES} bytes)."},
                    status_code=413,
                )
                return

            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"ok": False, "message": "Invalid JSON payload."}, status_code=400)
                return

            try:
                run_params = parse_igdb_fetch_params(payload)
            except ValueError as exc:
                self._send_json({"ok": False, "message": str(exc)}, status_code=400)
                return

            if self.runner.status().get("running"):
                self._send_json(
                    {"ok": False, "message": "Cannot fetch IGDB covers while analysis is running."},
                    status_code=409,
                )
                return
            if self.phase3_runner.status().get("running"):
                self._send_json(
                    {"ok": False, "message": "Cannot fetch IGDB covers while phase-3 analysis is running."},
                    status_code=409,
                )
                return

            started, message = self.igdb_runner.start(run_params)
            status_code = 202 if started else 409
            self._send_json(
                {"ok": started, "message": message, "params": run_params},
                status_code=status_code,
            )
            return

        self._send_json({"ok": False, "message": "Not Found"}, status_code=404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve CLIP explorer and run analysis from UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_DIR,
        help=f"Main dataset root used for analysis and IGDB operations (default: {DATASET_DIR}).",
    )
    parser.add_argument(
        "--demo-dataset-root",
        type=Path,
        default=DEMO_DATASET_DIR,
        help=f"Demo dataset root used when dataset_mode=demo (default: {DEMO_DATASET_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not WEB_DIR.exists():
        raise SystemExit(f"[ERROR] web directory not found: {WEB_DIR}")
    if not PIPELINE_PATH.exists():
        raise SystemExit(f"[ERROR] pipeline script not found: {PIPELINE_PATH}")
    if not IGDB_FETCH_PATH.exists():
        raise SystemExit(f"[ERROR] IGDB fetch script not found: {IGDB_FETCH_PATH}")
    if not PHASE3_PATH.exists():
        raise SystemExit(f"[ERROR] phase-3 script not found: {PHASE3_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runner = AnalysisRunner(
        project_root=PROJECT_ROOT,
        pipeline_path=PIPELINE_PATH,
        output_dir=OUTPUT_DIR,
        full_dataset_dir=args.dataset_root.expanduser().resolve(),
        demo_dataset_dir=args.demo_dataset_root.expanduser().resolve(),
    )
    phase3_runner = Phase3Runner(
        project_root=PROJECT_ROOT,
        phase3_script_path=PHASE3_PATH,
        output_dir=PHASE3_OUTPUT_DIR,
        full_dataset_dir=args.dataset_root.expanduser().resolve(),
        demo_dataset_dir=args.demo_dataset_root.expanduser().resolve(),
    )
    igdb_runner = IgdbFetchRunner(
        project_root=PROJECT_ROOT,
        fetch_script_path=IGDB_FETCH_PATH,
        dataset_dir=args.dataset_root.expanduser().resolve(),
        mapping_csv_path=DEFAULT_IGDB_MAPPING_CSV,
        report_json_path=DEFAULT_IGDB_REPORT_JSON,
        groups_csv_path=DEFAULT_GAME_GROUPS_CSV,
    )

    handler = partial(AppHandler, directory=str(WEB_DIR))
    AppHandler.runner = runner
    AppHandler.phase3_runner = phase3_runner
    AppHandler.igdb_runner = igdb_runner

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving app on http://{args.host}:{args.port}")
    print(f"Full dataset root: {args.dataset_root.expanduser().resolve()}")
    print(f"Demo dataset root: {args.demo_dataset_root.expanduser().resolve()}")
    print("Use 'Run Analysis' or IGDB buttons in the UI to start background jobs.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
