#!/usr/bin/env python3
"""Local web server for CLIP explorer with one-click pipeline execution.

Serves the `web/` app and provides:
- GET /api/status
- POST /api/run-analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
PIPELINE_PATH = PROJECT_ROOT / "src" / "clip_indie_pipeline.py"
OUTPUT_DIR = WEB_DIR / "data"
DEFAULT_STYLE_ADAPTER_CKPT = PROJECT_ROOT / "training_outputs" / "style_adapter" / "best_style_adapter.pt"
MAX_REQUEST_BYTES = 64 * 1024


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
    }

    model_name = str(data.get("model_name", "ViT-B/32")).strip()
    if len(model_name) == 0 or len(model_name) > 128:
        raise ValueError("'model_name' must be a non-empty string shorter than 129 characters.")
    normalized["model_name"] = model_name
    return normalized


class AnalysisRunner:
    """Stateful runner for a single in-flight analysis process."""

    def __init__(self, project_root: Path, pipeline_path: Path, output_dir: Path) -> None:
        self.project_root = project_root
        self.pipeline_path = pipeline_path
        self.output_dir = output_dir

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
            }

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
        cmd = [
            sys.executable,
            str(self.pipeline_path),
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

            started, message = self.runner.start(run_params)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not WEB_DIR.exists():
        raise SystemExit(f"[ERROR] web directory not found: {WEB_DIR}")
    if not PIPELINE_PATH.exists():
        raise SystemExit(f"[ERROR] pipeline script not found: {PIPELINE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runner = AnalysisRunner(
        project_root=PROJECT_ROOT,
        pipeline_path=PIPELINE_PATH,
        output_dir=OUTPUT_DIR,
    )

    handler = partial(AppHandler, directory=str(WEB_DIR))
    AppHandler.runner = runner

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving app on http://{args.host}:{args.port}")
    print("Use the 'Run Analysis' button in the UI to start processing.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
