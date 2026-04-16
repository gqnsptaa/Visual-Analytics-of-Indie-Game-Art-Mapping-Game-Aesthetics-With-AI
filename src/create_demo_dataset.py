#!/usr/bin/env python3
"""Build a balanced demo dataset for fast prompt iteration.

Creates a new dataset root with a fixed number of AAA and indie games,
copying or linking a small number of cover images per game.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATASET = PROJECT_ROOT / "indie_games_dataset"
DEFAULT_GROUPS_CSV = PROJECT_ROOT / "src" / "game_groups.csv"
DEFAULT_OUTPUT_DATASET = PROJECT_ROOT / "indie_games_dataset_demo_100x2"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


class DemoBuildError(RuntimeError):
    """Controlled, user-facing errors for demo dataset build."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced demo dataset for faster CLIP analysis runs."
    )
    parser.add_argument(
        "--source-dataset-root",
        type=Path,
        default=DEFAULT_SOURCE_DATASET,
        help=f"Input dataset root with one folder per game (default: {DEFAULT_SOURCE_DATASET}).",
    )
    parser.add_argument(
        "--game-groups-file",
        type=Path,
        default=DEFAULT_GROUPS_CSV,
        help=f"CSV with columns game,group (default: {DEFAULT_GROUPS_CSV}).",
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        default=DEFAULT_OUTPUT_DATASET,
        help=f"Output demo dataset root (default: {DEFAULT_OUTPUT_DATASET}).",
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=100,
        help="Number of games to select per group (default: 100).",
    )
    parser.add_argument(
        "--images-per-game",
        type=int,
        default=1,
        help="How many images to include per selected game (default: 1).",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="Use symlinks (fast, small) or physical copies (default: symlink).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace an existing output dataset root if present (default: true).",
    )
    return parser.parse_args()


def load_group_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        raise DemoBuildError(f"Game groups file not found: {path}")
    mapping: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = {h.strip().lower() for h in (reader.fieldnames or [])}
            if "game" not in fields or "group" not in fields:
                raise DemoBuildError(f"Expected columns 'game' and 'group' in: {path}")
            for row in reader:
                game = str(row.get("game", "")).strip()
                group = str(row.get("group", "")).strip().lower()
                if not game or group not in {"aaa", "indie"}:
                    continue
                mapping[game] = group
    except DemoBuildError:
        raise
    except Exception as exc:
        raise DemoBuildError(f"Failed reading groups file {path}: {exc}") from exc
    return mapping


def list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    images = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return images


def ensure_empty_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise DemoBuildError(
                f"Output dataset already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def run() -> int:
    args = parse_args()
    source_root = args.source_dataset_root.expanduser().resolve()
    output_root = args.output_dataset_root.expanduser().resolve()
    groups_file = args.game_groups_file.expanduser().resolve()

    if args.per_group < 1:
        raise DemoBuildError("--per-group must be >= 1.")
    if args.images_per_game < 1:
        raise DemoBuildError("--images-per-game must be >= 1.")
    if not source_root.exists() or not source_root.is_dir():
        raise DemoBuildError(f"Source dataset root not found or not a directory: {source_root}")

    group_map = load_group_mapping(groups_file)
    rng = random.Random(args.seed)

    candidates: dict[str, list[Path]] = defaultdict(list)
    for game_dir in sorted(source_root.iterdir()):
        if not game_dir.is_dir():
            continue
        group = group_map.get(game_dir.name)
        if group not in {"aaa", "indie"}:
            continue
        if not list_images(game_dir):
            continue
        candidates[group].append(game_dir)

    for group in ("aaa", "indie"):
        if len(candidates[group]) < args.per_group:
            raise DemoBuildError(
                f"Not enough '{group}' games with images: found {len(candidates[group])}, "
                f"needed {args.per_group}."
            )

    selected: dict[str, list[Path]] = {}
    for group in ("aaa", "indie"):
        pool = list(candidates[group])
        rng.shuffle(pool)
        selected[group] = sorted(pool[: args.per_group], key=lambda p: p.name.lower())

    ensure_empty_output(output_root, overwrite=args.overwrite)

    total_links = 0
    for group in ("aaa", "indie"):
        for game_dir in selected[group]:
            dest_game_dir = output_root / game_dir.name
            dest_game_dir.mkdir(parents=True, exist_ok=True)
            images = list_images(game_dir)[: args.images_per_game]
            for src_img in images:
                dst_img = dest_game_dir / src_img.name
                link_or_copy(src_img, dst_img, args.link_mode)
                total_links += 1

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_root": str(source_root),
        "output_dataset_root": str(output_root),
        "game_groups_file": str(groups_file),
        "per_group": int(args.per_group),
        "images_per_game": int(args.images_per_game),
        "link_mode": args.link_mode,
        "seed": int(args.seed),
        "counts": {
            "aaa": len(selected["aaa"]),
            "indie": len(selected["indie"]),
            "total_games": len(selected["aaa"]) + len(selected["indie"]),
            "total_images": int(total_links),
        },
        "games": {
            "aaa": [p.name for p in selected["aaa"]],
            "indie": [p.name for p in selected["indie"]],
        },
    }
    manifest_path = output_root / "demo_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Created demo dataset: {output_root}")
    print(
        "Counts: "
        f"AAA={manifest['counts']['aaa']}, "
        f"Indie={manifest['counts']['indie']}, "
        f"Images={manifest['counts']['total_images']}"
    )
    print(f"Manifest: {manifest_path}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except DemoBuildError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
