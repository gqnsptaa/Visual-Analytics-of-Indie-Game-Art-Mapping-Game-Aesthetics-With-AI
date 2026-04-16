#!/usr/bin/env python3
"""Compare Phase-2 centroid-distance metrics by IGDB genre.

This script joins:
1) per-image Phase-2 distance rows (AAA/Indie, distance to own-group centroid)
2) local_game -> IGDB game id mappings
3) IGDB game genres

Then it computes the same summary metrics used in Phase-2, but per genre.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib import error as urlerror
from urllib import request as urlrequest

IGDB_BASE_URL = "https://api.igdb.com/v4"
DEFAULT_DISTANCE_CSV = Path("web/data/phase2_overlap/distance_to_centroid_rows.csv")
DEFAULT_MAPPING_CSV = Path("src/igdb_game_mappings.csv")
DEFAULT_TOKEN_CACHE = Path("web/data/igdb_token_cache.json")
DEFAULT_OUTPUT_DIR = Path("web/data/phase2_overlap")


class GenreComparisonError(RuntimeError):
    """Raised for user-facing failures."""


def build_ssl_context() -> ssl.SSLContext:
    """Create SSL context with optional certifi CA bundle fallback."""
    ctx = ssl.create_default_context()
    env_cafile = os.getenv("SSL_CERT_FILE", "").strip()
    if env_cafile:
        try:
            ctx.load_verify_locations(cafile=env_cafile)
            return ctx
        except Exception:
            pass
    try:
        import certifi  # type: ignore

        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        pass
    return ctx


SSL_CONTEXT = build_ssl_context()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join Phase-2 centroid distances with IGDB genres and compute per-genre AAA/Indie comparisons."
    )
    parser.add_argument(
        "--distance-csv",
        type=Path,
        default=DEFAULT_DISTANCE_CSV,
        help=f"CSV from phase2_overlap_density_analysis.py (default: {DEFAULT_DISTANCE_CSV}).",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help=f"Mapping CSV with local_game -> igdb_game_id (default: {DEFAULT_MAPPING_CSV}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for genre comparison reports (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Sample size to analyze (0 = use max available sample size in distance CSV).",
    )
    parser.add_argument(
        "--min-images-per-group",
        type=int,
        default=20,
        help="Minimum images in each group (AAA and Indie) required for a genre to be included.",
    )
    parser.add_argument(
        "--top-n-genres",
        type=int,
        default=50,
        help="Maximum number of genres to keep after sorting by total images (0 = keep all).",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=os.getenv("IGDB_CLIENT_ID", "").strip(),
        help="IGDB/Twitch client id (default: IGDB_CLIENT_ID env var or token cache).",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default=os.getenv("IGDB_ACCESS_TOKEN", "").strip(),
        help="IGDB bearer token (default: IGDB_ACCESS_TOKEN env var or token cache).",
    )
    parser.add_argument(
        "--token-cache",
        type=Path,
        default=DEFAULT_TOKEN_CACHE,
        help=f"Token cache JSON (default: {DEFAULT_TOKEN_CACHE}).",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=4, help="Max retries for transient HTTP failures.")
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=3.5,
        help="Rate limit for IGDB requests (keep <= 4.0).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    client_id = args.client_id.strip()
    access_token = args.access_token.strip()

    if client_id and access_token:
        return client_id, access_token

    cache = load_json(args.token_cache)
    cached_client = str(cache.get("client_id") or "").strip()
    cached_token = str(cache.get("access_token") or "").strip()
    cached_exp = float(cache.get("expires_at_unix") or 0.0)
    now = time.time()

    if cached_client and not client_id:
        client_id = cached_client
    if cached_token and not access_token and cached_exp > now + 120:
        access_token = cached_token

    if not client_id or not access_token:
        raise GenreComparisonError(
            "Missing IGDB credentials. Provide --client-id and --access-token, "
            "or set IGDB_CLIENT_ID/IGDB_ACCESS_TOKEN, or use a valid token cache."
        )

    return client_id, access_token


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 0.0 if requests_per_second <= 0 else (1.0 / requests_per_second)
        self.next_time = 0.0

    def wait(self) -> None:
        if self.interval <= 0.0:
            return
        now = time.time()
        if now < self.next_time:
            time.sleep(self.next_time - now)
        self.next_time = max(now, self.next_time) + self.interval


def igdb_post(
    endpoint: str,
    query: str,
    *,
    client_id: str,
    access_token: str,
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> list[dict]:
    url = f"{IGDB_BASE_URL}/{endpoint.lstrip('/')}"
    data = query.encode("utf-8")
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "text/plain",
        "Accept": "application/json",
    }

    for attempt in range(max(1, int(max_retries)) + 1):
        rate_limiter.wait()
        req = urlrequest.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
                status = getattr(resp, "status", 200)
        except urlerror.HTTPError as exc:
            status = exc.code
            payload = exc.read().decode("utf-8", errors="replace")
        except urlerror.URLError as exc:
            if attempt < max_retries:
                time.sleep(min(8.0, 1.0 * (2**attempt)))
                continue
            raise GenreComparisonError(f"Network request failed for {url}: {exc}") from exc

        if 200 <= status < 300:
            try:
                rows = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise GenreComparisonError(f"Invalid JSON from IGDB endpoint={endpoint}: {payload[:400]}") from exc
            if not isinstance(rows, list):
                raise GenreComparisonError(f"Unexpected IGDB response shape for endpoint={endpoint}: {payload[:400]}")
            return [row for row in rows if isinstance(row, dict)]

        if status in {429, 500, 502, 503, 504} and attempt < max_retries:
            time.sleep(min(8.0, 1.0 * (2**attempt)))
            continue

        raise GenreComparisonError(f"IGDB request failed endpoint={endpoint} HTTP {status}: {payload[:400]}")

    raise GenreComparisonError(f"IGDB request failed after retries endpoint={endpoint}")


def chunked(values: Iterable[int], size: int) -> list[list[int]]:
    bucket: list[int] = []
    out: list[list[int]] = []
    for value in values:
        bucket.append(int(value))
        if len(bucket) >= size:
            out.append(bucket)
            bucket = []
    if bucket:
        out.append(bucket)
    return out


def read_distance_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise GenreComparisonError(f"Distance CSV not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                sample_size = int(row.get("sample_size", "0"))
                game = str(row.get("game", "")).strip()
                group = str(row.get("group", "")).strip().lower()
                dist = float(row.get("distance_to_group_centroid", "nan"))
            except Exception:
                continue
            if not game or group not in {"aaa", "indie"}:
                continue
            if not (dist == dist):  # NaN check
                continue
            rows.append(
                {
                    "sample_size": sample_size,
                    "game": game,
                    "group": group,
                    "distance": dist,
                }
            )
    if not rows:
        raise GenreComparisonError(f"No usable rows in distance CSV: {path}")
    return rows


def read_mapping(path: Path) -> dict[str, int]:
    if not path.exists():
        raise GenreComparisonError(f"Mapping CSV not found: {path}")
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            game = str(row.get("local_game", "")).strip()
            raw_id = str(row.get("igdb_game_id", "")).strip()
            if not game or not raw_id.isdigit():
                continue
            out[game] = int(raw_id)
    if not out:
        raise GenreComparisonError(f"No valid local_game -> igdb_game_id rows in: {path}")
    return out


def fetch_games_genres(
    game_ids: list[int],
    *,
    client_id: str,
    access_token: str,
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for group in chunked(game_ids, 400):
        ids_str = ",".join(str(gid) for gid in group)
        query = f"fields id,genres; where id = ({ids_str}); limit {len(group)};"
        rows = igdb_post(
            "games",
            query,
            client_id=client_id,
            access_token=access_token,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        for row in rows:
            gid = int(row.get("id") or 0)
            if gid <= 0:
                continue
            raw_genres = row.get("genres")
            genres: list[int] = []
            if isinstance(raw_genres, list):
                for item in raw_genres:
                    try:
                        genre_id = int(item)
                    except Exception:
                        continue
                    if genre_id > 0:
                        genres.append(genre_id)
            out[gid] = sorted(set(genres))
    return out


def fetch_genre_names(
    genre_ids: list[int],
    *,
    client_id: str,
    access_token: str,
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> dict[int, str]:
    out: dict[int, str] = {}
    if not genre_ids:
        return out
    for group in chunked(genre_ids, 400):
        ids_str = ",".join(str(gid) for gid in group)
        query = f"fields id,name; where id = ({ids_str}); limit {len(group)};"
        rows = igdb_post(
            "genres",
            query,
            client_id=client_id,
            access_token=access_token,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        for row in rows:
            gid = int(row.get("id") or 0)
            name = str(row.get("name") or "").strip()
            if gid > 0 and name:
                out[gid] = name
    return out


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def run() -> int:
    args = parse_args()
    client_id, access_token = resolve_credentials(args)

    all_rows = read_distance_rows(args.distance_csv)
    sample_sizes = sorted({int(r["sample_size"]) for r in all_rows})
    if not sample_sizes:
        raise GenreComparisonError("No sample_size values found in distance CSV.")

    chosen_size = int(args.sample_size) if args.sample_size > 0 else int(max(sample_sizes))
    selected = [r for r in all_rows if int(r["sample_size"]) == chosen_size]
    if not selected:
        raise GenreComparisonError(
            f"Requested sample size {chosen_size} not present in distance CSV. Available: {sample_sizes}"
        )

    mapping = read_mapping(args.mapping_csv)

    unique_games = sorted({str(r["game"]) for r in selected})
    game_to_igdb: dict[str, int] = {}
    for game in unique_games:
        gid = mapping.get(game)
        if isinstance(gid, int) and gid > 0:
            game_to_igdb[game] = gid

    if not game_to_igdb:
        raise GenreComparisonError("No games in the selected sample had IGDB ids in the mapping CSV.")

    rate_limiter = RateLimiter(max(0.1, float(args.requests_per_second)))
    igdb_ids = sorted(set(game_to_igdb.values()))
    game_id_to_genre_ids = fetch_games_genres(
        igdb_ids,
        client_id=client_id,
        access_token=access_token,
        timeout=float(args.timeout),
        max_retries=int(args.max_retries),
        rate_limiter=rate_limiter,
    )

    all_genre_ids = sorted({gid for vals in game_id_to_genre_ids.values() for gid in vals})
    genre_id_to_name = fetch_genre_names(
        all_genre_ids,
        client_id=client_id,
        access_token=access_token,
        timeout=float(args.timeout),
        max_retries=int(args.max_retries),
        rate_limiter=rate_limiter,
    )

    game_to_genres: dict[str, list[str]] = {}
    for game, igdb_id in game_to_igdb.items():
        genre_names = [
            genre_id_to_name[g]
            for g in game_id_to_genre_ids.get(igdb_id, [])
            if isinstance(genre_id_to_name.get(g), str)
        ]
        genre_names = sorted({name for name in genre_names if name})
        if genre_names:
            game_to_genres[game] = genre_names

    genre_group_dist: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows_with_genre = 0
    rows_with_mapping = 0
    for row in selected:
        game = str(row["game"])
        group = str(row["group"])
        dist = float(row["distance"])
        if game in game_to_igdb:
            rows_with_mapping += 1
        genres = game_to_genres.get(game, [])
        if not genres:
            continue
        rows_with_genre += 1
        for genre in genres:
            genre_group_dist[genre][group].append(dist)

    min_group_n = max(1, int(args.min_images_per_group))
    summary_rows: list[dict] = []
    dropped_by_threshold = 0
    for genre, grouped in genre_group_dist.items():
        aaa_values = grouped.get("aaa", [])
        indie_values = grouped.get("indie", [])
        if len(aaa_values) < min_group_n or len(indie_values) < min_group_n:
            dropped_by_threshold += 1
            continue
        aaa_mean = mean(aaa_values)
        indie_mean = mean(indie_values)
        summary_rows.append(
            {
                "genre": genre,
                "aaa_n_images": len(aaa_values),
                "indie_n_images": len(indie_values),
                "total_images": len(aaa_values) + len(indie_values),
                "aaa_mean_dist": aaa_mean,
                "indie_mean_dist": indie_mean,
                "indie_aaa_ratio": float(indie_mean / max(1e-12, aaa_mean)),
                "mean_gap_indie_minus_aaa": float(indie_mean - aaa_mean),
            }
        )

    summary_rows.sort(key=lambda r: (-int(r["total_images"]), -abs(float(r["mean_gap_indie_minus_aaa"])), str(r["genre"])))

    top_n = int(args.top_n_genres)
    if top_n > 0:
        summary_rows = summary_rows[:top_n]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / f"genre_distance_comparison_n{chosen_size}.csv"
    output_json = args.output_dir / f"genre_distance_comparison_n{chosen_size}.json"

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "genre",
                "aaa_n_images",
                "indie_n_images",
                "total_images",
                "aaa_mean_dist",
                "indie_mean_dist",
                "indie_aaa_ratio",
                "mean_gap_indie_minus_aaa",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "distance_csv": str(args.distance_csv),
            "mapping_csv": str(args.mapping_csv),
            "sample_size": chosen_size,
            "min_images_per_group": min_group_n,
            "top_n_genres": top_n,
        },
        "coverage": {
            "distance_rows_total_for_sample": len(selected),
            "distance_rows_with_mapping": rows_with_mapping,
            "distance_rows_with_genres": rows_with_genre,
            "unique_games_in_sample": len(unique_games),
            "unique_games_with_mapping": len(game_to_igdb),
            "unique_games_with_genres": len(game_to_genres),
            "unique_igdb_genre_ids": len(all_genre_ids),
            "named_genres_returned": len(genre_id_to_name),
            "genres_dropped_by_min_images_per_group": dropped_by_threshold,
            "genres_in_output": len(summary_rows),
        },
        "outputs": {
            "csv": str(output_csv),
            "json": str(output_json),
        },
        "rows": summary_rows,
    }
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")
    print(
        "Coverage: "
        f"rows_with_mapping={rows_with_mapping}/{len(selected)} | "
        f"rows_with_genres={rows_with_genre}/{len(selected)} | "
        f"genres_out={len(summary_rows)}"
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except GenreComparisonError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
