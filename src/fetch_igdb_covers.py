#!/usr/bin/env python3
"""Fetch game cover images from IGDB into local dataset folders.

This script is designed for backend/local use (no browser CORS issues).
It supports:
- Twitch OAuth app token (client credentials)
- IGDB game lookup by folder name or explicit mapping
- Cover image download from images.igdb.com
- Rate limiting, retries, and structured report output
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "indie_games_dataset"
TOKEN_ENDPOINT = "https://id.twitch.tv/oauth2/token"
IGDB_BASE_URL = "https://api.igdb.com/v4"
IMAGE_BASE_URL = "https://images.igdb.com/igdb/image/upload"
DEFAULT_TOKEN_CACHE = PROJECT_ROOT / "web" / "data" / "igdb_token_cache.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "web" / "data" / "igdb_cover_fetch_report.json"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "src" / "igdb_game_mappings.csv"
MAX_IGDB_SEARCH_RESULTS = 500
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
TOKEN_STOPWORDS = {
    "the",
    "of",
    "and",
    "edition",
    "deluxe",
    "definitive",
    "ultimate",
    "complete",
    "collection",
    "game",
    "cover",
}


def build_ssl_context() -> ssl.SSLContext:
    """Create SSL context with optional certifi CA bundle fallback."""
    ctx = ssl.create_default_context()
    env_cafile = os.getenv("SSL_CERT_FILE", "").strip()
    if env_cafile:
        try:
            ctx.load_verify_locations(cafile=env_cafile)
            return ctx
        except Exception:
            # Fall back to system/certifi if env path is invalid.
            pass

    try:
        import certifi  # type: ignore

        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        # If certifi is missing, system trust store is used.
        pass
    return ctx


SSL_CONTEXT = build_ssl_context()


class FetchError(RuntimeError):
    """Controlled, user-facing fetch error."""


@dataclass
class GameMapping:
    local_game: str
    igdb_game_id: int | None
    igdb_game_name: str | None


@dataclass
class CandidateGame:
    igdb_id: int
    name: str
    version_parent: int | None
    cover_image_id: str | None
    score: float
    token_overlap: float = 0.0


class RateLimiter:
    """Simple fixed-rate limiter for request pacing."""

    def __init__(self, requests_per_second: float) -> None:
        self.min_interval = 1.0 / max(0.1, requests_per_second)
        self._last_time = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_time = time.monotonic()


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download IGDB game covers into local dataset folders.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root dataset folder where each subfolder is a game label.",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=os.getenv("IGDB_CLIENT_ID", "").strip(),
        help="Twitch/IGDB client id (default: IGDB_CLIENT_ID env var).",
    )
    parser.add_argument(
        "--client-secret",
        type=str,
        default=os.getenv("IGDB_CLIENT_SECRET", "").strip(),
        help="Twitch/IGDB client secret (default: IGDB_CLIENT_SECRET env var).",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default=os.getenv("IGDB_ACCESS_TOKEN", "").strip(),
        help="Optional pre-generated app access token (default: IGDB_ACCESS_TOKEN env var).",
    )
    parser.add_argument(
        "--token-cache",
        type=Path,
        default=DEFAULT_TOKEN_CACHE,
        help="Path to cache OAuth token JSON.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help=(
            "Optional CSV mapping for disambiguation. "
            "Supported columns: local_game, igdb_game_id, igdb_game_name."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for JSON report output.",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default="cover_big",
        help="IGDB image size key (example: cover_big, 720p, 1080p).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Optional max number of game folders to process (0 = all).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=25.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retries per API/image request on transient errors.",
    )
    parser.add_argument(
        "--rate-limit-rps",
        type=float,
        default=3.8,
        help="Request rate cap (IGDB allows 4 req/sec).",
    )
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=0.72,
        help="Minimum name-match score to accept auto-matched search result.",
    )
    parser.add_argument(
        "--min-token-overlap",
        type=float,
        default=0.34,
        help="Minimum local-token overlap ratio in strict mode (0-1).",
    )
    parser.add_argument(
        "--strict-match-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require stronger token overlap and penalize ambiguous candidate names.",
    )
    parser.add_argument(
        "--auto-clean-local-names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean slug-like local folder names before IGDB search/matching.",
    )
    parser.add_argument(
        "--allow-low-confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow best match even if score < min-match-score.",
    )
    parser.add_argument(
        "--skip-if-any-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip game folder if it already contains any image file.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing destination file if same IGDB image_id is already present.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve matches without downloading files.",
    )
    parser.add_argument(
        "--seed-genre-name",
        type=str,
        default="",
        help="Optional IGDB genre name to auto-discover and create new game folders (example: Indie).",
    )
    parser.add_argument(
        "--seed-mode",
        type=str,
        default="genre",
        choices=("genre", "popular_year_range", "id_list", "name_list"),
        help="Seed strategy: 'genre', 'popular_year_range' (AAA approximation), 'id_list', or 'name_list'.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=0,
        help="Number of games to seed from --seed-genre-name (0 disables seeding).",
    )
    parser.add_argument(
        "--seed-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, after seeding process only seeded folders, not entire dataset root.",
    )
    parser.add_argument(
        "--seed-group-label",
        type=str,
        default="indie",
        help="Group label assigned to seeded games when --seed-update-groups is enabled.",
    )
    parser.add_argument(
        "--seed-groups-file",
        type=Path,
        default=Path("src/game_groups.csv"),
        help="CSV file to upsert seeded game group labels (columns: game,group).",
    )
    parser.add_argument(
        "--seed-update-groups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upsert seeded games into --seed-groups-file with --seed-group-label.",
    )
    parser.add_argument(
        "--seed-write-mappings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upsert seeded games into mapping CSV with IGDB id/name for deterministic future matches.",
    )
    parser.add_argument(
        "--seed-year-start",
        type=int,
        default=2010,
        help="Start year (inclusive) for 'popular_year_range' seed mode.",
    )
    parser.add_argument(
        "--seed-year-end",
        type=int,
        default=2020,
        help="End year (inclusive) for 'popular_year_range' seed mode.",
    )
    parser.add_argument(
        "--seed-min-total-rating-count",
        type=int,
        default=100,
        help="Minimum IGDB total_rating_count in 'popular_year_range' seed mode.",
    )
    parser.add_argument(
        "--seed-exclude-genre-name",
        type=str,
        default="Indie",
        help="Optional genre name to exclude in 'popular_year_range' mode (default: Indie).",
    )
    parser.add_argument(
        "--seed-game-ids",
        type=str,
        default="",
        help="Comma-separated IGDB game IDs used when --seed-mode id_list.",
    )
    parser.add_argument(
        "--seed-game-names",
        type=str,
        default="",
        help="List of game names for --seed-mode name_list (newline, comma, or semicolon separated).",
    )
    parser.add_argument(
        "--seed-game-list-file",
        type=Path,
        default=None,
        help="Optional text file of game names (one per line) for --seed-mode name_list.",
    )
    parser.add_argument(
        "--search-query",
        type=str,
        default="",
        help="Search IGDB games and print JSON results (no download run).",
    )
    parser.add_argument(
        "--search-company",
        type=str,
        default="",
        help="Optional studio filter(s) for search mode (comma/semicolon/newline separated).",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=20,
        help=f"Max results for --search-query mode (1-{MAX_IGDB_SEARCH_RESULTS}).",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def request_bytes(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 25.0,
    max_retries: int = 4,
    rate_limiter: RateLimiter | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    """HTTP request with retries for transient failures and rate limiting."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            request = urllib.request.Request(url=url, method=method, headers=headers or {}, data=body)
            with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
                status = int(getattr(response, "status", 200))
                payload = response.read()
                response_headers = {k.lower(): v for k, v in response.headers.items()}
                return status, payload, response_headers
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            if status in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(min(8.0, 0.6 * (2**attempt)))
                continue
            err_payload = exc.read() if hasattr(exc, "read") else b""
            return status, err_payload, {}
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            reason = getattr(exc, "reason", None)
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise FetchError(
                    "SSL certificate verification failed. "
                    "On macOS run the 'Install Certificates.command' for your Python install "
                    "or install certifi and set SSL_CERT_FILE to certifi.where()."
                ) from exc
            if attempt < max_retries:
                time.sleep(min(8.0, 0.6 * (2**attempt)))
                continue
            break
    raise FetchError(f"Network request failed for {url}: {last_error}")


def request_json(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 25.0,
    max_retries: int = 4,
    rate_limiter: RateLimiter | None = None,
) -> tuple[int, Any]:
    status, payload, _ = request_bytes(
        url=url,
        method=method,
        headers=headers,
        body=body,
        timeout=timeout,
        max_retries=max_retries,
        rate_limiter=rate_limiter,
    )
    if not payload:
        return status, None
    try:
        return status, json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return status, payload.decode("utf-8", errors="replace")


def get_token_from_twitch(client_id: str, client_secret: str, timeout: float, max_retries: int) -> tuple[str, int]:
    if not client_id or not client_secret:
        raise FetchError(
            "Missing IGDB credentials. Provide --client-id/--client-secret or set "
            "IGDB_CLIENT_ID and IGDB_CLIENT_SECRET environment variables."
        )
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        }
    )
    url = f"{TOKEN_ENDPOINT}?{query}"
    status, payload = request_json(url=url, method="POST", timeout=timeout, max_retries=max_retries, rate_limiter=None)
    if status != 200 or not isinstance(payload, dict):
        raise FetchError(f"Token request failed (HTTP {status}): {payload}")
    token = str(payload.get("access_token") or "").strip()
    expires_in = int(payload.get("expires_in") or 0)
    if not token:
        raise FetchError("Token request succeeded but access_token was missing.")
    return token, expires_in


def load_or_refresh_token(args: argparse.Namespace) -> str:
    if args.access_token:
        return args.access_token

    cache = load_json(args.token_cache)
    now = int(time.time())
    if isinstance(cache, dict):
        token = str(cache.get("access_token") or "").strip()
        expires_at = int(cache.get("expires_at_unix") or 0)
        client_id = str(cache.get("client_id") or "")
        if token and client_id == args.client_id and expires_at - now > 180:
            return token

    token, expires_in = get_token_from_twitch(
        client_id=args.client_id,
        client_secret=args.client_secret,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    expires_at = int(time.time()) + max(0, expires_in)
    save_json(
        args.token_cache,
        {
            "created_at_utc": utc_now_iso(),
            "client_id": args.client_id,
            "access_token": token,
            "expires_in_seconds": expires_in,
            "expires_at_unix": expires_at,
        },
    )
    return token


def igdb_headers(client_id: str, token: str) -> dict[str, str]:
    return {
        "Client-ID": client_id,
        "Authorization": f"Bearer {token}",
        "Content-Type": "text/plain",
    }


def igdb_post(
    endpoint: str,
    body_query: str,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> list[dict[str, Any]]:
    url = f"{IGDB_BASE_URL}/{endpoint.lstrip('/')}"
    status, payload = request_json(
        url=url,
        method="POST",
        headers=headers,
        body=body_query.encode("utf-8"),
        timeout=timeout,
        max_retries=max_retries,
        rate_limiter=rate_limiter,
    )
    if status != 200:
        raise FetchError(f"IGDB request failed endpoint={endpoint} HTTP {status}: {payload}")
    if not isinstance(payload, list):
        raise FetchError(f"IGDB response was not a list for endpoint={endpoint}: {payload}")
    return payload


def normalize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def collapse_single_letter_runs(tokens: list[str]) -> list[str]:
    """Join sequences like ['s','t','a','l','k','e','r'] -> ['stalker']."""
    out: list[str] = []
    run: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if run:
            out.append("".join(run) if len(run) >= 3 else " ".join(run))
            run = []
        out.append(token)
    if run:
        out.append("".join(run) if len(run) >= 3 else " ".join(run))
    return out


def clean_local_game_name(raw_name: str) -> str:
    name = raw_name.strip()
    name = re.sub(r"(?i)[\s_\-]*cover[\s_\-]*\d+\s*x\s*\d+$", "", name).strip()
    name = re.sub(r"(?i)[\s_\-]*\d+\s*x\s*\d+$", "", name).strip()
    name = re.sub(r"(?i)[\s_\-]*cover$", "", name).strip()
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()

    tokens = collapse_single_letter_runs(name.split())
    name = " ".join(tokens).strip()

    # Frequent slug spellings in this dataset.
    replacements = {
        "assassins creed": "assassin's creed",
        "tom clancys": "tom clancy's",
        "marvels ": "marvel's ",
    }
    lowered = name.lower()
    for src, dst in replacements.items():
        lowered = lowered.replace(src, dst)
    name = lowered.strip()
    return name or raw_name


def to_tokens(value: str) -> list[str]:
    normalized = normalize_name(value)
    return [token for token in normalized.split() if token and token not in TOKEN_STOPWORDS]


def token_overlap_ratio(local_tokens: list[str], remote_tokens: list[str]) -> float:
    if not local_tokens:
        return 0.0
    local_set = set(local_tokens)
    remote_set = set(remote_tokens)
    return len(local_set & remote_set) / max(1, len(local_set))


def matching_score(
    local_name: str,
    remote_name: str,
    version_parent: int | None,
    strict_match_mode: bool,
    min_token_overlap: float,
) -> tuple[float, float]:
    local_norm = normalize_name(local_name)
    remote_norm = normalize_name(remote_name)
    if not local_norm or not remote_norm:
        return 0.0, 0.0

    local_tokens = to_tokens(local_norm)
    remote_tokens = to_tokens(remote_norm)
    overlap = token_overlap_ratio(local_tokens, remote_tokens)
    if strict_match_mode and len(local_tokens) >= 2 and overlap < min_token_overlap:
        return -1.0, overlap

    set_local = set(local_tokens)
    set_remote = set(remote_tokens)
    jaccard = (
        len(set_local & set_remote) / max(1, len(set_local | set_remote))
        if set_local or set_remote
        else 0.0
    )
    ratio = difflib.SequenceMatcher(a=local_norm, b=remote_norm).ratio()
    score = (0.55 * ratio) + (0.45 * jaccard)
    if local_norm == remote_norm:
        score += 0.25
    elif local_norm in remote_norm or remote_norm in local_norm:
        score += 0.08

    local_nums = {token for token in local_tokens if token.isdigit()}
    remote_nums = {token for token in remote_tokens if token.isdigit()}
    if local_nums and not (local_nums & remote_nums):
        score -= 0.18

    if strict_match_mode:
        if len(local_tokens) >= 3 and len(set_local & set_remote) < 2:
            score -= 0.22
        if len(local_tokens) >= 3 and overlap < 0.5:
            score -= 0.12
    if version_parent is not None:
        score -= 0.12
    return score, overlap


def read_mapping_csv(path: Path) -> dict[str, GameMapping]:
    if not path.exists():
        return {}

    mapping: dict[str, GameMapping] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}

        for row in reader:
            local_game = str(row.get("local_game", "") or row.get("game", "")).strip()
            if not local_game:
                continue
            if local_game.startswith("#"):
                continue
            raw_id = str(row.get("igdb_game_id", "") or row.get("igdb_id", "")).strip()
            igdb_game_id = int(raw_id) if raw_id.isdigit() else None
            igdb_game_name = str(row.get("igdb_game_name", "") or row.get("igdb_name", "")).strip() or None
            mapping[local_game.lower()] = GameMapping(
                local_game=local_game,
                igdb_game_id=igdb_game_id,
                igdb_game_name=igdb_game_name,
            )
    return mapping


def collect_game_dirs(dataset_root: Path, max_games: int) -> list[Path]:
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FetchError(f"Dataset root missing or invalid: {dataset_root}")
    dirs = sorted([path for path in dataset_root.iterdir() if path.is_dir()], key=lambda p: p.name.lower())
    if max_games > 0:
        return dirs[:max_games]
    return dirs


def folder_has_any_image(folder: Path) -> bool:
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def find_best_candidate(
    local_game_name: str,
    records: list[dict[str, Any]],
    strict_match_mode: bool,
    min_token_overlap: float,
) -> CandidateGame | None:
    best: CandidateGame | None = None
    for rec in records:
        igdb_id = int(rec.get("id") or 0)
        name = str(rec.get("name") or "").strip()
        version_parent = rec.get("version_parent")
        cover_image_id = None
        cover_field = rec.get("cover")
        if isinstance(cover_field, dict):
            image_id = cover_field.get("image_id")
            if isinstance(image_id, str) and image_id.strip():
                cover_image_id = image_id.strip()
        score, overlap = matching_score(
            local_game_name,
            name,
            version_parent if isinstance(version_parent, int) else None,
            strict_match_mode=strict_match_mode,
            min_token_overlap=min_token_overlap,
        )
        if score < 0:
            continue
        candidate = CandidateGame(
            igdb_id=igdb_id,
            name=name,
            version_parent=version_parent if isinstance(version_parent, int) else None,
            cover_image_id=cover_image_id,
            score=score,
            token_overlap=overlap,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def fetch_game_by_id(
    game_id: int,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> CandidateGame | None:
    query = (
        "fields id,name,version_parent,cover.image_id;"
        f" where id = {game_id}; limit 1;"
    )
    records = igdb_post("games", query, headers, timeout, max_retries, rate_limiter)
    if not records:
        return None
    rec = records[0]
    cover_field = rec.get("cover")
    cover_image_id = None
    if isinstance(cover_field, dict):
        image_id = cover_field.get("image_id")
        if isinstance(image_id, str) and image_id.strip():
            cover_image_id = image_id.strip()
    return CandidateGame(
        igdb_id=int(rec.get("id") or 0),
        name=str(rec.get("name") or "").strip(),
        version_parent=rec.get("version_parent") if isinstance(rec.get("version_parent"), int) else None,
        cover_image_id=cover_image_id,
        score=1.0,
        token_overlap=1.0,
    )


def search_game_candidates(
    query_name: str,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> list[dict[str, Any]]:
    safe_query = query_name.replace('"', "").strip()
    query = (
        "search "
        f'"{safe_query}"; '
        "fields id,name,version_parent,cover.image_id;"
        "where cover != null;"
        "limit 25;"
    )
    return igdb_post("games", query, headers, timeout, max_retries, rate_limiter)


def sanitize_igdb_search_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\\", " ").replace('"', " ").strip())


def parse_company_filters(raw_value: str) -> list[str]:
    """Split a company search string into unique studio filters.

    Supports comma, semicolon, or newline delimiters so the UI can submit:
    "Rockstar, Ubisoft, CD Projekt".
    """
    normalized = str(raw_value or "").replace("\r\n", "\n").replace("\r", "\n")
    tokens = re.split(r"[,\n;]+", normalized)
    filters: list[str] = []
    for token in tokens:
        safe = sanitize_igdb_search_text(token)
        if len(safe) < 2:
            continue
        if safe not in filters:
            filters.append(safe)
    return filters


def company_name_match_score(query_name: str, candidate_name: str) -> float:
    """Score how well an IGDB company name matches the requested studio string."""
    query_norm = normalize_name(query_name)
    candidate_norm = normalize_name(candidate_name)
    if not query_norm or not candidate_norm:
        return 0.0

    query_tokens = to_tokens(query_norm)
    candidate_tokens = to_tokens(candidate_norm)
    overlap = token_overlap_ratio(query_tokens, candidate_tokens) if query_tokens else 0.0
    seq_ratio = difflib.SequenceMatcher(None, query_norm, candidate_norm).ratio()
    contains_bonus = 0.15 if (query_norm in candidate_norm or candidate_norm in query_norm) else 0.0
    return min(1.0, 0.6 * overlap + 0.35 * seq_ratio + contains_bonus)


def sort_game_rows_for_search(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, int, float, int]:
        category_raw = row.get("category")
        try:
            category = int(category_raw)
        except (TypeError, ValueError):
            category = -1
        category_priority = 1 if category in {0, 8, 9} else 0

        rating_count_raw = row.get("total_rating_count")
        try:
            rating_count = int(rating_count_raw)
        except (TypeError, ValueError):
            rating_count = 0

        rating_raw = row.get("total_rating")
        try:
            rating = float(rating_raw)
        except (TypeError, ValueError):
            rating = 0.0

        release_raw = row.get("first_release_date")
        try:
            release = int(release_raw)
        except (TypeError, ValueError):
            release = 0

        return (category_priority, rating_count, rating, release)

    return sorted(rows, key=sort_key, reverse=True)


def dedupe_game_rows(rows: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    seen: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        try:
            game_id = int(row.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if game_id <= 0 or game_id in seen:
            continue
        seen.add(game_id)
        deduped.append(row)
        if len(deduped) >= max_count:
            break
    return deduped


def fetch_games_rows_by_ids(
    game_ids: list[int],
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
    require_cover: bool = True,
) -> list[dict[str, Any]]:
    normalized_ids: list[int] = []
    for raw_id in game_ids:
        try:
            parsed = int(raw_id)
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in normalized_ids:
            normalized_ids.append(parsed)
    if not normalized_ids:
        return []

    base_fields = "fields id,name,first_release_date,total_rating,total_rating_count,category,cover.image_id,genres;"
    rows: list[dict[str, Any]] = []
    chunk_size = 120
    for start in range(0, len(normalized_ids), chunk_size):
        chunk = normalized_ids[start : start + chunk_size]
        id_csv = ",".join(str(game_id) for game_id in chunk)
        where_clause = f"id = ({id_csv})"
        if require_cover:
            where_clause += " & cover != null"
        query = (
            f"{base_fields}"
            f" where {where_clause};"
            f" limit {len(chunk)};"
        )
        chunk_rows = igdb_post("games", query, headers, timeout, max_retries, rate_limiter)
        if chunk_rows:
            rows.extend(chunk_rows)
    return rows


def search_games_by_name_api(
    query_name: str,
    limit: int,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> list[dict[str, Any]]:
    safe_query = sanitize_igdb_search_text(query_name)
    if not safe_query:
        return []
    bounded_limit = min(MAX_IGDB_SEARCH_RESULTS, max(1, int(limit)))
    escaped_query = safe_query.replace("\\", "\\\\")
    base_fields = "fields id,name,first_release_date,total_rating,total_rating_count,category,cover.image_id,genres;"

    tokens = [token for token in re.split(r"\s+", safe_query) if token]
    search_terms: list[str] = []
    for term in (safe_query, " ".join(tokens[:2]) if len(tokens) >= 2 else "", tokens[0] if tokens else ""):
        term = term.strip()
        if term and term not in search_terms:
            search_terms.append(term)

    merged: list[dict[str, Any]] = []
    for term in search_terms:
        if len(merged) >= bounded_limit:
            break
        escaped_term = sanitize_igdb_search_text(term)
        query_variants = [
            (
                "search "
                f'"{escaped_term}"; '
                f"{base_fields}"
                "where cover != null & category = (0,8,9);"
                f"limit {bounded_limit};"
            ),
            (
                "search "
                f'"{escaped_term}"; '
                f"{base_fields}"
                "where cover != null;"
                f"limit {bounded_limit};"
            ),
            (
                "search "
                f'"{escaped_term}"; '
                f"{base_fields}"
                f"limit {bounded_limit};"
            ),
            (
                f"{base_fields}"
                f'where name ~ *"{escaped_term}"* & cover != null; '
                "sort total_rating_count desc;"
                f"limit {bounded_limit};"
            ),
            (
                f"{base_fields}"
                f'where name ~ *"{escaped_term}"*; '
                "sort total_rating_count desc;"
                f"limit {bounded_limit};"
            ),
        ]
        for query in query_variants:
            rows = igdb_post("games", query, headers, timeout, max_retries, rate_limiter)
            if rows:
                merged.extend(rows)

    if not merged and escaped_query:
        # Last resort broad fallback.
        rows = igdb_post(
            "games",
            (
                f"{base_fields}"
                f'where name ~ *"{escaped_query}"*; '
                "sort total_rating_count desc;"
                f"limit {bounded_limit};"
            ),
            headers,
            timeout,
            max_retries,
            rate_limiter,
        )
        if rows:
            merged.extend(rows)

    return dedupe_game_rows(sort_game_rows_for_search(merged), bounded_limit)


def search_games_by_company_api(
    company_name: str,
    limit: int,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
    query_name: str = "",
) -> list[dict[str, Any]]:
    safe_company = sanitize_igdb_search_text(company_name)
    if not safe_company:
        return []
    bounded_limit = min(MAX_IGDB_SEARCH_RESULTS, max(1, int(limit)))

    company_terms: list[str] = []
    tokens = [token for token in re.split(r"\s+", safe_company) if token]
    for term in (safe_company, " ".join(tokens[:2]) if len(tokens) >= 2 else "", tokens[0] if tokens else ""):
        term = term.strip()
        if term and term not in company_terms:
            company_terms.append(term)

    company_rows: list[dict[str, Any]] = []
    for term in company_terms:
        escaped_term = sanitize_igdb_search_text(term)
        query_variants = [
            f'search "{escaped_term}"; fields id,name; limit 25;',
            f'fields id,name; where name ~ *"{escaped_term}"*; limit 25;',
        ]
        for query in query_variants:
            rows = igdb_post("companies", query, headers, timeout, max_retries, rate_limiter)
            if rows:
                company_rows.extend(rows)

    scored_companies: list[tuple[float, int]] = []
    scored_by_id: dict[int, float] = {}
    for row in company_rows:
        try:
            company_id = int(row.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if company_id <= 0:
            continue
        company_label = str(row.get("name") or "").strip()
        score = company_name_match_score(safe_company, company_label)
        prev = scored_by_id.get(company_id)
        if prev is None or score > prev:
            scored_by_id[company_id] = score

    for company_id, score in scored_by_id.items():
        scored_companies.append((score, company_id))
    scored_companies.sort(key=lambda item: item[0], reverse=True)

    strong_company_ids = [company_id for score, company_id in scored_companies if score >= 0.55]
    if strong_company_ids:
        company_ids = strong_company_ids[:12]
    else:
        # Fallback if IGDB names are noisy: keep top-scoring candidates only.
        company_ids = [company_id for _, company_id in scored_companies[:6]]

    if not company_ids:
        return []
    if len(company_ids) > 25:
        company_ids = company_ids[:25]

    company_id_csv = ",".join(str(value) for value in company_ids)
    discovered_game_ids: list[int] = []
    per_page = 500
    max_rows = min(5000, max(500, bounded_limit * 40))

    def collect_discovered_ids(require_primary_role: bool) -> list[int]:
        game_ids: list[int] = []
        offset = 0
        role_clause = " & (developer = true | publisher = true)" if require_primary_role else ""
        while offset < max_rows:
            query = (
                "fields game,company,publisher,developer;"
                f" where game != null & company = ({company_id_csv}){role_clause};"
                f" limit {per_page}; offset {offset};"
            )
            rows = igdb_post("involved_companies", query, headers, timeout, max_retries, rate_limiter)
            if not rows:
                break
            for row in rows:
                try:
                    game_id = int(row.get("game") or 0)
                except (TypeError, ValueError):
                    continue
                if game_id > 0 and game_id not in game_ids:
                    game_ids.append(game_id)
            if len(rows) < per_page:
                break
            offset += per_page
        return game_ids

    discovered_game_ids = collect_discovered_ids(require_primary_role=True)
    if not discovered_game_ids:
        discovered_game_ids = collect_discovered_ids(require_primary_role=False)

    if not discovered_game_ids:
        return []

    game_rows = fetch_games_rows_by_ids(
        game_ids=discovered_game_ids,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        rate_limiter=rate_limiter,
        require_cover=True,
    )

    safe_query = sanitize_igdb_search_text(query_name)
    if safe_query:
        safe_query_norm = normalize_name(safe_query)
        safe_query_tokens = to_tokens(safe_query)
        filtered_rows: list[dict[str, Any]] = []
        for row in game_rows:
            row_name = str(row.get("name") or "").strip()
            if not row_name:
                continue
            row_name_norm = normalize_name(row_name)
            if safe_query_norm and safe_query_norm in row_name_norm:
                filtered_rows.append(row)
                continue
            row_tokens = to_tokens(row_name)
            overlap = token_overlap_ratio(safe_query_tokens, row_tokens) if safe_query_tokens else 0.0
            if overlap >= 0.5:
                filtered_rows.append(row)
        game_rows = filtered_rows

    return dedupe_game_rows(sort_game_rows_for_search(game_rows), bounded_limit)


def search_games_api(
    query_name: str,
    limit: int,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
    company_name: str = "",
) -> list[dict[str, Any]]:
    bounded_limit = min(MAX_IGDB_SEARCH_RESULTS, max(1, int(limit)))
    safe_query = sanitize_igdb_search_text(query_name)
    company_filters = parse_company_filters(company_name)

    by_name_rows: list[dict[str, Any]] = []
    by_company_rows: list[dict[str, Any]] = []
    if safe_query:
        by_name_rows = search_games_by_name_api(
            query_name=safe_query,
            limit=bounded_limit,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
    if company_filters:
        per_company_rows: list[list[dict[str, Any]]] = []
        # Allow larger retrieval windows for company searches, including single-company >100 requests.
        if len(company_filters) == 1:
            per_company_limit = bounded_limit
        else:
            per_company_limit = min(
                bounded_limit,
                max(20, (bounded_limit + len(company_filters) - 1) // len(company_filters) + 12),
            )
        for company_filter in company_filters:
            rows = search_games_by_company_api(
                company_name=company_filter,
                limit=per_company_limit,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
                rate_limiter=rate_limiter,
                query_name=safe_query,
            )
            if rows:
                per_company_rows.append(rows)

        # Balance output across studios to avoid one large publisher dominating the result list.
        seen_ids: set[int] = set()
        by_company_rows = []
        round_idx = 0
        while len(by_company_rows) < bounded_limit:
            added_this_round = 0
            for rows in per_company_rows:
                if round_idx >= len(rows):
                    continue
                row = rows[round_idx]
                try:
                    row_id = int(row.get("id") or 0)
                except (TypeError, ValueError):
                    continue
                if row_id <= 0 or row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                by_company_rows.append(row)
                added_this_round += 1
                if len(by_company_rows) >= bounded_limit:
                    break
            if added_this_round == 0:
                break
            round_idx += 1

    if company_filters and not safe_query:
        return by_company_rows[:bounded_limit]
    if safe_query and not company_filters:
        return by_name_rows[:bounded_limit]

    # If both filters are supplied, prefer company-constrained matches and then top up with title-only matches.
    merged = dedupe_game_rows(sort_game_rows_for_search(by_company_rows + by_name_rows), bounded_limit)
    return merged[:bounded_limit]


def parse_seed_game_ids(raw_value: str) -> list[int]:
    value = str(raw_value or "").strip()
    if not value:
        return []
    ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError as exc:
            raise FetchError(f"Invalid seed game id: '{token}'") from exc
        if parsed <= 0:
            raise FetchError(f"Seed game id must be > 0: '{token}'")
        if parsed not in ids:
            ids.append(parsed)
    return ids


def parse_seed_game_names(raw_names: str, names_file: Path | None) -> list[str]:
    names: list[str] = []

    def add_token(token: str) -> None:
        value = token.strip()
        if not value:
            return
        if value.startswith("#"):
            return
        if value not in names:
            names.append(value)

    raw = str(raw_names or "")
    if raw.strip():
        normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
        if "\n" in normalized:
            tokens = normalized.split("\n")
        elif ";" in normalized:
            tokens = normalized.split(";")
        else:
            tokens = normalized.split(",")
        for token in tokens:
            add_token(token)

    if names_file is not None:
        if not names_file.exists():
            raise FetchError(f"seed-game-list-file not found: {names_file}")
        for line in names_file.read_text(encoding="utf-8").splitlines():
            add_token(line)

    return names


def normalize_requested_game_name(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    value = re.sub(r"\s+", " ", value)
    # Common copied-list separators where publisher/dev is appended.
    for separator in (" / ", " | "):
        if separator in value:
            value = value.split(separator, 1)[0].strip()
    return value.strip()


def fetch_games_by_ids(
    game_ids: list[int],
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> list[CandidateGame]:
    if not game_ids:
        return []
    normalized = [int(game_id) for game_id in game_ids if int(game_id) > 0]
    if not normalized:
        return []
    id_csv = ",".join(str(game_id) for game_id in normalized)
    query = (
        "fields id,name,version_parent,cover.image_id;"
        f" where id = ({id_csv}) & cover != null;"
        f" limit {max(25, len(normalized))};"
    )
    rows = igdb_post("games", query, headers, timeout, max_retries, rate_limiter)
    by_id: dict[int, CandidateGame] = {}
    for row in rows:
        igdb_id = int(row.get("id") or 0)
        if igdb_id <= 0:
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        cover_image_id = None
        cover_field = row.get("cover")
        if isinstance(cover_field, dict):
            image_id = cover_field.get("image_id")
            if isinstance(image_id, str) and image_id.strip():
                cover_image_id = image_id.strip()
        by_id[igdb_id] = CandidateGame(
            igdb_id=igdb_id,
            name=name,
            version_parent=row.get("version_parent") if isinstance(row.get("version_parent"), int) else None,
            cover_image_id=cover_image_id,
            score=1.0,
            token_overlap=1.0,
        )

    ordered: list[CandidateGame] = []
    for wanted_id in normalized:
        game = by_id.get(wanted_id)
        if game is not None:
            ordered.append(game)
    return ordered


def fetch_games_by_names(
    game_names: list[str],
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> tuple[list[CandidateGame], list[str]]:
    resolved: list[CandidateGame] = []
    unresolved: list[str] = []
    seen_ids: set[int] = set()

    total = len(game_names)
    for idx, raw_name in enumerate(game_names, start=1):
        cleaned_name = normalize_requested_game_name(raw_name)
        query_name = clean_local_game_name(cleaned_name)
        log(f"[IGDB] Name-list [{idx}/{total}] query='{query_name}'")

        # Fast path: single IGDB search query used elsewhere in pipeline.
        rows = search_game_candidates(
            query_name=query_name,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        candidate = find_best_candidate(
            local_game_name=query_name,
            records=rows,
            strict_match_mode=True,
            min_token_overlap=0.2,
        )
        # Fallback path: broader multi-query search.
        if candidate is None:
            fallback_rows = search_games_api(
                query_name=query_name,
                limit=25,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
                rate_limiter=rate_limiter,
            )
            candidate = find_best_candidate(
                local_game_name=query_name,
                records=fallback_rows,
                strict_match_mode=False,
                min_token_overlap=0.0,
            )
        if candidate is None:
            unresolved.append(raw_name)
            continue
        if candidate.score < 0.35:
            unresolved.append(raw_name)
            continue
        if candidate.igdb_id in seen_ids:
            continue
        seen_ids.add(candidate.igdb_id)
        resolved.append(candidate)

    return resolved, unresolved


def resolve_genre_id(
    genre_name: str,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> tuple[int, str]:
    safe_query = genre_name.replace('"', "").strip()
    if not safe_query:
        raise FetchError("seed-genre-name is empty.")

    escaped = safe_query.replace("\\", "\\\\").replace('"', '\\"')
    normalized_query = normalize_name(safe_query)
    slug_hint = re.sub(r"[^a-z0-9]+", "-", normalized_query).strip("-")

    # IGDB does not support `search` on the genres endpoint.
    rows: list[dict[str, Any]] = []
    query_variants = [
        f'fields id,name; where name = "{escaped}"; limit 25;',
        f'fields id,name; where slug = "{slug_hint}"; limit 25;' if slug_hint else "",
        f'fields id,name,slug; where name ~ *"{escaped}"*; limit 100;',
    ]
    for query in query_variants:
        if not query:
            continue
        rows = igdb_post(
            "genres",
            query,
            headers,
            timeout,
            max_retries,
            rate_limiter,
        )
        if rows:
            break
    if not rows:
        raise FetchError(f"No IGDB genre matches found for '{genre_name}'.")

    target = normalized_query
    best_row: dict[str, Any] | None = None
    best_score = -1.0
    for row in rows:
        name = str(row.get("name") or "").strip()
        gid = int(row.get("id") or 0)
        if gid <= 0 or not name:
            continue
        score = difflib.SequenceMatcher(a=target, b=normalize_name(name)).ratio()
        if normalize_name(name) == target:
            score += 0.35
        if score > best_score:
            best_score = score
            best_row = row

    if not best_row:
        raise FetchError(f"No valid IGDB genre records found for '{genre_name}'.")

    return int(best_row["id"]), str(best_row.get("name") or genre_name).strip()


def fetch_games_for_genre(
    genre_id: int,
    desired_count: int,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> list[CandidateGame]:
    if desired_count <= 0:
        return []

    chunk_size = min(150, max(30, desired_count))
    results: list[CandidateGame] = []
    seen_ids: set[int] = set()
    offset = 0
    while len(results) < desired_count:
        query = (
            "fields id,name,version_parent,cover.image_id,total_rating_count,first_release_date;"
            f" where cover != null & genres = ({genre_id}) & version_parent = null;"
            " sort total_rating_count desc;"
            f" limit {chunk_size}; offset {offset};"
        )
        rows = igdb_post("games", query, headers, timeout, max_retries, rate_limiter)
        if not rows:
            break

        for row in rows:
            igdb_id = int(row.get("id") or 0)
            if igdb_id <= 0 or igdb_id in seen_ids:
                continue
            seen_ids.add(igdb_id)

            name = str(row.get("name") or "").strip()
            if not name:
                continue

            cover_image_id = None
            cover_field = row.get("cover")
            if isinstance(cover_field, dict):
                image_id = cover_field.get("image_id")
                if isinstance(image_id, str) and image_id.strip():
                    cover_image_id = image_id.strip()

            results.append(
                CandidateGame(
                    igdb_id=igdb_id,
                    name=name,
                    version_parent=row.get("version_parent") if isinstance(row.get("version_parent"), int) else None,
                    cover_image_id=cover_image_id,
                    score=1.0,
                    token_overlap=1.0,
                )
            )
            if len(results) >= desired_count:
                break
        offset += chunk_size
    return results


def year_start_timestamp_utc(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())


def fetch_games_for_popular_year_range(
    year_start: int,
    year_end: int,
    desired_count: int,
    min_total_rating_count: int,
    exclude_genre_ids: set[int],
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> list[CandidateGame]:
    if desired_count <= 0:
        return []
    if year_end < year_start:
        raise FetchError("seed-year-end must be >= seed-year-start.")

    start_ts = year_start_timestamp_utc(year_start)
    end_ts_exclusive = year_start_timestamp_utc(year_end + 1)
    chunk_size = min(200, max(40, desired_count))
    results: list[CandidateGame] = []
    seen_ids: set[int] = set()
    fallback_thresholds = [min_total_rating_count, 80, 40, 10, 0]
    unique_thresholds: list[int] = []
    for value in fallback_thresholds:
        normalized = max(0, int(value))
        if normalized not in unique_thresholds:
            unique_thresholds.append(normalized)

    for threshold in unique_thresholds:
        if len(results) >= desired_count:
            break

        offset = 0
        pages_scanned = 0
        consecutive_no_add_pages = 0
        sort_field = "total_rating_count" if threshold > 0 else "total_rating"
        where_parts = [
            "cover != null",
            "category = 0",
            f"first_release_date >= {start_ts}",
            f"first_release_date < {end_ts_exclusive}",
        ]
        if threshold > 0:
            where_parts.append(f"total_rating_count >= {threshold}")
        if exclude_genre_ids:
            for excluded_genre_id in sorted(exclude_genre_ids):
                where_parts.append(f"genres != ({excluded_genre_id})")
        where_clause = " & ".join(where_parts)
        log(f"[IGDB] AAA seed pass threshold={threshold} sort={sort_field}")

        while len(results) < desired_count and pages_scanned < 40:
            query = (
                "fields id,name,version_parent,cover.image_id,total_rating_count,total_rating,first_release_date,genres,category;"
                f" where {where_clause};"
                f" sort {sort_field} desc;"
                f" limit {chunk_size}; offset {offset};"
            )
            rows = igdb_post("games", query, headers, timeout, max_retries, rate_limiter)
            if not rows:
                break
            pages_scanned += 1

            added_this_page = 0
            for row in rows:
                igdb_id = int(row.get("id") or 0)
                if igdb_id <= 0 or igdb_id in seen_ids:
                    continue

                if isinstance(row.get("version_parent"), int):
                    # Skip explicit variant entries and prefer base games.
                    continue

                name = str(row.get("name") or "").strip()
                if not name:
                    continue

                row_genres_raw = row.get("genres")
                row_genres: set[int] = set()
                if isinstance(row_genres_raw, list):
                    for item in row_genres_raw:
                        try:
                            gid = int(item)
                        except (TypeError, ValueError):
                            continue
                        if gid > 0:
                            row_genres.add(gid)
                if exclude_genre_ids and row_genres.intersection(exclude_genre_ids):
                    continue

                cover_image_id = None
                cover_field = row.get("cover")
                if isinstance(cover_field, dict):
                    image_id = cover_field.get("image_id")
                    if isinstance(image_id, str) and image_id.strip():
                        cover_image_id = image_id.strip()

                results.append(
                    CandidateGame(
                        igdb_id=igdb_id,
                        name=name,
                        version_parent=None,
                        cover_image_id=cover_image_id,
                        score=1.0,
                        token_overlap=1.0,
                    )
                )
                seen_ids.add(igdb_id)
                added_this_page += 1
                if len(results) >= desired_count:
                    break

            # Avoid infinite loops while still allowing a few sparse pages.
            if added_this_page == 0:
                consecutive_no_add_pages += 1
                if consecutive_no_add_pages >= 3:
                    break
            else:
                consecutive_no_add_pages = 0

            offset += chunk_size
    return results


def safe_folder_name(value: str) -> str:
    name = value.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    return name or "unnamed_game"


def choose_unique_folder(dataset_root: Path, desired_name: str, igdb_id: int) -> Path:
    base = safe_folder_name(desired_name)
    candidate = dataset_root / base
    if not candidate.exists():
        return candidate
    if candidate.is_dir():
        return candidate

    # Very rare edge case: conflicting file path.
    suffix = 2
    while True:
        trial = dataset_root / f"{base} ({igdb_id})" if suffix == 2 else dataset_root / f"{base} ({igdb_id}-{suffix})"
        if not trial.exists():
            return trial
        if trial.is_dir():
            return trial
        suffix += 1


def upsert_mapping_csv(path: Path, mapping: dict[str, GameMapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(mapping.values(), key=lambda item: item.local_game.lower())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["local_game", "igdb_game_id", "igdb_game_name"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "local_game": row.local_game,
                    "igdb_game_id": row.igdb_game_id or "",
                    "igdb_game_name": row.igdb_game_name or "",
                }
            )


def upsert_game_groups_csv(path: Path, games: list[str], group_label: str) -> None:
    current: dict[str, str] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                game = str(row.get("game", "")).strip()
                group = str(row.get("group", "")).strip()
                if game:
                    current[game] = group or "unassigned"

    for game in games:
        if game not in current:
            current[game] = group_label

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["game", "group"])
        writer.writeheader()
        for game in sorted(current.keys(), key=lambda g: g.lower()):
            writer.writerow({"game": game, "group": current[game]})


def seed_game_folders(
    dataset_root: Path,
    seed_mode: str,
    genre_name: str,
    seed_count: int,
    seed_year_start: int,
    seed_year_end: int,
    seed_min_total_rating_count: int,
    seed_exclude_genre_name: str,
    seed_game_ids: list[int],
    seed_game_names: list[str],
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    unresolved_names: list[str] = []
    if seed_mode == "genre":
        genre_id, resolved_genre_name = resolve_genre_id(
            genre_name=genre_name,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        log(f"[IGDB] Seeding from genre '{resolved_genre_name}' (id={genre_id}) count={seed_count}")
        games = fetch_games_for_genre(
            genre_id=genre_id,
            desired_count=seed_count,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        seed_descriptor = f"genre:{resolved_genre_name}"
    elif seed_mode == "popular_year_range":
        exclude_ids: set[int] = set()
        excluded_label = ""
        if seed_exclude_genre_name.strip():
            excluded_id, excluded_label = resolve_genre_id(
                genre_name=seed_exclude_genre_name,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
                rate_limiter=rate_limiter,
            )
            exclude_ids.add(excluded_id)
        log(
            "[IGDB] Seeding from popular year range "
            f"{seed_year_start}-{seed_year_end}, min_total_rating_count={seed_min_total_rating_count}, "
            f"exclude_genre={excluded_label or 'none'}, count={seed_count}"
        )
        games = fetch_games_for_popular_year_range(
            year_start=seed_year_start,
            year_end=seed_year_end,
            desired_count=seed_count,
            min_total_rating_count=seed_min_total_rating_count,
            exclude_genre_ids=exclude_ids,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        seed_descriptor = (
            f"popular_year_range:{seed_year_start}-{seed_year_end}"
            + (f":exclude={excluded_label}" if excluded_label else "")
        )
    elif seed_mode == "id_list":
        log(f"[IGDB] Seeding from explicit IGDB game ids count={len(seed_game_ids)}")
        games = fetch_games_by_ids(
            game_ids=seed_game_ids,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        if not games:
            raise FetchError("No IGDB games resolved from provided seed-game-ids.")
        seed_descriptor = "id_list"
    elif seed_mode == "name_list":
        log(f"[IGDB] Seeding from explicit game names count={len(seed_game_names)}")
        games, unresolved_names = fetch_games_by_names(
            game_names=seed_game_names,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        if not games:
            raise FetchError(
                "No IGDB games resolved from provided seed-game-names."
            )
        if unresolved_names:
            preview = ", ".join(unresolved_names[:8])
            suffix = " ..." if len(unresolved_names) > 8 else ""
            log(f"[IGDB] Unresolved names ({len(unresolved_names)}): {preview}{suffix}")
        seed_descriptor = "name_list"
    else:
        raise FetchError(f"Unsupported seed-mode: {seed_mode}")

    seeded_rows: list[dict[str, Any]] = []
    for game in games:
        folder_path = choose_unique_folder(dataset_root, game.name, game.igdb_id)
        created = False
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            created = True
        seeded_rows.append(
            {
                "local_game": folder_path.name,
                "folder": str(folder_path),
                "created": created,
                "igdb_game_id": game.igdb_id,
                "igdb_game_name": game.name,
                "cover_image_id": game.cover_image_id,
            }
        )
    return seed_descriptor, seeded_rows, unresolved_names


def fetch_cover_image_id_by_cover_id(
    cover_id: int,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> str | None:
    query = f"fields id,image_id; where id = {cover_id}; limit 1;"
    rows = igdb_post("covers", query, headers, timeout, max_retries, rate_limiter)
    if not rows:
        return None
    image_id = rows[0].get("image_id")
    if isinstance(image_id, str) and image_id.strip():
        return image_id.strip()
    return None


def resolve_cover_image_id(
    candidate: CandidateGame,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> str | None:
    if candidate.cover_image_id:
        return candidate.cover_image_id

    game_rows = igdb_post(
        "games",
        f"fields id,cover; where id = {candidate.igdb_id}; limit 1;",
        headers,
        timeout,
        max_retries,
        rate_limiter,
    )
    if not game_rows:
        return None

    cover_raw = game_rows[0].get("cover")
    if isinstance(cover_raw, dict):
        image_id = cover_raw.get("image_id")
        if isinstance(image_id, str) and image_id.strip():
            return image_id.strip()
    if isinstance(cover_raw, int):
        return fetch_cover_image_id_by_cover_id(cover_raw, headers, timeout, max_retries, rate_limiter)
    return None


def download_cover(
    image_id: str,
    image_size: str,
    destination_dir: Path,
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
    overwrite: bool,
    dry_run: bool,
) -> Path | None:
    for ext in (".jpg", ".png", ".webp"):
        url = f"{IMAGE_BASE_URL}/t_{image_size}/{image_id}{ext}"
        status, payload, headers = request_bytes(
            url=url,
            method="GET",
            headers={"User-Agent": "IGDB-Cover-Fetcher/1.0"},
            body=None,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        if status != 200:
            continue
        content_type = str(headers.get("content-type", "")).lower()
        if "image" not in content_type and not payload:
            continue

        output_path = destination_dir / f"igdb_cover_{image_id}{ext}"
        if output_path.exists() and not overwrite:
            return output_path
        if dry_run:
            return output_path
        output_path.write_bytes(payload)
        return output_path
    return None


def run_search_mode(args: argparse.Namespace) -> int:
    if args.search_limit < 1 or args.search_limit > MAX_IGDB_SEARCH_RESULTS:
        raise FetchError(f"search-limit must be between 1 and {MAX_IGDB_SEARCH_RESULTS}.")
    query = str(args.search_query or "").strip()
    company = str(args.search_company or "").strip()
    company_filters = parse_company_filters(company)
    if not query and not company:
        raise FetchError("Provide search-query or search-company for search mode.")

    rate_limiter = RateLimiter(args.rate_limit_rps)
    token = load_or_refresh_token(args)
    headers = igdb_headers(args.client_id, token)
    rows = search_games_api(
        query_name=query,
        limit=args.search_limit,
        headers=headers,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        rate_limiter=rate_limiter,
        company_name=company,
    )
    results: list[dict[str, Any]] = []
    for row in rows:
        first_release_raw = row.get("first_release_date")
        release_year = None
        if isinstance(first_release_raw, (int, float)) and first_release_raw > 0:
            try:
                release_year = datetime.fromtimestamp(float(first_release_raw), tz=timezone.utc).year
            except (OverflowError, OSError, ValueError):
                release_year = None
        cover_image_id = None
        cover_field = row.get("cover")
        if isinstance(cover_field, dict):
            image_id = cover_field.get("image_id")
            if isinstance(image_id, str) and image_id.strip():
                cover_image_id = image_id.strip()
        results.append(
            {
                "id": int(row.get("id") or 0),
                "name": str(row.get("name") or "").strip(),
                "release_year": release_year,
                "total_rating_count": int(row.get("total_rating_count") or 0),
                "total_rating": float(row.get("total_rating") or 0.0),
                "category": int(row.get("category") or 0),
                "has_cover": bool(cover_image_id),
            }
        )
    print(
        json.dumps(
            {
                "query": query,
                "company": company,
                "companies": company_filters,
                "count": len(results),
                "results": results,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def run() -> int:
    args = parse_args()
    if str(args.search_query or "").strip() or str(args.search_company or "").strip():
        return run_search_mode(args)

    if args.min_match_score < 0.0 or args.min_match_score > 1.0:
        raise FetchError("min-match-score must be between 0 and 1.")
    if args.min_token_overlap < 0.0 or args.min_token_overlap > 1.0:
        raise FetchError("min-token-overlap must be between 0 and 1.")
    if args.max_games < 0:
        raise FetchError("max-games must be >= 0.")
    if args.rate_limit_rps <= 0:
        raise FetchError("rate-limit-rps must be > 0.")
    if args.seed_count < 0:
        raise FetchError("seed-count must be >= 0.")
    if args.seed_min_total_rating_count < 0:
        raise FetchError("seed-min-total-rating-count must be >= 0.")
    seed_game_ids = parse_seed_game_ids(args.seed_game_ids)
    seed_game_names = parse_seed_game_names(args.seed_game_names, args.seed_game_list_file)
    if args.seed_count > 0 and args.seed_mode == "genre" and not args.seed_genre_name.strip():
        raise FetchError("seed-genre-name must be provided when seed-mode=genre and seed-count > 0.")
    if args.seed_count > 0 and args.seed_mode == "popular_year_range":
        if args.seed_year_start < 1970 or args.seed_year_start > 2100:
            raise FetchError("seed-year-start must be between 1970 and 2100.")
        if args.seed_year_end < 1970 or args.seed_year_end > 2100:
            raise FetchError("seed-year-end must be between 1970 and 2100.")
        if args.seed_year_end < args.seed_year_start:
            raise FetchError("seed-year-end must be >= seed-year-start.")
    if args.seed_mode == "id_list":
        if not seed_game_ids:
            raise FetchError("seed-game-ids must be provided when seed-mode=id_list.")
        if args.seed_count == 0:
            args.seed_count = len(seed_game_ids)
    if args.seed_mode == "name_list":
        if not seed_game_names:
            raise FetchError("seed-game-names or seed-game-list-file must be provided when seed-mode=name_list.")
        # Name-list mode is explicit: process the full provided list.
        args.seed_count = len(seed_game_names)

    rate_limiter = RateLimiter(args.rate_limit_rps)
    token = load_or_refresh_token(args)
    headers = igdb_headers(args.client_id, token)
    mappings = read_mapping_csv(args.mapping_csv)
    seeded_rows: list[dict[str, Any]] = []
    seed_unresolved_names: list[str] = []
    seeded_descriptor = ""
    if args.seed_count > 0:
        seeded_descriptor, seeded_rows, seed_unresolved_names = seed_game_folders(
            dataset_root=args.dataset_root,
            seed_mode=args.seed_mode,
            genre_name=args.seed_genre_name,
            seed_count=args.seed_count,
            seed_year_start=args.seed_year_start,
            seed_year_end=args.seed_year_end,
            seed_min_total_rating_count=args.seed_min_total_rating_count,
            seed_exclude_genre_name=args.seed_exclude_genre_name,
            seed_game_ids=seed_game_ids,
            seed_game_names=seed_game_names,
            headers=headers,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
            rate_limiter=rate_limiter,
        )
        if args.seed_write_mappings and seeded_rows:
            for row in seeded_rows:
                local_game = str(row["local_game"])
                mappings[local_game.lower()] = GameMapping(
                    local_game=local_game,
                    igdb_game_id=int(row["igdb_game_id"]),
                    igdb_game_name=str(row["igdb_game_name"]),
                )
            upsert_mapping_csv(args.mapping_csv, mappings)
            log(f"[IGDB] Updated mappings file with seeded games: {args.mapping_csv}")
        if args.seed_update_groups and seeded_rows:
            seeded_game_names = [str(row["local_game"]) for row in seeded_rows]
            upsert_game_groups_csv(args.seed_groups_file, seeded_game_names, args.seed_group_label)
            log(
                f"[IGDB] Updated game groups file with seeded games: {args.seed_groups_file} "
                f"(group={args.seed_group_label})"
            )
        # Refresh mappings after potential write to preserve normalized parsing.
        mappings = read_mapping_csv(args.mapping_csv)
        if not seeded_rows:
            raise FetchError(
                "Seeding returned 0 games. Try widening filters "
                "(lower --seed-min-total-rating-count, broaden years, or disable --seed-exclude-genre-name)."
            )

    if args.seed_only and seeded_rows:
        folders_by_name = {path.name: path for path in collect_game_dirs(args.dataset_root, 0)}
        game_dirs = [folders_by_name[row["local_game"]] for row in seeded_rows if row["local_game"] in folders_by_name]
    else:
        game_dirs = collect_game_dirs(args.dataset_root, args.max_games)
    log(f"[IGDB] Processing {len(game_dirs)} game folders from: {args.dataset_root}")
    log(f"[IGDB] Mapping rows loaded: {len(mappings)} from {args.mapping_csv}")

    report_rows: list[dict[str, Any]] = []
    summary = {
        "total_folders": len(game_dirs),
        "skipped_existing_images": 0,
        "matched": 0,
        "downloaded": 0,
        "dry_run_resolved": 0,
        "failed": 0,
        "seed_mode": args.seed_mode,
        "seeded_requested": int(args.seed_count),
        "seeded_descriptor": seeded_descriptor or None,
        "seeded_folders_total": len(seeded_rows),
        "seeded_folders_created": sum(1 for row in seeded_rows if row.get("created")),
        "seeded_folders_existing": sum(1 for row in seeded_rows if not row.get("created")),
        "seed_unresolved_names_count": len(seed_unresolved_names),
    }

    for idx, game_dir in enumerate(game_dirs, start=1):
        local_name = game_dir.name
        cleaned_local_name = clean_local_game_name(local_name) if args.auto_clean_local_names else local_name
        row: dict[str, Any] = {
            "local_game": local_name,
            "local_game_cleaned": cleaned_local_name,
            "folder": str(game_dir),
            "status": "unknown",
        }
        log(f"[IGDB] [{idx}/{len(game_dirs)}] {local_name} -> query='{cleaned_local_name}'")

        try:
            if args.skip_if_any_image and folder_has_any_image(game_dir):
                summary["skipped_existing_images"] += 1
                row["status"] = "skipped_existing_images"
                report_rows.append(row)
                continue

            mapping = mappings.get(local_name.lower()) or mappings.get(cleaned_local_name.lower())
            candidate: CandidateGame | None = None
            if mapping and mapping.igdb_game_id is not None:
                candidate = fetch_game_by_id(
                    game_id=mapping.igdb_game_id,
                    headers=headers,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    rate_limiter=rate_limiter,
                )
                if candidate is None:
                    raise FetchError(f"Mapped IGDB id not found: {mapping.igdb_game_id}")
            else:
                query_name = (
                    mapping.igdb_game_name
                    if mapping and mapping.igdb_game_name
                    else cleaned_local_name
                )
                candidates = search_game_candidates(
                    query_name=query_name,
                    headers=headers,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    rate_limiter=rate_limiter,
                )
                candidate = find_best_candidate(
                    cleaned_local_name,
                    candidates,
                    strict_match_mode=args.strict_match_mode,
                    min_token_overlap=args.min_token_overlap,
                )

            if candidate is None:
                raise FetchError("No IGDB search candidates found.")
            if candidate.score < args.min_match_score and not args.allow_low_confidence:
                raise FetchError(
                    f"Low confidence match: score={candidate.score:.3f} < min={args.min_match_score:.3f} "
                    f"(candidate='{candidate.name}', id={candidate.igdb_id}, overlap={candidate.token_overlap:.3f})"
                )

            cover_image_id = resolve_cover_image_id(
                candidate=candidate,
                headers=headers,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                rate_limiter=rate_limiter,
            )
            if not cover_image_id:
                raise FetchError(f"No cover image_id found for candidate id={candidate.igdb_id}")

            output_path = download_cover(
                image_id=cover_image_id,
                image_size=args.image_size,
                destination_dir=game_dir,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                rate_limiter=rate_limiter,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            if output_path is None:
                raise FetchError(f"Could not download cover image for image_id={cover_image_id}")

            summary["matched"] += 1
            row.update(
                {
                    "status": "dry_run_resolved" if args.dry_run else "downloaded",
                    "igdb_game_id": candidate.igdb_id,
                    "igdb_game_name": candidate.name,
                    "match_score": round(candidate.score, 4),
                    "match_token_overlap": round(candidate.token_overlap, 4),
                    "cover_image_id": cover_image_id,
                    "output_path": str(output_path),
                }
            )
            if args.dry_run:
                summary["dry_run_resolved"] += 1
            else:
                summary["downloaded"] += 1

        except Exception as exc:  # pragma: no cover - runtime guard
            summary["failed"] += 1
            row.update({"status": "failed", "error": str(exc)})
            log(f"[IGDB]    FAILED: {exc}")

        report_rows.append(row)

    report = {
        "created_at_utc": utc_now_iso(),
        "dataset_root": str(args.dataset_root),
        "mapping_csv": str(args.mapping_csv),
        "image_size": args.image_size,
        "dry_run": bool(args.dry_run),
        "parameters": {
            "max_games": int(args.max_games),
            "skip_if_any_image": bool(args.skip_if_any_image),
            "overwrite": bool(args.overwrite),
            "min_match_score": float(args.min_match_score),
            "min_token_overlap": float(args.min_token_overlap),
            "strict_match_mode": bool(args.strict_match_mode),
            "auto_clean_local_names": bool(args.auto_clean_local_names),
            "allow_low_confidence": bool(args.allow_low_confidence),
            "rate_limit_rps": float(args.rate_limit_rps),
            "seed_mode": args.seed_mode,
            "seed_genre_name": args.seed_genre_name,
            "seed_count": int(args.seed_count),
            "seed_only": bool(args.seed_only),
            "seed_update_groups": bool(args.seed_update_groups),
            "seed_groups_file": str(args.seed_groups_file),
            "seed_group_label": args.seed_group_label,
            "seed_write_mappings": bool(args.seed_write_mappings),
            "seed_year_start": int(args.seed_year_start),
            "seed_year_end": int(args.seed_year_end),
            "seed_min_total_rating_count": int(args.seed_min_total_rating_count),
            "seed_exclude_genre_name": args.seed_exclude_genre_name,
            "seed_game_ids": seed_game_ids,
            "seed_game_names_count": len(seed_game_names),
            "seed_game_list_file": str(args.seed_game_list_file) if args.seed_game_list_file else None,
        },
        "summary": summary,
        "seeded_rows": seeded_rows,
        "seed_unresolved_names": seed_unresolved_names,
        "rows": report_rows,
    }
    save_json(args.report_json, report)
    log(f"[IGDB] Report saved: {args.report_json}")
    log(
        "[IGDB] Summary: "
        f"folders={summary['total_folders']}, "
        f"seeded_created={summary['seeded_folders_created']}, "
        f"downloaded={summary['downloaded']}, "
        f"dry_run_resolved={summary['dry_run_resolved']}, "
        f"failed={summary['failed']}, "
        f"skipped_existing_images={summary['skipped_existing_images']}"
    )
    return 0 if summary["failed"] == 0 else 1


def main() -> None:
    try:
        raise SystemExit(run())
    except FetchError as exc:
        log(f"[ERROR] {exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
