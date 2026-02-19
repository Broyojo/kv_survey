#!/usr/bin/env python3
"""
Download config.json + full ModelInfo for ALL models on Hugging Face Hub.
Handles ~2.6M models with async I/O, rate limiting, and resume support.

Usage:
    pip install aiohttp aiofiles huggingface_hub tqdm
    # Optional: export HF_TOKEN=hf_... for higher rate limits
    python download_all_configs.py

Output:
    - model_list.jsonl:  full ModelInfo per line (phase 1)
    - configs.jsonl:     config.json + repo_id per line (phase 2)
    - failed.jsonl:      repos that failed config download (for retry)
"""

import asyncio
import aiohttp
import aiofiles
import json
import os
import time
import signal
from huggingface_hub import HfApi
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────
CONCURRENCY = 30           # parallel downloads
RATE_LIMIT_PER_SEC = 35   # max requests/sec
MODEL_LIST_FILE = "model_list.jsonl"  # full ModelInfo
CONFIG_FILE = "configs.jsonl"          # config.json contents
FAILED_FILE = "failed.jsonl"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0

# siblings (file listings) add ~28KB per model.
# True  → full scrape including file lists (~70GB+ for model list alone)
# False → everything except siblings (~2-3GB for model list)
INCLUDE_SIBLINGS = False


# ── JSON serializer for HF objects ────────────────────────────
def serialize_hf_object(obj):
    """Handle datetime, RepoSibling, dataclass-like HF objects."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def model_info_to_dict(model, include_siblings=INCLUDE_SIBLINGS) -> dict:
    """Convert a ModelInfo object to a fully serializable dict."""
    d = {}
    for k, v in vars(model).items():
        if not include_siblings and k == "siblings":
            d[k] = None  # placeholder so schema is consistent
            continue
        # Force serialization round-trip to catch any nested HF objects
        try:
            json.dumps(v)
            d[k] = v
        except (TypeError, ValueError):
            d[k] = json.loads(json.dumps(v, default=serialize_hf_object))
    return d


# ── Step 1: Enumerate all models ──────────────────────────────
def build_model_list() -> list[str]:
    """
    Fetch ALL models via list_models(full=True) and save full ModelInfo.
    Returns list of repo_ids for phase 2.
    """
    # If we already have a COMPLETE model list, just read repo_ids from it
    if os.path.exists(MODEL_LIST_FILE):
        print(f"Loading cached model list from {MODEL_LIST_FILE}...")
        repo_ids = []
        with open(MODEL_LIST_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    repo_ids.append(obj["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  Loaded {len(repo_ids):,} models from cache.")
        print(f"  (Delete {MODEL_LIST_FILE} to re-fetch.)")
        return repo_ids

    # Write to a temp file; only rename to final name on success.
    # This way a partial/interrupted run won't be mistaken for complete.
    tmp_file = MODEL_LIST_FILE + ".tmp"

    print("Fetching full model list from HF Hub API...")
    print("  (This takes 15-45 min for 2.6M models)")
    api = HfApi(token=HF_TOKEN)

    repo_ids = []
    count = 0

    with open(tmp_file, "w") as f:
        for model in api.list_models(full=True):
            entry = model_info_to_dict(model)
            f.write(json.dumps(entry, default=serialize_hf_object) + "\n")
            repo_ids.append(model.id)
            count += 1
            if count % 50000 == 0:
                print(f"  Listed {count:,} models...")
                f.flush()

    # Atomic rename: only exists as MODEL_LIST_FILE if enumeration completed
    os.rename(tmp_file, MODEL_LIST_FILE)
    print(f"  Done. Total: {len(repo_ids):,} models.")
    return repo_ids


# ── Step 2: Async download configs ────────────────────────────
class RateLimiter:
    """Token bucket rate limiter."""
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens < 1:
                await asyncio.sleep((1 - self.tokens) / self.rate)
                self.tokens = 0
            else:
                self.tokens -= 1


async def fetch_config(
    session: aiohttp.ClientSession,
    repo_id: str,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict | None, str | None]:
    """Download config.json for a single repo."""
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"

    for attempt in range(RETRY_ATTEMPTS):
        await rate_limiter.acquire()
        async with semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        try:
                            config = json.loads(text)
                            return (repo_id, config, None)
                        except json.JSONDecodeError:
                            return (repo_id, None, "invalid_json")
                    elif resp.status == 404:
                        return (repo_id, None, "no_config")
                    elif resp.status == 403:
                        return (repo_id, None, "gated")
                    elif resp.status == 429:
                        wait = RETRY_BACKOFF ** (attempt + 2)
                        await asyncio.sleep(wait)
                        continue
                    elif resp.status >= 500:
                        await asyncio.sleep(RETRY_BACKOFF ** attempt)
                        continue
                    else:
                        return (repo_id, None, f"http_{resp.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(RETRY_BACKOFF ** attempt)
                continue

    return (repo_id, None, "max_retries")


async def download_all_configs(repo_ids: list[str]):
    """Download config.json for all models with resume support."""

    # Resume: find already-processed repo_ids
    completed = set()
    for path in [CONFIG_FILE, FAILED_FILE]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        completed.add(obj.get("_repo_id") or obj.get("repo_id"))
                    except (json.JSONDecodeError, KeyError):
                        pass

    remaining = [r for r in repo_ids if r not in completed]
    print(f"  Already processed: {len(completed):,}")
    print(f"  Remaining: {len(remaining):,}")

    if not remaining:
        print("All done!")
        return

    rate_limiter = RateLimiter(RATE_LIMIT_PER_SEC)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    headers = {"User-Agent": "hf-config-scraper/1.0"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    stats = {"success": 0, "no_config": 0, "gated": 0, "error": 0}
    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        print("\nGraceful shutdown requested... finishing current batch.")
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)

    async with aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300) as connector:
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            async with aiofiles.open(CONFIG_FILE, "a") as out_f, \
                        aiofiles.open(FAILED_FILE, "a") as fail_f:

                pbar = tqdm(total=len(remaining), desc="Configs", unit="model")
                batch_size = CONCURRENCY * 4

                for i in range(0, len(remaining), batch_size):
                    if shutdown:
                        break

                    batch = remaining[i : i + batch_size]
                    tasks = [
                        fetch_config(session, repo_id, rate_limiter, semaphore)
                        for repo_id in batch
                    ]

                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        if isinstance(result, Exception):
                            stats["error"] += 1
                            pbar.update(1)
                            continue

                        repo_id, config, error = result

                        if config is not None:
                            if not isinstance(config, dict):
                                stats["error"] += 1
                                pbar.update(1)
                                continue
                            record = {"_repo_id": repo_id, **config}
                            await out_f.write(json.dumps(record) + "\n")
                            stats["success"] += 1
                        else:
                            await fail_f.write(
                                json.dumps({"repo_id": repo_id, "error": error}) + "\n"
                            )
                            if error == "no_config":
                                stats["no_config"] += 1
                            elif error == "gated":
                                stats["gated"] += 1
                            else:
                                stats["error"] += 1

                        pbar.update(1)

                    await out_f.flush()
                    await fail_f.flush()

                    pbar.set_postfix(
                        ok=stats["success"],
                        skip=stats["no_config"],
                        gated=stats["gated"],
                        err=stats["error"],
                    )

                pbar.close()

    print(f"\nConfig download complete:")
    print(f"  Configs saved:  {stats['success']:,}")
    print(f"  No config.json: {stats['no_config']:,}")
    print(f"  Gated/private:  {stats['gated']:,}")
    print(f"  Errors:         {stats['error']:,}")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HuggingFace Full Scraper - ModelInfo + config.json")
    print("=" * 60)

    # Phase 1: enumerate all models, save full ModelInfo
    print(f"\n[Phase 1] Model list → {MODEL_LIST_FILE}")
    repo_ids = build_model_list()

    # Phase 2: download config.json for each
    print(f"\n[Phase 2] Config download → {CONFIG_FILE}")
    print(f"  Total models: {len(repo_ids):,}")
    print(f"  Concurrency:  {CONCURRENCY}")
    print(f"  Rate limit:   {RATE_LIMIT_PER_SEC}/sec")
    print(f"  HF Token:     {'set' if HF_TOKEN else 'NOT SET (recommended)'}")
    print(f"  Resume:       {'yes' if os.path.exists(CONFIG_FILE) else 'fresh start'}")
    print(f"  Ctrl+C for graceful shutdown\n")

    asyncio.run(download_all_configs(repo_ids))

    print(f"\n{'=' * 60}")
    print(f"Output files:")
    print(f"  {MODEL_LIST_FILE}  - full ModelInfo for every model")
    print(f"  {CONFIG_FILE}      - config.json contents")
    print(f"  {FAILED_FILE}      - failed downloads (retry candidates)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
