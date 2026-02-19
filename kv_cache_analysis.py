#!/usr/bin/env python3
"""
KV Cache Analysis: bytes per token across HF model configs over time.

Handles MHA, GQA, MQA, MLA (DeepSeek-style), and hybrid linear attention.
Deduplicates fine-tunes to show unique architectures only.
"""

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).parent
CONFIGS_FILE = ROOT / "configs.jsonl"
MODEL_LIST_FILE = ROOT / "model_list.jsonl"
OUTPUT_PLOT = ROOT / "kv_bytes_per_token.png"

# Use fp16 for all models so the comparison reflects architecture, not dtype
BYTES_PER_ELEM = 2


# ── KV cache computation ─────────────────────────────────────────


def _resolve(raw: dict) -> dict:
    """Flatten text_config into top level for multimodal models."""
    tc = raw.get("text_config")
    if isinstance(tc, dict):
        merged = dict(raw)
        merged.update(tc)
        return merged
    return raw


def _first_int(c: dict, *keys: str) -> int | None:
    for k in keys:
        v = c.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return None


def compute_kv_bytes_per_token(raw: dict) -> tuple[str, int, int] | None:
    """
    Returns (attention_type, bytes_per_token, num_layers) or None.

    Formulas (all assume fp16, per token, summed over all layers):
      MHA:  2 * n_heads * d_head * L * 2
      GQA:  2 * n_kv_heads * d_head * L * 2
      MQA:  2 * 1 * d_head * L * 2
      MLA:  (kv_lora_rank + qk_rope_head_dim) * L * 2   (joint KV, no factor of 2)
      Hybrid: only full-attention layers counted (linear layers use fixed-size state)
    """
    c = _resolve(raw)

    # ── num_layers ──
    L = _first_int(c, "num_hidden_layers", "n_layer", "n_layers", "num_layers")
    if L is None:
        return None

    # ── MLA check (DeepSeek-V2/V3, GLM-MoE-DSA, etc.) ──
    kv_lr = c.get("kv_lora_rank")
    if isinstance(kv_lr, (int, float)) and kv_lr > 0:
        qk_rope = c.get("qk_rope_head_dim", 0)
        if not isinstance(qk_rope, (int, float)):
            qk_rope = 0
        bpt = int(kv_lr + qk_rope) * L * BYTES_PER_ELEM
        return ("MLA", bpt, L)

    # ── num_attention_heads ──
    n_q = _first_int(c, "num_attention_heads", "n_head", "n_heads", "num_heads")
    if n_q is None:
        return None

    # ── head_dim ──
    d_h = _first_int(c, "head_dim")
    if d_h is None:
        hs = _first_int(c, "hidden_size", "d_model", "n_embd", "dim")
        if hs is not None:
            d_h = hs // n_q
        else:
            d_h = _first_int(c, "d_kv", "d_head", "kv_channels")
    if d_h is None or d_h <= 0:
        return None

    # ── hybrid linear + full attention (Qwen3.5 style) ──
    lt = c.get("layer_types")
    if isinstance(lt, list) and "linear_attention" in lt:
        full = sum(1 for x in lt if x != "linear_attention")
        if full == 0:
            return None
        n_kv = c.get("num_key_value_heads")
        if not isinstance(n_kv, int) or n_kv <= 0:
            n_kv = n_q
        bpt = 2 * n_kv * d_h * full * BYTES_PER_ELEM
        sub = "GQA" if n_kv < n_q else "MHA"
        return (f"Hybrid ({sub}+Linear)", bpt, L)

    # ── num_kv_heads ──
    n_kv = c.get("num_key_value_heads")
    if n_kv is None:
        n_kv = c.get("num_kv_heads") or c.get("n_head_kv")
    if n_kv is None and c.get("multi_query_attention") is True:
        n_kv = 1
    if n_kv is None and c.get("multi_query") is True:
        n_kv = 1

    # per-layer varying kv heads (e.g. OpenELM)
    if isinstance(n_kv, list):
        if all(isinstance(x, (int, float)) for x in n_kv):
            bpt = 2 * sum(int(h) * d_h for h in n_kv) * BYTES_PER_ELEM
            return ("GQA (per-layer)", bpt, L)
        return None

    if not isinstance(n_kv, (int, float)) or n_kv <= 0:
        n_kv = n_q  # default: MHA
    n_kv = int(n_kv)

    # ── classify ──
    if n_kv == n_q:
        attn = "MHA"
    elif n_kv == 1:
        attn = "MQA"
    else:
        attn = "GQA"

    bpt = 2 * n_kv * d_h * L * BYTES_PER_ELEM
    return (attn, bpt, L)


# ── Data loading ──────────────────────────────────────────────────


@dataclass
class Result:
    repo_id: str
    attn_type: str
    bytes_per_token: int
    num_layers: int
    created_at: datetime | None = None


def load_dates() -> dict[str, datetime]:
    dates: dict[str, datetime] = {}
    with open(MODEL_LIST_FILE) as f:
        for line in f:
            try:
                obj = json.loads(line)
                rid = obj.get("id")
                ts = obj.get("created_at")
                if rid and ts:
                    dates[rid] = datetime.fromisoformat(ts)
            except (json.JSONDecodeError, ValueError):
                continue
    return dates


def process_all() -> list[Result]:
    print("Loading model release dates...")
    dates = load_dates()
    print(f"  {len(dates):,} models with dates")

    print("Computing KV cache bytes/token for all configs...")
    results: list[Result] = []
    skipped = 0

    with open(CONFIGS_FILE) as f:
        for line in f:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue

            repo_id = raw.get("_repo_id", "")
            out = compute_kv_bytes_per_token(raw)
            if out is None:
                skipped += 1
                continue

            attn, bpt, nl = out

            # sanity: skip clearly broken configs
            if bpt <= 0 or bpt > 500_000_000:
                skipped += 1
                continue

            results.append(
                Result(
                    repo_id=repo_id,
                    attn_type=attn,
                    bytes_per_token=bpt,
                    num_layers=nl,
                    created_at=dates.get(repo_id),
                )
            )

    print(f"  Computed: {len(results):,}  Skipped: {skipped:,}")
    return results


# ── Deduplication ─────────────────────────────────────────────────


def deduplicate(results: list[Result]) -> list[Result]:
    """Keep one entry per unique (attn_type, num_layers, bytes_per_token), earliest date."""
    dated = [r for r in results if r.created_at is not None]
    print(f"  With release dates: {len(dated):,}")

    best: dict[tuple, Result] = {}
    for r in dated:
        key = (r.attn_type, r.num_layers, r.bytes_per_token)
        if key not in best or r.created_at < best[key].created_at:
            best[key] = r

    unique = list(best.values())
    print(f"  Unique architectures: {len(unique):,}")
    return unique


# ── Plotting ──────────────────────────────────────────────────────

NOTABLE_MODELS = {
    "openai-community/gpt2": "GPT-2",
    "google-bert/bert-base-uncased": "BERT",
    "bigscience/bloom": "BLOOM 176B",
    "tiiuae/falcon-40b": "Falcon 40B",
    "meta-llama/Llama-2-7b-hf": "Llama 2 7B",
    "meta-llama/Llama-2-70b-hf": "Llama 2 70B",
    "mistralai/Mistral-7B-v0.1": "Mistral 7B",
    "microsoft/phi-2": "Phi-2",
    "meta-llama/Meta-Llama-3-8B": "Llama 3 8B",
    "meta-llama/Meta-Llama-3-70B": "Llama 3 70B",
    "meta-llama/Llama-3.1-405B": "Llama 3.1 405B",
    "google/gemma-2-27b": "Gemma 2 27B",
    "Qwen/Qwen2-72B": "Qwen2 72B",
    "deepseek-ai/DeepSeek-V2": "DeepSeek-V2",
    "deepseek-ai/DeepSeek-V3": "DeepSeek-V3",
    "Qwen/Qwen3-235B-A22B": "Qwen3 235B",
}

TYPE_COLORS = {
    "MHA": "#1f77b4",
    "GQA": "#2ca02c",
    "MQA": "#ff7f0e",
    "MLA": "#d62728",
    "GQA (per-layer)": "#9467bd",
    "Hybrid (GQA+Linear)": "#e377c2",
    "Hybrid (MHA+Linear)": "#bcbd22",
}

TYPE_ORDER = [
    "MHA",
    "GQA",
    "MQA",
    "MLA",
    "GQA (per-layer)",
    "Hybrid (GQA+Linear)",
    "Hybrid (MHA+Linear)",
]


def _bytes_fmt(val: float, _pos: object = None) -> str:
    if val >= 1_048_576:
        return f"{val / 1_048_576:.1f} MiB"
    if val >= 1024:
        return f"{val / 1024:.0f} KiB"
    return f"{val:.0f} B"


def plot(unique: list[Result]) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))

    for attn_type in TYPE_ORDER:
        pts = [r for r in unique if r.attn_type == attn_type]
        if not pts:
            continue
        xs = [r.created_at for r in pts]
        ys = [r.bytes_per_token for r in pts]
        ax.scatter(
            xs,
            ys,
            c=TYPE_COLORS.get(attn_type, "#7f7f7f"),
            label=f"{attn_type} ({len(pts):,})",
            alpha=0.45,
            s=18,
            edgecolors="none",
            zorder=2,
        )

    # annotate notable models
    for r in unique:
        label = NOTABLE_MODELS.get(r.repo_id)
        if label is None:
            continue
        ax.annotate(
            label,
            xy=(r.created_at, r.bytes_per_token),
            xytext=(12, 6),
            textcoords="offset points",
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            zorder=5,
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_bytes_fmt))
    ax.set_xlabel("Model Release Date", fontsize=12)
    ax.set_ylabel("KV Cache Bytes / Token (all layers, fp16)", fontsize=12)
    ax.set_title(
        "KV Cache Memory per Token Over Time\n"
        "Unique architectures, deduplicated across fine-tunes, assuming fp16",
        fontsize=13,
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=45)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, which="both")

    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot to {OUTPUT_PLOT}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    results = process_all()

    # type breakdown (all models)
    type_counts = Counter(r.attn_type for r in results)
    print("\nAttention type breakdown (all models):")
    for t, c in type_counts.most_common():
        print(f"  {t:30s} {c:>8,}")

    unique = deduplicate(results)

    # type breakdown (unique)
    u_counts = Counter(r.attn_type for r in unique)
    print("\nAttention type breakdown (unique architectures):")
    for t, c in u_counts.most_common():
        print(f"  {t:30s} {c:>8,}")

    bpts = np.array([r.bytes_per_token for r in unique])
    print(f"\nKV bytes/token (unique, fp16):")
    print(f"  min:    {_bytes_fmt(bpts.min()):>12s}")
    print(f"  median: {_bytes_fmt(np.median(bpts)):>12s}")
    print(f"  mean:   {_bytes_fmt(bpts.mean()):>12s}")
    print(f"  max:    {_bytes_fmt(bpts.max()):>12s}")

    plot(unique)


if __name__ == "__main__":
    main()
