#!/usr/bin/env python3
"""
KV Cache Analysis: bytes per token across HF model configs over time.

Handles MHA, GQA, MQA, MLA (DeepSeek-style), and hybrid linear attention.
Deduplicates fine-tunes to show unique architectures only.

Usage:
    python kv_cache_analysis.py                      # overall trend only
    python kv_cache_analysis.py --per-type            # + per-attention-type trends
    python kv_cache_analysis.py --per-type MHA GQA    # + only MHA and GQA trends
    python kv_cache_analysis.py --weighted            # weight by model popularity
    python kv_cache_analysis.py --weighted --per-type # both
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    L = _first_int(c, "num_hidden_layers", "n_layer", "n_layers", "num_layers",
                   "decoder_layers", "num_decoder_layers")
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
    n_q = _first_int(c, "num_attention_heads", "n_head", "n_heads", "num_heads",
                     "decoder_attention_heads", "encoder_attention_heads",
                     "attention_heads", "nhead", "num_decoder_attention_heads")
    if n_q is None:
        return None

    # ── head_dim ──
    d_h = _first_int(c, "head_dim")
    if d_h is None:
        hs = _first_int(c, "hidden_size", "d_model", "n_embd", "n_embed", "dim")
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
    downloads: int = 0


@dataclass
class ModelMeta:
    created_at: datetime
    downloads: int


def load_model_meta() -> dict[str, ModelMeta]:
    meta: dict[str, ModelMeta] = {}
    with open(MODEL_LIST_FILE) as f:
        for line in f:
            try:
                obj = json.loads(line)
                rid = obj.get("id")
                ts = obj.get("created_at")
                if rid and ts:
                    dl = obj.get("downloads") or 0
                    if not isinstance(dl, (int, float)):
                        dl = 0
                    meta[rid] = ModelMeta(
                        created_at=datetime.fromisoformat(ts),
                        downloads=int(dl),
                    )
            except (json.JSONDecodeError, ValueError):
                continue
    return meta


def process_all() -> list[Result]:
    print("Loading model metadata...")
    meta = load_model_meta()
    print(f"  {len(meta):,} models with metadata")

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

            m = meta.get(repo_id)
            results.append(
                Result(
                    repo_id=repo_id,
                    attn_type=attn,
                    bytes_per_token=bpt,
                    num_layers=nl,
                    created_at=m.created_at if m else None,
                    downloads=m.downloads if m else 0,
                )
            )

    print(f"  Computed: {len(results):,}  Skipped: {skipped:,}")
    return results


# ── Deduplication ─────────────────────────────────────────────────


def deduplicate(results: list[Result]) -> list[Result]:
    """
    Keep one entry per unique (attn_type, num_layers, bytes_per_token).
    Uses earliest date; sums downloads across all models sharing the fingerprint.
    """
    dated = [r for r in results if r.created_at is not None]
    print(f"  With release dates: {len(dated):,}")

    best: dict[tuple, Result] = {}
    earliest_date: dict[tuple, datetime] = {}
    agg_downloads: dict[tuple, int] = Counter()

    for r in dated:
        key = (r.attn_type, r.num_layers, r.bytes_per_token)
        agg_downloads[key] += r.downloads

        if key not in earliest_date or r.created_at < earliest_date[key]:
            earliest_date[key] = r.created_at

        if key not in best:
            best[key] = r
        elif r.created_at < best[key].created_at:
            best[key] = r

    # write aggregated downloads and earliest date back
    for key, r in best.items():
        r.downloads = agg_downloads[key]
        r.created_at = earliest_date[key]

    unique = list(best.values())
    print(f"  Unique architectures: {len(unique):,}")
    return unique


# ── Plotting ──────────────────────────────────────────────────────

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


def _log_linreg(
    dates: list[datetime],
    values: list[float],
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Linear regression on (timestamp, log(value)).
    Optional *weights* for weighted least squares.
    Returns (x_line, y_line) for plotting.
    """
    if len(dates) < 10:
        return None
    epoch = datetime(2020, 1, 1)
    x = np.array([(d.replace(tzinfo=None) - epoch).total_seconds() / 86400 for d in dates])
    y = np.log(np.array(values, dtype=float))

    coeffs = np.polyfit(x, y, 1, w=weights)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.exp(np.polyval(coeffs, x_line))
    d_line = np.array([epoch + timedelta(days=float(d)) for d in x_line])
    return d_line, y_line


def plot(
    unique: list[Result],
    *,
    log_scale: bool,
    out_path: Path,
    per_type_trends: list[str] | None = None,
    weighted: bool = False,
) -> None:
    """
    Args:
        per_type_trends: list of attention type names to draw individual trend
            lines for (e.g. ["MHA", "GQA"]).  ``None`` means no per-type lines.
        weighted: if True, weight regression by log(1+downloads) and scale
            scatter point sizes by popularity.
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    def _weights(pts: list[Result]) -> np.ndarray | None:
        if not weighted:
            return None
        return np.log1p(np.array([r.downloads for r in pts], dtype=float))

    def _sizes(pts: list[Result]) -> np.ndarray | float:
        if not weighted:
            return 14
        # scale marker area: log1p -> clamp to [6, 120]
        s = np.log1p(np.array([r.downloads for r in pts], dtype=float))
        s = 6 + (s / max(s.max(), 1)) * 114
        return s

    # ── scatter points ──
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
            alpha=0.35,
            s=_sizes(pts),
            edgecolors="none",
            zorder=2,
        )

    # ── regression trend lines (exponential fit: y = a·e^(bx)) ──
    all_dates = [r.created_at for r in unique]
    all_bpt = [float(r.bytes_per_token) for r in unique]

    trend_suffix = " (weighted)" if weighted else ""

    # overall
    overall = _log_linreg(all_dates, all_bpt, _weights(unique))
    if overall is not None:
        ax.plot(
            overall[0], overall[1],
            color="black", lw=2.5, alpha=0.7, zorder=4,
            label=f"Overall trend{trend_suffix}",
        )

    # per-type (only when requested)
    if per_type_trends:
        for attn_type in per_type_trends:
            pts = [r for r in unique if r.attn_type == attn_type]
            if len(pts) < 10:
                print(f"  Skipping {attn_type} trend (only {len(pts)} points)")
                continue
            result = _log_linreg(
                [r.created_at for r in pts],
                [float(r.bytes_per_token) for r in pts],
                _weights(pts),
            )
            if result is not None:
                ax.plot(
                    result[0], result[1],
                    color=TYPE_COLORS.get(attn_type, "#7f7f7f"),
                    lw=2, alpha=0.8, ls="--", zorder=4,
                    label=f"{attn_type} trend{trend_suffix}",
                )

    # ── axes ──
    if log_scale:
        ax.set_yscale("log")
        scale_label = "log"
    else:
        ax.set_yscale("linear")
        scale_label = "linear"
        # cap y-axis at 99th percentile so the exponential curves are visible
        p99 = float(np.percentile(all_bpt, 99))
        ax.set_ylim(0, p99 * 1.1)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_bytes_fmt))
    ax.set_xlabel("Model Release Date", fontsize=12)
    ax.set_ylabel(f"KV Cache Bytes / Token (all layers, fp16, {scale_label})", fontsize=12)
    ax.set_title(
        "KV Cache Memory per Token Over Time\n"
        "Unique architectures, deduplicated across fine-tunes, assuming fp16"
        f" — {scale_label} scale",
        fontsize=13,
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=45)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25, which="both" if log_scale else "major")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────

ALL_ATTN_TYPES = [t for t in TYPE_ORDER]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot KV cache bytes/token over time for HF models.",
    )
    p.add_argument(
        "--per-type",
        nargs="*",
        metavar="TYPE",
        default=None,
        help=(
            "Show per-attention-type trend lines.  "
            "Without arguments: show all types that have enough data.  "
            "With arguments: only the listed types.  "
            f"Choices: {', '.join(ALL_ATTN_TYPES)}"
        ),
    )
    p.add_argument(
        "--weighted",
        action="store_true",
        help=(
            "Weight regression by model popularity (log(1+downloads)).  "
            "Downloads are summed across all fine-tunes sharing an architecture.  "
            "Also scales scatter point sizes by popularity."
        ),
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # resolve --per-type: None=flag absent, []=flag with no args, [..]=explicit list
    if args.per_type is None:
        per_type: list[str] | None = None
    elif len(args.per_type) == 0:
        per_type = list(ALL_ATTN_TYPES)
    else:
        per_type = args.per_type

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

    print("\nPlotting...")
    plot_kw = dict(per_type_trends=per_type, weighted=args.weighted)
    plot(unique, log_scale=True, out_path=OUTPUT_PLOT, **plot_kw)
    plot(
        unique,
        log_scale=False,
        out_path=OUTPUT_PLOT.with_stem("kv_bytes_per_token_linear"),
        **plot_kw,
    )


if __name__ == "__main__":
    main()
