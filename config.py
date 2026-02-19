"""General HF model config data structure parsed from configs.jsonl."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterator

CONFIGS_FILE = Path(__file__).parent / "configs.jsonl"

# Names of fields on ModelConfig that map directly from the JSON
_KNOWN_FIELDS: set[str] | None = None


def _get_known_fields() -> set[str]:
    global _KNOWN_FIELDS
    if _KNOWN_FIELDS is None:
        _KNOWN_FIELDS = {f.name for f in fields(ModelConfig) if f.name != "extras"}
    return _KNOWN_FIELDS


@dataclass
class ModelConfig:
    # ── Metadata ──────────────────────────────────────────────────
    repo_id: str  # renamed from _repo_id
    model_type: str | None = None
    architectures: list[str] | None = None
    transformers_version: str | None = None
    torch_dtype: str | None = None
    name_or_path: str | None = None  # renamed from _name_or_path

    # ── Model dimensions ──────────────────────────────────────────
    vocab_size: int | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    head_dim: int | None = None
    hidden_act: str | None = None

    # ── Position / embedding ──────────────────────────────────────
    max_position_embeddings: int | None = None
    rope_theta: float | None = None
    rope_scaling: dict | None = None

    # ── Regularization ────────────────────────────────────────────
    initializer_range: float | None = None
    attention_dropout: float | None = None
    hidden_dropout_prob: float | None = None
    attention_probs_dropout_prob: float | None = None

    # ── Normalization ─────────────────────────────────────────────
    rms_norm_eps: float | None = None
    layer_norm_eps: float | None = None

    # ── Token IDs ─────────────────────────────────────────────────
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None

    # ── Flags ─────────────────────────────────────────────────────
    use_cache: bool | None = None
    tie_word_embeddings: bool | None = None
    attention_bias: bool | None = None
    mlp_bias: bool | None = None
    is_encoder_decoder: bool | None = None

    # ── Other common fields ───────────────────────────────────────
    sliding_window: int | None = None
    pretraining_tp: int | None = None
    quantization_config: dict | None = None
    id2label: dict | None = None
    label2id: dict | None = None

    # ── Everything else ───────────────────────────────────────────
    extras: dict = field(default_factory=dict)

    # Mapping from JSON key -> dataclass field name (for renamed fields)
    _ALIASES: dict[str, str] = field(
        default=None,  # type: ignore[assignment]
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_dict(cls, raw: dict) -> ModelConfig:
        aliases = {"_repo_id": "repo_id", "_name_or_path": "name_or_path"}
        known = _get_known_fields()
        kwargs: dict = {}
        extras: dict = {}

        for k, v in raw.items():
            field_name = aliases.get(k, k)
            if field_name in known:
                kwargs[field_name] = v
            else:
                extras[k] = v

        kwargs["extras"] = extras
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Reconstruct the original JSON-compatible dict."""
        reverse_aliases = {"repo_id": "_repo_id", "name_or_path": "_name_or_path"}
        d = {}
        for f in fields(self):
            if f.name in ("extras", "_ALIASES"):
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            key = reverse_aliases.get(f.name, f.name)
            d[key] = v
        d.update(self.extras)
        return d


def iter_configs(path: Path = CONFIGS_FILE) -> Iterator[ModelConfig]:
    """Lazily iterate over all configs from the JSONL file."""
    with open(path) as f:
        for line in f:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            yield ModelConfig.from_dict(raw)


def load_configs(path: Path = CONFIGS_FILE) -> list[ModelConfig]:
    """Load all configs into memory."""
    return list(iter_configs(path))
