#!/usr/bin/env python3
"""Shared configuration for the Phase 3 scripts (extraction, injection,
causal patching). All phase3*.py scripts import from here — this is what
keeps every script reading the SAME model/layer/hidden-dim without drift.

[CHANGED FOR THE EXPANDED PROJECT] This used to be a single active
TARGET_MODEL_NAME with the other two models commented out below it — the
exact pattern that caused the LMT/Qwen mix-up earlier (editing LAYER_IDX
without also editing TARGET_MODEL_NAME). It's now a MODEL_REGISTRY dict,
keyed by a short model id, each entry carrying its own name/layers AND a
MODEL_CATEGORY tag ("general_purpose" vs "translation_specialized") — the
category is now your headline independent variable, so it has to live in
the config, not just in your head. Switching models is one line
(ACTIVE_MODEL_KEY below), never multiple lines that can go out of sync.
"""
import torch
import os
from typing import TypedDict
from transformers import AutoModelForCausalLM


class _ModelReq(TypedDict):
    name: str
    layer_idx: int
    probe_layers: list[int]
    category: str


class ModelConfig(_ModelReq, total=False):
    carrier_token: str          # optional: pin after the sanity run, same token in both models of a family
    exclude_leading_n: int      # from the sanity sweep's recommended_exclude_leading_n; must be identical within a family
    secondary_carriers: dict[str, str]
# ----------------------------------------------------------------------------
# MODEL REGISTRY — one entry per model in the study, both existing and new.
# ----------------------------------------------------------------------------
# category: "general_purpose" (general instruct/chat model) or
#           "translation_specialized" (trained specifically for MT).
# This is the variable Step 2 of the research sequence is actually testing —
# keep it accurate per model, since your whole reframed hypothesis hangs on it.
#
# After the carrier sanity run, pin the chosen token per family, e.g.
#   carrier_token="<|endoftext|>"
# in BOTH registry entries of the family (lineage_check.py verifies they match).
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen3_8b": ModelConfig(
        name="Qwen/Qwen3-8B-Base",
        layer_idx=18, probe_layers=[23, 27, 31, 33, 35],
        category="general_purpose",
        exclude_leading_n = 1,
        carrier_token= "\n",
        secondary_carriers = {"dot": "."}
    ),
    "lmt_60_8b": ModelConfig(
        name="NiuTrans/LMT-60-8B-Base",
        layer_idx=18, probe_layers=[23, 27, 31, 33, 35],
        category="translation_specialized",
        exclude_leading_n = 1,
        carrier_token= "\n",
        secondary_carriers = {"dot": "."} 
    ),
    "gemma2_9b": ModelConfig(
        name="google/gemma-2-9b",
        layer_idx=21, probe_layers=[26, 31, 36, 38, 41],
        category="general_purpose",
        exclude_leading_n = 2,
        carrier_token= "<pad>"
    ),
    "gemmax2_28_9b": ModelConfig(
        name="ModelSpace/GemmaX2-28-9B-Pretrain",
        layer_idx=21, probe_layers=[26, 31, 36, 38, 41],
        category="translation_specialized",
        exclude_leading_n = 2,
        carrier_token= "<pad>"
    ),
    "llama2_7b": ModelConfig(
        name="meta-llama/Llama-2-7b-hf",
        layer_idx=16, probe_layers=[20, 24, 28, 29, 31],
        category="general_purpose",
        exclude_leading_n = 1,
        carrier_token= "<unk>"
    ),
    "towerbase_7b": ModelConfig(
        name="Unbabel/TowerBase-7B-v0.1",
        layer_idx=16, probe_layers=[20, 24, 28, 29, 31],
        category="translation_specialized",
        exclude_leading_n = 1,
        carrier_token= "<unk>"
    ),
}


# The three primary (general, specialized) pairs, by family.
# PAIRS = {
#     "qwen3":  ("qwen3_8b", "lmt_60_8b"),
#     "gemma2": ("gemma2_9b", "gemmax2_28_9b"),
#     "llama2": ("llama2_7b", "towerbase_7b"),
# }
# ----------------------------------------------------------------------------
# ACTIVE MODEL — the ONLY line you change to switch models.
# ----------------------------------------------------------------------------
ACTIVE_MODEL_KEY = os.environ.get("MODEL_KEY", "llama2_7b")

_active = MODEL_REGISTRY[ACTIVE_MODEL_KEY]
TARGET_MODEL_NAME = _active["name"]
LAYER_IDX = _active["layer_idx"]
PROBE_LAYERS = _active["probe_layers"]
MODEL_CATEGORY = _active["category"]          # "general_purpose" | "translation_specialized" — read this in any analysis script that groups by category.


PRIMARY_CARRIER_TOKEN = _active.get("carrier_token")
CARRIER_TAG = os.environ.get("CARRIER_TAG", "").strip()      # "" = primary run (default for every model)
if CARRIER_TAG:
    _sec = _active.get("secondary_carriers", {})
    if CARRIER_TAG not in _sec:                               # Llama/Gemma land here -> hard stop
        raise SystemExit(f"[config] CARRIER_TAG={CARRIER_TAG!r} is not defined for {ACTIVE_MODEL_KEY}. "
                         "Secondary carriers exist for the Qwen family only (qwen3_8b, lmt_60_8b).")
    if os.environ.get("CARRIER_TOKEN"):
        raise SystemExit("[config] CARRIER_TAG and CARRIER_TOKEN are mutually exclusive.")
    CARRIER_TOKEN, CARRIER_ROLE = _sec[CARRIER_TAG], "secondary"
else:
    CARRIER_TOKEN = os.environ.get("CARRIER_TOKEN") or PRIMARY_CARRIER_TOKEN
    CARRIER_ROLE = "primary"
CARRIER_SFX = f"_c{CARRIER_TAG}" if CARRIER_TAG else ""       # appended to carrier-dependent artefact names


def require_primary_carrier(script):
    """Call at the top of carrier-independent / primary-only scripts (extraction, 3b, compute_density_proxy).
    Refuses BOTH a secondary tag and a bare CARRIER_TOKEN override, either of which would write
    non-primary results into primary-named artefacts."""
    if CARRIER_TAG:
        raise SystemExit(f"[{script}] run without CARRIER_TAG (carrier-independent or primary-only).")
    if CARRIER_TOKEN != PRIMARY_CARRIER_TOKEN:
        raise SystemExit(f"[{script}] CARRIER_TOKEN override {CARRIER_TOKEN!r} differs from the pinned primary "
                         f"carrier {PRIMARY_CARRIER_TOKEN!r}; unset CARRIER_TOKEN (this script writes primary artefacts).")

def resolve_token_id(tok, s):
    """Token string -> id. Uses the vocab entry if one exists; otherwise encodes `s` as text and
    takes the last id (e.g. ' ' or '\\n' in sentencepiece vocabs). Returns None if unusable."""
    tid = tok.convert_tokens_to_ids(s)
    in_vocab = tid is not None and (tid != tok.unk_token_id or s == tok.unk_token)
    if in_vocab:
        return int(tid)
    if s.startswith("<") and s.endswith(">"):        # looks like a special token this tokenizer lacks
        return None
    ids = tok.encode(s, add_special_tokens=False)
    return int(ids[-1]) if ids else None


EXCLUDE_NORM_OUTLIER_FACTOR = 5.0
EXCLUDE_LEADING_N = _active.get("exclude_leading_n", 2)              
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# STORAGE — separate DB per model so re-running one model can never silently
# contaminate another's stored vectors (this used to be a single shared path).
# ----------------------------------------------------------------------------
CHROMA_DB_PATH = f"./phase3_chroma_db_{ACTIVE_MODEL_KEY}"

# ----------------------------------------------------------------------------
# DEVICE
# ----------------------------------------------------------------------------
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.bfloat16


def load_causal_lm():
    """Single loader for every script. Same dtype everywhere; eager attention for Gemma2 soft-capping."""
    return AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_NAME, torch_dtype=DTYPE, attn_implementation="eager"
    ).to(DEVICE).eval()


def assert_layers_exist(model):
    """Fail loudly if the loaded model doesn't actually have the layers this
    config assumes. Run this FIRST after loading any new model
    their true layer counts may differ slightly from the
    placeholder values above, and this will tell you exactly how to fix it."""
    n_layers = len(model.model.layers)
    needed = max([LAYER_IDX] + PROBE_LAYERS)
    if needed >= n_layers:
        raise ValueError(
            f"{TARGET_MODEL_NAME} has {n_layers} decoder layers (valid indices "
            f"0..{n_layers - 1}), but MODEL_REGISTRY['{ACTIVE_MODEL_KEY}'] needs "
            f"layer index {needed}. Update layer_idx/probe_layers for this key "
            f"in MODEL_REGISTRY to fit this model's actual depth."
        )
    print(f"[config] active model={TARGET_MODEL_NAME}  category={MODEL_CATEGORY}  "
          f"n_layers={n_layers}  layer_idx={LAYER_IDX}  probe_layers={PROBE_LAYERS}  "
          f"db={CHROMA_DB_PATH}")