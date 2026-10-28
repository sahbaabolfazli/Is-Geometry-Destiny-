#!/usr/bin/env python3
"""phase3d_interference_matrix.py (revised): same experiment, but stores three NLL aggregations.
Keys: nll{S}_<cond>, noise{S}_<cond>, centroid{S}_<cond>, none{S}_<lang>   with S in {"" (mean), "sum", "head"}.
Legacy keys (S="") are unchanged, so old downstream code keeps working."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import phase3_config as cfg
from phase3b_latent_injection import load_model_with_null_pad, INJECTION_POSITION, N_NULL_TOKENS
from phase3_sanity_checks import _all_rows_inject            # tiny hook helper; keep until moved to scoring.py
from scoring import batched_nll_stats
from analysis_common import (LANGS, TEXT_KEY, load_items, item_tables, load_concept_vectors,
                             cond_name, save_json)

CSFX = cfg.CARRIER_SFX
OUT_NPZ = Path(f"phase3d_interference_{cfg.ACTIVE_MODEL_KEY}{CSFX}.npz")
OUT_META = Path(f"phase3d_interference_meta_{cfg.ACTIVE_MODEL_KEY}{CSFX}.json")
VEC_NPZ = Path(f"phase3_concept_vectors_{cfg.ACTIVE_MODEL_KEY}.npz")
N_NOISE = int(os.environ.get("N_NOISE", 8))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
HEAD_K = int(os.environ.get("HEAD_K", 16))
SFX = ("", "sum", "head")


@torch.no_grad()
def score_texts(model, cid, sents, vec=None):
    """-> [mean, total, head], each an array over target texts."""
    hook = _all_rows_inject(model, vec) if vec is not None else None
    parts = []
    try:
        for s in range(0, len(sents), BATCH_SIZE):
            parts.append(batched_nll_stats(model, cid, sents[s:s + BATCH_SIZE], N_NULL_TOKENS, HEAD_K))
    finally:
        if hook is not None:
            hook.remove()
    return [np.concatenate([p[k] for p in parts]) for k in range(3)]


def main():
    t0 = time.time()
    
    # --- Guard: Prevent accidental overwrites without carrier tags ---
    if cfg.CARRIER_TOKEN != cfg.PRIMARY_CARRIER_TOKEN and not cfg.CARRIER_TAG:
        sys.exit("[3d] CARRIER_TOKEN override without CARRIER_TAG would overwrite primary outputs. "
                 "Unset CARRIER_TOKEN, or use CARRIER_TAG=dot (Qwen only).")

    model, tok, cid = load_model_with_null_pad()
    items = load_items()
    domain, source = item_tables(items)
    V = load_concept_vectors(db_path=cfg.CHROMA_DB_PATH)
    by_id = {r["id"]: r for r in items}
    ids = sorted(set(V["en"]) & set(V["zh"]) & set(by_id))
    n = len(ids)
    print(f"[3d] {cfg.TARGET_MODEL_NAME} concepts={n} layer={cfg.LAYER_IDX} pos={INJECTION_POSITION} "
          f"carrier={tok.convert_ids_to_tokens(cid)!r} role={cfg.CARRIER_ROLE}")

    sents = {l: [tok(by_id[i][TEXT_KEY[l]], add_special_tokens=False)["input_ids"] for i in ids] for l in LANGS}
    vecs = {l: np.stack([V[l][i] for i in ids]) for l in LANGS}
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    noise_dirs = rng.standard_normal((N_NOISE, vecs["en"].shape[1]))
    noise_dirs /= np.linalg.norm(noise_dirs, axis=1, keepdims=True)

    res = {}
    for t in LANGS:                                                   # no-injection baseline
        for sfx, arr in zip(SFX, score_texts(model, cid, sents[t])):
            res[f"none{sfx}_{t}"] = arr
    for src in LANGS:
        med_norm = float(np.median(np.linalg.norm(vecs[src], axis=1)))
        centroid = vecs[src].mean(0)
        for tgt in LANGS:
            c = cond_name(src, tgt)
            rows = []
            for j in range(n):
                rows.append(score_texts(model, cid, sents[tgt], vecs[src][j]))   # row j = vector j
                if (j + 1) % 8 == 0:
                    print(f"  [{c}] {j + 1}/{n} ({time.time() - t0:.0f}s)")
            nz = [score_texts(model, cid, sents[tgt], d * med_norm) for d in noise_dirs]
            ce = score_texts(model, cid, sents[tgt], centroid)
            for k, sfx in enumerate(SFX):
                res[f"nll{sfx}_{c}"] = np.stack([r[k] for r in rows])             # (vector j, text i)
                res[f"noise{sfx}_{c}"] = np.stack([z[k] for z in nz])             # (N_NOISE, text i)
                res[f"centroid{sfx}_{c}"] = ce[k]
            M = res[f"nll_{c}"]
            print(f"[3d] {c}: diag={np.diag(M).mean():.4f} offdiag={(M.sum() - np.trace(M)) / (n * n - n):.4f} "
                  f"noise={res[f'noise_{c}'].mean():.4f} none={res[f'none_{tgt}'].mean():.4f}")

    np.savez_compressed(OUT_NPZ, ids=np.array(ids), **res,
                        n_tokens_en=np.array([len(s) for s in sents["en"]]),
                        n_tokens_zh=np.array([len(s) for s in sents["zh"]]),
                        norm_en=np.linalg.norm(vecs["en"], axis=1), norm_zh=np.linalg.norm(vecs["zh"], axis=1))

    # --- Vector handling: primary writes, secondary validates ---
    if not cfg.CARRIER_TAG:
        np.savez_compressed(VEC_NPZ, ids=np.array(ids), en=vecs["en"], zh=vecs["zh"])
    else:
        Z = np.load(VEC_NPZ, allow_pickle=False)
        assert [str(i) for i in Z["ids"]] == ids and np.allclose(Z["en"], vecs["en"]) \
            and np.allclose(Z["zh"], vecs["zh"]), "ChromaDB vectors differ from the primary run"

    save_json(OUT_META, dict(
        model=cfg.TARGET_MODEL_NAME, model_key=cfg.ACTIVE_MODEL_KEY, category=cfg.MODEL_CATEGORY,
        layer_idx=cfg.LAYER_IDX, injection_position=INJECTION_POSITION,
        injection_position_note="0-indexed, i.e. the 5th carrier token",
        n_null_tokens=N_NULL_TOKENS, carrier_token_id=int(cid), carrier_token=tok.convert_ids_to_tokens(cid),
        carrier_token_pinned=cfg.CARRIER_TOKEN, n_noise=N_NOISE, head_k=HEAD_K, n_concepts=n, ids=ids,
        domain={i: domain[i] for i in ids}, source={i: source[i] for i in ids},
        carrier_tag=cfg.CARRIER_TAG or "primary", carrier_role=cfg.CARRIER_ROLE,
        primary_carrier_token=cfg.PRIMARY_CARRIER_TOKEN,
        nll_unit="mean: nats/token; sum: nats/passage; head: mean over first head_k tokens",
        runtime_s=round(time.time() - t0, 1)))

    # --- Audit trail log entry ---
    import json, datetime
    with open("carrier_manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(ts=datetime.datetime.now().isoformat(timespec="seconds"),
                                model_key=cfg.ACTIVE_MODEL_KEY, role=cfg.CARRIER_ROLE, tag=cfg.CARRIER_TAG or "primary",
                                token=cfg.CARRIER_TOKEN, token_id=int(cid), npz=str(OUT_NPZ), n_noise=N_NOISE)) + "\n")

    print(f"[done] {OUT_NPZ} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()