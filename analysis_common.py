#!/usr/bin/env python3
"""analysis_common.py — shared, MODEL-FREE helpers for the Phase 3d / 3e / 4 analysis layer.

Only numpy + scipy (chromadb is imported lazily, only when vectors are read from the DB).
Nothing here imports torch/transformers, so every statistics script can be re-run on a laptop
from the saved .npz/.json artefacts without touching a GPU.

Conventions used everywhere downstream:
  * NLL is mean nats/token over ALL continuation tokens (same scorer as phase3c / sanity C2).
  * "Potency" / "effect" values are signed so that POSITIVE = the injected vector helps more
    than the comparison (lower NLL), matching phase3c's `delta = noise - real` convention.
  * Interference conditions are named <vector_lang>2<text_lang>: en2en, zh2zh (monolingual),
    en2zh, zh2en (cross-lingual).
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

PHASE2_INPUT_PATH = Path(os.environ.get("PHASE2_INPUT_PATH", "phase2_final_48_local.json"))
APPROVED = ("APPROVED_FIRST_PASS", "APPROVED_AFTER_REVISION")
LANGS = ("en", "zh")
TEXT_KEY = {"en": "english", "zh": "chinese"}
DOMAINS = ("math_physics", "complex_systems", "philosophy", "syntax_baseline")
ABSTRACT_DOMAINS = ("math_physics", "complex_systems", "philosophy")
CONDITIONS = ("en2en", "zh2zh", "en2zh", "zh2en")
MONO_FOR_TEXT = {"en": "en2en", "zh": "zh2zh"}          # monolingual reference for a target-text language
CROSS_FOR_TEXT = {"en": "zh2en", "zh": "en2zh"}         # cross-lingual condition scored on that text language

# Mirrors phase3_config.MODEL_REGISTRY (duplicated so analysis never needs torch).
# lineage_check.py asserts these stay in sync with the registry.
FAMILIES = {
    "qwen3": ("qwen3_8b", "lmt_60_8b"),
    "gemma2": ("gemma2_9b", "gemmax2_28_9b"),
    "llama2": ("llama2_7b", "towerbase_7b"),
}
CATEGORY = {g: "general_purpose" for g, _ in FAMILIES.values()}
CATEGORY.update({s: "translation_specialized" for _, s in FAMILIES.values()})
FAMILY_OF = {k: fam for fam, pair in FAMILIES.items() for k in pair}
ALL_MODEL_KEYS = [k for pair in FAMILIES.values() for k in pair]

N_PERM = int(os.environ.get("N_PERM", 10000))
N_BOOT = int(os.environ.get("N_BOOT", 10000))
SEED = 42

# ------------------------------------------------------------------ NLL aggregation tag
# NLL aggregation used by phase3d/phase4: "mean" (nats/token, legacy), "sum" (nats/passage),
# "head" (mean over first head_k tokens). Mirrors the suffixes stored by
# phase3d_interference_matrix.py (SFX = "", "sum", "head").
NLL_KIND = os.environ.get("NLL_KIND", "mean")
if NLL_KIND not in ("mean", "sum", "head"):
    raise ValueError(f"NLL_KIND must be one of mean/sum/head, got {NLL_KIND!r}")
KIND_SFX = "" if NLL_KIND == "mean" else NLL_KIND 


# Optional suffix appended to every analysis artefact name (RUN_TAG env) and to the
# NLL_KIND-based tag. Empty by default so main-run file names are stable.
RUN_TAG = os.environ.get("RUN_TAG", "")
SECONDARY_CARRIERS = {"dot": "."}
SECONDARY_CARRIER_FAMILIES = ("qwen3",)
CARRIER_TAG = os.environ.get("CARRIER_TAG", "").strip()
if CARRIER_TAG and CARRIER_TAG not in SECONDARY_CARRIERS:
    raise ValueError(f"unknown CARRIER_TAG {CARRIER_TAG!r}")
CARRIER_SFX = f"_c{CARRIER_TAG}" if CARRIER_TAG else ""

def require_carrier_scope(model_key):
    """Secondary-carrier artefacts exist only for the Qwen family."""
    if CARRIER_TAG and FAMILY_OF[model_key] not in SECONDARY_CARRIER_FAMILIES:
        raise SystemExit(f"CARRIER_TAG={CARRIER_TAG} is Qwen-only; {model_key} is single-carrier by design.")

def out_tag(carrier_sfx=None):
    """NLL_KIND tag + carrier suffix + RUN_TAG. carrier_sfx=None -> current env; '' -> primary; '_cdot' -> secondary."""
    sfx = CARRIER_SFX if carrier_sfx is None else carrier_sfx
    return ("" if NLL_KIND == "mean" else f"_{NLL_KIND}") + sfx + RUN_TAG


def exclude_ids():
    """Concept ids to drop, from EXCLUDE_IDS_FILE (a JSON list of ids) or empty set."""
    p = os.environ.get("EXCLUDE_IDS_FILE")
    if p and Path(p).exists():
        return set(json.loads(Path(p).read_text(encoding="utf-8")))
    return set()


def cond_name(src, tgt):
    return f"{src}2{tgt}"


def chroma_path(model_key):
    """Must match phase3_config.CHROMA_DB_PATH."""
    return f"./phase3_chroma_db_{model_key}"


# ------------------------------------------------------------------ dataset
def load_items(path=PHASE2_INPUT_PATH):
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    return [r for r in recs if r["verification_status"] in APPROVED]


_SOURCE_KEYS = ("source_type", "source", "origin", "provenance", "collection_method",
                "generated_by", "author_type", "item_source", "creator")
_HUMAN_WORDS = ("human", "manual", "expert", "hand", "curated", "collected")
_AI_WORDS = ("ai", "llm", "gpt", "generated", "synthetic", "model", "claude", "gemini", "machine")


def _human_id_override():
    p = os.environ.get("HUMAN_IDS_FILE")
    if p and Path(p).exists():
        return set(json.loads(Path(p).read_text(encoding="utf-8")))
    return None


def source_of(rec, _override=None):
    if _override is not None:
        return "human" if rec["id"] in _override else "ai"
    st = str(rec.get("source_type", "")).strip().lower()      # explicit, unambiguous mapping first
    sw = st.replace("-", " ").replace("_", " ").split()
    if st == "human" or any(w.startswith(h) for w in sw for h in _HUMAN_WORDS):
        return "human"
    if st in ("synthetic", "ai", "llm", "generated") or any(w.startswith(a) for w in sw for a in _AI_WORDS):
        return "ai"
    if "is_human" in rec:
        return "human" if rec["is_human"] else "ai"
    for k in _SOURCE_KEYS[1:]:                                 # fall back to the old heuristic
        v = rec.get(k)
        if v is None:
            continue
        words = str(v).lower().replace("-", " ").replace("_", " ").split()
        if any(w.startswith(h) for w in words for h in _HUMAN_WORDS):
            return "human"
        if any(w.startswith(a) for w in words for a in _AI_WORDS):
            return "ai"
    return "unknown"


def item_tables(items):
    ov = _human_id_override()
    domain = {r["id"]: r.get("domain", "unknown") for r in items}
    source = {r["id"]: source_of(r, ov) for r in items}
    return domain, source


# ------------------------------------------------------------------ vectors
def load_concept_vectors(model_key=None, db_path=None):
    """{'en': {id: vec}, 'zh': {id: vec}} from the per-model ChromaDB written by the extraction pipeline."""
    import chromadb                                   # lazy: only needed when reading the DB
    client = chromadb.PersistentClient(path=db_path or chroma_path(model_key))
    out = {}
    for lang in LANGS:
        got = client.get_collection(f"concept_{lang}_geometry").get(include=["embeddings", "metadatas"])
        out[lang] = {m["sentence_id"]: np.asarray(v, dtype=np.float64)
                     for m, v in zip(got["metadatas"], got["embeddings"])}
    return out


def unit(a, axis=-1):
    a = np.asarray(a, dtype=np.float64)
    return a / (np.linalg.norm(a, axis=axis, keepdims=True) + 1e-12)


# ------------------------------------------------------------------ effect sizes / CIs / tests
def cohens_dz(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) + 1e-12))


def bootstrap_ci(x, stat=np.mean, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Percentile bootstrap over concepts (rows of x; x may be 1-D or (n, k) for ratio statistics)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(x[i]) for i in idx])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))]


def wilcoxon_1s(x, alternative="greater"):
    """One-sample Wilcoxon signed-rank vs 0 that never crashes on all-zero / tiny input."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0):
        return float("nan"), float("nan")
    try:
        r = stats.wilcoxon(x, zero_method="wilcox", alternative=alternative)
        return float(r.statistic), float(r.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def sign_flip_test(x, n_perm=N_PERM, seed=SEED, alternative="two-sided"):
    """Exact (n<=16) or Monte-Carlo paired sign-flip permutation test on the mean of x."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    obs = x.mean()
    exact = n <= 16
    if exact:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, n))
    null = (signs * x).mean(1)
    if alternative == "greater":
        k = (null >= obs - 1e-15).sum()
    elif alternative == "less":
        k = (null <= obs + 1e-15).sum()
    else:
        k = (np.abs(null) >= abs(obs) - 1e-15).sum()
    return float(k / len(null)) if exact else float((k + 1) / (len(null) + 1))


def summarize(x, alternative="greater"):
    """Standard per-metric block: mean, median, 95% bootstrap CI, dz, Wilcoxon, sign-flip p."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    w, wp = wilcoxon_1s(x, alternative)
    return dict(n=int(len(x)), mean=float(x.mean()) if len(x) else float("nan"),
                median=float(np.median(x)) if len(x) else float("nan"),
                ci95=bootstrap_ci(x), cohens_dz=cohens_dz(x),
                wilcoxon_stat=w, wilcoxon_p=wp, alternative=alternative,
                signflip_p=sign_flip_test(x, alternative=alternative))


def holm(pvals):
    """Holm-Bonferroni (FWER) adjusted p-values; NaNs pass through."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs pass through."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    if m == 0:
        return out.tolist()
    order = ok[np.argsort(p[ok])]
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.clip(q, 0, 1)
    return out.tolist()


# ------------------------------------------------------------------ geometry
def unbiased_cka(X, Y):
    """Linear CKA with the unbiased HSIC estimator (Song et al. 2012; Nguyen et al. 2021).
    Plain linear CKA is inflated when n (48) << d (~4k); this one is ~0 for unrelated X, Y."""
    def hsic_u(K, L):
        n = K.shape[0]
        K = K.copy(); L = L.copy()
        np.fill_diagonal(K, 0); np.fill_diagonal(L, 0)
        t1 = np.sum(K * L)
        t2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
        t3 = 2 * (K.sum(0) @ L.sum(0)) / (n - 2)
        return (t1 + t2 - t3) / (n * (n - 3))
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    K, L = X @ X.T, Y @ Y.T
    den = np.sqrt(max(hsic_u(K, K), 1e-30) * max(hsic_u(L, L), 1e-30))
    return float(hsic_u(K, L) / den)


def cka_permutation(X, Y, n_perm=2000, seed=SEED):
    """Null = shuffle the EN<->ZH concept pairing. Returns (cka, p, null_mean)."""
    rng = np.random.default_rng(seed)
    obs = unbiased_cka(X, Y)
    null = np.array([unbiased_cka(X, Y[rng.permutation(len(Y))]) for _ in range(n_perm)])
    return obs, float(((null >= obs).sum() + 1) / (n_perm + 1)), float(null.mean())


def rdm(X):
    """Correlation-distance RDM over concept vectors (rows)."""
    return 1.0 - np.corrcoef(np.asarray(X, dtype=np.float64))


def rsa_mantel(A, B, n_perm=N_PERM, seed=SEED):
    """Spearman rho between RDM upper triangles; Mantel p (rows+cols of A permuted jointly)."""
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    rb = stats.rankdata(B[iu])
    obs = float(np.corrcoef(stats.rankdata(A[iu]), rb)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        null[k] = np.corrcoef(stats.rankdata(A[np.ix_(p, p)][iu]), rb)[0, 1]
    return obs, float(((null >= obs).sum() + 1) / (n_perm + 1))


def cross_lingual_geometry(Ven, Vzh):
    """Centered-cosine alignment of paired EN/ZH concept vectors (per-language common mode removed,
    which the sanity check shows dominates raw cosines). Row i of Ven and Vzh = same concept."""
    a = unit(Ven - Ven.mean(0)); b = unit(Vzh - Vzh.mean(0))
    S = a @ b.T
    n = len(S)
    same = np.diag(S)
    off_row = (S.sum(1) - same) / (n - 1)
    r_e2z = (S > same[:, None]).sum(1)                 # 0 = matched ZH vector is the nearest
    r_z2e = (S.T > same[None, :]).sum(1)
    out = dict(
        per_concept_same_cos=same.tolist(),
        per_concept_cos_margin=(same - off_row).tolist(),
        mean_same_cos=float(same.mean()), mean_diff_cos=float(off_row.mean()),
        en2zh_top1=float((r_e2z == 0).mean()), zh2en_top1=float((r_z2e == 0).mean()),
        en2zh_mrr=float((1.0 / (r_e2z + 1)).mean()), zh2en_mrr=float((1.0 / (r_z2e + 1)).mean()),
        chance_top1=1.0 / n,
    )
    return out, S


def retrieval_permutation_p(S, n_perm=N_PERM, seed=SEED):
    """p for 'mean diagonal of S exceeds what a random EN<->ZH pairing gives'."""
    rng = np.random.default_rng(seed)
    obs = np.trace(S) / len(S)
    null = np.array([S[np.arange(len(S)), rng.permutation(len(S))].mean() for _ in range(n_perm)])
    return float(((null >= obs).sum() + 1) / (n_perm + 1))


def save_json(path, obj):
    def default(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default), encoding="utf-8")
