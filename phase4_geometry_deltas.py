#!/usr/bin/env python3
"""python3 phase4_geometry_deltas.py
Paired (specialized - general) deltas in unbiased CKA, top-1 retrieval, and RSA, with concept-subsampling
intervals. Subsampling WITHOUT replacement (80% of concepts), not bootstrap: duplicated rows bias the
unbiased-HSIC estimator. Intervals show concept-sampling stability, not a formal CI."""
import numpy as np
from scipy import stats
from analysis_common import FAMILIES, unbiased_cka, cross_lingual_geometry, rdm, save_json, SEED, RUN_TAG

N_REP, FRAC = 2000, 0.8

def load(key):
    Z = np.load(f"phase3_concept_vectors_{key}.npz", allow_pickle=False)
    return {str(i): (e, z) for i, e, z in zip(Z["ids"], Z["en"], Z["zh"])}

def metrics(en, zh):
    geo, _ = cross_lingual_geometry(en, zh)
    iu = np.triu_indices(len(en), 1)
    rsa = stats.spearmanr(rdm(en - en.mean(0))[iu], rdm(zh - zh.mean(0))[iu])[0]
    return dict(cka=unbiased_cka(en, zh), top1=(geo["en2zh_top1"] + geo["zh2en_top1"]) / 2, rsa=float(rsa))

out, rng = {}, np.random.default_rng(SEED)
for fam, (g, s) in FAMILIES.items():
    G, S_ = load(g), load(s)
    ids = sorted(set(G) & set(S_))
    A = [np.stack([D[i][k] for i in ids]) for D in (G, S_) for k in (0, 1)]    # g_en, g_zh, s_en, s_zh
    full = {m: metrics(A[2], A[3])[m] - metrics(A[0], A[1])[m] for m in ("cka", "top1", "rsa")}
    m = int(FRAC * len(ids))
    reps = {k: [] for k in full}
    for _ in range(N_REP):
        idx = rng.choice(len(ids), m, replace=False)                           # same concepts for both models (paired)
        mg, ms = metrics(A[0][idx], A[1][idx]), metrics(A[2][idx], A[3][idx])
        for k in reps:
            reps[k].append(ms[k] - mg[k])
    out[fam] = {k: dict(delta=float(full[k]), subsample_ci95=[float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                        frac_subsamples_positive=float(np.mean(np.array(v) > 0))) for k, v in reps.items()}
    print(fam, {k: f"{full[k]:+.3f}" for k in full})
save_json(f"phase4_geometry_deltas{RUN_TAG}.json", out)