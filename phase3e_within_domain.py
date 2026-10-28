#!/usr/bin/env python3
"""python3 phase3e_within_domain.py <model_key>
Is EN<->ZH alignment concept-specific BEYOND domain membership? Same centered-cosine similarity as 3e, but
(a) the contrast set is same-domain concepts only, (b) retrieval candidates are same-domain only,
(c) the permutation null shuffles the EN<->ZH pairing WITHIN domain."""
import sys
import numpy as np
from analysis_common import load_items, item_tables, unit, bootstrap_ci, save_json, N_PERM, SEED, RUN_TAG
from phase3e_geometry_alignment import load_vectors, MODEL_KEY

ids, Xen, Xzh = load_vectors()
domain, _ = item_tables(load_items())
keep = [k for k, i in enumerate(ids) if i in domain]
ids, Xen, Xzh = [ids[k] for k in keep], Xen[keep], Xzh[keep]
n, dom = len(ids), np.array([domain[i] for i in ids])

S = unit(Xen - Xen.mean(0)) @ unit(Xzh - Xzh.mean(0)).T        # S[i, j] = cos(EN_i, ZH_j), per-language common mode removed
same = dom[:, None] == dom[None, :]
others = same & ~np.eye(n, dtype=bool)
diag = np.diag(S)
wd_margin = diag - np.array([S[i, others[i]].mean() for i in range(n)])    # specific beyond domain

def best_in_domain(M):                                         # argmax over same-domain candidates
    return np.array([np.flatnonzero(same[i])[np.argmax(M[i, same[i]])] for i in range(n)])

b_e2z, b_z2e = best_in_domain(S), best_in_domain(S.T)
top1 = dict(en2zh=float((b_e2z == np.arange(n)).mean()), zh2en=float((b_z2e == np.arange(n)).mean()),
            chance=float(np.mean([1 / same[i].sum() for i in range(n)])))

rng = np.random.default_rng(SEED)
null_cos, null_top1 = np.empty(N_PERM), np.empty(N_PERM)
for k in range(N_PERM):
    perm = np.arange(n)
    for d in np.unique(dom):                                   # shuffle pairing inside each domain only
        idx = np.flatnonzero(dom == d)
        perm[idx] = rng.permutation(idx)
    null_cos[k] = S[np.arange(n), perm].mean()
    null_top1[k] = (b_e2z == perm).mean()                      # hit if the best candidate is the (shuffled) partner
obs_cos, obs_top1 = diag.mean(), top1["en2zh"]
save_json(f"phase3e_within_domain_{MODEL_KEY}{RUN_TAG}.json", dict(
    model_key=MODEL_KEY, n=n, top1_within_domain=top1,
    mean_same_cos=float(obs_cos), mean_same_cos_null=float(null_cos.mean()),
    perm_p_same_cos=float(((null_cos >= obs_cos).sum() + 1) / (N_PERM + 1)),
    perm_p_top1_en2zh=float(((null_top1 >= obs_top1).sum() + 1) / (N_PERM + 1)),
    within_domain_margin_mean=float(wd_margin.mean()), within_domain_margin_ci95=bootstrap_ci(wd_margin),
    per_concept=dict(ids=ids, within_domain_margin=wd_margin.tolist())))
print(f"[3e-wd] {MODEL_KEY}: within-domain top1 EN->ZH {top1['en2zh']:.2f} (chance {top1['chance']:.2f}), "
      f"margin {wd_margin.mean():+.3f}")