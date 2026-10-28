#!/usr/bin/env python3
"""python3 phase4_carrier_robustness.py [tag]    (default tag: dot)
Qwen-family-only carrier robustness. Every quantity is a WITHIN-carrier contrast (CSP = other - matched,
computed inside one carrier's run). Absolute NLL / potency is never compared across carriers.
Reports: (1) CSP per model x carrier, (2) per-concept agreement across carriers, (3) specialization delta
(spec - gen) per carrier, (4) carrier x specialization interaction, (5) CTR per carrier.
Primary inference stays on the primary carrier; this file is a descriptive robustness check (own family of tests)."""
import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats
from analysis_common import (FAMILIES, CONDITIONS, SECONDARY_CARRIERS, SECONDARY_CARRIER_FAMILIES, CARRIER_TAG,
                             summarize, bootstrap_ci, cohens_dz, sign_flip_test, save_json, out_tag)

if CARRIER_TAG:
    sys.exit("Run WITHOUT CARRIER_TAG: this script loads both carriers itself.")
TAG = sys.argv[1] if len(sys.argv) > 1 else "dot"
assert TAG in SECONDARY_CARRIERS, f"unknown tag {TAG}"
CARRIERS = {"primary": "", TAG: f"_c{TAG}"}                     # label -> file suffix
FAM = SECONDARY_CARRIER_FAMILIES[0]
GEN, SPE = FAMILIES[FAM]
METRICS = [f"{p}_{c}" for p in ("csp", "csp_rel", "within_domain_csp") for c in CONDITIONS]
OUT = Path(f"phase4_carrier_robustness_{TAG}{out_tag('')}.json")   # out_tag('') = NLL_KIND + RUN_TAG, no carrier


def load(key, label):
    p = Path(f"phase3d_per_concept_{key}{out_tag(CARRIERS[label])}.json")
    if not p.exists():
        sys.exit(f"[error] missing {p}: run phase3d_stats.py for {key} "
                 f"({'with CARRIER_TAG=' + TAG if CARRIERS[label] else 'primary'}) first.")
    P = json.loads(p.read_text(encoding="utf-8"))
    return {m: dict(zip(P["ids"], P[m])) for m in METRICS}, set(P["ids"])


D, idsets = {}, []
for key in (GEN, SPE):
    for lab in CARRIERS:
        D[key, lab], s = load(key, lab)
        idsets.append(s)
IDS = sorted(set.intersection(*idsets))                          # same concepts in all 4 runs (paired)
v = lambda key, lab, m: np.array([D[key, lab][m][i] for i in IDS])


def paired(d):
    return dict(n=int(len(d)), mean=float(d.mean()), ci95=bootstrap_ci(d), cohens_dz=cohens_dz(d),
                signflip_p_two_sided=sign_flip_test(d, alternative="two-sided"))


def ratio_of_means(xy):
    den = xy[:, 1].mean()
    return xy[:, 0].mean() / den if abs(den) > 1e-12 else np.nan


res = dict(scope=f"{FAM} family only ({GEN}, {SPE}); other families are single-carrier by design",
           primary_carrier="registry carrier_token ('\\n')", secondary_carrier=SECONDARY_CARRIERS[TAG],
           n_concepts=len(IDS), within_carrier={}, cross_carrier_agreement={}, ctr={},
           specialization_delta={}, carrier_x_specialization={}, replication_verdict={})

for key in (GEN, SPE):
    res["within_carrier"][key] = {lab: {m: summarize(v(key, lab, m)) for m in METRICS} for lab in CARRIERS}
    res["cross_carrier_agreement"][key] = {
        m: dict(zip(("spearman_rho", "p"), map(float, stats.spearmanr(v(key, "primary", m), v(key, TAG, m)))))
        for m in METRICS}
    res["ctr"][key] = {}
    for lab in CARRIERS:
        res["ctr"][key][lab] = {}
        for t, cross, mono in (("zh_text", "csp_en2zh", "csp_zh2zh"), ("en_text", "csp_zh2en", "csp_en2en")):
            xy = np.column_stack([v(key, lab, cross), v(key, lab, mono)])
            res["ctr"][key][lab][t] = dict(CTR=float(ratio_of_means(xy)),
                                           ci95=bootstrap_ci(xy, stat=ratio_of_means))
    # replication = matched beats mismatched (CSP>0, CI excluding 0) under BOTH carriers, per condition
    res["replication_verdict"][key] = {
        c: bool(all(res["within_carrier"][key][lab][f"csp_{c}"]["mean"] > 0 and
                    res["within_carrier"][key][lab][f"csp_{c}"]["ci95"][0] > 0 for lab in CARRIERS))
        for c in CONDITIONS}

delta = {lab: {m: v(SPE, lab, m) - v(GEN, lab, m) for m in METRICS} for lab in CARRIERS}   # within-carrier, paired
for lab in CARRIERS:
    res["specialization_delta"][lab] = {m: paired(delta[lab][m]) for m in METRICS}
for m in METRICS:                                                 # does the spec-gen effect depend on the carrier?
    inter = delta[TAG][m] - delta["primary"][m]
    blk = paired(inter)
    blk["same_sign_in_both_carriers"] = bool(np.sign(delta[TAG][m].mean()) == np.sign(delta["primary"][m].mean()))
    res["carrier_x_specialization"][m] = blk

save_json(OUT, res)
print(f"[carrier-robustness] {FAM}, n={len(IDS)}, tag={TAG}")
for key in (GEN, SPE):
    for c in CONDITIONS:
        a, b = (res["within_carrier"][key][lab][f"csp_{c}"]["mean"] for lab in CARRIERS)
        print(f"  {key:10s} {c}: CSP primary={a:+.4f}  {TAG}={b:+.4f}  replicated={res['replication_verdict'][key][c]}")
for c in CONDITIONS:
    print(f"  spec-gen csp_{c}: " + "  ".join(f"{lab}={res['specialization_delta'][lab][f'csp_{c}']['mean']:+.4f}" for lab in CARRIERS)
          + f"  same_sign={res['carrier_x_specialization'][f'csp_{c}']['same_sign_in_both_carriers']}")
print(f"[done] {OUT}")