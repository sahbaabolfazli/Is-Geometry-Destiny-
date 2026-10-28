#!/usr/bin/env python3
"""
Phase 3e: representational (geometric) EN<->ZH alignment at LAYER_IDX, with proper null models.
MODEL-FREE:  python3 phase3e_geometry_alignment.py qwen3_8b     (or MODEL_KEY=... env var)

Vectors: phase3_concept_vectors_<key>.npz (written by phase3d) or, if absent, the per-model ChromaDB.
Metrics (all on the same 48 paired concept vectors that are injected in 3b/3c/3d):
  * unbiased linear CKA (EN vs ZH)  + permutation p (shuffle EN<->ZH pairing)
      -> replaces the biased CKA in phase3_extraction_pipeline (inflated at n=48 << d)
  * centered-cosine translation retrieval: top-1, MRR both directions, binomial + permutation p
  * per-concept alignment margin = cos(EN_i, ZH_i) - mean_j cos(EN_i, ZH_j)   (consumed by phase4)
  * RSA: Spearman between EN-RDM and ZH-RDM (Mantel p) - language-invariant concept geometry
  * domain structure: RSA of each language RDM vs the domain-membership model RDM
Output: phase3e_geometry_<key>.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from analysis_common import (LANGS, load_items, item_tables, load_concept_vectors, chroma_path,
                             unbiased_cka, cka_permutation, cross_lingual_geometry, retrieval_permutation_p,
                             rdm, rsa_mantel, bootstrap_ci, save_json, RUN_TAG)

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_KEY", "llama2_7b")
VEC_NPZ = Path(f"phase3_concept_vectors_{MODEL_KEY}.npz")
OUT = Path(f"phase3e_geometry_{MODEL_KEY}{RUN_TAG}.json")


def load_vectors():
    if VEC_NPZ.exists():
        Z = np.load(VEC_NPZ, allow_pickle=False)
        return [str(i) for i in Z["ids"]], Z["en"].astype(np.float64), Z["zh"].astype(np.float64)
    V = load_concept_vectors(db_path=chroma_path(MODEL_KEY))
    ids = sorted(set(V["en"]) & set(V["zh"]))
    return ids, np.stack([V["en"][i] for i in ids]), np.stack([V["zh"][i] for i in ids])


def main():
    ids, Xen, Xzh = load_vectors()
    items = load_items()
    domain, source = item_tables(items)
    ids_ok = [i for i in ids if i in domain]
    keep = [ids.index(i) for i in ids_ok]
    ids, Xen, Xzh = ids_ok, Xen[keep], Xzh[keep]
    n = len(ids)
    dom = np.array([domain[i] for i in ids])

    cka, cka_p, cka_null = cka_permutation(Xen, Xzh)
    geo, S = cross_lingual_geometry(Xen, Xzh)
    k_e2z = int(round(geo["en2zh_top1"] * n)); k_z2e = int(round(geo["zh2en_top1"] * n))
    margin = np.asarray(geo["per_concept_cos_margin"])

    R_en, R_zh = rdm(Xen - Xen.mean(0)), rdm(Xzh - Xzh.mean(0))
    rsa_rho, rsa_p = rsa_mantel(R_en, R_zh)
    R_dom = (dom[:, None] != dom[None, :]).astype(float)       # 0 = same domain, 1 = different
    dom_rsa = {l: dict(zip(("rho", "p"), rsa_mantel(R, R_dom))) for l, R in (("en", R_en), ("zh", R_zh))}

    by_domain = {d: dict(n=int((dom == d).sum()), mean_margin=float(margin[dom == d].mean()),
                         ci95=bootstrap_ci(margin[dom == d])) for d in sorted(set(dom))}
    kw = stats.kruskal(*[margin[dom == d] for d in sorted(set(dom))])

    report = dict(
        model_key=MODEL_KEY, n_concepts=n,
        unbiased_cka=dict(value=cka, perm_p=cka_p, null_mean=cka_null,
                          note="pairing-shuffle null; compare with biased CKA in phase3_report_<key>.json"),
        retrieval=dict(en2zh_top1=geo["en2zh_top1"], zh2en_top1=geo["zh2en_top1"],
                       en2zh_mrr=geo["en2zh_mrr"], zh2en_mrr=geo["zh2en_mrr"], chance_top1=1 / n,
                       en2zh_binom_p=float(stats.binomtest(k_e2z, n, 1 / n, alternative="greater").pvalue),
                       zh2en_binom_p=float(stats.binomtest(k_z2e, n, 1 / n, alternative="greater").pvalue),
                       diag_perm_p=retrieval_permutation_p(S)),
        centered_cosine=dict(mean_same=geo["mean_same_cos"], mean_different=geo["mean_diff_cos"],
                             mean_margin=float(margin.mean()), margin_ci95=bootstrap_ci(margin)),
        rsa_en_vs_zh=dict(spearman_rho=rsa_rho, mantel_p=rsa_p),
        domain_structure_rsa=dom_rsa,
        margin_by_domain=by_domain, margin_domain_kruskal=dict(H=float(kw.statistic), p=float(kw.pvalue)),
        per_concept=dict(ids=ids, domain=dom.tolist(), source=[source[i] for i in ids],
                         alignment_margin=margin.tolist(), same_cos=geo["per_concept_same_cos"],
                         norm_en=np.linalg.norm(Xen, axis=1).tolist(), norm_zh=np.linalg.norm(Xzh, axis=1).tolist()),
    )
    save_json(OUT, report)
    print(f"[3e] {MODEL_KEY}: unbiased CKA={cka:.3f} (p={cka_p:.4f}, null {cka_null:.3f}) | "
          f"top1 EN->ZH={geo['en2zh_top1']:.2f} ZH->EN={geo['zh2en_top1']:.2f} (chance {1 / n:.3f}) | "
          f"RSA rho={rsa_rho:.3f} (p={rsa_p:.4f}) | margin={margin.mean():.3f}")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
