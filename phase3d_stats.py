#!/usr/bin/env python3
"""
Phase 3d statistics (MODEL-FREE; numpy/scipy only):
    MODEL_KEY=qwen3_8b python3 phase3d_stats.py        (or:  python3 phase3d_stats.py qwen3_8b)

Env switches (all optional):
    NLL_KIND=mean|sum|head   which NLL aggregation from the 3d npz to analyse (default mean = legacy).
                             mean = nats/token, sum = nats/passage, head = mean over first HEAD_K tokens.
    EXCLUDE_IDS_FILE=<json>  list of concept ids to drop (sensitivity run, e.g. flagged translations).
    RUN_TAG=<str>            suffix for output file names so sensitivity runs never overwrite the main run.

Reads phase3d_interference_<key>.npz and decomposes causal potency per target text i, condition c:

  M[j, i] = NLL of text i when concept vector j is injected (lower = better)
  diag_i            = M[i, i]                      matched concept
  other_i           = mean_{j != i} M[j, i]        any other concept (same src language)
  same_dom_other_i  = mean_{j != i, dom j == dom i} M[j, i]
  diff_dom_i        = mean_{dom j != dom i} M[j, i]
  noise_i           = mean_k noise[k, i]           norm-matched Gaussian
  centroid_i        = centroid[i]                  mean concept vector of the src language

  TOTAL potency        = noise_i  - diag_i       (what phase3c measures)
  GENERIC potency      = noise_i  - other_i      (any concept vector helps; not concept-specific)
  CSP (concept-specific potency) = other_i - diag_i     <-- primary DV
  CSP_rel              = CSP_i / NLL_i(no injection)   (scale-free; comparable across tokenizers)
  GAIN vs no injection = none_i - diag_i               (avoids the out-of-distribution noise baseline)
  CSP_vs_centroid      = centroid_i - diag_i
  DOMAIN potency       = diff_dom_i - same_dom_other_i  (vector carries domain info)
  WITHIN-DOMAIN CSP    = same_dom_other_i - diag_i      (specific beyond domain)
  AUC_i                = P_j(M[j,i] > M[i,i])           (0.5 = chance, 1 = matched always best)
  rank_i               = #{j: M[j,i] < M[i,i]}          -> top-1, MRR

Cross-lingual alignment (same TARGET text language, so text difficulty cancels):
  CTR_zh_text = CSP(en2zh) / CSP(zh2zh),   CTR_en_text = CSP(zh2en) / CSP(en2en)
  (ratio-of-means with concept bootstrap CI; 1 = EN and ZH vectors are causally interchangeable)
  plus paired gap CSP(mono) - CSP(cross) per concept.

Inference: concept-level one-sided Wilcoxon + sign-flip permutation, bootstrap 95% CI,
Cohen's dz; matrix-level row-permutation test; Holm across the primary family of tests.
Moderators: domain (Kruskal-Wallis; abstract vs syntax_baseline Mann-Whitney), source
(human vs AI, Mann-Whitney; robustness), length/perplexity (Spearman + OLS partial).
Output: phase3d_stats_<key><tag>.json and phase3d_per_concept_<key><tag>.json (consumed by phase4).
        <tag> = out_tag() = ("" if NLL_KIND == "mean" else "_<kind>") + RUN_TAG
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from analysis_common import (LANGS, CONDITIONS, CROSS_FOR_TEXT, MONO_FOR_TEXT, ABSTRACT_DOMAINS,
                             summarize, bootstrap_ci, sign_flip_test, holm, bh_fdr, save_json,
                             SEED, N_PERM, KIND_SFX, NLL_KIND, out_tag, exclude_ids, CARRIER_TAG, 
                             CARRIER_SFX, require_carrier_scope)

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_KEY", "llama2_7b")
require_carrier_scope(MODEL_KEY)
NPZ  = Path(f"phase3d_interference_{MODEL_KEY}{CARRIER_SFX}.npz")
META = Path(f"phase3d_interference_meta_{MODEL_KEY}{CARRIER_SFX}.json")
PPL = Path(f"phase3_perplexity_{MODEL_KEY}.json")
OUT = Path(f"phase3d_stats_{MODEL_KEY}{out_tag()}.json")
OUT_PC = Path(f"phase3d_per_concept_{MODEL_KEY}{out_tag()}.json")


def decompose(M, noise, centroid, dom):
    """Per-target-text potency components for one condition (see module docstring)."""
    n = M.shape[0]
    dom = np.asarray(dom)
    diag = np.diag(M)
    off = ~np.eye(n, dtype=bool)
    other = np.array([M[off[:, i], i].mean() for i in range(n)])
    same_dom = np.array([M[(dom == dom[i]) & off[:, i], i].mean() if ((dom == dom[i]) & off[:, i]).any()
                         else np.nan for i in range(n)])
    diff_dom = np.array([M[dom != dom[i], i].mean() if (dom != dom[i]).any() else np.nan for i in range(n)])
    nz = noise.mean(0)
    rank = (M < diag[None, :]).sum(0)
    auc = np.array([(M[off[:, i], i] > diag[i]).mean() + 0.5 * (M[off[:, i], i] == diag[i]).mean()
                    for i in range(n)])
    return dict(
        diag=diag, other=other, noise=nz, centroid=centroid,
        total=nz - diag, generic=nz - other, csp=other - diag, csp_vs_centroid=centroid - diag,
        domain_potency=diff_dom - same_dom, within_domain_csp=same_dom - diag,
        auc=auc, rank=rank, top1=(rank == 0).astype(float), rr=1.0 / (rank + 1),
    )


def matrix_permutation_p(M, n_perm=N_PERM, seed=SEED):
    """H0: matched vector is exchangeable with any other. Statistic = mean_i (other_i - M[i,i]);
    null = same statistic under a random reassignment of vectors to texts."""
    n = M.shape[0]
    colmean = M.mean(0)

    def stat(perm):
        d = M[perm, np.arange(n)]
        return float(((colmean * n - d) / (n - 1) - d).mean())
    obs = stat(np.arange(n))
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(n)) for _ in range(n_perm)])
    return obs, float(((null >= obs).sum() + 1) / (n_perm + 1))


def ratio_of_means(xy):
    den = xy[:, 1].mean()
    return xy[:, 0].mean() / den if abs(den) > 1e-12 else np.nan


def ols_partial(y, X, names):
    """OLS y ~ 1 + X with HC3 robust SEs (numpy only). Returns coef/se/t/p per predictor."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    XtX_inv = np.linalg.pinv(X1.T @ X1)
    h = np.einsum("ij,jk,ik->i", X1, XtX_inv, X1)
    meat = X1.T @ (X1 * (resid ** 2 / np.clip((1 - h) ** 2, 1e-12, None))[:, None])
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    dfree = len(y) - X1.shape[1]
    t = beta / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), dfree)
    return {nm: dict(coef=float(b), se=float(s), t=float(tt), p=float(pp))
            for nm, b, s, tt, pp in zip(["intercept"] + names, beta, se, t, p)}


def moderators(csp, ids, meta, ppl, ntok):
    """Domain / source / length / perplexity moderators of a per-concept potency vector."""
    dom = np.array([meta["domain"][i] for i in ids])
    src = np.array([meta["source"][i] for i in ids])
    out = {"by_domain": {}}
    groups = []
    for d in sorted(set(dom)):
        x = csp[dom == d]
        groups.append(x)
        out["by_domain"][d] = dict(n=int(len(x)), mean=float(x.mean()), ci95=bootstrap_ci(x),
                                   signflip_p=sign_flip_test(x, alternative="greater"))
    if len(groups) > 1 and all(len(g) > 1 for g in groups):
        kw = stats.kruskal(*groups)
        out["domain_kruskal"] = dict(H=float(kw.statistic), p=float(kw.pvalue))
    abstract = np.isin(dom, ABSTRACT_DOMAINS)
    if abstract.any() and (~abstract).any():
        mw = stats.mannwhitneyu(csp[abstract], csp[~abstract], alternative="two-sided")
        out["abstract_vs_syntax_baseline"] = dict(
            mean_abstract=float(csp[abstract].mean()), mean_syntax=float(csp[~abstract].mean()),
            U=float(mw.statistic), p=float(mw.pvalue),
            rank_biserial=float(2 * mw.statistic / (abstract.sum() * (~abstract).sum()) - 1))
    if {"human", "ai"} <= set(src):
        h, a = csp[src == "human"], csp[src == "ai"]
        mw = stats.mannwhitneyu(h, a, alternative="two-sided")
        out["human_vs_ai"] = dict(n_human=int(len(h)), n_ai=int(len(a)), mean_human=float(h.mean()),
                                  mean_ai=float(a.mean()), U=float(mw.statistic), p=float(mw.pvalue),
                                  note="n_human=8: descriptive / robustness only")
        out["ai_only_summary"] = summarize(a)
    else:
        out["human_vs_ai"] = "source labels unavailable; set HUMAN_IDS_FILE (json list of the 8 human ids)"
    rho, p = stats.spearmanr(ntok, csp)
    out["spearman_vs_n_tokens"] = dict(rho=float(rho), p=float(p))
    if ppl is not None:
        rho, p = stats.spearmanr(ppl, csp)
        out["spearman_vs_log_ppl"] = dict(rho=float(rho), p=float(p))
        X = np.column_stack([stats.zscore(np.log(ntok)), stats.zscore(ppl)])
        out["ols_hc3_csp_on_logtokens_logppl"] = ols_partial(csp, X, ["z_log_n_tokens", "z_log_ppl"])
    return out


def condition_block(d, M, none, n):
    mstat, mp = matrix_permutation_p(M)
    tot = d["total"].mean()
    return dict(
        mean_nll=dict(matched=float(d["diag"].mean()), other=float(d["other"].mean()),
                      noise=float(d["noise"].mean()), centroid=float(d["centroid"].mean()),
                      no_injection=float(none.mean())),
        total_potency=summarize(d["total"]), generic_potency=summarize(d["generic"]),
        concept_specific_potency=summarize(d["csp"]),
        csp_vs_centroid=summarize(d["csp_vs_centroid"]),
        domain_potency=summarize(d["domain_potency"]),
        within_domain_csp=summarize(d["within_domain_csp"]),
        specificity_share=dict(   # fraction of total potency that is concept-specific
            value=float(d["csp"].mean() / tot) if abs(tot) > 1e-12 else None,
            ci95=bootstrap_ci(np.column_stack([d["csp"], d["total"]]), stat=ratio_of_means)),
        auc=dict(mean=float(d["auc"].mean()), ci95=bootstrap_ci(d["auc"]),
                 signflip_p_vs_0_5=sign_flip_test(d["auc"] - 0.5, alternative="greater")),
        retrieval=dict(top1=float(d["top1"].mean()), mrr=float(d["rr"].mean()), chance_top1=1.0 / n,
                       binom_p_top1=float(stats.binomtest(int(d["top1"].sum()), n, 1.0 / n,
                                                          alternative="greater").pvalue)),
        matrix_permutation=dict(stat=mstat, p=mp),
    )


def load_subset_npz(path, drop):
    """Load the 3d npz fully into a plain dict (file handle closed) and drop concepts `drop`
    from the (n x n) matrices (rows AND columns), (k x n) noise stacks (columns) and (n,) vectors.
    Returns (dict, ids_kept)."""
    with np.load(path, allow_pickle=False) as Z0:
        arrays = {k: Z0[k] for k in Z0.files}
    all_ids = [str(x) for x in arrays["ids"]]
    n = len(all_ids)
    unknown = set(drop) - set(all_ids)
    if unknown:
        print(f"[warn] EXCLUDE ids not in this model's npz (ignored): {sorted(unknown)}")
    keep = np.array([k for k, i in enumerate(all_ids) if i not in drop])
    if len(keep) < 2:
        sys.exit("[error] fewer than 2 concepts left after exclusion.")
    out = {}
    for key, a in arrays.items():
        if a.ndim == 2 and a.shape == (n, n):
            a = a[np.ix_(keep, keep)]
        elif a.ndim == 2 and a.shape[1] == n:          # noise stacks: (N_NOISE, n)
            a = a[:, keep]
        elif a.ndim == 1 and len(a) == n:              # ids, none_*, centroid_*, n_tokens_*, norm_*
            a = a[keep]
        elif n in a.shape:
            sys.exit(f"[error] array '{key}' has unexpected shape {a.shape}; cannot subset safely.")
        out[key] = a
    return out, [all_ids[k] for k in keep]


def main():
    if not NPZ.exists():
        sys.exit(f"[error] {NPZ} not found - run phase3d_interference_matrix.py for {MODEL_KEY} first.")
    drop = exclude_ids()
    Z, ids = load_subset_npz(NPZ, drop)
    meta = json.loads(META.read_text(encoding="utf-8"))
    S = KIND_SFX
    needed = [f"{p}{S}_{c}" for c in CONDITIONS for p in ("nll", "noise", "centroid")] + \
             [f"none{S}_{t}" for t in LANGS]
    missing = [k for k in needed if k not in Z]
    if missing:
        sys.exit(f"[error] NLL_KIND={NLL_KIND}: keys {missing[:3]}... not in {NPZ}. "
                 "Re-run the revised phase3d_interference_matrix.py (it stores mean/sum/head).")
    dom = [meta["domain"][i] for i in ids]
    n = len(ids)
    ppl_by = {e["id"]: e for e in json.loads(PPL.read_text(encoding="utf-8"))} if PPL.exists() else {}

    comp = {c: decompose(Z[f"nll{S}_{c}"], Z[f"noise{S}_{c}"], Z[f"centroid{S}_{c}"], dom) for c in CONDITIONS}
    for c in CONDITIONS:
        none_c = Z[f"none{S}_{c[-2:]}"]
        comp[c]["csp_rel"] = comp[c]["csp"] / none_c            # CSP as a fraction of the text's un-injected NLL
        comp[c]["total_vs_none"] = none_c - comp[c]["diag"]     # gain vs NO injection (noise baseline can be OOD)

    report = dict(model_key=MODEL_KEY, model=meta["model"], category=meta["category"], n_concepts=n,
                  nll_kind=NLL_KIND, out_tag=out_tag(), excluded_ids=sorted(drop & set(meta["domain"])),
                  nll_unit=meta.get("nll_unit"),
                  layer_idx=meta["layer_idx"], injection_position=meta["injection_position"],
                  carrier_token=meta["carrier_token"], carrier_tag=CARRIER_TAG or "primary", conditions={})
    primary = []                                         # (label, p) -> Holm / BH

    for c in CONDITIONS:
        tgt = c[-2:]
        block = condition_block(comp[c], Z[f"nll{S}_{c}"], Z[f"none{S}_{tgt}"], n)
        block["relative_csp"] = summarize(comp[c]["csp_rel"])
        block["gain_vs_no_injection"] = summarize(comp[c]["total_vs_none"])
        tok = Z[f"n_tokens_{tgt}"].astype(float)
        lp = (np.array([np.log(ppl_by[i][f"{tgt}_ppl"]) for i in ids])
              if ppl_by and all(i in ppl_by for i in ids) else None)
        block["moderators"] = moderators(comp[c]["csp"], ids, meta, lp, tok)
        report["conditions"][c] = block
        primary.append((f"CSP>0 [{c}]", block["concept_specific_potency"]["signflip_p"]))

    # cross-lingual causal transfer: same TARGET-text language, so text difficulty cancels
    report["cross_lingual_transfer"] = {}
    for tgt in LANGS:
        mono, cross = MONO_FOR_TEXT[tgt], CROSS_FOR_TEXT[tgt]
        a, b = comp[cross]["csp"], comp[mono]["csp"]
        xy = np.column_stack([a, b])
        rho, rp = stats.spearmanr(a, b)
        blk = dict(mono=mono, cross=cross,
                   CTR=float(ratio_of_means(xy)), CTR_ci95=bootstrap_ci(xy, stat=ratio_of_means),
                   gap_mono_minus_cross=summarize(b - a, alternative="two-sided"),
                   total_potency_ratio=float(ratio_of_means(
                       np.column_stack([comp[cross]["total"], comp[mono]["total"]]))),
                   per_concept_csp_spearman=dict(rho=float(rho), p=float(rp)))
        report["cross_lingual_transfer"][f"{tgt}_text"] = blk
        primary.append((f"mono-cross gap [{tgt} text]", blk["gap_mono_minus_cross"]["signflip_p"]))

    report["direction_asymmetry_en2zh_minus_zh2en"] = summarize(
        comp["en2zh"]["csp"] - comp["zh2en"]["csp"], alternative="two-sided")
    report["mono_en_minus_mono_zh_csp"] = summarize(
        comp["en2en"]["csp"] - comp["zh2zh"]["csp"], alternative="two-sided")

    labels, ps = zip(*primary)
    report["multiplicity"] = dict(family="primary per-model tests", labels=list(labels), p_raw=list(ps),
                                  p_holm=holm(ps), q_bh=bh_fdr(ps))
    save_json(OUT, report)
    save_json(OUT_PC, dict(model_key=MODEL_KEY, category=meta["category"], nll_kind=NLL_KIND,
                           ids=ids, domain=dom, source=[meta["source"][i] for i in ids],
                           **{f"{k}_{c}": comp[c][k].tolist() for c in CONDITIONS
                              for k in ("total", "generic", "csp", "csp_rel", "auc",
                                        "within_domain_csp", "domain_potency")}))

    print(f"[3d-stats] {MODEL_KEY} ({meta['category']}), n={n}, NLL_KIND={NLL_KIND}"
          + (f", excluded={len(drop)}" if drop else ""))
    for c in CONDITIONS:
        b = report["conditions"][c]
        s = b["concept_specific_potency"]
        print(f"  {c}: CSP={s['mean']:+.4f} CI{np.round(s['ci95'], 4).tolist()} dz={s['cohens_dz']:.2f}"
              f" | rel={b['relative_csp']['mean']:+.4f} gain_vs_none={b['gain_vs_no_injection']['mean']:+.4f}"
              f" | total={b['total_potency']['mean']:+.4f} generic={b['generic_potency']['mean']:+.4f}"
              f" | AUC={b['auc']['mean']:.3f} top1={b['retrieval']['top1']:.3f}")
    for t, v in report["cross_lingual_transfer"].items():
        print(f"  CTR[{t}] = {v['CTR']:.3f} CI{np.round(v['CTR_ci95'], 3).tolist()}")
    print(f"[done] {OUT}, {OUT_PC}")


if __name__ == "__main__":
    main()