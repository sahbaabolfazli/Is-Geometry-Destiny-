#!/usr/bin/env python3
"""
Phase 4: lineage-controlled SPECIALIZATION contrasts across the 3 matched families.
MODEL-FREE:  python3 phase4_family_contrasts.py      (needs phase3d_per_concept_<key>.json and
                                                       phase3e_geometry_<key>.json for all 6 models)

Design: each family = (general base, translation-specialized continuation of that base). Because both
members see the SAME 48 concepts, every contrast is a within-concept, within-family paired difference:
    delta_f,i = metric(specialized_f, concept i) - metric(general_f, concept i)
Lineage is controlled by construction (architecture, tokenizer, layer index, carrier are shared within
a family); between-family heterogeneity is estimated, not assumed away.

Per family f and metric m:
    mean delta, bootstrap 95% CI (concepts), Cohen's dz, sign-flip p (two-sided)
Pooled across families:
    * stratified sign-flip test: per concept, the (spec - gen) sign is flipped jointly across families
      (H0: specialization label exchangeable within every family-concept cell)
    * DerSimonian-Laird random-effects meta-analysis of the 3 family deltas (+ tau^2, I^2)
    * sign consistency (k/3 families with the same sign; P(all 3 same sign | H0) = 0.25)
Interactions (difference-in-differences, all paired within family):
    * specialization x transfer: delta(CSP cross) - delta(CSP mono) for each target-text language
    * specialization x domain: delta(abstract) - delta(syntax_baseline)
    * specialization x direction: delta(CSP en2zh) - delta(CSP zh2en)
Model-level scalars (CTR, unbiased CKA, retrieval top-1, RSA): reported per model with deltas;
CTR delta gets a paired concept bootstrap CI.
Geometry->causality coupling: per model Spearman(alignment margin, cross-lingual CSP); delta per family.
All primary pooled p-values are Holm- and BH-corrected as one family.
Output: phase4_family_contrasts<tag>.json + .png   (<tag> = out_tag(): NLL_KIND and RUN_TAG)
Primary pooled p-values use the RMS-standardized stratified sign-flip test (*_std); the raw-unit p is kept.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from analysis_common import (FAMILIES, CONDITIONS, ABSTRACT_DOMAINS, bootstrap_ci, cohens_dz,
                             sign_flip_test, holm, bh_fdr, save_json, N_PERM, N_BOOT, SEED, out_tag, RUN_TAG, NLL_KIND, CARRIER_TAG)

if CARRIER_TAG:
    sys.exit("Pooled family contrasts are primary-carrier only. Use phase4_carrier_robustness.py for the Qwen check.")

OUT = Path(f"phase4_family_contrasts{out_tag()}.json")
FIG = Path(f"phase4_family_contrasts{out_tag()}.png")
PER_CONCEPT_METRICS = [f"csp_{c}" for c in CONDITIONS] + [f"total_{c}" for c in CONDITIONS] + \
                      [f"generic_{c}" for c in CONDITIONS] + [f"auc_{c}" for c in CONDITIONS] + \
                      [f"csp_rel_{c}" for c in CONDITIONS] + ["alignment_margin", "alignment_margin_wd"]
PRIMARY = ["csp_en2zh", "csp_zh2en", "csp_en2en", "csp_zh2zh", "alignment_margin", "alignment_margin_wd"]


def load_model(key):
    pc = Path(f"phase3d_per_concept_{key}{out_tag()}.json")
    st = Path(f"phase3d_stats_{key}{out_tag()}.json")
    ge = Path(f"phase3e_geometry_{key}{RUN_TAG}.json")
    wdp = Path(f"phase3e_within_domain_{key}{RUN_TAG}.json")
    missing = [str(p) for p in (pc, ge, st, wdp) if not p.exists()]
    if missing:
        sys.exit(f"[error] missing for {key}: {missing}. Run phase3d_stats.py, phase3e_geometry_alignment.py "
                 f"and phase3e_within_domain.py.")
    P = json.loads(pc.read_text(encoding="utf-8"))
    G = json.loads(ge.read_text(encoding="utf-8"))
    S = json.loads(st.read_text(encoding="utf-8"))
    gm = dict(zip(G["per_concept"]["ids"], G["per_concept"]["alignment_margin"]))
    rows = {k: dict(zip(P["ids"], P[k])) for k in P if k.split("_")[0] in ("csp", "total", "generic", "auc")}
    rows["alignment_margin"] = gm
    wd = json.loads(wdp.read_text(encoding="utf-8"))      # margin vs same-domain concepts only (domain-controlled)
    rows["alignment_margin_wd"] = dict(zip(wd["per_concept"]["ids"], wd["per_concept"]["within_domain_margin"]))
    return dict(ids=P["ids"], domain=dict(zip(P["ids"], P["domain"])), rows=rows, geo=G, stats=S,
                category=P["category"])


def paired_block(d):
    return dict(n=int(len(d)), mean_delta=float(d.mean()), ci95=bootstrap_ci(d), cohens_dz=cohens_dz(d),
                signflip_p=sign_flip_test(d, alternative="two-sided"))


def stratified_signflip(D, n_perm=N_PERM, seed=SEED):
    """D: (n_families, n_concepts) paired deltas. Flip each family-concept cell independently.
    Statistic: grand mean (raw units). Two-sided. Kept for reference; scale-dependent across tokenizers."""
    rng = np.random.default_rng(seed)
    obs = D.mean()
    null = np.array([(D * rng.choice([-1, 1], size=D.shape)).mean() for _ in range(n_perm)])
    return float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_perm + 1))


def stratified_signflip_std(D, n_perm=N_PERM, seed=SEED):
    """PRIMARY pooled test. nats/token (or nats/passage) are on different scales across tokenizers, so each
    family's deltas are divided by their own root-mean-square, sqrt(mean(d^2)), before pooling.
    The RMS (unlike the SD about the mean) is exactly invariant to sign flips, so the scale is identical
    in every permutation and the test stays exact under H0."""
    S = D / (np.sqrt((D ** 2).mean(axis=1, keepdims=True)) + 1e-12)
    rng = np.random.default_rng(seed)
    obs = S.mean()
    null = np.array([(S * rng.choice([-1, 1], size=S.shape)).mean() for _ in range(n_perm)])
    return float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_perm + 1))


def random_effects(means, ses):
    """DerSimonian-Laird random-effects pooling of per-family mean deltas."""
    y, v = np.asarray(means, float), np.asarray(ses, float) ** 2
    w = 1 / np.clip(v, 1e-30, None)
    fe = (w * y).sum() / w.sum()
    Q = (w * (y - fe) ** 2).sum()
    k = len(y)
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    ws = 1 / (v + tau2)
    mu = (ws * y).sum() / ws.sum()
    se = np.sqrt(1 / ws.sum())
    z = mu / se
    return dict(pooled=float(mu), se=float(se), ci95=[float(mu - 1.96 * se), float(mu + 1.96 * se)],
                z=float(z), p=float(2 * stats.norm.sf(abs(z))), tau2=float(tau2), Q=float(Q),
                I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0,
                note="k=3 families: treat as descriptive heterogeneity estimate")


def ctr_delta_ci(gen, spe, cross, mono, ids, n_boot=N_BOOT, seed=SEED):
    """Paired concept bootstrap of CTR(spec) - CTR(gen), CTR = mean CSP_cross / mean CSP_mono."""
    A = np.array([[gen["rows"][cross][i], gen["rows"][mono][i], spe["rows"][cross][i], spe["rows"][mono][i]]
                  for i in ids])
    f = lambda X: X[:, 2].mean() / X[:, 3].mean() - X[:, 0].mean() / X[:, 1].mean()
    return float(f(A)), bootstrap_ci(A, stat=f, n_boot=n_boot, seed=seed)


def model_scalars(m):
    S, G = m["stats"], m["geo"]
    out = dict(category=m["category"],
               unbiased_cka=G["unbiased_cka"]["value"], rsa_en_zh=G["rsa_en_vs_zh"]["spearman_rho"],
               retrieval_top1_en2zh=G["retrieval"]["en2zh_top1"], retrieval_top1_zh2en=G["retrieval"]["zh2en_top1"],
               alignment_margin=G["centered_cosine"]["mean_margin"])
    for t, v in S["cross_lingual_transfer"].items():
        out[f"CTR_{t}"] = v["CTR"]
    for c in CONDITIONS:
        b = S["conditions"][c]
        out[f"csp_{c}"] = b["concept_specific_potency"]["mean"]
        out[f"total_{c}"] = b["total_potency"]["mean"]
        out[f"specificity_share_{c}"] = b["specificity_share"]["value"]
        out[f"auc_{c}"] = b["auc"]["mean"]
    return out


def coupling(m):
    """Does geometric alignment predict cross-lingual causal potency, concept by concept?"""
    ids = m["ids"]
    g = np.array([m["rows"]["alignment_margin"][i] for i in ids])
    out = {}
    for c in ("en2zh", "zh2en"):
        rho, p = stats.spearmanr(g, [m["rows"][f"csp_{c}"][i] for i in ids])
        out[c] = dict(rho=float(rho), p=float(p))
    return out


def main():
    models = {k: load_model(k) for pair in FAMILIES.values() for k in pair}
    for fam, (g, s) in FAMILIES.items():
        assert models[g]["category"] == "general_purpose" and models[s]["category"] == "translation_specialized", \
            f"category tags wrong for family {fam}"
    ids = sorted(set.intersection(*[set(m["ids"]) for m in models.values()]))
    dom = models[FAMILIES["qwen3"][0]]["domain"]
    abstract = np.array([dom[i] in ABSTRACT_DOMAINS for i in ids])
    print(f"[4] {len(ids)} concepts shared by all 6 models")

    def delta(fam, metric):
        g, s = FAMILIES[fam]
        return np.array([models[s]["rows"][metric][i] - models[g]["rows"][metric][i] for i in ids])

    report = dict(n_concepts=len(ids), nll_kind=NLL_KIND, run_tag=RUN_TAG,
                  pooled_p_note="primary pooled p = *_std (RMS-standardized per family)",
                  families={f: dict(general=g, specialized=s) for f, (g, s) in FAMILIES.items()},
                  model_scalars={k: model_scalars(m) for k, m in models.items()},
                  coupling_geometry_vs_causal={k: coupling(m) for k, m in models.items()},
                  per_metric={}, interactions={}, ctr_deltas={})
    primary_p = []

    for metric in PER_CONCEPT_METRICS:
        D = np.stack([delta(f, metric) for f in FAMILIES])
        fams = {f: paired_block(D[k]) for k, f in enumerate(FAMILIES)}
        ses = [D[k].std(ddof=1) / np.sqrt(D.shape[1]) for k in range(len(FAMILIES))]
        signs = np.sign([fams[f]["mean_delta"] for f in FAMILIES])
        blk = dict(per_family=fams,
                   pooled_stratified_signflip_p=stratified_signflip(D),
                   pooled_stratified_signflip_p_std=stratified_signflip_std(D),
                   pooled_mean_delta=float(D.mean()),
                   pooled_ci95=bootstrap_ci(D.T, stat=lambda X: X.mean()),
                   random_effects=random_effects([fams[f]["mean_delta"] for f in FAMILIES], ses),
                   sign_consistency=dict(n_positive=int((signs > 0).sum()), n_negative=int((signs < 0).sum()),
                                         all_same=bool(abs(signs.sum()) == len(FAMILIES)),
                                         p_all_same_under_H0=2 * 0.5 ** len(FAMILIES)))
        report["per_metric"][metric] = blk
        if metric in PRIMARY:
            primary_p.append((f"spec-gen delta {metric}", blk["pooled_stratified_signflip_p_std"]))

    # difference-in-differences (all paired within family & concept)
    did_defs = {
        "transfer_x_spec_zh_text": ("csp_en2zh", "csp_zh2zh"),    # cross - mono on ZH text
        "transfer_x_spec_en_text": ("csp_zh2en", "csp_en2en"),    # cross - mono on EN text
        "direction_x_spec": ("csp_en2zh", "csp_zh2en"),
    }
    for name, (a, b) in did_defs.items():
        D = np.stack([delta(f, a) - delta(f, b) for f in FAMILIES])
        report["interactions"][name] = dict(
            definition=f"[spec-gen]({a}) - [spec-gen]({b})",
            per_family={f: paired_block(D[k]) for k, f in enumerate(FAMILIES)},
            pooled_mean=float(D.mean()), pooled_stratified_signflip_p=stratified_signflip(D),
            pooled_stratified_signflip_p_std=stratified_signflip_std(D))
        if name.startswith("transfer"):
            primary_p.append((name, report["interactions"][name]["pooled_stratified_signflip_p_std"]))
    for c in ("en2zh", "zh2en"):          # domain x specialization: concepts are between-subjects here
        per_f = {}
        for f in FAMILIES:
            d = delta(f, f"csp_{c}")
            mw = stats.mannwhitneyu(d[abstract], d[~abstract], alternative="two-sided")
            per_f[f] = dict(mean_abstract=float(d[abstract].mean()), mean_syntax=float(d[~abstract].mean()),
                            did=float(d[abstract].mean() - d[~abstract].mean()), mw_p=float(mw.pvalue))
        report["interactions"][f"domain_x_spec_csp_{c}"] = per_f

    for f, (g, s) in FAMILIES.items():
        report["ctr_deltas"][f] = {}
        for tag, cross, mono in (("zh_text", "csp_en2zh", "csp_zh2zh"), ("en_text", "csp_zh2en", "csp_en2en")):
            v, ci = ctr_delta_ci(models[g], models[s], cross, mono, ids)
            report["ctr_deltas"][f][tag] = dict(delta_CTR=v, ci95=ci)
        report["ctr_deltas"][f]["delta_unbiased_cka"] = (report["model_scalars"][s]["unbiased_cka"]
                                                         - report["model_scalars"][g]["unbiased_cka"])

    labels, ps = zip(*primary_p)
    report["multiplicity"] = dict(labels=list(labels), p_raw=list(ps), p_holm=holm(ps), q_bh=bh_fdr(ps))
    save_json(OUT, report)

    print("\n metric                 " + "  ".join(f"{f:>18}" for f in FAMILIES) + "   pooled p (std)")
    for metric in PRIMARY:
        b = report["per_metric"][metric]
        print(f" {metric:22s} " + "  ".join(f"{b['per_family'][f]['mean_delta']:+.4f} (p={b['per_family'][f]['signflip_p']:.3f})"
                                            for f in FAMILIES) + f"   {b['pooled_stratified_signflip_p_std']:.4f}")
    plot(report)
    print(f"[done] {OUT}, {FIG}")


def plot(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib missing; skipping figure")
        return
    fams = list(FAMILIES)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    ax = axes[0]                                          # A: CSP per condition per model
    w, x = 0.13, np.arange(len(CONDITIONS))
    for k, f in enumerate(fams):
        for j, (key, hatch) in enumerate(zip(FAMILIES[f], ("", "//"))):
            vals = [report["model_scalars"][key][f"csp_{c}"] for c in CONDITIONS]
            ax.bar(x + (2 * k + j - 2.5) * w, vals, w, color=f"C{k}", hatch=hatch, edgecolor="k", label=key)
    ax.set_xticks(x); ax.set_xticklabels(CONDITIONS); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel({"mean": "Concept-specific potency (nats/token)", "sum": "Concept-specific potency (nats/passage)",
                   "head": "Concept-specific potency (nats/token, first tokens)"}[NLL_KIND])
    ax.set_title("A. CSP by condition (hatched = specialized)"); ax.legend(fontsize=7, ncol=2)

    ax = axes[1]                                          # B: paired specialization deltas
    for k, f in enumerate(fams):
        m = [report["per_metric"][p]["per_family"][f] for p in PRIMARY]
        mu = np.array([b["mean_delta"] for b in m])
        ci = np.array([b["ci95"] for b in m])
        ax.errorbar(np.arange(len(PRIMARY)) + (k - 1) * 0.2, mu, yerr=np.abs(ci.T - mu),
                    fmt="o", capsize=3, color=f"C{k}", label=f)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(PRIMARY)))
    ax.set_xticklabels(PRIMARY, rotation=25, fontsize=8)
    ax.set_ylabel("specialized - general (paired, 95% CI)")
    ax.set_title("B. Lineage-controlled specialization effect"); ax.legend()

    ax = axes[2]                                          # C: causal transfer ratio
    for k, f in enumerate(fams):
        g, s = FAMILIES[f]
        for t, mk in (("zh_text", "o"), ("en_text", "s")):
            ax.plot([0, 1], [report["model_scalars"][g][f"CTR_{t}"], report["model_scalars"][s][f"CTR_{t}"]],
                    marker=mk, color=f"C{k}", label=f"{f} {t}")
    ax.axhline(1, color="gray", ls="--"); ax.set_xticks([0, 1]); ax.set_xticklabels(["general", "specialized"])
    ax.set_ylabel("CTR = CSP(cross) / CSP(mono)"); ax.set_title("C. Cross-lingual causal transfer")
    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG, dpi=200)


if __name__ == "__main__":
    main()