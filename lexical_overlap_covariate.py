#!/usr/bin/env python3
"""python3 lexical_overlap_covariate.py <family>   (qwen3 | gemma2 | llama2)
Per concept: fraction of ZH tokens that contain Latin letters AND also occur in the EN passage (shared proper
names / acronyms). Tests whether cross-lingual CSP depends on that overlap, and re-estimates it on the
low-overlap half."""
import json
import re
import sys
import numpy as np
from scipy import stats
from transformers import AutoTokenizer
import phase3_config as cfg
from analysis_common import FAMILIES, load_items, sign_flip_test, bootstrap_ci, save_json, out_tag, RUN_TAG

LATIN = re.compile(r"[A-Za-z]")
fam = sys.argv[1]
gen, spe = FAMILIES[fam]
tok = AutoTokenizer.from_pretrained(cfg.MODEL_REGISTRY[gen]["name"], trust_remote_code=True)   # same vocab within a family

cov = {}
for r in load_items():
    en = tok(r["english"], add_special_tokens=False)["input_ids"]
    zh = tok(r["chinese"], add_special_tokens=False)["input_ids"]
    en_set = set(en)
    leak = sum(1 for t in zh if t in en_set and LATIN.search(tok.decode([t])))
    cov[r["id"]] = leak / len(zh)

out = {}
for key in (gen, spe):
    P = json.load(open(f"phase3d_per_concept_{key}{out_tag()}.json", encoding="utf-8"))
    ids = P["ids"]
    x = np.array([cov[i] for i in ids])
    out[key] = {}
    for c in ("en2zh", "zh2en"):
        y = np.array(P[f"csp_{c}"])
        rho, p = stats.spearmanr(x, y)
        low = x <= np.median(x)
        out[key][c] = dict(spearman_overlap_vs_csp=dict(rho=float(rho), p=float(p)),
                           csp_low_overlap_half=dict(mean=float(y[low].mean()), ci95=bootstrap_ci(y[low]),
                                                     signflip_p_gt0=sign_flip_test(y[low], alternative="greater")),
                           csp_high_overlap_half=float(y[~low].mean()))
    print(key, {c: (round(v["spearman_overlap_vs_csp"]["rho"], 2), round(v["csp_low_overlap_half"]["mean"], 4))
                for c, v in out[key].items()})
save_json(f"lexical_overlap_{fam}{out_tag()}.json", dict(per_concept_overlap=cov, results=out))