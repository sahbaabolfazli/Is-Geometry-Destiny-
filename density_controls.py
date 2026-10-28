#!/usr/bin/env python3
"""python3 density_controls.py <model_key>
Does the EN-vs-ZH difference in monolingual CSP merely reflect token-count / perplexity (information density)
differences? One row per concept, so there is no pseudo-replication."""
import json
import sys
import numpy as np
from scipy import stats
from analysis_common import save_json, out_tag, NLL_KIND
from phase3d_stats import ols_partial                # reused HC3 OLS (reads the same argv[1] model key)

key = sys.argv[1]
P = json.load(open(f"phase3d_per_concept_{key}{out_tag()}.json", encoding="utf-8"))
ppl = {e["id"]: e for e in json.load(open(f"phase3_perplexity_{key}.json", encoding="utf-8"))}
ids = P["ids"]
y = np.array(P["csp_en2en"]) - np.array(P["csp_zh2zh"])                        # EN-ZH mono CSP gap per concept
log_tok = np.array([np.log(ppl[i]["en_tokens"] / ppl[i]["zh_tokens"]) for i in ids])
log_ppl = np.array([np.log(ppl[i]["en_ppl"]) - np.log(ppl[i]["zh_ppl"]) for i in ids])
res = dict(model_key=key, n=len(ids), nll_kind=NLL_KIND,
           mean_gap=float(y.mean()),
           spearman_vs_log_token_ratio=dict(zip(("rho", "p"), map(float, stats.spearmanr(log_tok, y)))),
           spearman_vs_log_ppl_ratio=dict(zip(("rho", "p"), map(float, stats.spearmanr(log_ppl, y)))),
           ols_hc3=ols_partial(y, np.column_stack([stats.zscore(log_tok), stats.zscore(log_ppl)]),
                               ["z_log_token_ratio", "z_log_ppl_ratio"]))
save_json(f"density_controls_{key}{out_tag()}.json", res)
print(res["spearman_vs_log_token_ratio"], res["spearman_vs_log_ppl_ratio"])