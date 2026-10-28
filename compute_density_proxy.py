#!/usr/bin/env python3
"""Step 6: Compute per-language passage perplexity, total nats, and explicit ratios,
outputting strictly to phase3_perplexity_<model_key>.json.

Note: HF's loss averages over n_tokens - 1 predicted positions (the first token is never
predicted), so total nats = mean_loss * (n_tokens - 1).
"""
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

import phase3_config as cfg

PHASE2_INPUT_PATH = Path("phase2_final_48_local.json")
OUT_PATH = Path(f"phase3_perplexity_{cfg.ACTIVE_MODEL_KEY}.json")


def load_items():
    records = json.loads(PHASE2_INPUT_PATH.read_text(encoding="utf-8"))
    return [
        r for r in records
        if r["verification_status"] in ("APPROVED_FIRST_PASS", "APPROVED_AFTER_REVISION")
    ]


@torch.no_grad()
def passage_metrics(model, tokenizer, text):
    """Compute per-token perplexity, mean loss (nats/token), and total nats."""
    enc = tokenizer(text, return_tensors="pt").to(cfg.DEVICE)
    input_ids = enc["input_ids"]
    n_tokens = input_ids.shape[1]
    out = model(**enc, labels=input_ids)
    mean_loss = float(out.loss.item())
    ppl = float(torch.exp(out.loss).item())
    total_nats = mean_loss * (n_tokens - 1)       # HF loss excludes the first token
    return ppl, mean_loss, total_nats, n_tokens


def main():
    cfg.require_primary_carrier("compute_density_proxy.py")   # carrier-independent, unsuffixed output
    items = load_items()
    print(f"[info] Loading {cfg.TARGET_MODEL_NAME} (category={cfg.MODEL_CATEGORY}) on {cfg.DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.TARGET_MODEL_NAME, trust_remote_code=True)
    model = cfg.load_causal_lm()
    cfg.assert_layers_exist(model)

    results = []
    for it in items:
        domain = it.get("domain", "unknown")
        en_ppl, en_loss, en_total_nats, en_tokens = passage_metrics(model, tokenizer, it["english"])
        zh_ppl, zh_loss, zh_total_nats, zh_tokens = passage_metrics(model, tokenizer, it["chinese"])

        # Explicitly calculate density control ratios & length-neutralized metrics
        ppl_ratio = en_ppl / (zh_ppl + 1e-9)
        total_nats_ratio = en_total_nats / (zh_total_nats + 1e-9)
        log_diff = math.log(en_ppl + 1e-9) - math.log(zh_ppl + 1e-9)

        results.append({
            "id": it["id"],
            "domain": domain,
            "en_ppl": en_ppl,
            "en_loss": en_loss,
            "en_total_nats": en_total_nats,
            "en_tokens": en_tokens,
            "zh_ppl": zh_ppl,
            "zh_loss": zh_loss,
            "zh_total_nats": zh_total_nats,
            "zh_tokens": zh_tokens,
            "perplexity_ratio": ppl_ratio,
            "total_nats_ratio": total_nats_ratio,
            "log_perplexity_diff": log_diff,
        })
        print(
            f"  {it['id']} [{domain}]: en_ppl={en_ppl:.2f} (nats={en_total_nats:.1f}) |"
            f" zh_ppl={zh_ppl:.2f} (nats={zh_total_nats:.1f}) |"
            f" ratio(en/zh)={ppl_ratio:.3f}"
        )

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] Wrote {len(results)} per-concept perplexity & total nats pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()