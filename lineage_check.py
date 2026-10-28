#!/usr/bin/env python3
"""lineage_check.py — verify that each (general, specialized) family pair is a valid matched contrast.
    python3 lineage_check.py               # config-level checks only (no downloads)
    CHECK_TOKENIZERS=1 python3 lineage_check.py   # also compare tokenizers + carrier token ids

Checks (FAIL = the within-family contrast is not lineage-controlled):
  * analysis_common.FAMILIES matches phase3_config.MODEL_REGISTRY keys and category tags
  * same layer_idx and probe_layers within a family
  * same pinned carrier_token within a family (and it is pinned at all)
  * secondary_carriers: identical within a family, present ONLY for the Qwen family, agree with
    analysis_common.SECONDARY_CARRIERS, and never equal to the primary carrier
  * [CHECK_TOKENIZERS] each carrier string (primary + secondary) maps to the same token id in both tokenizers,
    and vocab overlap / EN+ZH tokenization of the dataset is reported (LMT and GemmaX2 may extend vocab)
"""
import os
import sys

# This script validates the REGISTRY. An exported CARRIER_TAG/CARRIER_TOKEN would make phase3_config
# exit (or resolve a non-primary carrier) before any check runs, so refuse explicitly and clearly.
if os.environ.get("CARRIER_TAG", "").strip() or os.environ.get("CARRIER_TOKEN"):
    sys.exit("[lineage] run without CARRIER_TAG / CARRIER_TOKEN set: this script validates the registry, "
             "not a secondary-carrier run.")

import phase3_config as cfg
from analysis_common import (FAMILIES, CATEGORY, load_items, TEXT_KEY, LANGS,
                             SECONDARY_CARRIERS, SECONDARY_CARRIER_FAMILIES, CARRIER_SFX)


def main():
    fail, warn = [], []
    reg = cfg.MODEL_REGISTRY
    for fam, (g, s) in FAMILIES.items():
        for k in (g, s):
            if k not in reg:
                fail.append(f"{fam}: {k} missing from MODEL_REGISTRY")
            elif reg[k]["category"] != CATEGORY[k]:
                fail.append(f"{fam}: {k} category {reg[k]['category']} != {CATEGORY[k]}")
        if g not in reg or s not in reg:
            continue
        for field in ("layer_idx", "probe_layers", "exclude_leading_n"):
            if reg[g].get(field) != reg[s].get(field):
                fail.append(f"{fam}: {field} differs ({reg[g].get(field)} vs {reg[s].get(field)})")
        cg, cs = reg[g].get("carrier_token"), reg[s].get("carrier_token")
        if cg is None or cs is None:
            warn.append(f"{fam}: carrier_token not pinned (run choose_carrier.py {g} {s})")
        elif cg != cs:
            fail.append(f"{fam}: carrier_token differs ({cg!r} vs {cs!r})")

        # Secondary carrier & Qwen-only scope checks
        sg, ss = reg[g].get("secondary_carriers", {}), reg[s].get("secondary_carriers", {})
        if sg != ss:
            fail.append(f"{fam}: secondary_carriers differ within family ({sg} vs {ss})")
        if bool(sg) != (fam in SECONDARY_CARRIER_FAMILIES):
            fail.append(f"{fam}: secondary_carriers present={bool(sg)} violates the Qwen-only scope")
        if any(SECONDARY_CARRIERS.get(t) != v for t, v in sg.items()):
            fail.append(f"{fam}: registry secondary carriers disagree with analysis_common.SECONDARY_CARRIERS")
        if cg is not None and cg in sg.values():
            fail.append(f"{fam}: a secondary carrier equals the primary carrier {cg!r}; the robustness check would be vacuous")
        if cfg.CARRIER_SFX != CARRIER_SFX:
            fail.append("CARRIER_SFX mismatch between phase3_config and analysis_common")

    if os.environ.get("CHECK_TOKENIZERS"):
        from transformers import AutoTokenizer
        items = load_items()
        for fam, (g, s) in FAMILIES.items():
            tg = AutoTokenizer.from_pretrained(reg[g]["name"], trust_remote_code=True)
            ts = AutoTokenizer.from_pretrained(reg[s]["name"], trust_remote_code=True)
            vg, vs = set(tg.get_vocab()), set(ts.get_vocab())
            print(f"[{fam}] vocab {len(vg)} vs {len(vs)}  overlap={len(vg & vs) / max(len(vg), 1):.3f}")
            carriers = {"primary": reg[g].get("carrier_token")}
            carriers.update({f"secondary:{t}": v for t, v in reg[g].get("secondary_carriers", {}).items()})
            for label, ct in carriers.items():
                if not ct:
                    continue
                ig, is_ = cfg.resolve_token_id(tg, ct), cfg.resolve_token_id(ts, ct)
                if ig is None or is_ is None:
                    fail.append(f"{fam}: {label} carrier {ct!r} is unusable in a tokenizer (ids {ig} vs {is_})")
                elif ig != is_:
                    warn.append(f"{fam}: {label} carrier {ct!r} -> ids {ig} vs {is_} (embeddings differ anyway)")
            for l in LANGS:
                diff = sum(tg(r[TEXT_KEY[l]], add_special_tokens=False)["input_ids"]
                           != ts(r[TEXT_KEY[l]], add_special_tokens=False)["input_ids"] for r in items)
                msg = f"{fam}: {diff}/{len(items)} {l} passages tokenize differently"
                if diff:
                    warn.append(msg + " (vocab extension; concept vectors are still per-model, so OK, but report it)")
                else:
                    print("  " + msg)

    for x in fail:
        print("FAIL:", x)
    for x in warn:
        print("warn:", x)
    print("LINEAGE OK" if not fail else "LINEAGE CHECK FAILED")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()