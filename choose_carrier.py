#!/usr/bin/env python3
"""choose_carrier.py — python3 choose_carrier.py <parent_key> <child_key>
Reads phase3_sanity_<key>.json (full explore runs). Picks one carrier token that passes in BOTH models,
ranked by how closely parent and child baselines match (carrier NLL / plain NLL, per language)."""
import json
import sys


def load(k):
    return json.load(open(f"phase3_sanity_{k}.json", encoding="utf-8"))


def main():
    pk, ck = sys.argv[1:3]
    P, C = load(pk), load(ck)

    print("== pair-level gates ==")
    for k, d in ((pk, P), (ck, C)):
        v = d.get("verdict", {})
        print(f"  {k}: {v.get('status', 'no verdict (re-run sanity)')}  fails={v.get('fail', [])}  warns={v.get('warn', [])}")
    for lang in ("en", "zh"):
        a = P.get("position_sweep", {}).get(lang, {}).get("recommended_exclude_leading_n")
        b = C.get("position_sweep", {}).get(lang, {}).get("recommended_exclude_leading_n")
        print(f"  leading-sink run ({lang}): parent {a} | child {b} -> {'OK' if a == b and a is not None else 'MISMATCH, investigate'}")

    cb = {c["id"]: c for c in C["carrier"]["candidates"]}
    rows = []
    for ca in P["carrier"]["candidates"]:
        c2 = cb.get(ca["id"])
        if not c2:
            continue
        gap = sum(abs(ca[l]["ratio_to_plain"] - c2[l]["ratio_to_plain"]) for l in ("en", "zh")) / 2
        rows.append(dict(id=ca["id"], token=ca["token"], label=ca["label"] if ca["label"] != "DEFAULT" else c2["label"],
                         both_pass=ca["pass"] and c2["pass"], gap=gap,
                         sinkcos=max(ca["sink_cos"], c2["sink_cos"]), nmx=max(ca["nm_ratio"], c2["nm_ratio"]),
                         ratio=(ca["en"]["ratio_to_plain"], c2["en"]["ratio_to_plain"], ca["zh"]["ratio_to_plain"], c2["zh"]["ratio_to_plain"]),
                         flags=sorted(set(ca["flags"]) | set(c2["flags"]))))
    good = sorted([r for r in rows if r["both_pass"]], key=lambda r: (r["gap"], r["sinkcos"]))
    print("\n== candidates passing in BOTH models (best first: smallest parent/child baseline gap, then lowest sink cosine) ==")
    for r in good:
        print(f"  {r['token']!r:>12} id={r['id']:<6} gap={r['gap']:.2f} sinkcos={r['sinkcos']:.2f} nmx={r['nmx']:.2f} "
              f"ratio EN {r['ratio'][0]:.2f}/{r['ratio'][1]:.2f}  ZH {r['ratio'][2]:.2f}/{r['ratio'][3]:.2f}")
    if good:
        pin = good[0]["label"] if good[0]["label"] != "DEFAULT" else good[0]["token"]
        print(f"\nPIN in BOTH registry entries:  carrier_token={pin!r}")
        print("Then: python3 lineage_check.py ; PINNED_ONLY=1 SKIP_SWEEP=1 MODEL_KEY=<key> python3 phase3_sanity_checks.py (must be READY)")
    else:
        worst = sorted(rows, key=lambda r: (r["sinkcos"], r["gap"]))[:5]
        print("\nNo candidate passes in both models. Least sink-like options (report this as a limitation if you use one):")
        for r in worst:
            print(f"  {r['token']!r:>12} id={r['id']:<6} sinkcos={r['sinkcos']:.2f} flags={r['flags']}")


if __name__ == "__main__":
    main()