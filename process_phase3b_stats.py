#!/usr/bin/env python3
"""
Standalone Phase 3b Statistical Post-Processor.
Processes already-generated phase3b_injection_report_*.json files
to compute significance metrics without re-running model inference.
(Excludes the middle injection layer from probe/decay metrics).

If the report contains the in-distribution `mismatch_control` (another concept's EN vector), each layer
entry also gets mean_mismatch and an EN-vs-mismatch Wilcoxon test (retention beyond an in-distribution
vector). Reports written before that control existed are processed as before, without those fields.
"""

import json
from pathlib import Path
import numpy as np
import scipy.stats as stats

import phase3_config as cfg

REPORT_PATH = Path(f"phase3b_injection_report_{cfg.ACTIVE_MODEL_KEY}.json")
STATS_REPORT_PATH = Path(f"phase3b_significance_report_{cfg.ACTIVE_MODEL_KEY}.json")
PHASE2_INPUT_PATH = Path("phase2_final_48_local.json")


def main():
    cfg.require_primary_carrier("process_phase3b_stats.py")
    if not REPORT_PATH.exists():
        print(f"[error] Report file {REPORT_PATH} not found!")
        return

    print(f"[info] Loading injection report from {REPORT_PATH} ...")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    has_mm = all("mismatch_control" in e for e in report)
    if not has_mm:
        print("[warn] No mismatch_control in this report (older run); skipping mismatch statistics. "
              "Re-run phase3b_latent_injection.py to add them.")

    items_raw = json.loads(PHASE2_INPUT_PATH.read_text(encoding="utf-8"))
    domain_map = {
        r["id"]: r.get("domain", "unknown")
        for r in items_raw
        if r["verification_status"] in ("APPROVED_FIRST_PASS", "APPROVED_AFTER_REVISION")
    }

    raw_significance = []
    # Restrict statistical processing strictly to PROBE_LAYERS, excluding cfg.LAYER_IDX
    probe_layers_list = [str(L) for L in cfg.PROBE_LAYERS]
    num_tests_family = len(probe_layers_list)

    for L_str in probe_layers_list:
        L = int(L_str)

        en_vals, zh_vals, noise_vals, mm_vals = [], [], [], []
        for e in report:
            decay_en = e["en"]["decay_curve"]
            decay_zh = e["zh"]["decay_curve"]
            decay_noise = e["noise_control"]["decay_curve"]

            # Safely look up layer key as string or int to prevent KeyError
            val_en = decay_en.get(str(L), decay_en.get(L, None))
            val_zh = decay_zh.get(str(L), decay_zh.get(L, None))
            val_noise = decay_noise.get(str(L), decay_noise.get(L, None))
            if has_mm:
                decay_mm = e["mismatch_control"]["decay_curve"]
                val_mm = decay_mm.get(str(L), decay_mm.get(L, None))
            else:
                val_mm = 0.0                      # placeholder; never used when has_mm is False

            if (val_en is not None and val_zh is not None and val_noise is not None
                    and val_mm is not None):
                en_vals.append(val_en)
                zh_vals.append(val_zh)
                noise_vals.append(val_noise)
                if has_mm:
                    mm_vals.append(val_mm)        # same concept filter as the others -> paired

        if not en_vals:
            print(f"[warn] No valid data found for layer {L}, skipping.")
            continue

        en_arr, zh_arr = np.asarray(en_vals), np.asarray(zh_vals)
        noise_arr = np.asarray(noise_vals)
        diffs_ez = en_arr - zh_arr

        t_stat, t_p = stats.ttest_rel(en_arr, zh_arr)
        try:
            w_stat, w_p = stats.wilcoxon(en_arr, zh_arr, zero_method="wilcox")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        cohens_d_ez = float(diffs_ez.mean() / (diffs_ez.std(ddof=1) + 1e-12))

        _, en_noise_p = stats.wilcoxon(en_arr, noise_arr, zero_method="wilcox")
        _, zh_noise_p = stats.wilcoxon(zh_arr, noise_arr, zero_method="wilcox")

        domains = sorted(list(set(domain_map.values())))
        domain_stats = {}

        domain_en_groups = [
            [
                e["en"]["decay_curve"].get(str(L), e["en"]["decay_curve"].get(L))
                for e in report
                if e["domain"] == d and (e["en"]["decay_curve"].get(str(L), e["en"]["decay_curve"].get(L)) is not None)
            ]
            for d in domains
        ]
        if len(domains) > 1 and all(len(g) > 0 for g in domain_en_groups):
            kw_stat, kw_p = stats.kruskal(*domain_en_groups)
        else:
            kw_stat, kw_p = float("nan"), float("nan")

        for d in domains:
            d_en = [
                e["en"]["decay_curve"].get(str(L), e["en"]["decay_curve"].get(L))
                for e in report
                if e["domain"] == d and (e["en"]["decay_curve"].get(str(L), e["en"]["decay_curve"].get(L)) is not None)
            ]
            d_zh = [
                e["zh"]["decay_curve"].get(str(L), e["zh"]["decay_curve"].get(L))
                for e in report
                if e["domain"] == d and (e["zh"]["decay_curve"].get(str(L), e["zh"]["decay_curve"].get(L)) is not None)
            ]
            domain_stats[d] = {
                "mean_en": float(np.mean(d_en)) if d_en else 0.0,
                "mean_zh": float(np.mean(d_zh)) if d_zh else 0.0,
                "n": len(d_en),
            }

        entry = {
            "layer": L,
            "n": len(en_vals),
            "mean_en": float(en_arr.mean()),
            "mean_zh": float(zh_arr.mean()),
            "mean_noise": float(noise_arr.mean()),
            "mean_diff_en_zh": float(diffs_ez.mean()),
            "t_stat": float(t_stat),
            "t_p_raw": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p_raw": float(w_p),
            "cohens_d": cohens_d_ez,
            "en_vs_noise_wilcoxon_p_raw": float(en_noise_p),
            "zh_vs_noise_wilcoxon_p_raw": float(zh_noise_p),
            "domain_kruskal_wallis_p_raw": float(kw_p),
            "domains": domain_stats,
        }

        if has_mm:
            mm_arr = np.asarray(mm_vals)
            try:
                # retention beyond an in-distribution vector (another concept's EN vector)
                _, p_en_mm = stats.wilcoxon(en_arr, mm_arr, zero_method="wilcox")
            except ValueError:                    # e.g. all differences exactly zero
                p_en_mm = float("nan")
            entry["mean_mismatch"] = float(mm_arr.mean())
            entry["mean_diff_en_minus_mismatch"] = float((en_arr - mm_arr).mean())
            entry["en_vs_mismatch_wilcoxon_p_raw"] = float(p_en_mm)

        raw_significance.append(entry)

    significance_report = []
    for entry in raw_significance:
        corrected_entry = entry.copy()
        corrected_entry["t_p_bonferroni"] = float(
            min(1.0, entry["t_p_raw"] * num_tests_family)
        )
        corrected_entry["wilcoxon_p_bonferroni"] = float(
            min(1.0, entry["wilcoxon_p_raw"] * num_tests_family)
        )
        corrected_entry["en_vs_noise_wilcoxon_p_bonferroni"] = float(
            min(1.0, entry["en_vs_noise_wilcoxon_p_raw"] * num_tests_family)
        )
        corrected_entry["zh_vs_noise_wilcoxon_p_bonferroni"] = float(
            min(1.0, entry["zh_vs_noise_wilcoxon_p_raw"] * num_tests_family)
        )
        if "en_vs_mismatch_wilcoxon_p_raw" in entry:
            p_mm = entry["en_vs_mismatch_wilcoxon_p_raw"]
            corrected_entry["en_vs_mismatch_wilcoxon_p_bonferroni"] = (
                float("nan") if np.isnan(p_mm) else float(min(1.0, p_mm * num_tests_family))
            )
        significance_report.append(corrected_entry)

    STATS_REPORT_PATH.write_text(
        json.dumps(significance_report, indent=2), encoding="utf-8"
    )
    print(f"[done] Wrote fully corrected significance report to {STATS_REPORT_PATH}")


if __name__ == "__main__":
    main()