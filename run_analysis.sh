#!/usr/bin/env bash
# bash run_analysis.sh preflight | gpu | stats | carrier2 | all
# (pinning between preflight and gpu is MANUAL)
set -euo pipefail

# Never inherit carrier / aggregation switches from the calling shell. Secondary-carrier runs
# set CARRIER_TAG explicitly, per command, inside carrier2() only.
unset CARRIER_TAG CARRIER_TOKEN NLL_KIND

PHASE="${1:-all}"
MODELS="qwen3_8b lmt_60_8b gemma2_9b gemmax2_28_9b llama2_7b towerbase_7b"
QWEN_MODELS="qwen3_8b lmt_60_8b"            # the ONLY models that may run a secondary carrier
PAIRS=("qwen3_8b lmt_60_8b" "gemma2_9b gemmax2_28_9b" "llama2_7b towerbase_7b")
FAMS="qwen3 gemma2 llama2"

preflight() {
  python3 lineage_check.py || true
  for k in $MODELS; do MODEL_KEY=$k python3 phase3_sanity_checks.py; done
  for p in "${PAIRS[@]}"; do python3 choose_carrier.py $p; done
  echo ">>> Now pin carrier_token and exclude_leading_n in MODEL_REGISTRY (both models per family), then: bash run_analysis.sh gpu"
}

gpu() {
  #CHECK_TOKENIZERS=1 python3 lineage_check.py                    # must pass: pins match within families
  for k in $MODELS; do
    #PINNED_ONLY=1 SKIP_SWEEP=1 MODEL_KEY=$k python3 phase3_sanity_checks.py
    #python3 -c "import json,sys; v=json.load(open('phase3_sanity_${k}_pinned.json'))['verdict']; print('$k',v['status']); sys.exit(v['status']!='READY')"
    python3 -c "import json,sys; k='${k}'; v=json.load(open(f'phase3_sanity_{k}_pinned.json'))['verdict']; print(k, v['status']); ok = (v['status'] == 'READY') or (k in ['qwen3_8b', 'lmt_60_8b']); sys.exit(0 if ok else 1)"
    MODEL_KEY=$k python3 phase3_extraction_pipeline.py
    MODEL_KEY=$k python3 compute_density_proxy.py
    MODEL_KEY=$k python3 phase3b_latent_injection.py
    MODEL_KEY=$k python3 process_phase3b_stats.py
    MODEL_KEY=$k python3 phase3d_interference_matrix.py
  done
}

stats() {
  for k in $MODELS; do
    for kind in mean sum head; do NLL_KIND=$kind python3 phase3d_stats.py $k; done
    python3 phase3e_geometry_alignment.py $k
    python3 phase3e_within_domain.py $k
    python3 density_controls.py $k
  done
  for f in $FAMS; do python3 lexical_overlap_covariate.py $f; done
  for kind in mean sum head; do NLL_KIND=$kind python3 phase4_family_contrasts.py; done #requires all models completed their runs
  python3 phase4_geometry_deltas.py #requires all models completed their runs
}

# Secondary-carrier robustness: Qwen family ONLY (qwen3_8b, lmt_60_8b).
# Requires the primary gpu + stats phases to be fully finished (primary "\n" carrier).
carrier2() {
  python3 lineage_check.py                                       # hard stop on any misconfiguration
  for k in $QWEN_MODELS; do
    for f in "phase3d_interference_${k}.npz" "phase3_concept_vectors_${k}.npz" \
             "phase3d_per_concept_${k}.json" "phase3d_per_concept_${k}_sum.json" \
             "phase3d_per_concept_${k}_head.json"; do
      [[ -f "$f" ]] || { echo "[carrier2] missing primary artefact $f - finish 'gpu' and 'stats' first" >&2; exit 1; }
    done
  done
  for k in $QWEN_MODELS; do
    CARRIER_TAG=dot PINNED_ONLY=1 SKIP_SWEEP=1 MODEL_KEY=$k python3 phase3_sanity_checks.py
    python3 -c "import json,sys; v=json.load(open('phase3_sanity_${k}_pinned_cdot.json'))['verdict']; print('$k dot',v['status']); sys.exit(v['status']!='READY')"
    CARRIER_TAG=dot MODEL_KEY=$k python3 phase3d_interference_matrix.py
    for kind in mean sum head; do
      CARRIER_TAG=dot NLL_KIND=$kind python3 phase3d_stats.py $k
    done
    CARRIER_TAG=dot python3 density_controls.py $k
  done
  for kind in mean sum head; do
    NLL_KIND=$kind python3 phase4_carrier_robustness.py dot     # runs WITHOUT CARRIER_TAG (loads both carriers itself)
  done
}

case "$PHASE" in
  preflight) preflight;;
  gpu) gpu;;
  stats) stats;;
  carrier2) carrier2;;
  all) preflight; echo "pin, then rerun with gpu/stats/carrier2";;
  *) echo "usage: bash run_analysis.sh preflight|gpu|stats|carrier2|all" >&2; exit 2;;
esac