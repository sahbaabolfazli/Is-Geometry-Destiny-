# Analysis framework: concept-specific causal potency & lineage-controlled specialization

The experimental setup is unchanged: the same 48 EN/ZH concept vectors at LAYER_IDX, stored in
ChromaDB, injected at position 5 (which is 4 when indexed from 0) of the same null carrier at the same layer.
The new files only add analysis.

## New files
| File | GPU? | Purpose |
|---|---|---|
| `analysis_common.py` | no | stats helpers (bootstrap, sign-flip, Holm/BH, unbiased CKA, RSA, retrieval) |
| `phase3d_interference_matrix.py` | yes | 48x48 NLL matrices x 4 conditions (en2en, zh2zh, en2zh, zh2en) + none/noise/centroid controls |
| `phase3d_stats.py` | no | total / generic / **concept-specific potency (CSP)**, AUC, top-1, CTR, moderators |
| `phase3e_geometry_alignment.py` | no | unbiased CKA + permutation null, retrieval, alignment margin, RSA |
| `phase4_family_contrasts.py` | no | paired (specialized - general) deltas per family, stratified pooled test, DiD, meta-analysis, figure |
| `lineage_check.py` | no* | checks family matching (layers, carrier, categories); referenced by your config but was missing |
| `run_analysis.sh` | - | runs the full pipeline for all 6 models |


\* `CHECK_TOKENIZERS=1` downloads the tokenizers.

## Run
```bash
python3 lineage_check.py
for k in qwen3_8b lmt_60_8b gemma2_9b gemmax2_28_9b llama2_7b towerbase_7b; do
  MODEL_KEY=$k python3 compute_density_proxy.py
  MODEL_KEY=$k python3 phase3d_interference_matrix.py      # GPU, ~4x48 + 34 scoring sweeps
  python3 phase3d_stats.py $k
  python3 phase3e_geometry_alignment.py $k
done
python3 phase4_family_contrasts.py
```
If your phase-2 JSON has no provenance field, set `HUMAN_IDS_FILE=human_ids.json`, a JSON list of
the 8 human-collected ids. Without it, the human-vs-AI moderator is reported as unavailable.


