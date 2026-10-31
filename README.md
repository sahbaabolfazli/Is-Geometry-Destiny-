# Is Geometry Destiny?
### Cross-lingual concept-specific causal potency in matched general-purpose and translation-specialized LLMs

> **Live app:** https://isgeometrydestiny.streamlit.app/

## Motivation

This project began with my curiosity about the "black box" of cognition. During a
conversation with my mother, I wondered how the language a person speaks might shape how
they process information. That inspired a related question about artificial language
models: how do they represent abstract concepts across languages, and how does
translation-focused training affect those representations? This connection is an
inspiration for the study, not a claim that language models reproduce human brain
mechanisms.

## Research question

Do English and Chinese representations of abstract concepts exhibit concept-specific
causal potency, and does geometric alignment between those representations correspond to
similar functional effects? How do these relationships differ between general-purpose and
translation-specialized models within the Qwen, Gemma and Llama families?

The matched model pairs are: Qwen3-8B-Base vs LMT-60-8B-Base, Gemma-2-9B vs
GemmaX2-28-9B-Pretrain, and Llama-2-7b-hf vs TowerBase-7B-v0.1. Pairing models within a
family helps account for lineage, but siblings can differ in more than translation
training, so the pairing is an approximate control (see Limitations).

## Design

- **Concepts.** 48 abstract concept passages, each in English and Chinese
  (`phase2_final_48_local.json`): 40 synthetic + 8 human-collected, spanning 4 domains
  (math/physics, complex systems, philosophy, syntax baseline; 12 each). Human-collected
  items carry a verification history (`verification_status`, `revision_rounds`).
- **Concept vectors.** For each model, a concept vector is pooled from the model's
  activations while it reads that concept's passage (excluding leading tokens and norm
  outliers), at the model's intervention layer (`phase3_extraction_pipeline.py`). Vectors
  are stored in ChromaDB and exported here as `phase3_concept_vectors_<model>.npz`.
- **Intervention site.** One middle layer per model (e.g. layer 16 of 32 for Llama-2),
  one injection position (the 5th carrier token), one carrier token per model
  (`phase3_config.py`); carrier choices are recorded in `carrier_manifest.jsonl`.

The study then uses two complementary experimental setups at the same intervention site.

### Experiment 1: null-carrier injection - direction persistence and generation
(`phase3b_latent_injection.py`)

- The input is only 10 repetitions of the null carrier token; no concept passage is
  present anywhere in the input.
- The concept vector replaces the hidden state at the injection position at the
  intervention layer.
- **Direction persistence:** probe hooks at 5 later layers record cosine similarity
  between the injected vector and the surviving activation (the "decay curve").
- **Generation:** the model generates up to 150 tokens from the null carrier; the text is
  recorded as samples (no automated scoring of generated content).
- **Controls per concept:** norm-matched Gaussian noise, and a mismatch control (a
  different concept's vector), so persistence can be compared against both
  out-of-distribution and in-distribution alternatives.
- Outputs: `phase3b_injection_report_<model>.json`,
  `phase3b_significance_report_<model>.json`.

### Experiment 2: interference matrix - concept-specific potency
(`phase3d_interference_matrix.py`, `scoring.py`, `phase3d_stats.py`)

- The input is the null carrier followed by one supplied concept passage; the concept
  vector is injected at the same site while the model scores the passage
  (teacher-forced NLL over the passage tokens).
- Repeating for all 48 vectors x 48 passages in 4 conditions (EN->EN, ZH->ZH, EN->ZH,
  ZH->EN), plus none / noise / centroid controls, builds a 48x48 interference matrix per
  model and condition.
- **Decomposition.** For each target passage *i*, compare its NLL with the matched
  concept vector (matrix diagonal) against the average NLL with all other vectors on the
  same passage (off-diagonal mean): **concept-specific potency (CSP)** = other - matched.
  Influences shared by all concept vectors on that passage cancel in the subtraction.
  Norm-matched Gaussian noise and the mean (centroid) vector are separate baselines for
  the generic effect of injecting anything concept-like (generic potency = noise - other;
  total potency = noise - matched).
- **Geometry.** Unbiased CKA with a permutation null, centered-cosine alignment margin
  (per concept: cosine of the matched EN-ZH pair minus the mean cosine against the 47
  mismatched vectors, after per-language centering), RSA, and EN<->ZH retrieval
  (`phase3e_geometry_alignment.py`, `analysis_common.py`).
- **Family contrasts.** Paired specialized-minus-general deltas per family with sign-flip
  permutation tests, bootstrap CIs, stratified pooled tests, and Holm/BH multiplicity
  control (`phase4_family_contrasts.py`).
- **Controls.** Density/perplexity covariates (`density_controls.py`,
  `compute_density_proxy.py`), lexical-overlap covariate (`lexical_overlap_covariate.py`),
  attention-sink and position sweeps plus environment capture
  (`phase3_sanity_<model>.json`), and a human-vs-synthetic source moderator. The
  density checks test whether each concept's EN-minus-ZH monolingual CSP gap tracks the
  log EN/ZH token-count ratio and the log EN/ZH perplexity difference of its two
  passages (Spearman correlations, unadjusted p-values, `density_controls.py`): no
  significant association in four models, and significant associations in Gemma-2-9B
  (rho = -0.46, p = 0.001 and rho = +0.43, p = 0.002) and LMT-60-8B (rho = -0.38,
  p = 0.008 and rho = +0.33, p = 0.023). In those two models, part of the EN-ZH
  monolingual CSP gap covaries with tokenization and perplexity differences between the
  language versions (see Limitations). Null checks do not prove absence of confounds.

## Key results (recorded outputs)

1. **Similar cross-lingual and monolingual potency in the Qwen family.** Recorded mean CSP
   for English vectors on Chinese passages was similar to Chinese vectors on Chinese
   passages (Qwen3-8B: 0.00439 vs 0.00434 mean nats/token; LMT-60-8B: 0.00449 vs 0.00422).
   These are near-equal point estimates from one setup (one layer, one injection position,
   one carrier per model); no equivalence test was run, so this is observed similarity in
   these conditions, not established equality. In the Gemma family the cross-lingual effect
   was smaller than the monolingual one, and the Llama family was near zero throughout.
2. **Geometry and causal potency appear weakly coupled.** Unbiased CKA between EN and ZH
   concept geometry ranges from 0.935 to 0.992 across the six models (permutation
   p <= 5e-4), while the recorded per-concept Spearman correlations between alignment margin and cross-lingual CSP are small,
   mixed in sign, and mostly not statistically significant at n = 48
   (`phase4_family_contrasts.json -> coupling_geometry_vs_causal`). This is consistent with
   weak coupling in these data; it does not prove the absence of any relationship.
3. **Lower measured potency in the specialized Gemma model.** Mean CSP falls 92-97% from
   Gemma-2-9B to GemmaX2-28-9B across the four transfer conditions, while unbiased CKA
   remains high in both models (0.969 and 0.974 respectively). The Qwen pair does not show
   a comparable collapse; the Llama family is null at baseline in both models.
4. **Null-carrier direction persistence.** In each model's recorded phase3b report,
   retention of the injected direction at the first probe layer exceeded the norm-matched
   noise control in both languages (e.g. Qwen3-8B layer 23: 0.83 EN / 0.83 ZH vs 0.35
   noise; Gemma-2-9B layer 26: 0.70 / 0.70 vs 0.44; Llama-2-7B layer 20: 0.42 / 0.38 vs
   0.24). Injection also altered free generation.
5. **Documented null result, cause unresolved.** Llama-2-7B showed near-zero CSP in every
   condition, and its phase3b generations degenerated. Within its own activation scale, its concept vectors are small
   relative to ordinary-token norms (0.44-0.52 vs 0.71-0.87 in the other families);
   however, norm-matched random vectors shifted NLL more than the concept vectors did
   (~0.044 vs ~0.0003 nats), which a magnitude-only explanation would not predict. Carrier
   choice (`<unk>` for this model), intervention layer, and Chinese representation quality
   remain untested alternative explanations.

## Status

The initially planned experimental phase is complete. The recorded results characterize
representation geometry, direction persistence, generated outputs and concept-specific
predictive effects under the tested conditions. Broader interpretations remain
provisional: further robustness checks and refined evaluations are needed to assess
generalization and to distinguish competing explanations for the observed effects.
Follow-up experiments are proposed, not yet completed: a CSP layer sweep, carrier-
robustness tests, injection-strength rescaling for the Llama family, and task-based
evaluation of reasoning performance and generated content.

## Methods references and related work

These are the methods implemented in the code and related published work on activation
interventions. This list is not a record of sources consulted while building the
project.

- Unbiased linear CKA: Kornblith et al., "Similarity of Neural Network Representations
  Revisited" (ICML 2019), https://arxiv.org/abs/1905.00414
- The CKA implementation's unbiased HSIC estimator is cited in `analysis_common.py` as
  "Song et al. 2012": Song et al., "Feature Selection via Dependence Maximization"
  (JMLR 2012), https://jmlr.org/papers/volume13/song12a/song12a.pdf. The same code
  comment cites "Nguyen et al. 2021"; the intended paper has not been identified.
- Related work on activation interventions: Turner et al., "Activation Addition:
  Steering Language Models Without Optimization" (2023),
  https://arxiv.org/abs/2308.10248; Todd et al., "Function Vectors in Large Language
  Models" (2023), https://arxiv.org/abs/2310.15213; Meng et al., "Locating and Editing
  Factual Associations in GPT" (NeurIPS 2022); Vig et al., "Causal Mediation Analysis
  for Interpreting Neural NLP" (2020), https://arxiv.org/abs/2004.12265.

## Contributions and credits

The project-specific contributions of this study are its design, implementation and
analysis choices:

- A 48x48 concept-vector x passage evaluation with the matched-versus-mismatched CSP
  calculation, plus norm-matched noise and centroid baselines that separate
  concept-specific effects from the generic effect of injecting anything concept-like.
- A matched-family study design comparing general-purpose and translation-specialized
  model pairs within the Qwen, Gemma and Llama families.
- Two complementary experiments at one intervention site: null-carrier injection
  (direction persistence and free generation) and passage scoring under injection
  (teacher-forced NLL).
- A coupling analysis relating per-concept cross-lingual alignment margins to
  cross-lingual CSP, connecting representational geometry to predictive effects.

These build on established methods (see Methods references and related work); listing
them as contributions of this study is a statement about its design and implementation,
not a claim of methodological novelty.

## Repository layout

| Path | Contents |
|---|---|
| `phase2_final_48_local.json` | The 48 EN/ZH concept items with provenance |
| `phase3*.py`, `phase4*.py` | Extraction, injection, interference, stats, geometry, family contrasts |
| `analysis_common.py` | Shared stats: bootstrap, sign-flip, Holm/BH, unbiased CKA, RSA |
| `<model>/` | Per-model recorded outputs (reports, sanity checks, matrices, per-concept stats) |
| `All_family/` | Cross-model contrasts, figures, lexical-overlap covariates |
| `run_analysis.sh`, `run_modal.py` | Full pipeline; Modal (A100-80GB) runner for GPU phases |
| `app.py` | Streamlit dashboard over the recorded outputs (no recomputation) |
| `README_ANALYSIS.md` | Notes on the analysis-layer files |

## Reproducing (not independently verified)

The commands below come from the repo's own notes (`README_ANALYSIS.md`, `run_modal.py`)
and have **not** been re-run as part of preparing this README. The per-model sanity files
record the environment used for each run (e.g. torch 2.14.1, transformers 5.20.2,
bfloat16, eager attention). GPU phases were run on Modal A100-80GB (`run_modal.py`, logs
in `modal_logs/`). Gated models (Llama-2, Gemma-2) require accepted Hugging Face
licenses, and the concept-vector store uses ChromaDB.

```bash
pip install -r requirements.txt   # app dependencies; research environment differs, see above
python3 lineage_check.py
for k in qwen3_8b lmt_60_8b gemma2_9b gemmax2_28_9b llama2_7b towerbase_7b; do
  MODEL_KEY=$k python3 compute_density_proxy.py
  MODEL_KEY=$k python3 phase3d_interference_matrix.py      # GPU
  python3 phase3d_stats.py $k
  python3 phase3e_geometry_alignment.py $k
done
python3 phase4_family_contrasts.py
```

These steps do not cover building the ChromaDB concept-vector store, which is not fully
recorded in the scripts; the exported `.npz` files preserve the extracted vectors.

## License

The code in this repository is released under the MIT License (see `LICENSE`). This
covers the author's own code only: model weights (Llama-2, Gemma, Qwen and other
families) and any third-party code remain under their own licenses. The concept
passages (`phase2_final_48_local.json`) and recorded outputs are data/text rather than
code and are not covered by this license grant.

## Limitations

- One intervention layer, one injection position, and one carrier per model; results may
  not transfer across layers or carriers (carrier-robustness tooling exists:
  `phase4_carrier_robustness.py`).
- Direction persistence (cosine similarity at probe layers) is not CSP: the two
  experiments measure propagation of the injected direction and predictive effect on
  passages separately.
- Recorded generation outputs are samples; they are not an evaluation of whether the
  generated text expresses the intended concept, and no reasoning performance is measured.
- In Gemma-2-9B and LMT-60-8B, part of the EN-vs-ZH monolingual CSP gap covaries with
  EN/ZH token-count and perplexity ratios (`density_controls_*.json`, unadjusted tests);
  cross-language gap conclusions for those models should be read with that association
  in mind.
- The matched pairs differ in more than translation training alone, so family contrasts
  are an approximate lineage control, not isolation of a single cause.
- Base models, not instruction-tuned variants; 48 concepts in 4 domains.
- The ChromaDB vector store is not included (excluded to keep the repo light); concept
  vectors are preserved as `.npz`.
- All results are recorded outputs of the pipeline; see per-model sanity files for the
  captured environments.
