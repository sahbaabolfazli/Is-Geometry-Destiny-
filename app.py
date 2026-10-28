#!/usr/bin/env python3
"""
app.py - Interactive dashboard for the cross-lingual concept-specific causal potency study.

    streamlit run app.py

Research question: when a model processes an abstract concept in EN vs ZH, do its internal
representations carry concept-specific causal potency (CSP), and how does translation
specialization change this, with lineage controlled across 3 matched (general, specialized)
families?

Design notes
  * Path-agnostic discovery: every .json/.npz/.jsonl under the project root is indexed by file name
    (shallowest path wins), so flat roots, per-model sub-folders and All_family/ all work. PNGs are ignored.
  * All file reads go through @st.cache_data.
  * Nothing here recomputes the experiment; the only live computations are cheap re-derivations
    (CSP from the 48x48 matrices, PCA of concept vectors, grouping) so that filters by domain/source work.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cross-lingual causal potency", page_icon="🧬", layout="wide")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
FAMILIES = {
    "qwen3": ("qwen3_8b", "lmt_60_8b"),
    "gemma2": ("gemma2_9b", "gemmax2_28_9b"),
    "llama2": ("llama2_7b", "towerbase_7b"),
}
FAMILY_OF = {k: f for f, pair in FAMILIES.items() for k in pair}
CATEGORY = {g: "general_purpose" for g, _ in FAMILIES.values()}
CATEGORY.update({s: "translation_specialized" for _, s in FAMILIES.values()})
KNOWN_MODELS = [k for pair in FAMILIES.values() for k in pair]
CONDS = ["en2en", "zh2zh", "en2zh", "zh2en"]
COND_LABEL = {
    "en2en": "EN vector → EN text (mono)",
    "zh2zh": "ZH vector → ZH text (mono)",
    "en2zh": "EN vector → ZH text (cross)",
    "zh2en": "ZH vector → EN text (cross)",
}
KIND_SFX = {"mean": "", "sum": "_sum", "head": "_head"}      # json file suffix (analysis_common.out_tag)
NPZ_AGG = {"mean": "", "sum": "sum", "head": "head"}         # key infix inside the interference .npz
KIND_LABEL = {"mean": "mean (nats/token)", "sum": "sum (nats/passage)", "head": "head (first 16 tokens)"}
CARRIER_SFX = {"primary": "", "dot": "_cdot"}
DOMAINS = ["math_physics", "complex_systems", "philosophy", "syntax_baseline"]
SKIP_DIR_PARTS = {".git", "venv", ".venv", "__pycache__", "node_modules", "site-packages"}
PHASE2 = "phase2_final_48_local.json"


# --------------------------------------------------------------------------------------
# Discovery + cached loading
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Indexing workspace...")
def scan(root: str) -> dict[str, str]:
    """name -> path for every .json/.npz/.jsonl under root (png ignored; shallowest path wins)."""
    found = []
    for p in Path(root).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".json", ".npz", ".jsonl"}:
            continue
        if set(p.parts) & SKIP_DIR_PARTS or any(x.startswith("phase3_chroma_db") for x in p.parts):
            continue
        found.append(p)
    found.sort(key=lambda q: (len(q.parts), str(q)))
    idx: dict[str, str] = {}
    for p in found:
        idx.setdefault(p.name, str(p))
    return idx


@st.cache_data(show_spinner=False)
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_jsonl(path: str):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner="Loading tensor archive...")
def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as Z:
        return {k: Z[k] for k in Z.files}


class Workspace:
    def __init__(self, root: str):
        self.root = root
        self.idx = scan(root)

    def path(self, name: str):
        return self.idx.get(name)

    def json(self, name: str):
        p = self.path(name)
        return load_json(p) if p else None

    def npz(self, name: str):
        p = self.path(name)
        return load_npz(p) if p else None

    def has(self, name: str) -> bool:
        return name in self.idx

    def models(self) -> list[str]:
        extra = set()
        for n in self.idx:
            m = re.match(r"phase3d_(?:stats|per_concept|interference_meta)_(.+?)(?:_sum|_head)?(?:_cdot)?\.json$", n)
            if m:
                extra.add(m.group(1))
            m = re.match(r"phase3d_interference_(.+?)(?:_cdot)?\.npz$", n)
            if m:
                extra.add(m.group(1))
        keys = [k for k in KNOWN_MODELS if k in extra or any(k in n for n in self.idx)]
        keys += sorted(k for k in extra if k not in KNOWN_MODELS)
        return keys


# name builders ------------------------------------------------------------------------
def n_stats(k, kind, car): return f"phase3d_stats_{k}{KIND_SFX[kind]}{CARRIER_SFX[car]}.json"
def n_pc(k, kind, car): return f"phase3d_per_concept_{k}{KIND_SFX[kind]}{CARRIER_SFX[car]}.json"
def n_npz(k, car): return f"phase3d_interference_{k}{CARRIER_SFX[car]}.npz"
def n_meta(k, car): return f"phase3d_interference_meta_{k}{CARRIER_SFX[car]}.json"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def show(fig, height=None):
    if height:
        fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def ci_excludes_zero(ci):
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def meta_df(meta) -> pd.DataFrame:
    ids = meta["ids"]
    return pd.DataFrame({"id": ids, "domain": [meta["domain"][i] for i in ids],
                         "source": [meta["source"][i] for i in ids]})


def pc_df(pc) -> pd.DataFrame:
    base = {k: pc[k] for k in ("ids", "domain", "source")}
    df = pd.DataFrame(base).rename(columns={"ids": "id"})
    for k, v in pc.items():
        if isinstance(v, list) and len(v) == len(df) and k not in base:
            df[k] = v
    return df


def apply_filter(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    return df[df["domain"].isin(f["domains"]) & df["source"].isin(f["sources"])]


def forest(df, x, lo, hi, y, color=None, title="", height=None):
    d = df.copy()
    d["_ep"] = d[hi] - d[x]
    d["_em"] = d[x] - d[lo]
    fig = px.scatter(d, x=x, y=y, color=color, error_x="_ep", error_x_minus="_em", title=title)
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_traces(marker=dict(size=10))
    show(fig, height)


def cond_table(stats) -> pd.DataFrame:
    rows = []
    for c in CONDS:
        b = stats["conditions"][c]
        s, r = b["concept_specific_potency"], b["relative_csp"]
        rows.append({
            "condition": c, "CSP mean": s["mean"], "CSP 95% CI lo": s["ci95"][0], "CSP 95% CI hi": s["ci95"][1],
            "dz": s["cohens_dz"], "sign-flip p": s["signflip_p"],
            "rel. CSP mean": r["mean"], "rel. CI lo": r["ci95"][0], "rel. CI hi": r["ci95"][1],
            "AUC-ROC": b["auc"]["mean"], "top-1": b["retrieval"]["top1"], "MRR": b["retrieval"]["mrr"],
            "total potency": b["total_potency"]["mean"], "generic potency": b["generic_potency"]["mean"],
            "within-domain CSP": b["within_domain_csp"]["mean"],
            "matched NLL": b["mean_nll"]["matched"], "other NLL": b["mean_nll"]["other"],
            "noise NLL": b["mean_nll"]["noise"], "no-inj NLL": b["mean_nll"]["no_injection"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🧬 Controls")
    default_root = str(Path(__file__).resolve().parent) if "__file__" in globals() else "."
    root = st.sidebar.text_input("Project root", default_root)
    if not Path(root).exists():
        st.sidebar.error("Path does not exist.")
        st.stop()
    ws = Workspace(root)
    models = ws.models()
    if not models:
        st.error("No model artifacts found under the project root.")
        st.stop()

    mode = st.sidebar.radio("View", ["Single-model deep dive", "Cross-family comparison"])
    model = st.sidebar.selectbox("Model", models, format_func=lambda k: f"{k}  ({CATEGORY.get(k, '?').replace('_', ' ')})")
    kind = st.sidebar.selectbox("NLL aggregation", list(KIND_LABEL), format_func=KIND_LABEL.get)
    carrier = "primary"
    if any(ws.has(n_stats(k, "mean", "dot")) or ws.has(n_npz(k, "dot")) for k in models):
        carrier = st.sidebar.radio("Carrier token", ["primary", "dot"],
                                   help="'dot' (secondary '.') exists for the Qwen family only.",
                                   horizontal=True)
        if carrier == "dot" and FAMILY_OF.get(model) != "qwen3":
            st.sidebar.warning("Secondary carrier exists for Qwen family only; this model falls back to primary.")
    st.sidebar.markdown("**Concept filters**")
    domains = st.sidebar.multiselect("Domain", DOMAINS, default=DOMAINS)
    sources = st.sidebar.multiselect("Source", ["human", "ai"], default=["human", "ai"])
    with st.sidebar.expander("Workspace inventory"):
        st.caption(f"{len(ws.idx)} data files indexed")
        st.write(f"Models: {', '.join(models)}")
        fam_files = [n for n in ws.idx if n.startswith(("phase4_", "lexical_overlap_"))]
        st.write("Aggregate files:", ", ".join(sorted(fam_files)) or "none")
    return ws, models, mode, model, kind, carrier, dict(domains=domains, sources=sources)


def eff_carrier(ws, model, carrier):
    if carrier == "dot" and not ws.has(n_stats(model, "mean", "dot")) and not ws.has(n_npz(model, "dot")):
        return "primary"
    return carrier


# --------------------------------------------------------------------------------------
# Tab: Overview
# --------------------------------------------------------------------------------------
def tab_overview(ws, models, kind, carrier, flt):
    st.header("When a Large Language Model processes an abstract concept in English vs. Chinese, do its internal representations carry concept-specific causal potency (CSP), and how does architectural specialization alter this alignment?")
    st.markdown(
        r"""
### 1. Introduction & Motivation
As Large Language Models (LLMs) are increasingly deployed across diverse linguistic landscapes, a fundamental question in mechanistic interpretability arises: When a model processes an abstract concept in English versus Chinese, do its internal representations carry concept-specific causal potency, and how is this semantic alignment altered by architectural specialization?

To answer this, our research investigates the geometry and causal functionality of cross-lingual latent representations. Specifically, we evaluate whether semantic representations are stored in a language-neutral format and how structural adaptations—such as transitioning from a general-purpose base model to a multilingual or translation-specialized variant—modulate a model's ability to transfer concepts seamlessly across English and Mandarin.

---

### 2. Experimental Design & Test Material
To maintain strict experimental control, the study evaluates a standardized benchmark of **48 abstract concepts** distributed evenly across **4 distinct domains**:
* `math_physics`
* `complex_systems`
* `philosophy`
* `syntax_baseline`

This concept set comprises a robust mix of **8 human-collected** and **40 AI-generated** concepts (`phase2_final_48_local.json`), each rigorously defined and tested in both **English (EN)** and **Mandarin Chinese (ZH)**.

Our comparative analysis spans **6 distinct models organized into 3 matched lineages** (pairing a general-purpose base against its translation-specialized counterpart):
1. **Qwen3 Family:** `qwen3_8b` (General) vs. `lmt_60_8b` (Specialized)
2. **Gemma Family:** `gemma2_9b` (General) vs. `GemmaX2-28-9B-Pretrain` (Specialized)
3. **Llama Family:** `llama2_7b-hf` (General) vs. `TowerBase-7B-v0.1` (Specialized)

*By keeping lineage, tokenizers, and architectures closely matched within each family, we isolate the specific mechanical impact that translation-focused specialization has on internal model representations.*

---

### 3. Core Methodology: The Causal Injection Framework
To measure whether an internal model representation actively drives semantic understanding rather than merely correlating with it, we employ a causal intervention pipeline:
1. **Activation Extraction:** Intermediate layer activations (concept vectors) for all 48 concepts across both languages are extracted and stored in compressed numpy archives (`.npz`) and vector stores.
2. **The Causal Injection Test:** For a given evaluation, a saved concept vector is force-injected into **position 4** (0-indexed, the 5th token) of a neutral 10-token null carrier phrase at target layer indices.
3. **Scoring via Negative Log-Likelihood (NLL):** We measure how effectively the model reads and predicts the passage under the influence of the injected vector using **Negative Log-Likelihood (NLL)**, where a lower NLL indicates that the model finds the text easier to predict.

---

### 4. Key Measurements & Statistical Metrics
To quantify these causal effects rigorously across models and conditions, we rely on several key measurement metrics:
* **Concept-Specific Potency (CSP):** Defined as the difference in prediction loss when injecting other (incorrect) concept vectors versus injecting the matched (correct) concept vector ($\text{NLL}_{\text{other}} - \text{NLL}_{\text{matched}}$).
  * *What it proves:* A positive CSP confirms that the vector holds specific, functional data about that exact concept, providing a causal boost beyond what a generic, random vector would achieve.
* **Relative CSP (`rel_csp`):** Normalizes CSP against the un-injected baseline NLL of the target text ($\text{CSP} / \text{Baseline NLL}$), yielding a scale-free metric that allows fair comparisons across different model tokenizers and architectures.
* **Retrieval Performance (AUC & Top-1):** Evaluates how reliably the correct concept vector outperforms all 47 alternative concept vectors in minimizing prediction loss for a given passage.
* **Cross-Lingual Transfer Ratio (CTR):** Measures cross-lingual potency against monolingual potency ($\text{CSP}_{\text{cross}} / \text{CSP}_{\text{mono}}$) to test whether English and Chinese concept vectors are causally interchangeable within the model's latent space.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Models", len(models))
    c2.metric("Concepts", 48)
    c3.metric("Aggregation / carrier", f"{kind} / {carrier}")

    rows = []
    for k in models:
        s = ws.json(n_stats(k, kind, eff_carrier(ws, k, carrier)))
        if not s:
            continue
        for c in CONDS:
            b = s["conditions"][c]
            r = b["relative_csp"]
            rows.append({"model": k, "family": FAMILY_OF.get(k, "?"), "category": CATEGORY.get(k, s.get("category")),
                         "condition": c, "rel_csp": r["mean"], "lo": r["ci95"][0], "hi": r["ci95"][1],
                         "AUC-ROC": b["auc"]["mean"], "top1": b["retrieval"]["top1"],
                         "CSP>0 (CI excl. 0)": r["ci95"][0] > 0})
    if not rows:
        st.info("No phase3d_stats files for this selection.")
        return
    df = pd.DataFrame(rows)
    st.subheader("Relative CSP by model and condition")
    st.caption("Relative CSP = CSP / un-injected NLL of the same text: scale-free, so comparable across tokenizers. "
               "Error bars: bootstrap 95% CI over the 48 concepts.")
    d = df.copy()
    d["_ep"], d["_em"] = d["hi"] - d["rel_csp"], d["rel_csp"] - d["lo"]
    fig = px.bar(d, x="condition", y="rel_csp", color="model", barmode="group", error_y="_ep", error_y_minus="_em",
                 pattern_shape="category", hover_data=["AUC-ROC", "top1"])
    fig.add_hline(y=0, line_color="grey")
    show(fig, 450)

    null_models = [k for k in models if not df[(df.model == k)]["CSP>0 (CI excl. 0)"].any() and (df.model == k).any()]
    strong = df.groupby("model")["CSP>0 (CI excl. 0)"].sum()
    st.subheader("Automatic reading")
    st.markdown("\n".join([
        f"- **{k}** shows CSP>0 (CI excludes 0) in **{int(n)}/4** conditions." for k, n in strong.items()]))
    if null_models:
        st.warning(f"Models with **no** detectable CSP in any condition: {', '.join(null_models)}. "
                   "Specialization deltas involving these models are differences between a signal and ~0 (or two ~0s) "
                   "and should be read accordingly.")
    st.dataframe(df.drop(columns=["_ep", "_em"], errors="ignore").round(5), use_container_width=True)


# --------------------------------------------------------------------------------------
# Tab: Concept explorer (general users)
# --------------------------------------------------------------------------------------
def tab_concepts(ws, models, kind, carrier, flt):
    st.header("Concept explorer")
    st.caption("Pick a concept and see whether *its own* injected vector helps the model predict *its own* text "
               "better than vectors of other concepts do.")
    p2 = ws.json(PHASE2)
    if not p2:
        st.info(f"{PHASE2} not found.")
        return
    items = {r["id"]: r for r in p2 if r["verification_status"] in ("APPROVED_FIRST_PASS", "APPROVED_AFTER_REVISION")}
    ids = [i for i, r in items.items() if r.get("domain") in flt["domains"]
           and ("human" if r.get("source_type") == "human" else "ai") in flt["sources"]]
    if not ids:
        st.warning("No concepts match the filters.")
        return
    cid = st.selectbox("Concept", ids, format_func=lambda i: f"{i} - {items[i].get('topic') or items[i]['english'][:60]}")
    r = items[cid]
    a, b = st.columns(2)
    a.markdown(f"**English** · *{r.get('domain')}* · {r.get('source_type')}")
    a.write(r["english"])
    b.markdown("**中文**")
    b.write(r["chinese"])

    rows = []
    for k in models:
        pc = ws.json(n_pc(k, kind, eff_carrier(ws, k, carrier)))
        if not pc or cid not in pc["ids"]:
            continue
        j = pc["ids"].index(cid)
        for c in CONDS:
            rows.append({"model": k, "condition": c, "rel_csp": pc[f"csp_rel_{c}"][j], "AUC-ROC": pc[f"auc_{c}"][j],
                         "rank_pct": float(np.mean(np.array(pc[f"csp_rel_{c}"]) <= pc[f"csp_rel_{c}"][j]))})
    if not rows:
        st.info("No per-concept files found.")
        return
    df = pd.DataFrame(rows)
    st.subheader("How much did this concept's own vector help?")
    pv = df.pivot(index="model", columns="condition", values="rel_csp")[CONDS]
    fig = px.imshow(pv, text_auto=".3f", color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect="auto",
                    labels=dict(color="relative CSP"))
    show(fig, 120 + 60 * len(pv))
    st.caption("Blue = own vector beat other concepts' vectors; red = it did not. "
               "AUC-ROC = fraction of the other 47 vectors that gave a *worse* NLL than the matched vector (0.5 = chance).")
    auc = df.pivot(index="model", columns="condition", values="AUC-ROC")[CONDS]
    st.dataframe(auc.style.format("{:.2f}").background_gradient(cmap="Blues", vmin=0.3, vmax=1.0), use_container_width=True)

    st.subheader("English vs Chinese in plain terms")
    mono_cross = df.copy()
    mono_cross["type"] = np.where(mono_cross.condition.isin(["en2en", "zh2zh"]), "same language", "cross-language")
    fig = px.box(mono_cross, x="type", y="rel_csp", color="type", points="all", hover_data=["model", "condition"])
    fig.add_hline(y=0, line_color="grey")
    show(fig, 350)
    st.caption("If cross-language potency ≈ same-language potency, the English and Chinese vectors for a concept "
               "are interchangeable for the model, which is evidence for a shared, language-neutral concept representation.")


# --------------------------------------------------------------------------------------
# Single-model sub-tabs
# --------------------------------------------------------------------------------------
def sm_potency(ws, model, kind, car, flt):
    s = ws.json(n_stats(model, kind, car))
    pc = ws.json(n_pc(model, kind, car))
    if not s:
        st.info(f"Missing {n_stats(model, kind, car)}")
        return
    t = cond_table(s)
    st.subheader("Potency decomposition per condition")
    st.dataframe(t.round(5), use_container_width=True)
    c1, c2 = st.columns(2)
    d = t.copy()
    d["_ep"], d["_em"] = d["CSP 95% CI hi"] - d["CSP mean"], d["CSP mean"] - d["CSP 95% CI lo"]
    fig = px.bar(d, x="condition", y="CSP mean", error_y="_ep", error_y_minus="_em", color="condition",
                 title=f"CSP ({KIND_LABEL[kind]})")
    fig.add_hline(y=0, line_color="grey")
    with c1:
        show(fig, 380)
    comp = t.melt(id_vars="condition", value_vars=["total potency", "generic potency", "CSP mean"],
                  var_name="component", value_name="value")
    with c2:
        show(px.bar(comp, x="condition", y="value", color="component", barmode="group",
                    title="Total = generic + concept-specific"), 380)
    st.markdown("**Cross-lingual transfer ratio (CTR)** = CSP(cross) / CSP(mono) on the same target text; "
                "1 means EN and ZH vectors are causally interchangeable.")
    ctr = pd.DataFrame([{"target text": k, "CTR": v["CTR"], "lo": v["CTR_ci95"][0], "hi": v["CTR_ci95"][1],
                         "mono−cross gap": v["gap_mono_minus_cross"]["mean"],
                         "gap sign-flip p": v["gap_mono_minus_cross"]["signflip_p"]}
                        for k, v in s["cross_lingual_transfer"].items()])
    st.dataframe(ctr.round(4), use_container_width=True)
    forest(ctr, "CTR", "lo", "hi", "target text", title="CTR with bootstrap CI")
    st.markdown("**Multiplicity-corrected primary tests**")
    m = s["multiplicity"]
    st.dataframe(pd.DataFrame({"test": m["labels"], "p_raw": m["p_raw"], "p_holm": m["p_holm"], "q_BH": m["q_bh"]}),
                 use_container_width=True)
    if pc:
        st.markdown("**Per-concept relative CSP (filtered)**")
        df = apply_filter(pc_df(pc), flt)
        long = df.melt(id_vars=["id", "domain", "source"], value_vars=[f"csp_rel_{c}" for c in CONDS],
                       var_name="condition", value_name="rel_csp")
        long["condition"] = long["condition"].str.replace("csp_rel_", "")
        show(px.box(long, x="condition", y="rel_csp", color="domain", points="all", hover_data=["id", "source"]), 420)


def sm_heatmap(ws, model, kind, car, flt):
    Z, meta = ws.npz(n_npz(model, car)), ws.json(n_meta(model, car))
    if Z is None or meta is None:
        st.info("Interference npz / meta file missing for this model/carrier.")
        return
    ids = [str(i) for i in Z["ids"]]
    agg = NPZ_AGG[kind]
    c1, c2, c3 = st.columns(3)
    cond = c1.selectbox("Condition", CONDS, format_func=COND_LABEL.get, key="hm_cond")
    norm = c2.selectbox("Transform", ["Raw NLL", "Column-centered (NLL − column mean): concept specificity",
                                      "Gain vs no injection (none − NLL)"], key="hm_norm")
    order_by = c3.selectbox("Order by", ["domain, id", "id"], key="hm_ord")
    M = Z[f"nll{agg}_{cond}"].astype(float)       # rows: injected vector j, cols: target text i
    tgt = cond[-2:]
    keep = [k for k, i in enumerate(ids) if meta["domain"][i] in flt["domains"] and meta["source"][i] in flt["sources"]]
    if len(keep) < 3:
        st.warning("Need at least 3 concepts after filtering.")
        return
    keep.sort(key=(lambda k: (meta["domain"][ids[k]], ids[k])) if order_by.startswith("domain") else (lambda k: ids[k]))
    S = M[np.ix_(keep, keep)]
    sub = [ids[k] for k in keep]
    if norm.startswith("Column"):
        S, cs, zmid = S - S.mean(0, keepdims=True), "RdBu_r", 0
    elif norm.startswith("Gain"):
        none = Z[f"none{agg}_{tgt}"].astype(float)[keep]
        S, cs, zmid = none[None, :] - S, "RdBu", 0
    else:
        cs, zmid = "Viridis_r", None
    labels = [f"{i} ({meta['domain'][i][:4]})" for i in sub]
    fig = go.Figure(go.Heatmap(z=S, x=labels, y=labels, colorscale=cs, zmid=zmid,
                               colorbar=dict(title=norm.split(":")[0]),
                               hovertemplate="vector %{y}<br>text %{x}<br>%{z:.4f}<extra></extra>"))
    fig.update_layout(xaxis_title=f"target text ({tgt.upper()})", yaxis_title=f"injected vector ({cond[:2].upper()})",
                      yaxis_autorange="reversed", title=f"{cond} · {KIND_LABEL[kind]} · {len(sub)}×{len(sub)}")
    show(fig, 720)
    st.caption("A dark diagonal in the column-centered view = each text is best served by its own concept vector. "
               "Domain blocks along the diagonal = vectors carry domain information too (see 'within-domain CSP').")

    Ms = M[np.ix_(keep, keep)]
    n = len(keep)
    diag = np.diag(Ms)
    off = ~np.eye(n, dtype=bool)
    other = np.array([Ms[off[:, i], i].mean() for i in range(n)])
    rank = (Ms < diag[None, :]).sum(0)
    auc = np.array([(Ms[off[:, i], i] > diag[i]).mean() for i in range(n)])
    st.markdown(f"**Live recomputation on the filtered {n}-concept sub-matrix** (the 'other' set shrinks with the filter)")
    a, b, c, d = st.columns(4)
    a.metric("Mean CSP", f"{(other - diag).mean():+.4f}")
    b.metric("Mean AUC-ROC", f"{auc.mean():.3f}")
    c.metric("Top-1", f"{(rank == 0).mean():.2f}", help=f"chance = {1 / n:.3f}")
    d.metric("MRR", f"{(1 / (rank + 1)).mean():.3f}")
    nz = Z.get(f"noise{agg}_{cond}")
    if nz is not None:
        st.caption(f"Noise reference (8 norm-matched Gaussian vectors): mean NLL {nz[:, keep].mean():.4f} vs matched "
                   f"{diag.mean():.4f}, other {other.mean():.4f}.")
    live = pd.DataFrame({"id": sub, "domain": [meta["domain"][i] for i in sub], "source": [meta["source"][i] for i in sub],
                         "matched NLL": diag, "other NLL": other, "CSP": other - diag, "rank (0=best)": rank, "AUC-ROC": auc})
    st.dataframe(live.round(4), use_container_width=True)


def sm_agg(ws, model, car):
    st.subheader("Does the conclusion depend on how NLL is aggregated?")
    st.caption("mean = nats/token (length-diluted), sum = nats/passage, head = first 16 tokens, where a single injected "
               "vector acts most. Unit-free metrics (AUC-ROC, top-1) are directly comparable across aggregations.")
    rows = []
    for kind in KIND_LABEL:
        s = ws.json(n_stats(model, kind, car))
        if not s:
            continue
        t = cond_table(s)
        t["aggregation"] = kind
        rows.append(t)
    if not rows:
        st.info("No stats files.")
        return
    df = pd.concat(rows)
    c1, c2 = st.columns(2)
    with c1:
        show(px.bar(df, x="condition", y="AUC-ROC", color="aggregation", barmode="group", title="AUC-ROC (0.5 = chance)"), 360)
    with c2:
        show(px.bar(df, x="condition", y="rel. CSP mean", color="aggregation", barmode="group",
                    title="Relative CSP"), 360)
    st.dataframe(df[["aggregation", "condition", "CSP mean", "CSP 95% CI lo", "CSP 95% CI hi", "rel. CSP mean",
                     "AUC-ROC", "top-1", "MRR", "sign-flip p"]].round(5), use_container_width=True)
    Z = ws.npz(n_npz(model, car))
    if Z is not None:
        st.markdown("**Same quantity recomputed directly from the .npz (all 48 concepts)**")
        out = []
        for kind, agg in NPZ_AGG.items():
            for c in CONDS:
                k = f"nll{agg}_{c}"
                if k in Z:
                    M = Z[k].astype(float)
                    n = len(M)
                    other = (M.sum(0) - np.diag(M)) / (n - 1)
                    out.append({"aggregation": kind, "condition": c, "CSP (npz)": float((other - np.diag(M)).mean())})
        st.dataframe(pd.DataFrame(out).pivot(index="condition", columns="aggregation", values="CSP (npz)").round(5),
                     use_container_width=True)


def decay_frame(rep, probe_only=True):
    rows = []
    for e in rep:
        for series in ("en", "zh", "noise_control", "mismatch_control"):
            if series not in e:
                continue
            for L, v in e[series]["decay_curve"].items():
                rows.append({"id": e["id"], "domain": e["domain"], "series": series, "layer": int(L), "cos": v})
    return pd.DataFrame(rows)


def sm_layers(ws, model, flt):
    rep = ws.json(f"phase3b_injection_report_{model}.json")
    pr = ws.json(f"phase3_report_{model}.json")
    sig = ws.json(f"phase3b_significance_report_{model}.json")
    meta = ws.json(n_meta(model, "primary"))
    inj_layer = meta["layer_idx"] if meta else (pr or {}).get("primary_layer")
    if rep:
        st.subheader("Injection decay: cosine(injected vector, state at deeper layer)")
        df = decay_frame(rep)
        df = df[df.domain.isin(flt["domains"])]
        by = st.radio("Group by", ["series", "domain × series"], horizontal=True)
        show_mm = st.checkbox("Show mismatch control", value=True)
        if not show_mm:
            df = df[df.series != "mismatch_control"]
        fig = go.Figure()
        if by == "series":
            groups = [(s, g) for s, g in df.groupby("series")]
        else:
            groups = [(f"{d}·{s}", g) for (d, s), g in df.groupby(["domain", "series"])]
        for name, g in groups:
            a = g.groupby("layer")["cos"].agg(["mean", "std"]).reset_index()
            fig.add_trace(go.Scatter(x=a.layer, y=a["mean"], mode="lines+markers", name=name,
                                     error_y=dict(type="data", array=a["std"].fillna(0), visible=True)))
        if inj_layer is not None:
            fig.add_vline(x=inj_layer, line_dash="dot", annotation_text="injection layer")
        fig.update_layout(xaxis_title="layer", yaxis_title="cosine to injected vector", title="Mean ± SD over concepts")
        show(fig, 460)
        mm = [e for e in rep if "mismatch_control" in e]
        if mm:
            diff = max(abs(e["en"]["decay_curve"][k] - next(x for x in rep if x["id"] == _prev(rep, e["id"]))["en"]["decay_curve"][k])
                       for e in mm[:1] for k in e["en"]["decay_curve"])
            _ = diff
            ids_ = [e["id"] for e in rep]
            by_id = {e["id"]: e for e in rep}
            mx = 0.0
            for e in rep:
                other = by_id[e["mismatch_control"]["vector_of"]]
                for k, v in e["mismatch_control"]["decay_curve"].items():
                    mx = max(mx, abs(v - other["en"]["decay_curve"][k]))
            if mx < 1e-9:
                st.warning("**Mismatch control is an identity here.** Each concept's mismatch run injects another concept's "
                           "EN vector, and that run is already that concept's own EN run, so the paired EN-vs-mismatch "
                           "test compares the same 48 numbers in permuted order (mean difference ≡ 0). Use it only as an "
                           "unpaired in-distribution reference, not as a significance test.")
        st.markdown("**Generated continuations from the null carrier (degeneracy check)**")
        gen = []
        for e in rep:
            if e["domain"] not in flt["domains"]:
                continue
            for s in ("en", "zh", "noise_control", "mismatch_control"):
                if s in e:
                    t = e[s]["generated_text"]
                    gen.append({"id": e["id"], "series": s, "unique-char ratio": len(set(t)) / max(len(t), 1),
                                "len": len(t), "text[:80]": t[:80]})
        st.dataframe(pd.DataFrame(gen).round(3), use_container_width=True)
    else:
        st.info(f"phase3b_injection_report_{model}.json not found.")

    if pr:
        st.subheader("Per-layer EN↔ZH CKA (extraction pipeline)")
        ck = pd.DataFrame([{"layer": int(L), **v} for L, v in pr["cka_by_layer"].items()]).sort_values("layer")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ck.layer, y=ck.raw, name="raw (biased at n≪d)", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=ck.layer, y=ck.debiased, name="debiased (top-2 SVs removed)", mode="lines+markers"))
        if inj_layer is not None:
            fig.add_vline(x=inj_layer, line_dash="dot")
        fig.update_layout(xaxis_title="layer", yaxis_title="linear CKA", yaxis_range=[0, 1.02])
        show(fig, 380)
        c1, c2 = st.columns(2)
        c1.markdown("**Participation ratio (token-level)**")
        c1.json({"raw": pr["pr1_raw"], "equalized+debiased": pr["pr2_equalized_debiased"]})
        c2.markdown("**Token counts**")
        c2.json({"EN": pr["n_en_tokens_raw"], "ZH": pr["n_zh_tokens_raw"], "monitored layers": pr["monitored_layers"]})
    if sig:
        st.subheader("Layer-wise significance report (EN vs ZH retention, vs noise)")
        sd = pd.DataFrame(sig).drop(columns=["domains"], errors="ignore")
        st.dataframe(sd.round(5), use_container_width=True)
        dom_rows = [{"layer": e["layer"], "domain": d, **v} for e in sig for d, v in e["domains"].items()]
        dd = pd.DataFrame(dom_rows)
        dd = dd[dd.domain.isin(flt["domains"])]
        show(px.line(dd.melt(id_vars=["layer", "domain", "n"], value_vars=["mean_en", "mean_zh"], var_name="lang"),
                     x="layer", y="value", color="domain", line_dash="lang", markers=True,
                     title="Mean retention by domain and language"), 400)


def _prev(rep, i):  # helper kept trivial: returns the id itself
    return i


def sm_geometry(ws, model, flt):
    G = ws.json(f"phase3e_geometry_{model}.json")
    W = ws.json(f"phase3e_within_domain_{model}.json")
    V = ws.npz(f"phase3_concept_vectors_{model}.npz")
    if not G:
        st.info("phase3e_geometry file missing.")
        return
    c = st.columns(5)
    c[0].metric("Unbiased CKA", f"{G['unbiased_cka']['value']:.3f}", help=f"null mean {G['unbiased_cka']['null_mean']:.3f}")
    c[1].metric("Top-1 EN→ZH", f"{G['retrieval']['en2zh_top1']:.2f}", help=f"chance {G['retrieval']['chance_top1']:.3f}")
    c[2].metric("MRR", f"{G['retrieval']['en2zh_mrr']:.2f}")
    c[3].metric("RSA ρ (EN vs ZH)", f"{G['rsa_en_vs_zh']['spearman_rho']:.3f}")
    c[4].metric("Margin", f"{G['centered_cosine']['mean_margin']:.3f}")
    if G["retrieval"]["en2zh_top1"] >= 0.999:
        st.info("Retrieval is at ceiling (top-1 = 1.0 at n=48): it cannot separate models. Use the alignment margin, RSA "
                "and CKA instead.")
    pcg = pd.DataFrame({"id": G["per_concept"]["ids"], "domain": G["per_concept"]["domain"],
                        "source": G["per_concept"]["source"], "margin": G["per_concept"]["alignment_margin"],
                        "same_cos": G["per_concept"]["same_cos"], "norm_en": G["per_concept"]["norm_en"],
                        "norm_zh": G["per_concept"]["norm_zh"]})
    if W:
        pcg = pcg.merge(pd.DataFrame({"id": W["per_concept"]["ids"], "within-domain margin": W["per_concept"]["within_domain_margin"]}), on="id")
    pcg = apply_filter(pcg, flt)
    ycols = [c for c in ("margin", "within-domain margin") if c in pcg]
    long = pcg.melt(id_vars=["id", "domain", "source"], value_vars=ycols, var_name="metric")
    c1, c2 = st.columns(2)
    with c1:
        show(px.box(long, x="domain", y="value", color="metric", points="all", hover_data=["id", "source"],
                    title="Alignment margin: all concepts vs same-domain only"), 420)
    with c2:
        dr = pd.DataFrame([{"language": l, "rho": v["rho"]} for l, v in G["domain_structure_rsa"].items()])
        show(px.bar(dr, x="language", y="rho", title="RSA vs domain-membership model"), 420)
    if W:
        st.caption(f"Within-domain: mean same-cos {W['mean_same_cos']:.3f} vs permutation null {W['mean_same_cos_null']:.3f}; "
                   f"top-1 {W['top1_within_domain']['en2zh']:.2f} (chance {W['top1_within_domain']['chance']:.3f}); "
                   f"margin {W['within_domain_margin_mean']:+.3f} CI {np.round(W['within_domain_margin_ci95'], 3).tolist()}")
    if V is not None:
        st.subheader("Concept-vector map (per-language mean-centered, joint PCA)")
        ids = [str(i) for i in V["ids"]]
        meta = ws.json(n_meta(model, "primary"))
        en, zh = V["en"].astype(float), V["zh"].astype(float)
        en_c, zh_c = en - en.mean(0), zh - zh.mean(0)
        X = np.vstack([en_c, zh_c])
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        P = U[:, :3] * S[:3]
        n = len(ids)
        dom = [meta["domain"][i] if meta else "?" for i in ids]
        src = [meta["source"][i] if meta else "?" for i in ids]
        d = pd.DataFrame({"id": ids * 2, "lang": ["EN"] * n + ["ZH"] * n, "domain": dom * 2, "source": src * 2,
                          "PC1": P[:, 0], "PC2": P[:, 1], "PC3": P[:, 2]})
        d = d[d.domain.isin(flt["domains"]) & d.source.isin(flt["sources"])]
        fig = px.scatter(d, x="PC1", y="PC2", color="domain", symbol="lang", hover_data=["id", "source"])
        segx, segy = [], []
        for i in d.id.unique():
            g = d[d.id == i]
            if len(g) == 2:
                segx += [g.PC1.iloc[0], g.PC1.iloc[1], None]
                segy += [g.PC2.iloc[0], g.PC2.iloc[1], None]
        fig.add_trace(go.Scatter(x=segx, y=segy, mode="lines", line=dict(color="rgba(120,120,120,.35)", width=1),
                                 hoverinfo="skip", showlegend=False))
        ev = (S ** 2) / (S ** 2).sum()
        fig.update_layout(title=f"PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}; grey line joins the EN/ZH vectors of one concept")
        show(fig, 560)
        a = en_c / np.linalg.norm(en_c, axis=1, keepdims=True)
        b = zh_c / np.linalg.norm(zh_c, axis=1, keepdims=True)
        S_ = a @ b.T
        order = np.argsort(dom, kind="stable")
        lab = [f"{ids[i]}" for i in order]
        hm = go.Figure(go.Heatmap(z=S_[np.ix_(order, order)], x=lab, y=lab, colorscale="RdBu", zmid=0,
                                  colorbar=dict(title="cos")))
        hm.update_layout(xaxis_title="ZH concept", yaxis_title="EN concept", yaxis_autorange="reversed",
                         title="Centered cosine, EN × ZH (diagonal = translation pair), ordered by domain")
        show(hm, 640)
        raw = en @ zh.T / (np.linalg.norm(en, axis=1)[:, None] * np.linalg.norm(zh, axis=1)[None, :])
        st.caption(f"Raw (uncentered) pairwise cosine is dominated by a shared direction: mean off-diagonal "
                   f"{(raw.sum() - np.trace(raw)) / (n * n - n):.3f} vs same-concept {np.trace(raw) / n:.3f}; "
                   "that is why all alignment metrics here are computed after per-language centering.")


def sm_controls(ws, model, kind, car, flt):
    pc = ws.json(n_pc(model, kind, car))
    ppl = ws.json(f"phase3_perplexity_{model}.json")
    dc = ws.json(f"density_controls_{model}{'' if kind == 'mean' else KIND_SFX[kind]}{CARRIER_SFX[car]}.json") \
        or ws.json(f"density_controls_{model}.json")
    st.subheader("Density / perplexity confound")
    st.caption("Does the EN−ZH difference in same-language CSP just reflect token-count or perplexity differences?")
    if dc:
        c = st.columns(3)
        c[0].metric("Mean EN−ZH CSP gap", f"{dc['mean_gap']:.4f}")
        c[1].metric("ρ vs log token ratio", f"{dc['spearman_vs_log_token_ratio']['rho']:.2f}",
                    f"p={fmt_p(dc['spearman_vs_log_token_ratio']['p'])}")
        c[2].metric("ρ vs log ppl ratio", f"{dc['spearman_vs_log_ppl_ratio']['rho']:.2f}",
                    f"p={fmt_p(dc['spearman_vs_log_ppl_ratio']['p'])}")
        st.dataframe(pd.DataFrame(dc["ols_hc3"]).T.round(4), use_container_width=True)
    if pc and ppl:
        P = pd.DataFrame(ppl).set_index("id")
        df = apply_filter(pc_df(pc), flt).set_index("id").join(P[["en_tokens", "zh_tokens", "en_ppl", "zh_ppl",
                                                                  "en_total_nats", "zh_total_nats"]])
        df["gap"] = df["csp_en2en"] - df["csp_zh2zh"]
        df["log_token_ratio"] = np.log(df.en_tokens / df.zh_tokens)
        df["log_ppl_ratio"] = np.log(df.en_ppl / df.zh_ppl)
        c1, c2 = st.columns(2)
        with c1:
            show(px.scatter(df.reset_index(), x="log_token_ratio", y="gap", color="domain", hover_name="id",
                            trendline="ols", title="EN−ZH CSP gap vs log(EN/ZH tokens)"), 380)
        with c2:
            show(px.scatter(df.reset_index(), x="log_ppl_ratio", y="gap", color="domain", hover_name="id",
                            trendline="ols", title="EN−ZH CSP gap vs log(EN/ZH perplexity)"), 380)
    st.subheader("Human vs AI-generated concepts (robustness)")
    if pc:
        df = apply_filter(pc_df(pc), flt)
        long = df.melt(id_vars=["id", "domain", "source"], value_vars=[f"csp_rel_{c}" for c in CONDS],
                       var_name="condition", value_name="rel_csp")
        long["condition"] = long.condition.str.replace("csp_rel_", "")
        show(px.box(long, x="condition", y="rel_csp", color="source", points="all", hover_data=["id"]), 400)
        st.caption("n_human = 8, so this is descriptive only. A large human/AI gap would mean CSP partly reflects "
                   "properties of synthetic text rather than concepts per se.")
    st.subheader("Shared-name (lexical) overlap covariate")
    fam = FAMILY_OF.get(model)
    lo = ws.json(f"lexical_overlap_{fam}.json") if fam else None
    if lo and model in lo["results"]:
        rows = []
        for c, v in lo["results"][model].items():
            rows.append({"cross cond": c, "ρ(overlap, CSP)": v["spearman_overlap_vs_csp"]["rho"],
                         "p": v["spearman_overlap_vs_csp"]["p"], "low-overlap CSP": v["csp_low_overlap_half"]["mean"],
                         "CI lo": v["csp_low_overlap_half"]["ci95"][0], "CI hi": v["csp_low_overlap_half"]["ci95"][1],
                         "p(>0)": v["csp_low_overlap_half"]["signflip_p_gt0"],
                         "high-overlap CSP": v["csp_high_overlap_half"]})
        st.dataframe(pd.DataFrame(rows).round(5), use_container_width=True)
        ov = pd.Series(lo["per_concept_overlap"])
        st.caption(f"{(ov > 0).sum()} of {len(ov)} concepts have any Latin-token overlap between EN and ZH passages, "
                   "so cross-lingual CSP cannot be explained by copied names.")
    st.subheader("Sanity checks & carrier selection")
    sp = ws.json(f"phase3_sanity_{model}_pinned.json")
    sx = ws.json(f"phase3_sanity_{model}.json")
    if sp:
        v = sp["verdict"]
        (st.success if v["status"] == "READY" else st.error)(f"Pinned-carrier verdict: {v['status']}")
        for w in v.get("warn", []):
            st.warning(w)
        for f in v.get("fail", []):
            st.error(f)
    if sx and "carrier" in sx:
        rows = [{"token": c["token"], "id": c["id"], "norm×": c["norm_ratio"], "sink cos": c["sink_cos"],
                 "entropy": c["entropy"], "EN NLL/plain": c["en"]["ratio_to_plain"], "ZH NLL/plain": c["zh"]["ratio_to_plain"],
                 "flags": ",".join(c["flags"]) or "PASS"} for c in sx["carrier"]["candidates"]]
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True)
        for lang, s in sx.get("position_sweep", {}).items():
            st.caption(f"{lang}: attention-sink positions {np.round(s['systematic_sink_rate_by_position'], 2).tolist()} → "
                       f"exclude first {s['recommended_exclude_leading_n']} tokens")
    elif not sx:
        st.caption("No full sanity report found.")


def tab_single(ws, model, kind, carrier, flt):
    car = eff_carrier(ws, model, carrier)
    st.header(f"{model} · {CATEGORY.get(model, '')} · {KIND_LABEL[kind]} · carrier {car}")
    t = st.tabs(["Causal potency", "Interference heatmaps", "NLL aggregations", "Layer dynamics",
                 "Geometry", "Controls & sanity"])
    with t[0]:
        sm_potency(ws, model, kind, car, flt)
    with t[1]:
        sm_heatmap(ws, model, kind, car, flt)
    with t[2]:
        sm_agg(ws, model, car)
    with t[3]:
        sm_layers(ws, model, flt)
    with t[4]:
        sm_geometry(ws, model, flt)
    with t[5]:
        sm_controls(ws, model, kind, car, flt)


# --------------------------------------------------------------------------------------
# Tab: Cross-family
# --------------------------------------------------------------------------------------
def fam_name(prefix_kind):
    return f"phase4_family_contrasts{KIND_SFX[prefix_kind]}.json"


def tab_cross(ws, models, kind, carrier, flt):
    st.header("Lineage-controlled specialization contrasts")
    st.caption("Each family = (general base, translation-specialized continuation). Architecture, tokenizer, layer and carrier "
               "are shared inside a family, so contrasts are paired within concept and within family: Δ = specialized − general.")
    F = ws.json(fam_name(kind))
    if not F:
        st.info(f"{fam_name(kind)} not found.")
    else:
        metrics = list(F["per_metric"])
        default = metrics.index("csp_en2zh") if "csp_en2zh" in metrics else 0
        metric = st.selectbox("Metric", metrics, index=default)
        blk = F["per_metric"][metric]
        rows = [{"family": f, "Δ": v["mean_delta"], "lo": v["ci95"][0], "hi": v["ci95"][1], "dz": v["cohens_dz"],
                 "p (sign-flip)": v["signflip_p"]} for f, v in blk["per_family"].items()]
        d = pd.DataFrame(rows)
        forest(d, "Δ", "lo", "hi", "family", color="family", title=f"Δ({metric}) specialized − general, 95% CI", height=340)
        c = st.columns(4)
        c[0].metric("Pooled Δ", f"{blk['pooled_mean_delta']:.4g}")
        c[1].metric("Pooled p (RMS-std, primary)", fmt_p(blk["pooled_stratified_signflip_p_std"]))
        c[2].metric("Random-effects p", fmt_p(blk["random_effects"]["p"]), f"I²={blk['random_effects']['I2']:.2f}")
        sc = blk["sign_consistency"]
        c[3].metric("Sign consistency", f"{max(sc['n_positive'], sc['n_negative'])}/3", "all same" if sc["all_same"] else "mixed")
        st.dataframe(d.round(5), use_container_width=True)
        if "csp_" in metric and not metric.startswith("csp_rel"):
            st.warning("Raw-unit CSP is not comparable across tokenizers (e.g. Gemma ≈ 0.1 nats/token vs Llama ≈ 1e-4). "
                       "Prefer **csp_rel_*** or the RMS-standardized pooled p for cross-family statements.")
        ndf = pd.DataFrame(F["multiplicity"])
        st.subheader("Primary-family multiplicity correction")
        st.dataframe(ndf.round(5), use_container_width=True)

        st.subheader("Model-level scalars")
        ms = pd.DataFrame(F["model_scalars"]).T
        st.dataframe(ms.round(4), use_container_width=True)
        sel = st.multiselect("Plot scalars", [c for c in ms.columns if c != "category"],
                             default=["unbiased_cka", "rsa_en_zh", "alignment_margin"])
        if sel:
            long = ms.reset_index().rename(columns={"index": "model"}).melt(id_vars=["model", "category"], value_vars=sel)
            show(px.bar(long, x="model", y="value", color="category", facet_col="variable", barmode="group"), 380)

        st.subheader("Interactions (difference-in-differences)")
        rows = []
        for name, v in F["interactions"].items():
            if "per_family" in v and "pooled_stratified_signflip_p_std" in v:
                for f, b in v["per_family"].items():
                    rows.append({"interaction": name, "family": f, "mean": b["mean_delta"], "lo": b["ci95"][0],
                                 "hi": b["ci95"][1], "p": b["signflip_p"], "pooled p (std)": v["pooled_stratified_signflip_p_std"]})
        if rows:
            ir = pd.DataFrame(rows)
            forest(ir, "mean", "lo", "hi", "interaction", color="family", height=420)
            st.dataframe(ir.round(5), use_container_width=True)
        dom_rows = [{"interaction": n, "family": f, **b} for n, v in F["interactions"].items()
                    if n.startswith("domain_x_spec") for f, b in v.items()]
        if dom_rows:
            st.markdown("**Specialization × domain (abstract vs syntax_baseline)**")
            st.dataframe(pd.DataFrame(dom_rows).round(5), use_container_width=True)

        st.subheader("Change in cross-lingual transfer ratio and geometry–causality coupling")
        ctr = [{"family": f, "text": t, "ΔCTR": v["delta_CTR"], "lo": v["ci95"][0], "hi": v["ci95"][1]}
               for f, d_ in F["ctr_deltas"].items() for t, v in d_.items() if isinstance(v, dict)]
        cd = pd.DataFrame(ctr)
        forest(cd, "ΔCTR", "lo", "hi", "family", color="text", title="Δ CTR (very wide when mono-CSP ≈ 0)", height=340)
        cp = [{"model": m, "cond": c, "ρ(alignment margin, CSP)": v["rho"], "p": v["p"]}
              for m, d_ in F["coupling_geometry_vs_causal"].items() for c, v in d_.items()]
        cpd = pd.DataFrame(cp)
        show(px.bar(cpd, x="model", y="ρ(alignment margin, CSP)", color="cond", barmode="group",
                    title="Does geometric alignment predict causal transfer concept by concept?"), 380)

    G = ws.json("phase4_geometry_deltas.json")
    if G:
        st.subheader("Geometry deltas (specialized − general)")
        rows = [{"family": f, "metric": m, "Δ": v["delta"], "lo": v["subsample_ci95"][0], "hi": v["subsample_ci95"][1],
                 "frac subsamples >0": v["frac_subsamples_positive"]} for f, d_ in G.items() for m, v in d_.items()]
        gd = pd.DataFrame(rows)
        forest(gd, "Δ", "lo", "hi", "metric", color="family", height=400,
               title="Subsampling (80%, no replacement) intervals, not formal CIs")
        st.caption("Top-1 retrieval is at ceiling for all models, so its Δ is exactly 0 and uninformative.")
        st.dataframe(gd.round(5), use_container_width=True)

    st.subheader("Lexical-overlap covariate across families")
    rows = []
    for fam in FAMILIES:
        lo = ws.json(f"lexical_overlap_{fam}.json")
        if not lo:
            continue
        for m, d_ in lo["results"].items():
            for c, v in d_.items():
                rows.append({"family": fam, "model": m, "cond": c, "ρ(overlap,CSP)": v["spearman_overlap_vs_csp"]["rho"],
                             "ρ p": v["spearman_overlap_vs_csp"]["p"], "low-overlap CSP": v["csp_low_overlap_half"]["mean"],
                             "lo": v["csp_low_overlap_half"]["ci95"][0], "hi": v["csp_low_overlap_half"]["ci95"][1]})
    if rows:
        lo_df = pd.DataFrame(rows)
        forest(lo_df, "low-overlap CSP", "lo", "hi", "model", color="cond", height=420,
               title="Cross-lingual CSP on the low-overlap half (raw units; read per model)")
        st.dataframe(lo_df.round(5), use_container_width=True)

    st.subheader("Model-level comparison from per-model stats")
    rows = []
    for k in models:
        s = ws.json(n_stats(k, kind, "primary"))
        if s:
            for t, v in s["cross_lingual_transfer"].items():
                rows.append({"model": k, "category": CATEGORY.get(k), "family": FAMILY_OF.get(k), "text": t,
                             "CTR": v["CTR"], "lo": v["CTR_ci95"][0], "hi": v["CTR_ci95"][1]})
    if rows:
        cd = pd.DataFrame(rows)
        forest(cd, "CTR", "lo", "hi", "model", color="text", height=420, title="CTR by model")


# --------------------------------------------------------------------------------------
# Tab: Carrier robustness
# --------------------------------------------------------------------------------------
def tab_carrier(ws, models, kind, flt):
    st.header("Carrier robustness")
    man = ws.path("carrier_manifest.jsonl")
    if man:
        st.subheader("Carrier manifest (run audit trail)")
        manifest_df = pd.DataFrame(load_jsonl(man))
        
        # Target the exact column name 'token' from the manifest
        if "token" in manifest_df.columns:
            manifest_df["token"] = manifest_df.apply(
                lambda row: "\\n" if row.get("model_key") in ("qwen3_8b", "lmt_60_8b") and (not row.get("token") or str(row.get("token")).strip() == "")
                else row.get("token"), axis=1
            )
        st.dataframe(manifest_df, use_container_width=True)

    qwen = [k for k in FAMILIES["qwen3"] if k in models]
    have = [k for k in qwen if ws.has(n_pc(k, kind, "dot")) and ws.has(n_pc(k, kind, "primary"))]
    
    # Helper to map primary/dot carrier names to custom display labels
    def get_display_carrier(model_name: str, car: str) -> str:
        if model_name in ("qwen3_8b", "lmt_60_8b") and car == "primary":
            return "\\n"
        return car

    if not have:
        st.info("No secondary-carrier ('_cdot') per-concept files found. Only the Qwen family supports this check.")
    else:
        rows = []
        for k in have:
            for car in ("primary", "dot"):
                s = ws.json(n_stats(k, kind, car))
                if s:
                    display_car = get_display_carrier(k, car)
                    for c in CONDS:
                        r = s["conditions"][c]["relative_csp"]
                        rows.append({"model": k, "carrier": display_car, "condition": c, "rel_csp": r["mean"],
                                     "lo": r["ci95"][0], "hi": r["ci95"][1]})
                                     
        d = pd.DataFrame(rows)
        d["_ep"], d["_em"] = d.hi - d.rel_csp, d.rel_csp - d.lo
        show(px.bar(d, x="condition", y="rel_csp", color="carrier", facet_col="model", barmode="group",
                    error_y="_ep", error_y_minus="_em", title="Relative CSP, '\\n' (primary) vs '.' carrier"), 400)
        st.caption("Only within-carrier contrasts are meaningful; absolute NLL is not compared across carriers.")
        
        k = st.selectbox("Per-concept agreement for", have)
        cond = st.selectbox("Condition", CONDS, key="car_cond")
        
        a = pc_df(ws.json(n_pc(k, kind, "primary")))
        b = pc_df(ws.json(n_pc(k, kind, "dot")))
        m = a[["id", "domain", "source", f"csp_rel_{cond}"]].merge(b[["id", f"csp_rel_{cond}"]], on="id",
                                                                    suffixes=("_primary", "_dot"))
        m = apply_filter(m, flt)
        rho = m[f"csp_rel_{cond}_primary"].corr(m[f"csp_rel_{cond}_dot"], method="spearman")
        
        primary_label = get_display_carrier(k, "primary")
        show(px.scatter(m, x=f"csp_rel_{cond}_primary", y=f"csp_rel_{cond}_dot", color="domain", hover_name="id",
                        title=f"Per-concept agreement ({primary_label} vs dot) · Spearman ρ = {rho:.2f}"), 450)
                        
    for tag in ("", "_sum", "_head"):
        p = ws.json(f"phase4_carrier_robustness_dot{tag}.json")
        if p:
            with st.expander(f"phase4_carrier_robustness_dot{tag}.json"):
                st.json(p.get("replication_verdict", {}))
                sd = pd.DataFrame({lab: {m: v["mean"] for m, v in d_.items()} for lab, d_ in p["specialization_delta"].items()})
                st.dataframe(sd.round(5), use_container_width=True)

# --------------------------------------------------------------------------------------
# Tab: Data tables
# --------------------------------------------------------------------------------------
def tab_tables(ws, models, model, kind, carrier, flt):
    st.header("Raw data browser")
    car = eff_carrier(ws, model, carrier)
    what = st.selectbox("Table", ["Per-concept metrics", "Concept metadata (domain/source)", "Interference matrix (.npz array)",
                                  "Injection decay (long format)", "Perplexity / density", "Phase-2 concepts",
                                  "Sanity: carrier candidates", "Workspace file index"])
    if what == "Per-concept metrics":
        pc = ws.json(n_pc(model, kind, car))
        if pc:
            df = apply_filter(pc_df(pc), flt)
            cols = st.multiselect("Columns", [c for c in df.columns], default=list(df.columns)[:12])
            st.dataframe(df[cols], use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), f"{model}_per_concept.csv")
    elif what.startswith("Concept metadata"):
        meta = ws.json(n_meta(model, car))
        if meta:
            st.json({k: v for k, v in meta.items() if k not in ("ids", "domain", "source")})
            st.dataframe(apply_filter(meta_df(meta), flt), use_container_width=True)
    elif what.startswith("Interference"):
        Z = ws.npz(n_npz(model, car))
        if Z:
            key = st.selectbox("Array", sorted(Z.keys()))
            A = Z[key]
            st.caption(f"shape {A.shape}, dtype {A.dtype}")
            ids = [str(i) for i in Z["ids"]]
            if A.ndim == 2 and A.shape[1] == len(ids):
                idx = [f"row {i}" for i in (ids if A.shape[0] == len(ids) else range(A.shape[0]))]
                st.dataframe(pd.DataFrame(A, index=idx, columns=ids).round(4), use_container_width=True)
            else:
                st.dataframe(pd.DataFrame(A.reshape(len(A), -1) if A.ndim > 1 else A), use_container_width=True)
    elif what.startswith("Injection"):
        rep = ws.json(f"phase3b_injection_report_{model}.json")
        if rep:
            df = decay_frame(rep)
            df = df[df.domain.isin(flt["domains"])]
            layers = sorted(df.layer.unique())
            sel = st.multiselect("Layers", layers, default=layers)
            ser = st.multiselect("Series", sorted(df.series.unique()), default=sorted(df.series.unique()))
            st.dataframe(df[df.layer.isin(sel) & df.series.isin(ser)].round(5), use_container_width=True)
    elif what.startswith("Perplexity"):
        p = ws.json(f"phase3_perplexity_{model}.json")
        if p:
            st.dataframe(pd.DataFrame(p).round(4), use_container_width=True)
    elif what.startswith("Phase-2"):
        p2 = ws.json(PHASE2)
        if p2:
            df = pd.DataFrame([{"id": r["id"], "domain": r["domain"], "source": r["source_type"], "topic": r["topic"],
                                "revision_rounds": r.get("revision_rounds", 0), "status": r["verification_status"],
                                "english": r["english"], "chinese": r["chinese"]} for r in p2])
            st.dataframe(df[df.domain.isin(flt["domains"])], use_container_width=True)
    elif what.startswith("Sanity"):
        sx = ws.json(f"phase3_sanity_{model}.json")
        if sx:
            st.dataframe(pd.DataFrame([{"token": c["token"], "id": c["id"], **{k: c[k] for k in
                         ("norm_ratio", "sink_cos", "entropy", "top1_prob", "flags", "pass")}} for c in sx["carrier"]["candidates"]]),
                         use_container_width=True)
    else:
        st.dataframe(pd.DataFrame({"file": list(ws.idx), "path": list(ws.idx.values())}), use_container_width=True)


# --------------------------------------------------------------------------------------
# Tab: Audit notes
# --------------------------------------------------------------------------------------
def tab_audit(ws, models):
    st.header("Methods and audit notes")
    st.markdown("""
**Metric definitions.** `M[j,i]` = NLL of text *i* with concept vector *j* injected. *Matched* = `M[i,i]`; *other* = mean of
`M[j≠i,i]`. **CSP = other − matched.** *Total* = noise − matched; *generic* = noise − other (any vector helps);
*within-domain CSP* = same-domain-other − matched; *AUC-ROC* = share of other vectors worse than the matched one.

**Things worth knowing when reading the results** (several are checked live elsewhere in the app):
1. **Scale.** NLL units differ by tokenizer and model, so raw CSP is not comparable across families. Use relative CSP, AUC-ROC,
   or the RMS-standardized pooled tests.
2. **Models with CSP ≈ 0.** Where CI covers 0 in all four conditions, specialization deltas and CTR ratios are ratios of noise
   (the CTR CIs are enormous) and should not be interpreted.
3. **Total vs specific potency.** Total potency uses a Gaussian noise baseline that is out-of-distribution, so it can be negative
   even when CSP is positive; "specificity share" can then explode or flip sign. CSP and gain-vs-no-injection are the safer readouts.
4. **Mean-NLL dilution.** EN and ZH passages differ in token count; the *head* aggregation (first 16 tokens) removes 1/length
   dilution, the *sum* aggregation is length-sensitive. Conclusions that hold in all three are robust.
5. **Mismatch control (phase 3b).** The cyclic mismatch runs reuse another concept's EN vector, which duplicates that concept's
   own EN run, so the paired EN-vs-mismatch test is an identity (see Layer dynamics).
6. **Geometry saturation.** Top-1 retrieval is 1.0 for every model, and unbiased CKA / RSA are ≥ 0.9, so geometry differs little
   between models; deltas are small (subsampling intervals, not formal CIs).
7. **Shared direction.** Raw concept vectors have pairwise cosine ≈ 0.9, so all alignment metrics use per-language centering.
8. **Human vs AI.** Only 8 human concepts; compare descriptively. In some models human concepts show near-zero CSP.
9. **k = 3 families.** Random-effects pooling over 3 families is descriptive only; the stratified sign-flip is the primary pooled test.
10. **Secondary carrier** is defined for the Qwen family only by design; other families are single-carrier.
""")
    st.subheader("Registry check (from meta files)")
    rows = []
    for k in models:
        m = ws.json(n_meta(k, "primary"))
        if m:
            rows.append({"model": k, "name": m["model"], "category": m["category"], "layer": m["layer_idx"],
                         "inject pos (0-idx)": m["injection_position"], "carrier": repr(m["carrier_token"]),
                         "carrier id": m["carrier_token_id"], "n_noise": m["n_noise"], "head_k": m["head_k"],
                         "runtime (s)": m.get("runtime_s")})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        for f, (g, s) in FAMILIES.items():
            a, b = df[df.model == g], df[df.model == s]
            if len(a) and len(b):
                same = (a.layer.iloc[0] == b.layer.iloc[0]) and (a.carrier.iloc[0] == b.carrier.iloc[0])
                (st.success if same else st.error)(f"{f}: layer and carrier {'match' if same else 'DIFFER'} within family")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ws, models, mode, model, kind, carrier, flt = sidebar()
    st.title("Cross-Lingual Concept Representations: Geometry and Causal Potency")
    audience = st.radio("Audience", ["General", "Research reviewer"], horizontal=True,
                        help="General: concept-level, plain-language views first. Reviewer: all statistical tabs.")
    tabs = [("🧭 Overview", lambda: tab_overview(ws, models, kind, carrier, flt)),
            ("💬 Concept explorer", lambda: tab_concepts(ws, models, kind, carrier, flt))]
    if audience == "Research reviewer":
        if mode.startswith("Single"):
            tabs.append(("🔬 Single model", lambda: tab_single(ws, model, kind, carrier, flt)))
        else:
            tabs.append(("🏛️ Cross-family", lambda: tab_cross(ws, models, kind, carrier, flt)))
        tabs += [("🔁 Carrier robustness", lambda: tab_carrier(ws, models, kind, flt)),
                 ("📋 Data tables", lambda: tab_tables(ws, models, model, kind, carrier, flt)),
                 ("🧪 Methods & audit", lambda: tab_audit(ws, models))]
    else:
        tabs.append(("🏛️ Specialized vs general", lambda: tab_cross(ws, models, kind, carrier, flt)))
        tabs.append(("🔬 Look inside one model", lambda: tab_single(ws, model, kind, carrier, flt)))
    ui = st.tabs([t for t, _ in tabs])
    for tab, (_, fn) in zip(ui, tabs):
        with tab:
            fn()


if __name__ == "__main__":
    main()
