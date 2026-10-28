#!/usr/bin/env python3
"""
Phase 3: Tensor extraction, mathematical alignment, and ChromaDB indexing.
Cleaned to remove unused token counts report generation.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import chromadb
import phase3_config as cfg

PHASE2_INPUT_PATH = Path("phase2_final_48_local.json")
TOP_K_SINGULAR_REMOVE = 2
N_BOOTSTRAP_DRAWS = 20
REPORT_PATH = Path(f"phase3_report_{cfg.ACTIVE_MODEL_KEY}.json")


def load_approved_items(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    approved = [r for r in records if r["verification_status"] in ("APPROVED_FIRST_PASS", "APPROVED_AFTER_REVISION")]
    dropped = len(records) - len(approved)
    if dropped:
        print(f"[warn] Excluded {dropped} item(s) that were not cleanly approved.")
    print(f"[info] {len(approved)} approved item(s) going into Phase 3.")
    return approved


def profile_tokens(tokenizer, items):
    en_counts, zh_counts = [], []
    for it in items:
        en_ids = tokenizer(it["english"], add_special_tokens=True)["input_ids"]
        zh_ids = tokenizer(it["chinese"], add_special_tokens=True)["input_ids"]
        en_counts.append(len(en_ids))
        zh_counts.append(len(zh_ids))
    n_en_total, n_zh_total = sum(en_counts), sum(zh_counts)
    print(f"[tokens] N_en_total={n_en_total}  N_zh_total={n_zh_total}  ratio(en/zh)={n_en_total / n_zh_total:.3f}")
    return n_en_total, n_zh_total, en_counts, zh_counts


class MultiLayerActivationCapture:
    def __init__(self, layer_indices):
        self.layer_indices = layer_indices
        self.last_captured = {}

    def make_hook(self, layer_idx):
        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.last_captured[layer_idx] = hidden.detach().to(torch.float32).cpu().numpy()
        return hook_fn

    def reset(self):
        self.last_captured = {}


def load_model_and_tokenizer():
    print(f"[info] Loading {cfg.TARGET_MODEL_NAME} on device={cfg.DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.TARGET_MODEL_NAME)
    model = cfg.load_causal_lm()
    cfg.assert_layers_exist(model)
    monitored_layers = sorted(list(set([cfg.LAYER_IDX] + cfg.PROBE_LAYERS)))
    capture = MultiLayerActivationCapture(monitored_layers)

    for layer_idx in monitored_layers:
        target_layer = model.model.layers[layer_idx]
        target_layer.register_forward_hook(capture.make_hook(layer_idx))

    return model, tokenizer, capture, monitored_layers


def extract_all(model, tokenizer, capture, items, monitored_layers):
    token_rows = {"en": [], "zh": []}
    concept_vecs = {layer: {"en": {}, "zh": {}} for layer in monitored_layers}

    with torch.no_grad():
        for lang, text_key in (("en", "english"), ("zh", "chinese")):
            for it in items:
                enc = tokenizer(it[text_key], return_tensors="pt").to(cfg.DEVICE)
                
                capture.reset()
                model(**enc)

                start_pos = cfg.EXCLUDE_LEADING_N
                token_ids = enc["input_ids"][0].cpu().tolist()
                token_strs = tokenizer.convert_ids_to_tokens(token_ids)

                for layer_idx in monitored_layers:
                    hidden = capture.last_captured[layer_idx][0]
                    
                    passage_rows = []
                    for pos in range(start_pos, hidden.shape[0]):
                        row_data = {
                            "id": it["id"],
                            "position": pos,
                            "token_str": token_strs[pos],
                            "vector": hidden[pos],
                        }
                        passage_rows.append(row_data)
                        if layer_idx == cfg.LAYER_IDX:
                            token_rows[lang].append(row_data)

                    if passage_rows:
                        passage_norms = np.array([np.linalg.norm(r["vector"]) for r in passage_rows])
                        passage_median = np.median(passage_norms)
                        max_allowed_norm = passage_median * cfg.EXCLUDE_NORM_OUTLIER_FACTOR

                        final_vectors = [r["vector"] for r, norm in zip(passage_rows, passage_norms) if norm <= max_allowed_norm]
                        if final_vectors:
                            concept_vecs[layer_idx][lang][it["id"]] = np.stack(final_vectors).mean(axis=0)
                        else:
                            concept_vecs[layer_idx][lang][it["id"]] = hidden[start_pos:].mean(axis=0)
                    else:
                        concept_vecs[layer_idx][lang][it["id"]] = hidden[start_pos:].mean(axis=0)

    print(f"[info] Extracted representations across layers: {monitored_layers}")
    return token_rows, concept_vecs


def participation_ratio(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    eig = s ** 2
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def subsample_rows(X, n, seed):
    rng = random.Random(seed)
    idx = rng.sample(range(X.shape[0]), n)
    return X[idx]


def bootstrap_pr(X, target_n, n_boot, base_seed):
    scores = [participation_ratio(subsample_rows(X, target_n, base_seed + i)) for i in range(n_boot)]
    return float(np.mean(scores)), float(np.std(scores))


def debias(X, top_k=TOP_K_SINGULAR_REMOVE):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    S_clean = S.copy()
    S_clean[:top_k] = 0.0
    X_clean = (U * S_clean) @ Vt
    norms = np.linalg.norm(X_clean, axis=1, keepdims=True)
    return X_clean / np.clip(norms, 1e-8, None)


def linear_cka(X, Y):
    assert X.shape[0] == Y.shape[0], "linear_cka() requires paired rows."
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(cross / denom)


def ingest_to_chroma(token_rows, concept_vecs):
    client = chromadb.PersistentClient(path=cfg.CHROMA_DB_PATH)
    for name in ["token_en_activations", "token_zh_activations", "concept_en_geometry", "concept_zh_geometry"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    token_en = client.get_or_create_collection("token_en_activations")
    token_zh = client.get_or_create_collection("token_zh_activations")
    concept_en = client.get_or_create_collection("concept_en_geometry")
    concept_zh = client.get_or_create_collection("concept_zh_geometry")

    CHROMA_MAX_BATCH = 2000
    for lang, collection in (("en", token_en), ("zh", token_zh)):
        rows = token_rows[lang]
        for start in range(0, len(rows), CHROMA_MAX_BATCH):
            chunk = rows[start:start + CHROMA_MAX_BATCH]
            collection.add(
                ids=[f"{lang}_{r['id']}_{r['position']}" for r in chunk],
                embeddings=[r["vector"].tolist() for r in chunk],
                metadatas=[{
                    "sentence_id": r["id"],
                    "position": r["position"],
                    "token_str": r["token_str"],
                    "language": lang,
                    "raw_l2_norm": float(np.linalg.norm(r["vector"])),
                } for r in chunk],
            )

    primary_vecs = concept_vecs[cfg.LAYER_IDX]
    for lang, collection in (("en", concept_en), ("zh", concept_zh)):
        ids = list(primary_vecs[lang].keys())
        collection.add(
            ids=[f"{lang}_{sid}" for sid in ids],
            embeddings=[primary_vecs[lang][sid].tolist() for sid in ids],
            metadatas=[{"sentence_id": sid, "language": lang} for sid in ids],
        )

    print(f"[info] Ingested into ChromaDB at {cfg.CHROMA_DB_PATH}")


def main():
    cfg.require_primary_carrier("phase3_extraction_pipeline.py")
    items = load_approved_items(PHASE2_INPUT_PATH)
    model, tokenizer, capture, monitored_layers = load_model_and_tokenizer()
    profile_tokens(tokenizer, items)

    token_rows, concept_vecs = extract_all(model, tokenizer, capture, items, monitored_layers)

    X_en_raw = np.stack([r["vector"] for r in token_rows["en"]])
    X_zh_raw = np.stack([r["vector"] for r in token_rows["zh"]])

    pr1_en = participation_ratio(X_en_raw)
    pr1_zh = participation_ratio(X_zh_raw)

    target_n = min(X_en_raw.shape[0], X_zh_raw.shape[0])
    pr2_en_mean, pr2_en_std = bootstrap_pr(debias(X_en_raw), target_n, N_BOOTSTRAP_DRAWS, cfg.RANDOM_SEED)
    pr2_zh_mean, pr2_zh_std = bootstrap_pr(debias(X_zh_raw), target_n, N_BOOTSTRAP_DRAWS + 1000, cfg.RANDOM_SEED )

    cka_by_layer = {}
    print(f"\n[info] Computing CKA across layers ({monitored_layers}):")
    for layer_idx in monitored_layers:
        shared_ids = sorted(set(concept_vecs[layer_idx]["en"]) & set(concept_vecs[layer_idx]["zh"]))
        Xs_en = np.stack([concept_vecs[layer_idx]["en"][sid] for sid in shared_ids])
        Xs_zh = np.stack([concept_vecs[layer_idx]["zh"][sid] for sid in shared_ids])

        c_raw = linear_cka(Xs_en, Xs_zh)
        k_top = min(TOP_K_SINGULAR_REMOVE, len(shared_ids) - 1)
        c_debiased = linear_cka(debias(Xs_en, top_k=k_top), debias(Xs_zh, top_k=k_top))

        cka_by_layer[str(layer_idx)] = {
            "raw": c_raw,
            "debiased": c_debiased,
            "n_pairs": len(shared_ids)
        }
        print(f"    Layer {layer_idx:2d} -> raw CKA: {c_raw:.4f} | debiased CKA: {c_debiased:.4f}")

    report_data = {
        "model": cfg.TARGET_MODEL_NAME,
        "category": cfg.MODEL_CATEGORY,
        "primary_layer": cfg.LAYER_IDX,
        "monitored_layers": monitored_layers,
        "n_en_tokens_raw": int(X_en_raw.shape[0]),
        "n_zh_tokens_raw": int(X_zh_raw.shape[0]),
        "pr1_raw": {"en": pr1_en, "zh": pr1_zh},
        "pr2_equalized_debiased": {
            "n_per_language": target_n,
            "en_mean": pr2_en_mean, "en_std": pr2_en_std,
            "zh_mean": pr2_zh_mean, "zh_std": pr2_zh_std,
        },
        "cka_by_layer": cka_by_layer,
    }

    REPORT_PATH.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\n[done] Comprehensive report with multi-layer CKA written to {REPORT_PATH}")

    ingest_to_chroma(token_rows, concept_vecs)


if __name__ == "__main__":
    main()