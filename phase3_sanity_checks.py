#!/usr/bin/env python3
"""phase3_sanity_checks.py — run once per model:  MODEL_KEY=gemma2_9b python3 phase3_sanity_checks.py

Env vars:
  MODEL_KEY      registry key of the model to check
  SKIP_SWEEP=1   skip the position-exclusion sweep (carrier candidates only)
  CARRIER_TOKEN  evaluate just this one candidate (writes phase3_sanity_<key>_single.json)
"""
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import phase3_config as cfg
from phase3b_latent_injection import load_model_with_null_pad, N_NULL_TOKENS, INJECTION_POSITION, InjectionHook

PHASE2 = Path("phase2_final_48_local.json")
OUT = Path(f"phase3_sanity_{cfg.ACTIVE_MODEL_KEY}.json")
LANGS = (("en", "english"), ("zh", "chinese"))

SWEEP_N = [0, 1, 2, 3, 4]
PROFILE_POS = 6                     # positions 0..5 profiled against the median of positions >= 6
FACTOR = cfg.EXCLUDE_NORM_OUTLIER_FACTOR
SYSTEMATIC_RATE = 0.5               # a leading position is a "systematic sink" if > 50% of passages are outliers there
MASSIVE_DIM_FACTOR = 20.0           # a dim is "massive" if mean |sink state| > 20x the median dim magnitude
REF_TOKEN_SAMPLE = 4000


CANDIDATES = ["<pad>", "<unk>", "</s>", "<eos>", "<|endoftext|>", "<|im_end|>", " ", "\n", ".", "the"]   # edit freely
SINK_RATIO_MAX = 5.0                # carrier norm / ordinary-token norm above this = sink-like
SINK_COS_MAX = 0.9                  # cosine to the natural sink direction above this = sink-like
ENTROPY_MIN = 0.5                   # next-token entropy below this (nats) = degenerate


def load_items():
    recs = json.loads(PHASE2.read_text(encoding="utf-8"))
    items = [r for r in recs if r["verification_status"] in ("APPROVED_FIRST_PASS", "APPROVED_AFTER_REVISION")]
    if not items:   # fail early with a readable message instead of an obscure numpy error later
        from collections import Counter as _C
        raise SystemExit(f"[load_items] 0 of {len(recs)} records in {PHASE2} have an approved verification_status "
                         f"(found: {dict(_C(r['verification_status'] for r in recs))}). "
                         "Expected APPROVED_FIRST_PASS or APPROVED_AFTER_REVISION.")
    return items


def participation_ratio(X):
    Xc = X.astype(np.float32) - X.mean(0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(Xc.T @ Xc), 0, None)       # eigenvalues of the covariance (= singular values^2)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


# ---------------------------------------------------------------- natural-text hidden states + references
@torch.no_grad()
def layer_hidden(model, tok, text):
    enc = tok(text, return_tensors="pt").to(cfg.DEVICE)             # default special tokens, as in extraction
    hs = model(**enc, output_hidden_states=True).hidden_states      # hs[0]=embeddings, hs[L+1]=output of layer L
    assert cfg.LAYER_IDX + 1 < len(hs) - 1, "hidden_states[-1] is post-final-norm; LAYER_IDX must be an interior layer"
    return enc["input_ids"][0].cpu().tolist(), hs[cfg.LAYER_IDX + 1][0].float().cpu().numpy()


def collect(model, tok, items):
    return {lang: [layer_hidden(model, tok, it[key]) for it in items] for lang, key in LANGS}


def reference_stats(passages):
    """Ordinary-token and natural-sink (position 0) references at the injection layer, both languages pooled."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    allp = passages["en"] + passages["zh"]
    ordinary = np.concatenate([h[2:] for _, h in allp])
    if len(ordinary) > REF_TOKEN_SAMPLE:
        ordinary = ordinary[rng.choice(len(ordinary), REF_TOKEN_SAMPLE, replace=False)]
    sink = np.stack([h[0] for _, h in allp])
    mag = np.abs(sink).mean(0)
    massive = np.where(mag > MASSIVE_DIM_FACTOR * np.median(mag))[0]
    sink_dir = sink.mean(0)
    sink_dir = sink_dir / np.linalg.norm(sink_dir)
    on = np.linalg.norm(ordinary, axis=1)
    return dict(
        ref_norm=float(np.median(on)),
        ref_norm_nm=float(np.median(np.linalg.norm(np.delete(ordinary, massive, axis=1), axis=1))),
        sink_norm=float(np.median(np.linalg.norm(sink, axis=1))),
        ord_sink_cos=float(np.median(ordinary @ sink_dir / on)),
        massive_dims=massive, sink_dir=sink_dir,
    )


# ---------------------------------------------------------------- A. position-exclusion sweep
def sweep(passages, tok):
    res = {}
    for lang, _ in LANGS:
        ps = passages[lang]
        has_bos = float(np.mean([ids[0] == tok.bos_token_id for ids, _ in ps]))
        first_tokens = [tok.convert_ids_to_tokens(ids[:3]) for ids, _ in ps[:3]]

        # norm profile of leading positions relative to the passage's own typical token
        prof = np.array([np.linalg.norm(h[:PROFILE_POS], axis=1) / np.median(np.linalg.norm(h[PROFILE_POS:], axis=1))
                         for ids, h in ps if len(ids) > PROFILE_POS + 2])
        sys_rate = (prof > FACTOR).mean(0)                          # per position: share of passages with a sink there
        rec = 0
        while rec < PROFILE_POS and sys_rate[rec] > SYSTEMATIC_RATE:
            rec += 1                                                # leading run of systematic sink positions

        rows = {}
        for N in SWEEP_N:
            raw, kept, removed, total = [], [], 0, 0
            for ids, h in ps:
                x = h[N:]
                n = np.linalg.norm(x, axis=1)
                keep = n <= FACTOR * np.median(n)                   # same filter as the extraction pipeline
                raw.append(x); kept.append(x[keep]); removed += int((~keep).sum()); total += len(n)
            rows[N] = dict(n_tokens=total, pr_raw=participation_ratio(np.concatenate(raw)),
                           pr_filtered=participation_ratio(np.concatenate(kept)),
                           frac_removed_by_filter=removed / total)

        # late (non-leading) outliers that no position exclusion can remove; the norm filter handles them
        late, n_late, n_tot = Counter(), 0, 0
        for ids, h in ps:
            n = np.linalg.norm(h[rec:], axis=1)
            bad = np.where(n > FACTOR * np.median(n))[0]
            n_tot += len(n); n_late += len(bad)
            for b in bad:
                late[tok.convert_ids_to_tokens(ids[rec + b])] += 1

        pr_stable = rec in rows and (rec + 1) in rows and \
            abs(rows[rec]["pr_filtered"] - rows[rec + 1]["pr_filtered"]) / rows[rec + 1]["pr_filtered"] < 0.05
        res[lang] = dict(frac_passages_starting_with_bos=has_bos, first_tokens_examples=first_tokens,
                         norm_ratio_by_position_mean=prof.mean(0).tolist(),
                         systematic_sink_rate_by_position=sys_rate.tolist(),
                         recommended_exclude_leading_n=rec, filtered_pr_stable_at_rec=bool(pr_stable),
                         late_outlier_rate=n_late / n_tot, late_outlier_top_tokens=late.most_common(8),
                         sweep=rows)

        print(f"\n[A:{lang}] BOS present in {has_bos:.0%} of passages; example first tokens: {first_tokens}")
        print("   norm / median-norm by position 0..5 (mean):", np.round(prof.mean(0), 2))
        print("   share of passages with a sink at position 0..5:", np.round(sys_rate, 2))
        print("   N | tokens | PR raw | PR filtered | removed by %.0fx filter" % FACTOR)
        for N, v in rows.items():
            print(f"   {N} | {v['n_tokens']:6d} | {v['pr_raw']:6.2f} | {v['pr_filtered']:11.2f} | {v['frac_removed_by_filter']:8.4f}")
        print(f"   => recommended EXCLUDE_LEADING_N for {lang}: {rec} (config currently {cfg.EXCLUDE_LEADING_N}); "
              f"late outliers at pos>={rec}: {n_late / n_tot:.4f} of tokens, top: {late.most_common(4)} "
              f"(removed by the norm filter, not by position)")
    return res


# ---------------------------------------------------------------- B. carrier candidates
@torch.no_grad()
def token_nlls(model, prefix_ids, sent_ids):
    """Per-token NLL of every sentence token given a prefix (the first token is predicted from the prefix)."""
    ids = torch.tensor([list(prefix_ids) + list(sent_ids)], device=cfg.DEVICE)
    logits = model(input_ids=ids).logits[0].float()
    s = len(prefix_ids)
    nll = F.cross_entropy(logits[s - 1:-1], ids[0, s:], reduction="none")
    return nll.cpu().numpy(), bool(torch.isfinite(logits).all())


def plain_nll_tail(model, tok, text):
    """Plain-text NLL over sentence tokens 1..n-1: the tokens every setup can score
    (Qwen has no BOS, so token 0 is unscorable there). Matches the carrier 'tail' NLL."""
    ids = tok(text)["input_ids"]
    sent = tok(text, add_special_tokens=False)["input_ids"]
    k = len(ids) - len(sent)
    assert ids[k:] == sent, "special tokens are not a pure prefix; plain/carrier NLL would not be comparable"
    if k == 0:
        nll, _ = token_nlls(model, ids[:1], ids[1:])
        return float(nll.mean())
    nll, _ = token_nlls(model, ids[:k], sent)
    return float(nll[1:].mean())


def candidate_list(tok, default_id):
    if os.environ.get("PINNED_ONLY"):                  # final check: only the carrier the pipeline will really use
        return [(cfg.CARRIER_TOKEN or "DEFAULT", default_id)]
    env = os.environ.get("CARRIER_TOKEN")
    out, seen = [], set()
    for s in ([env] if env else CANDIDATES):
        cid = cfg.resolve_token_id(tok, s)
        if cid is not None and cid not in seen:
            seen.add(cid); out.append((s, cid))
    if default_id not in seen:                         # always include the loader's current default
        out.append(("DEFAULT", default_id))
    return out


@torch.no_grad()
def eval_candidate(model, tok, sents, cid, ref, V, plain):
    ids = torch.full((1, N_NULL_TOKENS), cid, dtype=torch.long, device=cfg.DEVICE)
    o = model(input_ids=ids, output_hidden_states=True)
    h = o.hidden_states[cfg.LAYER_IDX + 1][0].float()
    p = F.softmax(o.logits[0, -1].float(), dim=-1)
    hn = h.cpu().numpy()
    nrm = np.linalg.norm(hn, axis=1)
    cos_sink = hn @ ref["sink_dir"] / nrm                          # direction: sink-like states are nearly parallel
    nm = np.linalg.norm(np.delete(hn, ref["massive_dims"], axis=1), axis=1)   # norm with the sink's massive dims removed

    r = dict(token=tok.convert_ids_to_tokens(cid), id=cid,
             finite=bool(torch.isfinite(o.logits).all() and torch.isfinite(h).all()),
             norm_ratio=float(nrm[1:].mean() / ref["ref_norm"]),                  # positions 1..N-1
             pos0_norm_ratio=float(nrm[0] / ref["ref_norm"]),
             inj_site_norm_ratio=float(nrm[INJECTION_POSITION] / ref["ref_norm"]),
             sink_cos=float(cos_sink[1:].mean()), pos0_sink_cos=float(cos_sink[0]),
             nm_ratio=float(nm[1:].mean() / ref["ref_norm_nm"]),                  # ~1 = ordinary apart from massive dims
             norm_by_position=[round(float(x / ref["ref_norm"]), 2) for x in nrm],
             sink_cos_by_position=[round(float(x), 3) for x in cos_sink],
             positions_indistinguishable=bool(nrm.max() / max(nrm.min(), 1e-9) < 1.01),
             entropy=float(-(p * torch.log(p + 1e-12)).sum()), top1_prob=float(p.max()),
             top5=[tok.decode([i]) for i in p.topk(5).indices.tolist()])
    finite_nll = True
    for lang, _ in LANGS:
        all_m, tail_m = [], []
        for s in sents[lang]:
            nll, ok = token_nlls(model, [cid] * N_NULL_TOKENS, s)
            finite_nll &= ok
            all_m.append(nll.mean()); tail_m.append(nll[1:].mean())
        r[lang] = dict(nll_all=float(np.mean(all_m)), nll_tail=float(np.mean(tail_m)),
                       nll_std=float(np.std(all_m)), ratio_to_plain=float(np.mean(tail_m) / plain[lang]))
    flags = []
    if not (r["finite"] and finite_nll): flags.append("NONFINITE")
    if max(r["norm_ratio"], r["inj_site_norm_ratio"]) > SINK_RATIO_MAX or r["sink_cos"] > SINK_COS_MAX:
        flags.append("SINK-LIKE")
    if r["entropy"] < ENTROPY_MIN: flags.append("DEGENERATE")
    if max(r["en"]["nll_all"], r["zh"]["nll_all"]) > 0.9 * math.log(V): flags.append("NEAR-UNIFORM")
    r["flags"], r["pass"] = flags, not flags
    return r


def carrier_candidates(model, tok, default_id, items, ref):
    V = model.config.vocab_size
    sents = {lang: [tok(it[key], add_special_tokens=False)["input_ids"] for it in items] for lang, key in LANGS}
    plain = {lang: float(np.mean([plain_nll_tail(model, tok, it[key]) for it in items])) for lang, key in LANGS}
    print(f"\n[B] references @ layer {cfg.LAYER_IDX}: ordinary norm {ref['ref_norm']:.1f} | natural sink norm {ref['sink_norm']:.1f} "
          f"| ordinary-token cos to sink dir {ref['ord_sink_cos']:.2f} | massive dims {len(ref['massive_dims'])}")
    print(f"    plain-text NLL (tail tokens): en {plain['en']:.2f} zh {plain['zh']:.2f} | uniform {math.log(V):.2f}")
    print(f"    normx = norm/ordinary (positions 1..{N_NULL_TOKENS - 1}); pos0x, injx = same at position 0 / injection site; "
          f"sinkcos = cos to sink dir; nmx = norm without massive dims / ordinary; r_en,r_zh = carrier NLL / plain NLL (same tokens)")
    print(f"    {'token':>14} {'id':>6} {'normx':>7} {'pos0x':>7} {'injx':>7} {'sinkcos':>7} {'nmx':>6} {'entropy':>7} "
          f"{'r_en':>5} {'r_zh':>5}  verdict | top-3 next tokens")
    rows = []
    for s, cid in candidate_list(tok, default_id):
        r = eval_candidate(model, tok, sents, cid, ref, V, plain)
        r["label"] = s; rows.append(r)
        print(f"    {repr(r['token']):>14} {cid:6d} {r['norm_ratio']:7.2f} {r['pos0_norm_ratio']:7.2f} {r['inj_site_norm_ratio']:7.2f} "
              f"{r['sink_cos']:7.2f} {r['nm_ratio']:6.2f} {r['entropy']:7.2f} "
              f"{r['en']['ratio_to_plain']:5.2f} {r['zh']['ratio_to_plain']:5.2f}  "
              f"{'PASS' if r['pass'] else ','.join(r['flags'])} | {r['top5'][:3]}"
              f"{'  [all positions indistinguishable]' if r['positions_indistinguishable'] else ''}")
    return dict(plain_nll_tail=plain, uniform_nll=math.log(V), default_id=default_id, candidates=rows)

 # ---------------------------------------------------------------- C. functional tests of the real pipeline pieces
def _out_h(o):
    return o[0] if isinstance(o, tuple) else o


def environment(model):
    import transformers
    p = next(model.parameters())
    env = dict(torch=torch.__version__, transformers=transformers.__version__, device=str(p.device),
               param_dtype=str(p.dtype), expected_dtype=str(cfg.DTYPE), model_type=model.config.model_type,
               attn_impl=getattr(model.config, "_attn_implementation", None),
               n_layers=len(model.model.layers), hidden_size=model.config.hidden_size, vocab_size=model.config.vocab_size)
    env["dtype_ok"] = bool(p.dtype == cfg.DTYPE)
    env["attn_ok"] = bool(env["attn_impl"] == "eager") if env["model_type"].startswith("gemma2") else True
    print(f"\n[env] torch {env['torch']} | transformers {env['transformers']} | dtype {env['param_dtype']} | "
          f"attn {env['attn_impl']} | layers {env['n_layers']} | device {env['device']}")
    return env


def concept_vectors(passages):
    """Same recipe as the extraction pipeline: drop leading positions, 5x-median norm filter, mean."""
    N0, V = cfg.EXCLUDE_LEADING_N, {}
    for lang, _ in LANGS:
        rows = []
        for ids, h in passages[lang]:
            x = h[N0:]
            n = np.linalg.norm(x, axis=1)
            keep = n <= FACTOR * np.median(n)
            rows.append((x[keep] if keep.any() else x).mean(0))
        V[lang] = np.stack(rows)
    return V


@torch.no_grad()
def hook_and_injection_tests(model, cid, passages, vec):
    """Checks the exact objects 3b/3c/3d rely on: layer-output hook, InjectionHook, causal locality."""
    L, pos, layers = cfg.LAYER_IDX, INJECTION_POSITION, model.model.layers
    res = {}

    # (a) the layer-output hook sees the same tensor as hidden_states[L+1] (extraction vs. injection layer agree)
    ids = torch.tensor([passages["en"][0][0]], device=cfg.DEVICE)
    cap = {}
    h = layers[L].register_forward_hook(lambda m, i, o: cap.__setitem__("h", _out_h(o).detach().float().cpu()))
    try:
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states[L + 1].detach().float().cpu()
    finally:
        h.remove()
    res["hook_vs_hidden_states_max_abs_diff"] = float((cap["h"] - hs).abs().max())

    # (b) inject `vec` at the injection position of the null carrier, capture layers L and L+1
    carrier = torch.full((1, N_NULL_TOKENS), cid, dtype=torch.long, device=cfg.DEVICE)
    mask = torch.ones_like(carrier)

    def run(inject):
        caps, hooks = {}, []

        def mk(name):
            def f(m, i, o):
                caps[name] = _out_h(o).detach().float().cpu()
            return f
        if inject:
            inj = InjectionHook(pos, vec)
            inj.reset()
            hooks.append(layers[L].register_forward_hook(inj.hook_fn))
        hooks.append(layers[L].register_forward_hook(mk("L")))        # registered after: sees the injected state
        hooks.append(layers[L + 1].register_forward_hook(mk("L1")))
        try:
            out = model(input_ids=carrier, attention_mask=mask)
        finally:
            for x in hooks:
                x.remove()
        return caps, out.logits[0, -1].float().cpu()

    base, lb = run(False)
    inj, li = run(True)
    v32 = torch.tensor(vec, dtype=torch.float32)
    target = v32.to(next(model.parameters()).dtype).float()
    got = inj["L"][0, pos]
    others, before, after = [i for i in range(N_NULL_TOKENS) if i != pos], list(range(pos)), list(range(pos + 1, N_NULL_TOKENS))
    rel = lambda a, b: float((a - b).norm() / b.norm())
    res.update(
        inject_exact_vs_cast_vector_max_abs_diff=float((got - target).abs().max()),
        inject_rel_err_vs_fp32_vector=rel(got, v32),
        layerL_other_positions_rel_change=rel(inj["L"][0, others], base["L"][0, others]),
        layerL1_before_pos_rel_change=rel(inj["L1"][0, before], base["L1"][0, before]),   # must be ~0 (causality)
        layerL1_at_pos_rel_change=rel(inj["L1"][0, pos], base["L1"][0, pos]),             # must be > 0 (propagation)
        layerL1_after_pos_rel_change=rel(inj["L1"][0, after], base["L1"][0, after]),
        final_logits_max_abs_change=float((li - lb).abs().max()),
    )
    print("\n[C1] hook/injection tests: " + ", ".join(f"{k}={v:.3g}" for k, v in res.items()))
    return res


@torch.no_grad()
def batched_mean_nll(model, cid, sent_list):
    """Mirrors the 3d scorer: right-padded batch, [carrier | sentence], every sentence token scored."""
    C, B, Lm = N_NULL_TOKENS, len(sent_list), max(len(s) for s in sent_list)
    ids = torch.full((B, C + Lm), cid, dtype=torch.long)
    att = torch.zeros((B, C + Lm), dtype=torch.long); att[:, :C] = 1
    tgt = torch.full((B, Lm), -100, dtype=torch.long)
    for b, s in enumerate(sent_list):
        t = torch.tensor(s)
        ids[b, C:C + len(s)] = t; att[b, C:C + len(s)] = 1; tgt[b, :len(s)] = t
    ids, att, tgt = ids.to(cfg.DEVICE), att.to(cfg.DEVICE), tgt.to(cfg.DEVICE)
    logits = model(input_ids=ids, attention_mask=att).logits[:, C - 1:-1, :].float()
    nll = F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none")
    return (nll.sum(1) / (tgt != -100).sum(1)).cpu().numpy()


def _all_rows_inject(model, vec):
    v = torch.tensor(vec, dtype=torch.float32)

    def f(m, i, o):
        h = _out_h(o)
        if h.shape[1] <= INJECTION_POSITION:
            return o
        h = h.clone()
        h[:, INJECTION_POSITION, :] = v.to(device=h.device, dtype=h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    return model.model.layers[cfg.LAYER_IDX].register_forward_hook(f)


@torch.no_grad()
def batch_equivalence_test(model, cid, sents, vec, k=3):
    """Batched (padded) scoring must match one-at-a-time scoring, with and without injection."""
    order = sorted(range(len(sents["en"])), key=lambda i: len(sents["en"][i]))
    batch = [sents["en"][i] for i in order[:k] + order[-k:]]            # shortest + longest = maximal padding
    res = {"lengths": [int(len(s)) for s in batch]}
    for name, inject in (("no_injection", False), ("with_injection", True)):
        hook = _all_rows_inject(model, vec) if inject else None
        try:
            single = np.array([token_nlls(model, [cid] * N_NULL_TOKENS, s)[0].mean() for s in batch])
            batched = batched_mean_nll(model, cid, batch)
        finally:
            if hook is not None:
                hook.remove()
        res[name] = dict(max_abs_diff=float(np.abs(single - batched).max()), mean_nll_single=float(single.mean()))
    print(f"\n[C2] batched vs single NLL max |diff| (nats/token): "
          f"no injection {res['no_injection']['max_abs_diff']:.4f} | with injection {res['with_injection']['max_abs_diff']:.4f}")
    return res


def concept_vector_diagnostics(V, ref):
    """What will actually be injected: scale, and how much of it is a shared common-mode direction."""
    unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)

    def offdiag(S):
        n = len(S)
        return float((S.sum() - np.trace(S)) / (n * (n - 1)))
    res = {}
    for lang, _ in LANGS:
        v = V[lang]; c = v - v.mean(0)
        res[lang] = dict(norm_ratio_to_ordinary=float(np.median(np.linalg.norm(v, axis=1)) / ref["ref_norm"]),
                         cos_to_sink_dir=float(np.median(unit(v) @ ref["sink_dir"])),
                         mean_pairwise_cos_raw=offdiag(unit(v) @ unit(v).T),
                         mean_pairwise_cos_centered=offdiag(unit(c) @ unit(c).T))
    for tag, (a, b) in (("raw", (V["en"], V["zh"])), ("centered", (V["en"] - V["en"].mean(0), V["zh"] - V["zh"].mean(0)))):
        S = unit(a) @ unit(b).T
        res[f"cross_{tag}"] = dict(same_concept_cos=float(np.trace(S) / len(S)), different_concept_cos=offdiag(S),
                                   en_to_zh_top1=float((S.argmax(1) == np.arange(len(S))).mean()), chance=1 / len(S))
    print("\n[C3] concept vectors (preview only, not a result): "
          + " | ".join(f"{l}: norm/ord {res[l]['norm_ratio_to_ordinary']:.2f}, pairwise cos raw {res[l]['mean_pairwise_cos_raw']:.2f}"
                       for l, _ in LANGS)
          + f" | EN->ZH top1 (centered) {res['cross_centered']['en_to_zh_top1']:.2f} (chance {res['cross_centered']['chance']:.2f})")
    return res


def verdict(out, mode):
    bad, warn = [], []
    e, f, b, c = out["environment"], out.get("functional"), out.get("batch_equivalence"), out.get("concept_vectors")
    if not e["dtype_ok"]:
        bad.append(f"model dtype {e['param_dtype']} != expected {e['expected_dtype']}")
    if not e["attn_ok"]:
        bad.append("Gemma-2 must run with eager attention")
    if f:
        if f["hook_vs_hidden_states_max_abs_diff"] > 1e-2: bad.append("layer hook output != hidden_states[L+1]")
        if f["inject_exact_vs_cast_vector_max_abs_diff"] > 1e-6: bad.append("injected state is not the vector")
        if f["inject_rel_err_vs_fp32_vector"] > 0.01: bad.append("injected state deviates >1% from the fp32 vector")
        if f["layerL_other_positions_rel_change"] > 5e-3 or f["layerL1_before_pos_rel_change"] > 5e-3:
            bad.append("injection leaked to other/earlier positions")
        if f["layerL1_at_pos_rel_change"] < 1e-3 or f["final_logits_max_abs_change"] == 0:
            bad.append("injection has no downstream effect")
    if b:
        for name in ("no_injection", "with_injection"):
            d = b[name]["max_abs_diff"]
            if d > 0.2: bad.append(f"batched vs single NLL differs by {d:.3f} nats ({name})")
            elif d > 0.05: warn.append(f"batched vs single NLL differs by {d:.3f} nats ({name})")
    for lang, s in out.get("position_sweep", {}).items():
        rec = s["recommended_exclude_leading_n"]
        if rec > cfg.EXCLUDE_LEADING_N: bad.append(f"{lang}: config EXCLUDE_LEADING_N={cfg.EXCLUDE_LEADING_N} < systematic sink run {rec}")
        if rec == 0: warn.append(f"{lang}: no systematic sink at position 0 (unexpected)")
    rows = out["carrier"]["candidates"]
    if mode == "pinned":
        if not rows[0]["pass"]: bad.append(f"pinned carrier fails: {rows[0]['flags']}")
    elif not any(r["pass"] for r in rows):
        bad.append("no carrier candidate passes")
    if c:
        for lang, _ in LANGS:
            if c[lang]["mean_pairwise_cos_raw"] > 0.9:
                warn.append(f"{lang}: concept vectors share a dominant direction (pairwise cos {c[lang]['mean_pairwise_cos_raw']:.2f}); "
                            "uncentered cosine metrics will be uninformative, rely on the mismatch matrix")
            if c[lang]["norm_ratio_to_ordinary"] < 0.25:
                warn.append(f"{lang}: concept vectors are small relative to ordinary token states "
                            f"({c[lang]['norm_ratio_to_ordinary']:.2f}x)")
    status = "READY" if not bad else "REVIEW"
    print(f"\n[verdict:{mode}] {out['key']}: {status}")
    for x in bad: print("   FAIL:", x)
    for x in warn: print("   warn:", x)
    return dict(status=status, fail=bad, warn=warn, mode=mode)

def main():
    items = load_items()
    model, tok, default_id = load_model_with_null_pad()
    passages = collect(model, tok, items)
    ref = reference_stats(passages)
    mode = "pinned" if os.environ.get("PINNED_ONLY") else "explore"
    out = {"model": cfg.TARGET_MODEL_NAME, "key": cfg.ACTIVE_MODEL_KEY, "layer_idx": cfg.LAYER_IDX,
           "exclude_leading_n_in_config": cfg.EXCLUDE_LEADING_N, "mode": mode, "environment": environment(model),
           "reference": {k: v for k, v in ref.items() if k not in ("sink_dir", "massive_dims")}
                        | {"n_massive_dims": int(len(ref["massive_dims"]))}}
    if not os.environ.get("SKIP_SWEEP"):
        out["position_sweep"] = sweep(passages, tok)
    out["carrier"] = carrier_candidates(model, tok, default_id, items, ref)

    if not os.environ.get("SKIP_FUNCTIONAL"):
        V = concept_vectors(passages)
        sents = {lang: [tok(it[key], add_special_tokens=False)["input_ids"] for it in items] for lang, key in LANGS}
        out["functional"] = hook_and_injection_tests(model, default_id, passages, V["zh"][0])
        out["batch_equivalence"] = batch_equivalence_test(model, default_id, sents, V["en"][0])
        out["concept_vectors"] = concept_vector_diagnostics(V, ref)

    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "position_sweep" not in out and "position_sweep" in old:
        out["position_sweep"] = old["position_sweep"]            # keep an earlier sweep
    out["verdict"] = verdict(out, mode)

    suffix = "_pinned" if mode == "pinned" else "_single" if os.environ.get("CARRIER_TOKEN") else ""
    suffix += cfg.CARRIER_SFX                            # -> phase3_sanity_<key>_pinned_cdot.json
    out["carrier_tag"] = cfg.CARRIER_TAG or "primary"            # before the write
    out_final = OUT.with_name(OUT.stem + suffix + ".json")       # only the full explore run writes the main file
    out_final.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {out_final}")
    
if __name__ == "__main__":
    main()