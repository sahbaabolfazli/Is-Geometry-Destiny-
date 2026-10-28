#!/usr/bin/env python3
"""
Phase 3b: Latent injection into a NULL carrier, layer-by-layer cosine-similarity
decay curves, noise-control validation, multi-level statistical reporting,
keyed with ACTIVE_MODEL_KEY.

Controls run for every concept (same carrier / layer / position):
  * noise_control    : random Gaussian vector, norm-matched to the concept's EN vector
                       (out-of-distribution, so "real > noise" is almost guaranteed)
  * mismatch_control : the EN vector of a DIFFERENT concept (next id in sorted order, wrapping).
                       In-distribution and norm-realistic, so retention beyond it is evidence that
                       survival is not just a generic property of any concept-like vector.
"""

import json
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import chromadb
import phase3_config as cfg

NULL_PAD_TOKEN = "<NULL_PAD>"
N_NULL_TOKENS = 10
INJECTION_POSITION = 4
MAX_NEW_TOKENS = 150
CONCEPT_IDS_TO_RUN = None
REPORT_PATH = Path(f"phase3b_injection_report_{cfg.ACTIVE_MODEL_KEY}.json")
STATS_REPORT_PATH = Path(f"phase3b_significance_report_{cfg.ACTIVE_MODEL_KEY}.json")
PHASE2_INPUT_PATH = Path("phase2_final_48_local.json")

RNG = np.random.default_rng(cfg.RANDOM_SEED)


def load_model_with_null_pad():
  tokenizer = AutoTokenizer.from_pretrained(
      cfg.TARGET_MODEL_NAME, trust_remote_code=True
  )
  if tokenizer.pad_token_id is not None:
    null_pad_id = tokenizer.pad_token_id
    chosen_token = tokenizer.pad_token or "PAD"
  elif tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
    null_pad_id = tokenizer.eos_token_id
    chosen_token = tokenizer.eos_token
  else:
    tokenizer.pad_token = tokenizer.unk_token
    null_pad_id = tokenizer.unk_token_id
    chosen_token = tokenizer.unk_token

  if cfg.CARRIER_TOKEN:
    null_pad_id = cfg.resolve_token_id(tokenizer, cfg.CARRIER_TOKEN)
    if null_pad_id is None:
      raise ValueError(f"CARRIER_TOKEN {cfg.CARRIER_TOKEN!r} not usable for {cfg.TARGET_MODEL_NAME}")
    chosen_token = tokenizer.convert_ids_to_tokens(null_pad_id)
  print(
      f"[config] Using universal pad token: '{chosen_token}' (ID:"
      f" {null_pad_id})"
  )
  dtype = torch.bfloat16 if cfg.DEVICE == "mps" else "auto"
  model = cfg.load_causal_lm()
  cfg.assert_layers_exist(model)
  return model, tokenizer, null_pad_id


def build_null_carrier(tokenizer, null_pad_id):
  input_ids = torch.full(
      (1, N_NULL_TOKENS), null_pad_id, dtype=torch.long, device=cfg.DEVICE
  )
  attention_mask = torch.ones(
      (1, N_NULL_TOKENS), dtype=torch.long, device=cfg.DEVICE
  )
  return input_ids, attention_mask


class InjectionHook:

  def __init__(self, position, injected_vector):
    self.position = position
    self.injected_vector = torch.tensor(injected_vector, dtype=torch.float32)
    self.fired = False

  def hook_fn(self, module, inputs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    if self.fired or self.position >= hidden.shape[1]:
      return output
    injected = self.injected_vector.to(hidden.dtype).to(hidden.device)
    hidden = hidden.clone()
    hidden[0, self.position, :] = injected
    self.fired = True
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

  def reset(self):
    self.fired = False


class ProbeHooks:

  def __init__(self, position):
    self.position = position
    self.captured = {}

  def make_hook(self, layer_idx):
    def hook_fn(module, inputs, output):
      hidden = output[0] if isinstance(output, tuple) else output
      if self.position >= hidden.shape[1]:
        return output
      self.captured[layer_idx] = (
          hidden.clone()[0, self.position, :]
          .detach()
          .to(torch.float32)
          .cpu()
          .numpy()
      )
      return output

    return hook_fn

  def reset(self):
    self.captured = {}


def cosine_similarity(a, b):
  a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
  return float(
      np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
  )


def run_one_injection(model, tokenizer, null_pad_id, injected_vector):
  input_ids, attention_mask = build_null_carrier(tokenizer, null_pad_id)
  injection_hook = InjectionHook(INJECTION_POSITION, injected_vector)
  probes = ProbeHooks(INJECTION_POSITION)
  handles = [
      model.model.layers[cfg.LAYER_IDX].register_forward_hook(
          injection_hook.hook_fn
      )
  ]
  handles += [
      model.model.layers[L].register_forward_hook(probes.make_hook(L))
      for L in cfg.PROBE_LAYERS
  ]
  try:
    injection_hook.reset()
    probes.reset()
    with torch.no_grad():
      model(input_ids=input_ids, attention_mask=attention_mask)

    decay_curve = {cfg.LAYER_IDX: 1.0}
    for L in cfg.PROBE_LAYERS:
      decay_curve[L] = cosine_similarity(probes.captured[L], injected_vector)

    injection_hook.reset()
    with torch.no_grad():
      generated_ids = model.generate(
          input_ids=input_ids,
          attention_mask=attention_mask,
          max_new_tokens=MAX_NEW_TOKENS,
          do_sample=False,
          pad_token_id=null_pad_id,
      )
    generated_text = tokenizer.decode(
        generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    )
  finally:
    for h in handles:
      h.remove()

  return decay_curve, generated_text


def make_noise_control_vector(dim, reference_vector):
  noise = RNG.normal(size=dim)
  target_norm = float(np.linalg.norm(reference_vector))
  return noise * (target_norm / (np.linalg.norm(noise) + 1e-12))


def main():
  cfg.require_primary_carrier("phase3b_latent_injection.py")
  model, tokenizer, null_pad_id = load_model_with_null_pad()

  items_raw = json.loads(PHASE2_INPUT_PATH.read_text(encoding="utf-8"))
  domain_map = {
      r["id"]: r.get("domain", "unknown")
      for r in items_raw
      if r["verification_status"]
      in ("APPROVED_FIRST_PASS", "APPROVED_AFTER_REVISION")
  }

  client = chromadb.PersistentClient(path=cfg.CHROMA_DB_PATH)
  concept_en = client.get_collection("concept_en_geometry")
  concept_zh = client.get_collection("concept_zh_geometry")

  en_all = concept_en.get(include=["embeddings", "metadatas"])
  zh_all = concept_zh.get(include=["embeddings", "metadatas"])
  en_by_id = {
      m["sentence_id"]: np.array(v)
      for m, v in zip(en_all["metadatas"], en_all["embeddings"])
  }
  zh_by_id = {
      m["sentence_id"]: np.array(v)
      for m, v in zip(zh_all["metadatas"], zh_all["embeddings"])
  }

  ids_to_run = CONCEPT_IDS_TO_RUN or sorted(set(en_by_id) & set(zh_by_id))
  print(f"[info] Running latent injection on {len(ids_to_run)} concept(s).")

  report = []
  for concept_id in ids_to_run:
    entry = {
        "id": concept_id,
        "domain": domain_map.get(concept_id, "unknown"),
    }

    for lang, vectors in (("en", en_by_id), ("zh", zh_by_id)):
      decay, text = run_one_injection(
          model, tokenizer, null_pad_id, vectors[concept_id]
      )
      entry[lang] = {"decay_curve": decay, "generated_text": text}

    noise_vec = make_noise_control_vector(
        en_by_id[concept_id].shape[0], en_by_id[concept_id]
    )
    noise_decay, noise_text = run_one_injection(
        model, tokenizer, null_pad_id, noise_vec
    )
    entry["noise_control"] = {
        "decay_curve": noise_decay,
        "generated_text": noise_text,
    }

    # In-distribution control: another concept's EN vector (next id in sorted order, wrapping).
    # Needs >= 2 concepts, otherwise it would be the matched vector itself.
    if len(ids_to_run) > 1:
      other_id = ids_to_run[(ids_to_run.index(concept_id) + 1) % len(ids_to_run)]
      mm_decay, mm_text = run_one_injection(
          model, tokenizer, null_pad_id, en_by_id[other_id]
      )
      entry["mismatch_control"] = {
          "decay_curve": mm_decay,
          "generated_text": mm_text,
          "vector_of": other_id,
      }
    report.append(entry)

  REPORT_PATH.write_text(
      json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(f"[done] Wrote injection report to {REPORT_PATH}")

if __name__ == "__main__":
  main()