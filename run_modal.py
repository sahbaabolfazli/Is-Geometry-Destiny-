#!/usr/bin/env python3
"""run_modal.py - run `bash run_analysis.sh gpu` for the remaining models on Modal (A100-80GB).

One-time setup (local):
    pip install modal && modal setup
    modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxx      # token must have access to the gated
                                                                      # repos (Llama-2, Gemma-2): accept their
                                                                      # licenses on huggingface.co first

Run (from the project folder):
    modal run run_modal.py::main                          # 4 remaining models, in parallel, then download
    modal run run_modal.py::main --sequential             # one A100 at a time
    modal run run_modal.py::main --models "gemma2_9b llama2_7b"
    modal run run_modal.py::main --force                  # re-run models already marked DONE in the volume
    modal run --detach run_modal.py::main                 # survives closing the laptop; fetch later with:
    modal run run_modal.py::download                      # (optional: --include-chroma)
    caffeinate -i modal run run_modal.py::main            # alternative: keep the Mac awake

Design:
  * Each model is ONE container (A100-80GB) executing `MODELS=<key> bash run_analysis.sh gpu`.
    run_analysis.sh hardcodes MODELS, so a patched COPY inside the container makes it overridable;
    your local file is never modified. Models are independent, so they can run in parallel; cost is per
    second, so parallel vs sequential costs the same, only wall-clock differs.
  * Outputs (new/changed files only) are copied to the results Volume under /<model_key>/ as soon as the
    model finishes (or fails). A model with a _DONE.json marker is skipped on the next run (resume).
  * Weights live on a separate HF-cache Volume, so re-runs never re-download them.
  * Download goes straight into LOCAL_DIR, flat, exactly where the local scripts would have written them.
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

# ------------------------------------------------------------------------------------------ config
LOCAL_DIR = Path(os.environ.get("FRAMEWORK_DIR",
                                "/Users/sovereignabundance/Desktop/language_analysis_framework"))
DEFAULT_MODELS = "gemma2_9b gemmax2_28_9b llama2_7b towerbase_7b"      # qwen3_8b / lmt_60_8b already done locally
ALL_MODELS = ("qwen3_8b", "lmt_60_8b", "gemma2_9b", "gemmax2_28_9b", "llama2_7b", "towerbase_7b")
QWEN_MODELS = ("qwen3_8b", "lmt_60_8b")    # run_analysis.sh waives the pinned-sanity READY check for these

GPU = "A100-80GB"
TIMEOUT_S = 6 * 3600                       # per model; Modal kills the container (no cleanup) after this
RESULTS_VOLUME = "language-analysis-results"
HF_VOLUME = "language-analysis-hf-cache"
HF_SECRET = "huggingface-secret"

SRC, WORK, RESULTS, HF_HOME = "/src", "/work", "/results", "/cache/hf"

# Match these to your local `pip freeze` (torch / transformers especially) for the closest reproduction.
PIP_PACKAGES = [
    "torch>=2.3,<2.9", "transformers>=4.51,<5", "accelerate", "sentencepiece", "protobuf",
    "numpy", "scipy", "matplotlib", "chromadb", "huggingface_hub",
]

# Not uploaded: previous outputs / big artefacts the GPU stage never reads.
IGNORE = ["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/venv", "**/.DS_Store", "**/._*",
          "**/*.npz", "**/*.png", "phase3_chroma_db_*", "phase3_chroma_db_*/**", "modal_logs", "modal_logs/**"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl")             # build tools = insurance for chromadb/hnswlib
    .pip_install(*PIP_PACKAGES)
    .env({
        "HF_HOME": HF_HOME,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "ANONYMIZED_TELEMETRY": "False",                        # chromadb telemetry
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # less fragmentation with 256k-vocab logits
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(str(LOCAL_DIR), remote_path=SRC, ignore=IGNORE)   # keep last (copy=False)
)

app = modal.App("language-analysis-gpu")
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)


# ------------------------------------------------------------------------------------------ helpers
def _snapshot(root: Path) -> dict:
    snap = {}
    for p in root.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            st = p.stat()
            snap[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return snap


def _hf_identity() -> str:
    """Normalise the token from the Modal secret and report which HF account it belongs to."""
    from huggingface_hub import HfApi
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip().strip("'\"")
    if not tok:
        raise RuntimeError(f"No HF_TOKEN in the environment: Modal secret '{HF_SECRET}' is missing the key HF_TOKEN.")
    os.environ["HF_TOKEN"] = tok
    try:
        who = HfApi().whoami(token=tok)
    except Exception as e:                                           # noqa: BLE001
        raise RuntimeError(f"Token rejected by Hugging Face (starts '{tok[:7]}...', length {len(tok)}): {e!r}. "
                           f"The Modal secret holds an invalid/old/placeholder token; recreate it with --force.")
    return f"{who.get('name')} (token '{tok[:7]}...', type={who.get('auth', {}).get('accessToken', {}).get('role', '?')})"


def _prefetch(repo: str):
    """Download weights into the HF volume; doubles as a fail-fast auth check."""
    from huggingface_hub import snapshot_download
    sys.stdout.write(f"[hf] authenticated as {_hf_identity()}\n")
    try:
        path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                                       "*.py", "tokenizer*"])
        if not list(Path(path).glob("*.safetensors")):          # repo without safetensors -> fetch everything usable
            path = snapshot_download(repo, ignore_patterns=["*.pt", "*.pth", "original/*", "consolidated*",
                                                            "*.gguf", "*.h5", "*.msgpack", "*.onnx"])
    except Exception as e:
        raise RuntimeError(
            f"Could not download {repo}: {e!r}. If this is a gated repo, accept its license on "
            f"huggingface.co with the account that owns the token stored in Modal secret '{HF_SECRET}'.")
    return path


# ------------------------------------------------------------------------------------------ remote
@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    memory=65536,                                   # MiB host RAM: weights load on CPU in bf16 before .to(cuda)
    cpu=8,
    volumes={RESULTS: results_vol, HF_HOME.rsplit("/", 1)[0]: hf_vol},   # /cache
    secrets=[modal.Secret.from_name(HF_SECRET)],
    retries=0,
)
def run_model(model_key: str, force: bool = False) -> dict:
    t0 = time.time()
    out_dir = Path(RESULTS) / model_key
    results_vol.reload()
    if model_key not in ALL_MODELS:
        return dict(model_key=model_key, status="failed", detail=f"unknown model key; valid: {ALL_MODELS}")
    if (out_dir / "_DONE.json").exists() and not force:
        return dict(model_key=model_key, status="skipped", detail="already complete in volume (use --force)")

    if Path(WORK).exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK)                      # writable working copy; cwd for every script
    work = Path(WORK)

    # run_analysis.sh hardcodes MODELS=...; make it overridable in this COPY only.
    sh_path = work / "run_analysis.sh"
    txt = sh_path.read_text(encoding="utf-8")
    if "${MODELS:-" not in txt:
        txt, n = re.subn(r'^MODELS="([^"]*)"[ \t]*$', lambda m: f'MODELS="${{MODELS:-{m.group(1)}}}"',
                         txt, count=1, flags=re.M)
        if n != 1:
            return dict(model_key=model_key, status="failed", detail="could not patch MODELS= line in run_analysis.sh")
        sh_path.write_text(txt, encoding="utf-8")

    man_path = work / "carrier_manifest.jsonl"
    man_before = man_path.read_text(encoding="utf-8").splitlines() if man_path.exists() else []
    before = _snapshot(work)                        # taken AFTER patching, so only real outputs show up as changed

    log_path = Path("/tmp") / f"{model_key}.log"
    log = open(log_path, "w", encoding="utf-8")
    env = {**os.environ, "MODELS": model_key, "PYTHONUNBUFFERED": "1"}
    result = dict(model_key=model_key, status="failed", detail="")

    def run(cmd, extra_env=None, name=None):
        header = f"\n===== [{model_key}] {name or ' '.join(cmd)} =====\n"
        sys.stdout.write(header); log.write(header)
        p = subprocess.Popen(cmd, cwd=WORK, env={**env, **(extra_env or {})}, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); log.write(line)
        log.flush()
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"step failed (exit {rc}): {name or ' '.join(cmd)}")

    try:
        subprocess.run(["nvidia-smi"], stdout=log, stderr=subprocess.STDOUT)
        run(["python3", "-c", "import torch; assert torch.cuda.is_available(), 'no CUDA'; "
             "print(torch.__version__, torch.cuda.get_device_name(0), "
             "round(torch.cuda.get_device_properties(0).total_memory/2**30), 'GiB')"], name="cuda check")

        # repo id comes from the project's own registry, so there is no second copy to drift
        repo = subprocess.run(["python3", "-c", "import phase3_config as c; print(c.TARGET_MODEL_NAME)"],
                              cwd=WORK, env={**env, "MODEL_KEY": model_key}, capture_output=True, text=True)
        if repo.returncode != 0:
            raise RuntimeError(f"cannot read registry for {model_key}: {repo.stderr[-500:]}")
        repo_id = repo.stdout.strip().splitlines()[-1]
        sys.stdout.write(f"[{model_key}] weights: {repo_id}\n")
        _prefetch(repo_id)
        hf_vol.commit()

        # run_analysis.sh gpu aborts unless phase3_sanity_<key>_pinned.json says READY (Qwen family exempt).
        pinned = work / f"phase3_sanity_{model_key}_pinned.json"
        if model_key not in QWEN_MODELS:
            if not pinned.exists():
                run(["python3", "phase3_sanity_checks.py"],
                    extra_env={"PINNED_ONLY": "1", "SKIP_SWEEP": "1", "MODEL_KEY": model_key},
                    name="pinned sanity check (file was not in the uploaded folder)")
            v = json.loads(pinned.read_text(encoding="utf-8"))["verdict"]
            if v["status"] != "READY":
                raise RuntimeError(f"pinned sanity verdict {v['status']}: fail={v['fail']} warn={v['warn']}")

        run(["bash", "run_analysis.sh", "gpu"], name=f"MODELS={model_key} bash run_analysis.sh gpu")
        result.update(status="ok")
    except Exception as e:                           # noqa: BLE001 - report, keep partial outputs
        result["detail"] = repr(e)
        sys.stdout.write(f"\n[{model_key}] FAILED: {e!r}\n")
        log.write(f"\nFAILED: {e!r}\n")
    finally:
        log.close()
        out_dir.mkdir(parents=True, exist_ok=True)
        after = _snapshot(work)
        changed = sorted(rel for rel, sig in after.items() if before.get(rel) != sig)
        copied = []
        for rel in changed:
            if rel == "carrier_manifest.jsonl":
                continue                             # handled below: only THIS model's appended lines
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work / rel, dst)
            copied.append(rel)
        if man_path.exists():
            new_lines = man_path.read_text(encoding="utf-8").splitlines()[len(man_before):]
            (out_dir / "_manifest_lines.jsonl").write_text("\n".join(new_lines) + ("\n" if new_lines else ""),
                                                           encoding="utf-8")
        shutil.copy2(log_path, out_dir / "_run.log")
        result.update(runtime_min=round((time.time() - t0) / 60, 1), n_files=len(copied))
        marker = out_dir / ("_DONE.json" if result["status"] == "ok" else "_FAILED.json")
        stale = out_dir / ("_FAILED.json" if result["status"] == "ok" else "_DONE.json")
        marker.write_text(json.dumps({**result, "files": copied}, indent=2), encoding="utf-8")
        stale.unlink(missing_ok=True)
        for attempt in range(3):
            try:
                results_vol.commit(); hf_vol.commit(); break
            except Exception as e:                   # noqa: BLE001
                sys.stdout.write(f"[commit retry {attempt}] {e!r}\n"); time.sleep(5)
    return result


@app.function(image=image, timeout=600, secrets=[modal.Secret.from_name(HF_SECRET)])
def check_hf(models: list) -> list:
    """CPU-only, costs ~nothing: which HF account does the Modal secret use, and can it read each gated repo?"""
    from huggingface_hub import hf_hub_download
    lines = []
    try:
        lines.append(f"identity: {_hf_identity()}")
    except Exception as e:                                           # noqa: BLE001
        return [f"FAIL identity: {e}"]
    for k in models:
        r = subprocess.run([sys.executable, "-c", "import phase3_config as c; print(c.TARGET_MODEL_NAME)"],
                           cwd=SRC, env={**os.environ, "MODEL_KEY": k}, capture_output=True, text=True)
        if r.returncode != 0:
            lines.append(f"FAIL {k}: cannot read registry: {r.stderr[-200:]}"); continue
        repo = r.stdout.strip().splitlines()[-1]
        try:
            hf_hub_download(repo, "config.json", cache_dir="/tmp/hfcheck")
            lines.append(f"OK   {k:14s} {repo}")
        except Exception as e:                                       # noqa: BLE001
            lines.append(f"FAIL {k:14s} {repo}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    return lines


@app.local_entrypoint()
def check(models: str = DEFAULT_MODELS):
    """modal run run_modal.py::check   -> verify the HF token/licences before spending GPU time."""
    for line in check_hf.remote(models.replace(",", " ").split()):
        print(line)


# ------------------------------------------------------------------------------------------ local
def _download(local_dir: Path, include_chroma: bool = False) -> int:
    vol = modal.Volume.from_name(RESULTS_VOLUME)
    FILE, DIRECTORY = modal.volume.FileEntryType.FILE, modal.volume.FileEntryType.DIRECTORY
    local_dir.mkdir(parents=True, exist_ok=True)
    tops = sorted(e.path.strip("/") for e in vol.listdir("/") if e.type == DIRECTORY)
    tops = [t for t in tops if not t.startswith("_")]
    if not tops:
        print("[download] results volume is empty"); return 0
    n_new = n_skip = 0
    manifest_lines = []
    for top in tops:
        for e in vol.listdir(top, recursive=True):
            if e.type != FILE:
                continue
            rel = e.path.strip("/")
            rel = rel[len(top) + 1:] if rel.startswith(top + "/") else rel
            if rel == "_manifest_lines.jsonl":
                buf = io.BytesIO(); vol.read_file_into_fileobj(e.path, buf)
                manifest_lines += [l for l in buf.getvalue().decode("utf-8").splitlines() if l.strip()]
                continue
            if rel == "_run.log":
                dest = local_dir / "modal_logs" / f"{top}.log"
            elif rel.startswith("_"):
                continue                                              # _DONE.json / _FAILED.json markers
            elif rel.startswith("phase3_chroma_db_") and not include_chroma:
                continue
            else:
                dest = local_dir / rel
            if dest.exists() and dest.stat().st_size == e.size and rel != "_run.log":
                n_skip += 1; continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            with open(tmp, "wb") as f:
                vol.read_file_into_fileobj(e.path, f)
            if tmp.stat().st_size != e.size:
                raise RuntimeError(f"size mismatch downloading {e.path}: {tmp.stat().st_size} != {e.size}")
            tmp.replace(dest)
            n_new += 1
            print(f"  [{top}] {rel}  ({e.size / 1e6:.1f} MB)")
    if manifest_lines:                                                # merge, never overwrite, the local audit log
        mp = local_dir / "carrier_manifest.jsonl"
        have = set(mp.read_text(encoding="utf-8").splitlines()) if mp.exists() else set()
        add = [l for l in manifest_lines if l not in have]
        if add:
            with open(mp, "a", encoding="utf-8") as f:
                f.write("\n".join(add) + "\n")
    print(f"[download] {n_new} file(s) written, {n_skip} already up to date -> {local_dir}"
          + ("" if include_chroma else "   (Chroma DBs left in the volume; use --include-chroma)"))
    return n_new


@app.local_entrypoint()
def main(models: str = DEFAULT_MODELS, sequential: bool = False, force: bool = False,
         include_chroma: bool = False, local_dir: str = str(LOCAL_DIR)):
    keys = models.replace(",", " ").split()
    bad = [k for k in keys if k not in ALL_MODELS]
    if bad:
        sys.exit(f"unknown model key(s) {bad}; valid: {ALL_MODELS}")
    need = ["run_analysis.sh", "phase2_final_48_local.json", "phase3_config.py", "analysis_common.py", "scoring.py",
            "phase3_extraction_pipeline.py", "compute_density_proxy.py", "phase3b_latent_injection.py",
            "process_phase3b_stats.py", "phase3d_interference_matrix.py", "phase3_sanity_checks.py"]
    missing = [f for f in need if not (LOCAL_DIR / f).exists()]
    if missing:
        sys.exit(f"missing in {LOCAL_DIR}: {missing}")
    for k in keys:
        if k not in QWEN_MODELS and not (LOCAL_DIR / f"phase3_sanity_{k}_pinned.json").exists():
            print(f"[note] {k}: no local phase3_sanity_{k}_pinned.json; the pinned sanity check will run in the cloud first.")
    print(f"[modal] {GPU} x {len(keys)} model(s), {'sequential' if sequential else 'parallel'}: {keys}")

    if sequential:
        results = []
        for k in keys:
            try:
                results.append(run_model.remote(k, force))
            except Exception as e:                                   # noqa: BLE001
                results.append(dict(model_key=k, status="failed", detail=repr(e)))
    else:
        raw = list(run_model.starmap([(k, force) for k in keys], return_exceptions=True))
        results = [r if isinstance(r, dict) else dict(model_key=k, status="failed", detail=repr(r))
                   for k, r in zip(keys, raw)]

    print("\n[modal] summary")
    for r in results:
        print(f"  {r['model_key']:15s} {r['status']:8s} {r.get('runtime_min', '-')} min  {r.get('detail', '')}")
    _download(Path(local_dir), include_chroma)                      # fetch whatever exists, even after failures
    if any(r["status"] == "failed" for r in results):
        sys.exit("some models failed; see modal_logs/<model>.log (partial outputs were downloaded)")


@app.local_entrypoint()
def download(include_chroma: bool = False, local_dir: str = str(LOCAL_DIR)):
    """Fetch everything from the results volume (use after `modal run --detach`)."""
    _download(Path(local_dir), include_chroma)