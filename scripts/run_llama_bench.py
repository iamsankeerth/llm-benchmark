"""
llama-bench Integration Runner
=============================
Runs llama-bench against all GGUF-backed models in the queue.
This REPLACES the custom TPS measurement with the industry-standard
llama-bench tool for raw hardware performance benchmarking.

Usage:
    python scripts/run_llama_bench.py              # Bench all pending models
    python scripts/run_llama_bench.py --dry-run     # Show what would run
    python scripts/run_llama_bench.py --model tinyllama  # Single model
"""

import os
import sys
import json
import subprocess
import glob
import time
import argparse
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["ANSI_COLORS_DISABLED"] = "1"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_QUEUE
from src.artifact_store import BenchmarkArtifactStore
from src.lifecycle import RuntimeHandle, acquire_runtime, cleanup_runtime
from src.model_entry import as_model_entry

# ─── Configuration ───────────────────────────────────────────────────────────

# Path to llama-bench.exe — auto-detected from tools/ or set manually
LLAMA_BENCH_EXE = os.environ.get("LLAMA_BENCH_EXE", "")
if not LLAMA_BENCH_EXE:
    _auto = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tools", "llama-bench", "bin", "llama-bench.exe")
    if os.path.isfile(_auto):
        LLAMA_BENCH_EXE = _auto

# Ollama model storage location (blobs are raw GGUF files)
OLLAMA_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".ollama", "models")

# Benchmark settings tuned for RTX 2050 (4GB VRAM)
BENCH_SETTINGS = {
    "n_prompt": 512,       # Tokens for prompt processing (pp) test
    "n_gen": 128,          # Tokens for text generation (tg) test
    "n_gpu_layers": 99,    # Full GPU offload
    "flash_attn": 1,       # Enable flash attention
    "repetitions": 3,      # Statistical averaging (5 is default, 3 is faster)
    "batch_size": 2048,    # Default batch size
}

artifact_store = BenchmarkArtifactStore()
PERF_DIR = str(artifact_store.perf_dir)
artifact_store.perf_dir.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[llama-bench] {msg}")


def find_ollama_gguf(ollama_tag: str) -> str | None:
    """
    Locate the GGUF file for an Ollama model.
    Ollama stores models as blobs in ~/.ollama/models/blobs/
    The manifest maps tags → digests → blob files.
    """
    # Parse tag → name:tag
    parts = ollama_tag.split(":")
    name = parts[0]
    tag = parts[1] if len(parts) > 1 else "latest"

    # Handle library/ prefix
    if "/" not in name:
        manifest_dir = os.path.join(OLLAMA_MODELS_DIR, "manifests", "registry.ollama.ai", "library", name)
    else:
        namespace, model_name = name.split("/", 1)
        manifest_dir = os.path.join(OLLAMA_MODELS_DIR, "manifests", "registry.ollama.ai", namespace, model_name)

    manifest_path = os.path.join(manifest_dir, tag)
    if not os.path.isfile(manifest_path):
        return None

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Find the model layer (mediaType contains "model")
        for layer in manifest.get("layers", []):
            media_type = layer.get("mediaType", "")
            if "model" in media_type:
                digest = layer["digest"]
                # Digest format: sha256:abc123... → blob file: sha256-abc123...
                blob_filename = digest.replace(":", "-")
                blob_path = os.path.join(OLLAMA_MODELS_DIR, "blobs", blob_filename)
                if os.path.isfile(blob_path):
                    return blob_path
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        pass

    return None


def find_hf_gguf(model_entry: dict) -> str | None:
    """Find GGUF file from HuggingFace download directory."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    requested_name = model_entry.get("requested_name", "").lower().replace(" ", "-")
    local_dir = os.path.join(models_dir, requested_name.replace(":", "_").replace("/", "_"))

    if os.path.isdir(local_dir):
        gguf_files = glob.glob(os.path.join(local_dir, "*.gguf"))
        if gguf_files:
            return gguf_files[0]

    # Also check HF cache
    hf_repo = model_entry.get("hf_repo", "")
    if hf_repo:
        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        repo_dir_name = f"models--{hf_repo.replace('/', '--')}"
        repo_cache = os.path.join(hf_cache, repo_dir_name)
        if os.path.isdir(repo_cache):
            for root, dirs, files in os.walk(repo_cache):
                for f in files:
                    if f.endswith(".gguf"):
                        return os.path.join(root, f)

    return None


def resolve_gguf_path(model_entry: dict) -> str | None:
    """
    Resolve the GGUF file path for a model, checking Ollama blobs first,
    then HF downloads.
    """
    runtime = model_entry.get("resolved_runtime", "")

    # Ollama models: check blob storage
    if runtime in ("ollama", "huggingface_gguf"):
        ollama_tag = model_entry.get("ollama_tag") or model_entry.get("resolved_model_ref", "")
        if ollama_tag:
            path = find_ollama_gguf(ollama_tag)
            if path:
                return path

    # HF GGUF models: check download directory
    if runtime == "huggingface_gguf":
        path = find_hf_gguf(model_entry)
        if path:
            return path

    return None


def run_llama_bench(gguf_path: str, model_name: str, settings: dict = None) -> dict | None:
    """
    Execute llama-bench and return parsed JSON results.

    Returns dict with keys: pp_tps, tg_tps, pp_stddev, tg_stddev, raw_results
    """
    if not LLAMA_BENCH_EXE or not os.path.isfile(LLAMA_BENCH_EXE):
        log(f"ERROR: llama-bench.exe not found at: {LLAMA_BENCH_EXE}")
        log(f"  Set LLAMA_BENCH_EXE env var or place binary in tools/llama-bench/bin/")
        return None

    s = {**BENCH_SETTINGS, **(settings or {})}

    cmd = [
        LLAMA_BENCH_EXE,
        "-m", gguf_path,
        "-p", str(s["n_prompt"]),
        "-n", str(s["n_gen"]),
        "-ngl", str(s["n_gpu_layers"]),
        "-fa", str(s["flash_attn"]),
        "-r", str(s["repetitions"]),
        "-b", str(s["batch_size"]),
        "-o", "json",
        "--progress",
    ]

    log(f"Running: {' '.join(cmd[-10:])}")  # Show last 10 args for readability

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,  # 10 minute timeout per model
        )
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT: llama-bench exceeded 10 minutes for {model_name}")
        return None

    if result.returncode != 0:
        log(f"  ERROR (exit {result.returncode}): {result.stderr[-500:]}")
        return None

    # Parse JSON output from stdout
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        log(f"  ERROR: Could not parse JSON output")
        log(f"  stdout: {result.stdout[:500]}")
        return None

    # Extract pp and tg metrics
    pp_result = None
    tg_result = None

    for entry in raw:
        if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
            pp_result = entry
        elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
            tg_result = entry

    parsed = {
        "model_name": model_name,
        "gguf_path": gguf_path,
        "timestamp": datetime.now().isoformat(),
        "gpu_info": raw[0].get("gpu_info", "unknown") if raw else "unknown",
        "cpu_info": raw[0].get("cpu_info", "unknown") if raw else "unknown",
        "backend": raw[0].get("backends", "unknown") if raw else "unknown",
        "model_type": raw[0].get("model_type", "unknown") if raw else "unknown",
        "model_size_bytes": raw[0].get("model_size", 0) if raw else 0,
        "model_params": raw[0].get("model_n_params", 0) if raw else 0,
        "n_gpu_layers": s["n_gpu_layers"],
        "flash_attn": bool(s["flash_attn"]),
        "pp_tokens": s["n_prompt"],
        "tg_tokens": s["n_gen"],
        "repetitions": s["repetitions"],
    }

    if pp_result:
        parsed["pp_tps"] = round(pp_result["avg_ts"], 2)
        parsed["pp_stddev"] = round(pp_result["stddev_ts"], 2)
        parsed["pp_samples"] = pp_result.get("samples_ts", [])
    else:
        parsed["pp_tps"] = 0
        parsed["pp_stddev"] = 0
        parsed["pp_samples"] = []

    if tg_result:
        parsed["tg_tps"] = round(tg_result["avg_ts"], 2)
        parsed["tg_stddev"] = round(tg_result["stddev_ts"], 2)
        parsed["tg_samples"] = tg_result.get("samples_ts", [])
    else:
        parsed["tg_tps"] = 0
        parsed["tg_stddev"] = 0
        parsed["tg_samples"] = []

    parsed["raw_results"] = raw

    return parsed


def download_model(model_entry: dict) -> RuntimeHandle | None:
    """
    Acquire the model runtime to ensure the GGUF blob exists on disk.
    Returns a runtime handle on success, None on failure.
    """
    return acquire_runtime(model_entry, log=log)


def delete_model(ollama_tag: str):
    """Delete a model from Ollama to free disk space."""
    cleanup_runtime(RuntimeHandle(ollama_tag, None, {"resolved_runtime": "ollama"}), log=log)


def run_all_benchmarks(dry_run=False, single_model=None):
    """
    Run llama-bench on all eligible models using ephemeral lifecycle:
      1. Download model (ollama pull)
      2. Find GGUF blob on disk
      3. Run llama-bench
      4. Save results
      5. Delete model (ollama rm)
    """
    log("=" * 60)
    log("llama-bench Performance Benchmark")
    log(f"GPU: RTX 2050 | Settings: pp={BENCH_SETTINGS['n_prompt']}, tg={BENCH_SETTINGS['n_gen']}, ngl={BENCH_SETTINGS['n_gpu_layers']}, fa={BENCH_SETTINGS['flash_attn']}")
    log("Lifecycle: Download → Bench → Delete (one model at a time)")
    log("=" * 60)

    # Identify eligible models (Ollama or GGUF-backed)
    eligible = []
    for m in MODEL_QUEUE:
        entry = as_model_entry(m)
        if not entry.is_runnable:
            continue
        if not entry.is_runtime("ollama", "huggingface_gguf"):
            continue
        if single_model and single_model.lower() not in entry.requested_name.lower():
            continue
        eligible.append(m)

    log(f"\nFound {len(eligible)} eligible models")

    if dry_run:
        log("\n--- DRY RUN MODE ---")
        for m in eligible:
            entry = as_model_entry(m)
            tag = entry.ollama_tag or entry.resolved_model_ref or "?"
            log(f"  Would bench: {entry.requested_name} (tag: {tag})")
        return

    # Run benchmarks with ephemeral lifecycle
    all_results = {}
    for idx, m in enumerate(eligible, 1):
        entry = as_model_entry(m)
        name = entry.requested_name
        queue_id = entry.queue_id
        
        target_model = os.environ.get("BENCHMARK_SINGLE_MODEL")
        if target_model and queue_id != target_model:
            continue
            
        skip_lifecycle = os.environ.get("BENCHMARK_SKIP_LIFECYCLE") == "1"
        
        ollama_tag = entry.ollama_tag or entry.resolved_model_ref

        log(f"\n[{idx}/{len(eligible)}] {name}")

        # Check if already benchmarked
        result_file = artifact_store.llama_bench_result_path(queue_id)
        if result_file.is_file():
            log(f"  Already benchmarked, skipping (delete {result_file.name} to re-run)")
            existing = artifact_store.load_llama_bench_result(queue_id)
            if existing:
                all_results[queue_id] = existing
            continue

        # ── STEP 1: DOWNLOAD ──
        if skip_lifecycle:
            runtime_handle = RuntimeHandle(ollama_tag, None, m)
            log(f"  [Orchestrator] Skipping download, assuming {runtime_handle.model_ref} is ready.")
        else:
            runtime_handle = download_model(m)
            
        if not runtime_handle:
            log(f"  ✗ Skipping (download failed)")
            continue

        try:
            # ── STEP 2: FIND GGUF ──
            gguf_path = resolve_gguf_path(m)
            if not gguf_path:
                log(f"  ✗ No GGUF blob found after download")
                continue

            size_gb = os.path.getsize(gguf_path) / (1024**3)
            log(f"  GGUF: {gguf_path} ({size_gb:.2f} GB)")

            # ── STEP 3: BENCH + STORE ──
            start = time.perf_counter()
            result = run_llama_bench(gguf_path, name)
            elapsed = time.perf_counter() - start

            if result:
                result["queue_id"] = queue_id
                result["bench_duration_sec"] = round(elapsed, 1)

                artifact_store.save_llama_bench_result(queue_id, result)

                all_results[queue_id] = result

                log(f"  ✓ pp={result['pp_tps']} t/s (±{result['pp_stddev']})")
                log(f"  ✓ tg={result['tg_tps']} t/s (±{result['tg_stddev']})")
                log(f"  ✓ Completed in {elapsed:.1f}s")
            else:
                log(f"  ✗ Benchmark failed")

        except Exception as e:
            log(f"  ✗ Unexpected error: {e}")
        finally:
            # ── STEP 4: DELETE (always runs) ──
            if not skip_lifecycle:
                cleanup_runtime(runtime_handle, log=log)

    # Save summary
    artifact_store.save_llama_bench_summary(BENCH_SETTINGS, all_results)

    # Print summary table
    log("\n" + "=" * 80)
    log(f"{'Model':<30} {'pp t/s':>10} {'tg t/s':>10} {'Size':>8}")
    log("-" * 80)
    for qid, r in sorted(all_results.items(), key=lambda x: x[1].get("tg_tps", 0), reverse=True):
        name = r.get("model_name", qid)[:28]
        pp = f"{r['pp_tps']:.1f}" if r.get("pp_tps") else "N/A"
        tg = f"{r['tg_tps']:.1f}" if r.get("tg_tps") else "N/A"
        size = f"{r.get('model_size_bytes', 0) / (1024**3):.1f}G"
        log(f"  {name:<28} {pp:>10} {tg:>10} {size:>8}")
    log("=" * 80)
    log(f"Results saved to: {PERF_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run llama-bench on all GGUF models")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without running")
    parser.add_argument("--model", type=str, help="Filter to a specific model name")
    args = parser.parse_args()

    run_all_benchmarks(dry_run=args.dry_run, single_model=args.model)
