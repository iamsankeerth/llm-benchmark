import sys
import os
import shutil
import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    os.environ['ANSI_COLORS_DISABLED'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Force UTF-8 output on Windows
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Enable flash attention for better GPU performance
os.environ.setdefault('OLLAMA_FLASH_ATTENTION', '1')

import re
def strip_ansi(text):
    """Remove ANSI escape codes from text."""
    if text is None:
        return ""
    ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_pattern.sub('', text)

def log(msg, end=None, flush=False):
    """Simple print wrapper with optional end parameter."""
    if end:
        print(msg, end=end, flush=flush)
    else:
        print(msg)
    sys.stdout.flush()

from src.benchmarker import Benchmarker
from src.structured_output import StructuredOutputTester
from src.model_comparator import ModelComparator
from config import MODEL_QUEUE, PROMPTS_DIR, RESULTS_DIR, TEMPS_TO_TEST, MODELS_DIR, PHASE2_PROMPT_LIMIT
try:
    from config import SMOKE_RUN_PROMPTS, SLOW_MODEL_THRESHOLD_SECS
except ImportError:
    SMOKE_RUN_PROMPTS = 0
    SLOW_MODEL_THRESHOLD_SECS = 0
from scripts.test_tracker import TestTracker

# Optional runtime backends — graceful fallback if not installed
try:
    from src.hf_runner import HFTransformersRunner
    HAS_HF_RUNNER = True
except ImportError:
    HAS_HF_RUNNER = False

try:
    from src.vllm_runner import VLLMRunner
    HAS_VLLM_RUNNER = True
except ImportError:
    HAS_VLLM_RUNNER = False

tracker = TestTracker()

def _phase2_limit() -> int:
    """Return the effective Phase 2 limit. None means all prompts."""
    if PHASE2_PROMPT_LIMIT is None:
        return 2**62
    return PHASE2_PROMPT_LIMIT

def get_free_disk_space_gb() -> float:
    """Get free disk space in GB for the drive containing MODELS_DIR."""
    import shutil
    target = MODELS_DIR if os.path.exists(MODELS_DIR) else BASE_DIR
    total, used, free = shutil.disk_usage(target)
    return free / (1024 ** 3)

def clear_huggingface_cache():
    """Clear HuggingFace downloads from models directory to free disk space."""
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    log(f"Cleared HuggingFace cache: {item}")
            except Exception as e:
                log(f"Warning: Could not clear {item}: {e}")

def check_disk_space(required_gb: float = 5.0) -> bool:
    """Check if sufficient disk space is available. Returns True if OK."""
    free_gb = get_free_disk_space_gb()
    if free_gb < required_gb:
        log(f"Low disk space: {free_gb:.2f} GB free (need {required_gb} GB)")
        log("Clearing HuggingFace cache...")
        clear_huggingface_cache()
        free_gb = get_free_disk_space_gb()
        log(f"After cleanup: {free_gb:.2f} GB free")
    if free_gb < required_gb:
        log(f"ERROR: Insufficient disk space ({free_gb:.2f} GB). Skipping model.")
        return False
    return True

def check_ollama_model_exists(model_tag: str) -> bool:
    """Check if model exists in Ollama using exact name matching."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            return False
        # Parse each line: "NAME    ID    SIZE    MODIFIED"
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if not parts:
                continue
            listed_name = parts[0]  # e.g. "qwen2.5:3b-instruct"
            # Exact match or match with :latest suffix
            if listed_name == model_tag or listed_name == f"{model_tag}:latest":
                return True
            # Handle case where model_tag has no tag (implies :latest)
            if ':' not in model_tag and listed_name == f"{model_tag}:latest":
                return True
        return False
    except Exception:
        return False

def download_model(model_entry: dict) -> tuple:
    """
    Download/load model based on its resolved_runtime.
    Returns (model_ref, runtime_client) tuple, or (None, None) on failure.

    runtime_client is:
      - None for Ollama (uses the shared benchmarker.client)
      - HFTransformersRunner instance for hf_transformers
      - VLLMRunner instance for vllm
    """
    source = model_entry.get("source", "ollama")
    resolved_runtime = model_entry.get("resolved_runtime", "")

    if source == "provider_unsupported" or model_entry.get("status") == "provider_unsupported":
        log(f"  SKIP: provider_unsupported")
        return None, None

    # --- Ollama runtime ---
    if resolved_runtime == "ollama":
        ollama_tag = model_entry.get("ollama_tag") or model_entry.get("resolved_model_ref", "")

        if check_ollama_model_exists(ollama_tag):
            log(f"  Model {ollama_tag} already exists in Ollama")
            return ollama_tag, None

        log(f"  Pulling {ollama_tag} from Ollama...")
        result = subprocess.run(
            ["ollama", "pull", ollama_tag],
            capture_output=True, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            log(f"  Successfully downloaded {ollama_tag}")
        else:
            log(f"  Pull output: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")

        if check_ollama_model_exists(ollama_tag):
            return ollama_tag, None
        else:
            log(f"  Failed to pull {ollama_tag}")
            return None, None

    # --- HuggingFace GGUF runtime (download GGUF → create in Ollama) ---
    if resolved_runtime == "huggingface_gguf":
        hf_repo = model_entry.get("hf_repo", "")
        ollama_tag = model_entry.get("requested_name", "").lower().replace(" ", "-")

        if check_ollama_model_exists(ollama_tag):
            log(f"  Model {ollama_tag} already exists in Ollama (from GGUF)")
            return ollama_tag, None

        if not hf_repo:
            log(f"  SKIP: No hf_repo for GGUF download")
            return None, None

        if not check_disk_space(required_gb=5.0):
            return None, None

        log(f"  Downloading {hf_repo} from HuggingFace (GGUF)...")
        local_dir = os.path.join(MODELS_DIR, ollama_tag.replace(':', '_').replace('/', '_'))
        result = subprocess.run(
            [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
             "download", hf_repo, "--local-dir", local_dir,
             "--local-dir-use-symlinks", "False"],
            capture_output=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            log(f"  HuggingFace download error: {result.stderr[-500:]}")
            return None, None

        gguf_files = [f for f in os.listdir(local_dir) if f.endswith('.gguf')]
        if not gguf_files:
            log(f"  SKIP: No GGUF file found in {hf_repo}")
            try:
                shutil.rmtree(local_dir)
            except Exception:
                pass
            return None, None

        modelfile_path = os.path.join(local_dir, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(f"FROM ./{gguf_files[0]}")

        log(f"  Creating Ollama model from GGUF: {gguf_files[0]}")
        subprocess.run(
            ["ollama", "create", ollama_tag, "-f", modelfile_path],
            encoding='utf-8', errors='replace'
        )

        if check_ollama_model_exists(ollama_tag):
            log(f"  Successfully loaded {ollama_tag} from GGUF")
            return ollama_tag, None
        else:
            log(f"  Failed to create Ollama model from GGUF")
            return None, None

    # --- HuggingFace Transformers runtime ---
    if resolved_runtime == "hf_transformers":
        if not HAS_HF_RUNNER:
            log(f"  SKIP: HF Transformers runner not installed (pip install transformers accelerate bitsandbytes)")
            return None, None

        hf_repo = model_entry.get("hf_repo", "")
        if not hf_repo:
            log(f"  SKIP: No hf_repo for Transformers download")
            return None, None

        log(f"  Loading {hf_repo} via HuggingFace Transformers (4-bit)...")
        runner = HFTransformersRunner()
        is_vision = model_entry.get("category", "") == "Vision"
        error = runner.load_model(hf_repo, is_vision=is_vision)
        if error:
            log(f"  HF load error: {error}")
            return None, None

        log(f"  Successfully loaded {hf_repo} via Transformers")
        return hf_repo, runner

    # --- vLLM runtime ---
    if resolved_runtime == "vllm":
        if not HAS_VLLM_RUNNER:
            log(f"  SKIP: vLLM runner not installed (pip install vllm)")
            return None, None

        hf_repo = model_entry.get("hf_repo", "")
        if not hf_repo:
            log(f"  SKIP: No hf_repo for vLLM")
            return None, None

        log(f"  Starting vLLM server for {hf_repo}...")
        runner = VLLMRunner(gpu_memory_utilization=0.85)
        error = runner.start_server(hf_repo, max_model_len=2048)
        if error:
            log(f"  vLLM start error: {error}")
            return None, None

        log(f"  Successfully started vLLM for {hf_repo}")
        return hf_repo, runner

    log(f"  SKIP: Unknown runtime '{resolved_runtime}'")
    return None, None

def cleanup_model(model_entry: dict, model_tag: str, runtime_client=None):
    """Clean up model resources after benchmarking.

    Ephemeral lifecycle: DELETE the model to free disk space.
    For Ollama: run `ollama rm` to remove the model entirely.
    For HF Transformers: unload model from GPU, clear cache.
    For vLLM: stop server process.
    """
    
    skip_lifecycle = os.environ.get("BENCHMARK_SKIP_LIFECYCLE") == "1"
    
    resolved_runtime = model_entry.get("resolved_runtime", "")

    if runtime_client is not None:
        if resolved_runtime == "hf_transformers" and hasattr(runtime_client, 'unload_model'):
            runtime_client.unload_model()
            log(f"  Unloaded HF Transformers model")
            # Also clean HF cache to save disk
            hf_repo = model_entry.get("hf_repo", "")
            if hf_repo:
                try:
                    from src.hf_runner import cleanup_hf_model_cache
                    if cleanup_hf_model_cache(hf_repo):
                        log(f"  Cleared HF cache for {hf_repo}")
                except Exception:
                    pass
        elif resolved_runtime == "vllm" and hasattr(runtime_client, 'stop_server'):
            runtime_client.stop_server()
            log(f"  Stopped vLLM server")
            # Clean vLLM HF cache too
            hf_repo = model_entry.get("hf_repo", "")
            if hf_repo:
                _cleanup_hf_cache(hf_repo)
    else:
        # Ollama / GGUF path — DELETE the model from Ollama
        if model_tag and not skip_lifecycle:
            log(f"  Deleting model from Ollama: {model_tag}")
            try:
                result = subprocess.run(
                    ["ollama", "rm", model_tag],
                    capture_output=True, encoding='utf-8', errors='replace',
                    timeout=60,
                )
                if result.returncode == 0:
                    log(f"  ✓ Deleted {model_tag} from Ollama")
                else:
                    log(f"  Warning: ollama rm failed: {result.stderr.strip()}")
            except Exception as e:
                log(f"  Warning: Could not delete model: {e}")

        # Also clear any local HF download dir
        source = model_entry.get("source", "ollama")
        if source == "huggingface" or resolved_runtime == "huggingface_gguf":
            local_dir_name = model_tag.replace(':', '_').replace('/', '_') if model_tag else None
            if local_dir_name:
                local_dir = os.path.join(MODELS_DIR, local_dir_name)
                if os.path.exists(local_dir):
                    try:
                        shutil.rmtree(local_dir)
                        log(f"  Cleared HuggingFace download dir: {local_dir_name}")
                    except Exception as e:
                        log(f"  Warning: Could not clear HF dir: {e}")

    free_gb = get_free_disk_space_gb()
    log(f"  Free disk space after cleanup: {free_gb:.2f} GB")


def _cleanup_hf_cache(hf_repo: str):
    """Remove a model from the HuggingFace hub cache."""
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    repo_dir_name = f"models--{hf_repo.replace('/', '--')}"
    repo_cache = os.path.join(hf_cache, repo_dir_name)
    if os.path.isdir(repo_cache):
        try:
            shutil.rmtree(repo_cache)
            log(f"  Cleared HF cache: {repo_dir_name}")
        except Exception as e:
            log(f"  Warning: Could not clear HF cache: {e}")

def save_unified_result(results_list: list, model: str) -> str:
    """
    Saves the aggregated results of a model's run to a CSV formatted specifically for Phase 3 aggregation.
    Returns the path to the saved CSV file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace(":", "_").replace("/", "_")
    output_dir = os.path.join(RESULTS_DIR, "phase1")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{safe_model}_MegaBench_{timestamp}.csv")
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    log(f"Saved results for '{model}' to {csv_path}")
    return csv_path

def _save_checkpoint_csv(results_list: list, queue_id: str) -> str:
    """Save partial results to a deterministic checkpoint path (overwrites)."""
    safe_id = queue_id.replace(":", "_").replace("/", "_")
    output_dir = os.path.join(RESULTS_DIR, "phase1")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{safe_id}_checkpoint.csv")
    pd.DataFrame(results_list).to_csv(csv_path, index=False)
    return csv_path

def load_existing_results(csv_path: str) -> list:
    """Load existing results from a CSV file to resume testing."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.to_dict('records')
    return None

def run_project_pipeline():
    log("Starting Unified LLM Mega-Benchmark Pipeline with Checkpoint/Resume")
    log("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    
    prompts_path = os.path.join(PROMPTS_DIR, "benchmark_prompts.json")
    if not os.path.exists(prompts_path):
        log("prompts.json not found!")
        return
        
    with open(prompts_path, 'r') as f:
        all_prompts = json.load(f)
    
    total_all_prompts = len(all_prompts)
    log(f"Loaded {total_all_prompts} prompts from dataset")
    
    total_models = len(MODEL_QUEUE)
    runnable = sum(1 for m in MODEL_QUEUE if m["status"] == "pending")
    log(f"Model queue: {total_models} total ({runnable} runnable)")
        
    benchmarker = Benchmarker()
    tester = StructuredOutputTester()
    
    phase2_limit = _phase2_limit()
    
    cat_map = {
        "Coding": "Coding Generation",
        "Reasoning": "Medium Reasoning",
        "Chat": "Chat & Generation",
        "Vision": "Multimodal Vision",
    }
    
    for idx, model_entry in enumerate(MODEL_QUEUE, 1):
        queue_id = model_entry["queue_id"]
        
        target_model = os.environ.get("BENCHMARK_SINGLE_MODEL")
        if target_model and queue_id != target_model:
            continue

        model_cat = model_entry["category"]
        model_status = model_entry.get("status", "pending")
        display_name = model_entry["requested_name"]
        resolved_runtime = model_entry.get("resolved_runtime", "")
        resolved_ref = model_entry.get("resolved_model_ref", "")

        target_category = cat_map.get(model_cat)
        if target_category is None:
            target_category = "Multimodal Vision"
        filtered_prompts = [p for p in all_prompts if p["category"] == target_category]

        tracker.init_model(queue_id, model_cat, total_prompts=len(filtered_prompts),
                           model_metadata=model_entry)

        st, completed_prompts, total = tracker.get_model_progress(queue_id)

        # --- Already handled models ---
        if st == "completed":
            log(f"\n[{idx}/{total_models}] {display_name} ({model_cat}) - Already completed, skipping")
            continue

        if st == "failed":
            err = tracker.get_all_progress()["models"].get(queue_id, {}).get("error", "Unknown")
            log(f"\n[{idx}/{total_models}] {display_name} ({model_cat}) - Previously failed: {err[:80]}")
            continue

        if st in ("skipped", "provider_unsupported", "deferred_vision"):
            reason = tracker.get_all_progress()["models"].get(queue_id, {}).get("error", "")
            log(f"\n[{idx}/{total_models}] {display_name} ({model_cat}) - Skipped: {reason[:80]}")
            continue

        # --- Pre-run skips ---
        if model_status == "deferred_vision":
            log(f"\n[{idx}/{total_models}] {display_name} ({model_cat}) - Deferred (multimodal/vision)")
            tracker.skip_model(queue_id, "Multimodal/Vision model – deferred until real image assets exist", "deferred_vision")
            tracker.save_status_to_file()
            continue

        if model_status == "provider_unsupported":
            reason = model_entry.get("variant_note", "Provider unsupported")
            log(f"\n[{idx}/{total_models}] {display_name} ({model_cat}) - Unsupported: {reason[:80]}")
            tracker.skip_model(queue_id, reason, "provider_unsupported")
            tracker.save_status_to_file()
            continue

        # --- Start benchmark ---
        log(f"\n[{idx}/{total_models}] Evaluating {display_name} ({resolved_runtime})")
        log(f"  Resolved ref: {resolved_ref}")
        log(f"  Status: {st}, Completed prompts: {completed_prompts}/{total}")

        model_tag, runtime_client = download_model(model_entry)
        if not model_tag:
            error_msg = "Download/load failed"
            log(f"  ERROR: {error_msg}")
            tracker.fail_model(queue_id, error_msg)
            tracker.save_status_to_file()
            continue

        tracker.start_model(queue_id)

        # Create runtime-specific benchmark function
        if runtime_client is not None:
            # HF Transformers or vLLM — use the runner directly
            def run_benchmark(prompt_text, max_tok, is_vis=False, img_path=None):
                return runtime_client.generate_benchmark(
                    model_tag, prompt_text, max_tokens=max_tok,
                    is_vision=is_vis, image_path=img_path
                )
            def run_structured(prompt_text, schema_json, temp, is_vis=False, img_path=None):
                content, error = runtime_client.generate_structured(
                    model_tag, prompt_text, schema_json,
                    temperature=temp, is_vision=is_vis, image_path=img_path
                )
                return {"success": error is None and len(content) > 0, "error": error, "output": content}
            log(f"  Using {resolved_runtime} runtime")
        else:
            # Ollama — use the shared benchmarker client
            # Pre-load model into GPU memory
            log(f"  Pre-loading model into GPU memory (Ollama)...")
            try:
                _ = benchmarker.client.generate_benchmark(model_tag, "", max_tokens=1)
            except Exception:
                pass
            run_benchmark = None  # Will use benchmarker directly
            run_structured = None  # Will use tester directly

        model_results = []
        start_index = 0

        if st == "in_progress" and completed_prompts > 0:
            existing_csv = tracker.get_all_progress()["models"].get(queue_id, {}).get("csv_file")
            if existing_csv and os.path.exists(existing_csv):
                existing_results = load_existing_results(existing_csv)
                if existing_results:
                    model_results = existing_results
                    start_index = completed_prompts
                    log(f"  Resuming from prompt {start_index + 1}")

        log(f"  Running {len(filtered_prompts) - start_index} remaining prompts...")
        smoke_prompts = SMOKE_RUN_PROMPTS if SMOKE_RUN_PROMPTS > 0 else None

        import time
        loop_start_time = time.perf_counter()
        checkpoint_csv = None

        try:
            for i, p in enumerate(filtered_prompts):
                if i < start_index:
                    continue

                if smoke_prompts and (i - start_index) >= smoke_prompts:
                    log(f"\n  Smoke run: stopping after {smoke_prompts} prompts")
                    break

                prompts_done = i - start_index
                if prompts_done > 0:
                    elapsed = time.perf_counter() - loop_start_time
                    avg_time_per_prompt = elapsed / prompts_done
                    prompts_left = len(filtered_prompts) - i
                    eta_mins = (avg_time_per_prompt * prompts_left) / 60
                    log(f"  Prompt {i+1}/{len(filtered_prompts)} | Elapsed: {elapsed/60:.1f}m | ETA: {eta_mins:.1f}m", end="\r")

                    # Slow-model guard
                    if SLOW_MODEL_THRESHOLD_SECS > 0 and prompts_done >= 3:
                        if avg_time_per_prompt > SLOW_MODEL_THRESHOLD_SECS:
                            err = f"Average prompt time {avg_time_per_prompt:.0f}s > threshold {SLOW_MODEL_THRESHOLD_SECS}s"
                            log(f"\n  SLOW: {err}")
                            if model_results:
                                ckpt_path = _save_checkpoint_csv(model_results, queue_id)
                                tracker.update_checkpoint(queue_id, len(model_results), ckpt_path)
                            tracker.fail_model(queue_id, err)
                            cleanup_model(model_entry, model_tag, runtime_client)
                            tracker.save_status_to_file()
                            break  # skip this model, continue with next
                else:
                    log(f"  Prompt {i+1}/{len(filtered_prompts)} | Calculating ETA...", end="\r")

                unified_data = benchmarker.benchmark_single(model_tag, p, model_entry=model_entry)

                # Override with runtime-specific client if not Ollama
                if run_benchmark is not None:
                    category = p.get("category", "")
                    is_vis = (category == "Multimodal Vision")
                    image_path = p.get("image_path", None)
                    from config import MAX_NEW_TOKENS
                    res = run_benchmark(p["prompt"], MAX_NEW_TOKENS, is_vis=is_vis, img_path=image_path)
                    # Overwrite metrics from the runtime-specific result
                    unified_data["tps"] = res.tps
                    unified_data["ttft"] = res.ttft
                    unified_data["latency"] = res.latency
                    unified_data["output"] = res.content
                    unified_data["error"] = res.error

                if i < (start_index + phase2_limit):
                    prompt_image_path = p.get("image_path", None)
                    for temp in TEMPS_TO_TEST:
                        if run_structured is not None:
                            # Use runtime-specific structured output
                            schema_class = __import__('src.schemas', fromlist=['get_schema_for_category']).get_schema_for_category(target_category)
                            schema_json = schema_class.model_json_schema()
                            temp_res = run_structured(p["prompt"], schema_json, temp,
                                                      is_vis=(target_category == "Multimodal Vision"),
                                                      img_path=prompt_image_path)
                        else:
                            temp_res = tester.generate_single_with_retry(
                                model_tag, p["prompt"],
                                category=target_category,
                                temperature=temp,
                                image_path=prompt_image_path
                            )
                        unified_data[f"temp_{temp}_success"] = temp_res["success"]
                        unified_data[f"temp_{temp}_error"] = temp_res["error"]
                        unified_data[f"temp_{temp}_output"] = temp_res["output"]

                model_results.append(unified_data)

                # Save checkpoint CSV after every prompt
                checkpoint_csv = _save_checkpoint_csv(model_results, queue_id)
                tracker.update_checkpoint(queue_id, i + 1, checkpoint_csv)

        except KeyboardInterrupt:
            log(f"\n  Interrupted! Saving checkpoint at prompt {len(model_results)}...")
            if model_results:
                csv_path = save_unified_result(model_results, queue_id)
                tracker.update_checkpoint(queue_id, len(model_results), csv_path)
            cleanup_model(model_entry, model_tag, runtime_client)
            tracker.save_status_to_file()
            log("  Checkpoint saved. Run again to resume.")
            return

        except Exception as e:
            error_msg = str(e)
            log(f"\n  ERROR during testing: {error_msg}")
            if model_results:
                csv_path = save_unified_result(model_results, queue_id)
                tracker.update_checkpoint(queue_id, len(model_results), csv_path)
            tracker.fail_model(queue_id, error_msg)
            tracker.save_status_to_file()
            cleanup_model(model_entry, model_tag, runtime_client)
            continue

        log(f"\n  Testing complete! ({len(model_results)} prompts)")
        error_count = sum(1 for r in model_results if r.get("error"))
        csv_path = save_unified_result(model_results, queue_id)

        if error_count == len(model_results) and len(model_results) > 0:
            sample_errors = [r["error"] for r in model_results if r.get("error")][:3]
            error_msg = f"All {len(model_results)} prompts failed. Sample: {'; '.join(sample_errors)}"
            log(f"  ERROR: {error_msg}")
            tracker.fail_model(queue_id, error_msg)
            log(f"  Marked as failed (all prompts errored)")
        else:
            tracker.complete_model(queue_id, csv_path)
            if error_count > 0:
                log(f"  Marked as completed ({error_count}/{len(model_results)} errors)")
            else:
                log(f"  Marked as completed")

        # Clean up checkpoint CSV now that final CSV is saved
        if checkpoint_csv and os.path.exists(checkpoint_csv) and checkpoint_csv != csv_path:
            try:
                os.remove(checkpoint_csv)
                log(f"  Cleaned up checkpoint: {os.path.basename(checkpoint_csv)}")
            except Exception:
                pass

        cleanup_model(model_entry, model_tag, runtime_client)
        tracker.save_status_to_file()
        log("  Status saved to TEST_STATUS.md")
        log("\n" + "-" * 60)
        log(tracker.generate_status_report())
    
    log("\n" + "=" * 60)
    log("ALL MODELS PROCESSED!")
    log("=" * 60)
    
    log("\nGenerating Final Report...")
    comparator = ModelComparator()
    comparator.run_offline_report()

if __name__ == "__main__":
    try:
        run_project_pipeline()
    except KeyboardInterrupt:
        log("\n\nPipeline interrupted. Run again to resume from last checkpoint.")
