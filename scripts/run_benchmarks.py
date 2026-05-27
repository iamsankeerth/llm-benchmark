import sys
import os
import json

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
from src.artifact_store import BenchmarkArtifactStore
from src.lifecycle import acquire_runtime, cleanup_runtime
from src.model_entry import as_model_entry
from config import MODEL_QUEUE, PROMPTS_DIR, TEMPS_TO_TEST, MODELS_DIR, PHASE2_PROMPT_LIMIT
try:
    from config import SMOKE_RUN_PROMPTS, SLOW_MODEL_THRESHOLD_SECS
except ImportError:
    SMOKE_RUN_PROMPTS = 0
    SLOW_MODEL_THRESHOLD_SECS = 0
from scripts.test_tracker import TestTracker

tracker = TestTracker()
artifact_store = BenchmarkArtifactStore()

def _phase2_limit() -> int:
    """Return the effective Phase 2 limit. None means all prompts."""
    if PHASE2_PROMPT_LIMIT is None:
        return 2**62
    return PHASE2_PROMPT_LIMIT

def save_unified_result(results_list: list, model: str) -> str:
    """
    Saves the aggregated results of a model's run to a CSV formatted specifically for Phase 3 aggregation.
    Returns the path to the saved CSV file.
    """
    csv_path = artifact_store.save_unified_result(results_list, model)
    log(f"Saved results for '{model}' to {csv_path}")
    return csv_path

def _save_checkpoint_csv(results_list: list, queue_id: str) -> str:
    """Save partial results to a deterministic checkpoint path (overwrites)."""
    return artifact_store.save_checkpoint_csv(results_list, queue_id)

def load_existing_results(csv_path: str) -> list:
    """Load existing results from a CSV file to resume testing."""
    return artifact_store.load_existing_results(csv_path)

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
    runnable = sum(1 for m in MODEL_QUEUE if as_model_entry(m).is_runnable)
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
        entry = as_model_entry(model_entry)
        queue_id = entry.queue_id
        
        target_model = os.environ.get("BENCHMARK_SINGLE_MODEL")
        if target_model and queue_id != target_model:
            continue

        model_cat = entry.category
        model_status = entry.status
        display_name = entry.requested_name
        resolved_runtime = entry.resolved_runtime
        resolved_ref = entry.resolved_model_ref

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
        if entry.is_deferred_vision:
            log(f"\n[{idx}/{total_models}] {display_name} ({model_cat}) - Deferred (multimodal/vision)")
            tracker.skip_model(queue_id, "Multimodal/Vision model â€“ deferred until real image assets exist", "deferred_vision")
            tracker.save_status_to_file()
            continue

        if entry.is_provider_unsupported:
            reason = entry.variant_note or "Provider unsupported"
            log(f"\n[{idx}/{total_models}] {display_name} ({model_cat}) - Unsupported: {reason[:80]}")
            tracker.skip_model(queue_id, reason, "provider_unsupported")
            tracker.save_status_to_file()
            continue

        # --- Start benchmark ---
        log(f"\n[{idx}/{total_models}] Evaluating {display_name} ({resolved_runtime})")
        log(f"  Resolved ref: {resolved_ref}")
        log(f"  Status: {st}, Completed prompts: {completed_prompts}/{total}")

        runtime_handle = acquire_runtime(model_entry, log=log)
        if runtime_handle is None:
            error_msg = "Download/load failed"
            log(f"  ERROR: {error_msg}")
            tracker.fail_model(queue_id, error_msg)
            tracker.save_status_to_file()
            continue
        model_tag = runtime_handle.model_ref
        runtime_client = runtime_handle.runtime_client

        tracker.start_model(queue_id)

        # Create runtime-specific benchmark function
        if runtime_client is not None:
            # HF Transformers or vLLM â€” use the runner directly
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
            # Ollama â€” use the shared benchmarker client
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
                            cleanup_runtime(runtime_handle, log=log)
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
            cleanup_runtime(runtime_handle, log=log)
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
            cleanup_runtime(runtime_handle, log=log)
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

        cleanup_runtime(runtime_handle, log=log)
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
