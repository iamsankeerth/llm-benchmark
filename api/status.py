import os
import csv
import json
import glob
import subprocess
import psutil
from pathlib import Path
from datetime import datetime
from collections import deque
from fastapi import APIRouter, HTTPException

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
PROGRESS_FILE = BASE_DIR / "test_progress.json"
LOG_FILE = BASE_DIR / "logs" / "benchmarks.log"
PID_FILE = BASE_DIR / "logs" / "benchmark.pid"
RESULTS_DIR = BASE_DIR / "results" / "phase1"
QUALITY_DIR = BASE_DIR / "results" / "quality"


def _get_gpu_info() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "utilization": parts[0].strip(),
                "vram_used": parts[1].strip(),
                "vram_total": parts[2].strip(),
            }
    except Exception:
        pass
    return {"utilization": "N/A", "vram_used": "N/A", "vram_total": "N/A"}


def _get_benchmark_pid() -> int | None:
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            pass
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)
            if "run_full_benchmark" in cmdline_str or "run_benchmarks" in cmdline_str:
                if "python" in (proc.info.get("name") or "").lower():
                    return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def _is_process_alive(pid: int) -> bool:
    return psutil.pid_exists(pid)


def _get_log_tail(n: int = 30) -> list[str]:
    if not LOG_FILE.exists():
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            return list(deque(f, n))
    except OSError:
        return []


def _load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {"models": {}}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"models": {}}


def _load_model_queue() -> list[dict]:
    try:
        from src.model_queue import build_model_queue
        return build_model_queue()
    except Exception:
        return []


def _calc_eta(started_at: str, completed: int, total: int) -> str:
    if not started_at or completed <= 0:
        return "Calculating..."
    try:
        start = datetime.fromisoformat(started_at)
        now = datetime.now()
        elapsed = (now - start).total_seconds()
        if elapsed <= 0:
            return "Calculating..."
        time_per_prompt = elapsed / completed
        remaining = total - completed
        eta_sec = time_per_prompt * remaining
        if eta_sec < 60:
            return f"~{int(eta_sec)}s"
        elif eta_sec < 3600:
            return f"~{int(eta_sec / 60)}m"
        else:
            hours = int(eta_sec // 3600)
            mins = int((eta_sec % 3600) // 60)
            return f"~{hours}h {mins}m"
    except Exception:
        return "Calculating..."


def _format_elapsed(started_at: str) -> str:
    if not started_at:
        return "-"
    try:
        start = datetime.fromisoformat(started_at)
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed < 60:
            return f"{int(elapsed)}s"
        elif elapsed < 3600:
            return f"{int(elapsed / 60)}m"
        else:
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            return f"{hours}h {mins}m"
    except Exception:
        return "-"


def _compute_csv_metrics(csv_path: str) -> dict:
    """Read a checkpoint/final CSV and compute aggregate metrics."""
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    try:
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            return {}

        tps_vals = [float(r["tps"]) for r in rows if r.get("tps") and float(r.get("tps", 0) or 0) > 0]
        lat_vals = [float(r["latency"]) for r in rows if r.get("latency") and float(r.get("latency", 0) or 0) > 0]
        vram_vals = [float(r["peak_vram_mb"]) for r in rows if r.get("peak_vram_mb") and float(r.get("peak_vram_mb", 0) or 0) > 0]

        t0_vals = [r.get("temp_0.0_success", "") for r in rows]
        t0_ok = sum(1 for v in t0_vals if v.strip().lower() in ("true", "1", "yes"))

        return {
            "tps_avg": round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0,
            "latency_avg": round(sum(lat_vals) / len(lat_vals), 2) if lat_vals else 0,
            "vram_peak": round(max(vram_vals), 0) if vram_vals else 0,
            "json_success_rate": round(t0_ok / len(rows) * 100, 1) if rows else 0,
        }
    except Exception:
        return {}


def _compute_quality_metrics(model_name: str) -> dict:
    """Parse promptfoo raw results and compute quality metrics."""
    if not QUALITY_DIR.exists():
        return {}
    try:
        # Find the raw results file for this model
        safe_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
        raw_file = QUALITY_DIR / f"{safe_name}_promptfoo_raw.json"
        if not raw_file.exists():
            # Try alternative naming patterns
            for f in QUALITY_DIR.glob("*_promptfoo_raw.json"):
                if safe_name.replace("_", "") in f.name.replace("_", ""):
                    raw_file = f
                    break
        if not raw_file.exists():
            return {}

        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", {}).get("results", [])
        if not results:
            return {}

        passed = sum(1 for r in results if r.get("success"))
        total = len(results)

        # Category breakdown
        cats = {}
        for r in results:
            desc = r.get("testCase", {}).get("description", "")
            cat = desc.split(":")[0].strip() if ":" in desc else "Other"
            if cat not in cats:
                cats[cat] = {"passed": 0, "total": 0}
            cats[cat]["total"] += 1
            if r.get("success"):
                cats[cat]["passed"] += 1

        def _cat_rate(name):
            c = cats.get(name, {"passed": 0, "total": 0})
            return round(c["passed"] / c["total"] * 100, 1) if c["total"] > 0 else 0

        return {
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "coding_pass_rate": _cat_rate("Coding"),
            "chat_pass_rate": _cat_rate("Chat"),
            "reasoning_pass_rate": _cat_rate("Reasoning"),
            "structured_pass_rate": _cat_rate("Structured"),
            "total_tests": total,
            "passed": passed,
        }
    except Exception:
        return {}


def _find_csv_for_model(queue_id: str) -> str | None:
    """Find the most recent CSV file for a given queue_id."""
    if not RESULTS_DIR.exists():
        return None
    safe = queue_id.replace(":", "_").replace("/", "_")
    patterns = [
        f"{safe}_MegaBench_*.csv",
        f"{safe}_checkpoint.csv",
    ]
    for pat in patterns:
        matches = sorted(RESULTS_DIR.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return str(matches[0])
    return None


def _detect_phase(log_tail: list[str]) -> tuple[str, str]:
    """Parse log tail bottom-up to find the most recent phase marker."""
    phase_markers = [
        (">>> Downloading", "download", "Download"),
        ("--- Running Test A", "test_a", "Test A: CSV Data"),
        ("--- Running Test B", "test_b", "Test B: llama-bench"),
        ("--- Running Test C", "test_c", "Test C: promptfoo"),
        (">>> Deleting", "delete", "Delete"),
    ]
    for line in reversed(log_tail):
        stripped = line.strip()
        for marker, phase_id, phase_label in phase_markers:
            if marker in stripped:
                return phase_id, phase_label
    return "idle", "Idle"


def _calc_per_prompt_eta(elapsed_sec: float, prompts_completed: int) -> str:
    """Calculate average time per prompt."""
    if prompts_completed <= 0:
        return "Calculating..."
    avg = elapsed_sec / prompts_completed
    if avg < 60:
        return f"~{avg:.0f}s"
    else:
        return f"~{avg/60:.1f}m"


def _calc_total_model_eta(phase: str, prompts_completed: int, total_prompts: int,
                           elapsed_sec: float, avg_test_b: float = 120,
                           avg_test_c: float = 300) -> str:
    """Calculate total ETA for current model including all remaining phases."""
    if prompts_completed <= 0:
        return "Calculating..."
    avg_per_prompt = elapsed_sec / prompts_completed
    remaining_prompts = total_prompts - prompts_completed

    if phase == "test_a":
        eta = (remaining_prompts * avg_per_prompt) + avg_test_b + avg_test_c
    elif phase == "test_b":
        eta = avg_test_b + avg_test_c
    elif phase == "test_c":
        eta = avg_test_c
    elif phase == "download":
        eta = (total_prompts * avg_per_prompt) + avg_test_b + avg_test_c
    else:
        eta = 0

    if eta < 60:
        return f"~{int(eta)}s"
    elif eta < 3600:
        return f"~{int(eta/60)}m"
    else:
        return f"~{int(eta//3600)}h {int((eta%3600)//60)}m"


def _compute_difficulty_stats(csv_path: str, total_per_diff: int = 10) -> dict:
    """Compute per-difficulty timing stats from a CSV file."""
    stats = {}
    for diff in ["easy", "medium", "hard", "adversarial", "long_context"]:
        stats[diff] = {"count": total_per_diff, "avg_latency": 0, "completed": 0}
    if not csv_path or not os.path.isfile(csv_path):
        return stats
    try:
        by_diff = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                diff = row.get("difficulty", "")
                lat = float(row.get("latency", 0) or 0)
                if diff not in by_diff:
                    by_diff[diff] = []
                if lat > 0:
                    by_diff[diff].append(lat)
        for diff, vals in by_diff.items():
            if diff in stats:
                stats[diff]["avg_latency"] = round(sum(vals)/len(vals), 2) if vals else 0
                stats[diff]["completed"] = len(vals)
    except Exception:
        pass
    return stats


@router.get("/status")
def get_status():
    progress = _load_progress()
    tracked_models = progress.get("models", {})
    queue = _load_model_queue()

    pid = _get_benchmark_pid()
    alive = _is_process_alive(pid) if pid else False

    gpu = _get_gpu_info()
    log_tail = _get_log_tail(30)

    current_model = None
    current_model_index = 0
    prompts_completed = 0
    total_prompts = 0
    status = "idle"
    elapsed_time = "-"
    eta = "-"
    elapsed_sec = 0
    last_updated = progress.get("last_updated", "")

    completed_models = []
    failed_models = []

    for key, data in tracked_models.items():
        model_status = data.get("status", "pending")
        model_name = data.get("model_name") or data.get("requested_name", key)

        if model_status == "completed":
            csv_path = data.get("csv_file", "")
            metrics = _compute_csv_metrics(csv_path)
            quality = _compute_quality_metrics(model_name)
            completed_models.append({
                "name": model_name,
                "category": data.get("category", ""),
                "prompts_completed": data.get("prompts_completed", 0),
                "csv_file": csv_path,
                **metrics,
                **quality,
            })
        elif model_status == "failed":
            failed_models.append({
                "name": model_name,
                "category": data.get("category", ""),
                "error": (data.get("error") or "Unknown")[:200],
            })
        elif model_status == "in_progress":
            current_model = model_name
            prompts_completed = data.get("prompts_completed", 0)
            total_prompts = data.get("total_prompts", 50)
            status = "in_progress"
            elapsed_time = _format_elapsed(data.get("started_at"))
            eta = _calc_eta(data.get("started_at"), prompts_completed, total_prompts)
            last_updated = data.get("last_checkpoint", last_updated)
            current_model_index = len(completed_models) + 1
            # Calculate elapsed seconds for ETA calculations
            if data.get("started_at"):
                try:
                    start = datetime.fromisoformat(data["started_at"])
                    elapsed_sec = (datetime.now() - start).total_seconds()
                except Exception:
                    elapsed_sec = 0

    if alive and status == "idle":
        status = "running"

    if not alive and status == "in_progress":
        status = "stopped"

    if current_model_index == 0:
        current_model_index = len(completed_models) + (1 if status == "in_progress" else 0)

    # Build all_models list from queue + tracked progress
    all_models = []
    for idx, m in enumerate(queue):
        qid = m["queue_id"]
        name = m.get("requested_name", m.get("name", ""))
        category = m.get("category", "")
        tracked = tracked_models.get(qid, {})
        model_status = tracked.get("status", m.get("status", "pending"))

        entry = {
            "queue_id": qid,
            "name": name,
            "category": category,
            "status": model_status,
            "source": m.get("source", ""),
            "resolved_runtime": m.get("resolved_runtime", ""),
            "size": m.get("size", "?"),
            "prompts_completed": tracked.get("prompts_completed", 0),
            "total_prompts": tracked.get("total_prompts", 50),
            "elapsed": _format_elapsed(tracked.get("started_at")) if tracked.get("started_at") else "-",
            "tps_avg": 0,
            "latency_avg": 0,
            "vram_peak": 0,
            "json_success_rate": 0,
            "pass_rate": 0,
            "coding_pass_rate": 0,
            "chat_pass_rate": 0,
            "reasoning_pass_rate": 0,
            "structured_pass_rate": 0,
            "total_tests": 0,
            "passed": 0,
            "error": "",
        }

        csv_path = tracked.get("csv_file", "")
        if not csv_path and model_status in ("completed", "in_progress"):
            csv_path = _find_csv_for_model(qid)

        if csv_path:
            metrics = _compute_csv_metrics(csv_path)
            entry.update(metrics)

        # Add quality metrics for completed models
        if model_status == "completed":
            quality = _compute_quality_metrics(name)
            entry.update(quality)

        if model_status == "failed":
            entry["error"] = (tracked.get("error") or "")[:200]

        if tracked.get("started_at"):
            entry["elapsed"] = _format_elapsed(tracked["started_at"])

        all_models.append(entry)

    pending_count = sum(1 for m in all_models if m["status"] == "pending")

    # Phase detection from log
    current_phase, current_phase_label = _detect_phase(log_tail)

    # Enhanced ETA calculations
    per_prompt_eta = _calc_per_prompt_eta(elapsed_sec, prompts_completed) if status == "in_progress" else "-"
    total_model_eta = _calc_total_model_eta(current_phase, prompts_completed, total_prompts, elapsed_sec) if status == "in_progress" else "-"
    phase_eta = eta  # Reuse existing phase ETA

    # Difficulty stats for current model
    difficulty_stats = {}
    if status == "in_progress":
        # Find CSV for current in-progress model
        for key, data in tracked_models.items():
            if data.get("status") == "in_progress":
                csv_path = data.get("csv_file", "")
                if not csv_path:
                    csv_path = _find_csv_for_model(key)
                difficulty_stats = _compute_difficulty_stats(csv_path)
                break

    return {
        "process_alive": alive,
        "process_pid": pid,
        "gpu": gpu,
        "current_model": current_model,
        "current_model_index": current_model_index,
        "total_models": len(all_models),
        "prompts_completed": prompts_completed,
        "total_prompts": total_prompts,
        "status": status,
        "elapsed_time": elapsed_time,
        "eta": eta,
        "current_phase": current_phase,
        "current_phase_label": current_phase_label,
        "per_prompt_eta": per_prompt_eta,
        "total_model_eta": total_model_eta,
        "phase_eta": phase_eta,
        "difficulty_stats": difficulty_stats,
        "last_updated": last_updated,
        "completed_models": completed_models,
        "failed_models": failed_models,
        "pending_count": pending_count,
        "all_models": all_models,
        "log_tail": [line.rstrip() for line in log_tail],
    }


@router.get("/model/{queue_id:path}/prompts")
def get_model_prompts(queue_id: str):
    """Return per-prompt progress for a specific model from its CSV."""
    tracked = _load_progress().get("models", {}).get(queue_id, {})
    csv_path = tracked.get("csv_file", "")

    if not csv_path or not os.path.isfile(csv_path):
        csv_path = _find_csv_for_model(queue_id)

    if not csv_path or not os.path.isfile(csv_path):
        raise HTTPException(status_code=404, detail=f"No CSV found for model: {queue_id}")

    prompts = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                def _safe_float(v):
                    try:
                        return round(float(v), 2) if v and v.strip() else None
                    except (ValueError, TypeError):
                        return None

                def _safe_bool(v):
                    return str(v).strip().lower() in ("true", "1", "yes") if v and v.strip() else None

                prompts.append({
                    "prompt_id": _safe_float(row.get("prompt_id")),
                    "difficulty": row.get("difficulty", ""),
                    "tps": _safe_float(row.get("tps")),
                    "ttft": _safe_float(row.get("ttft")),
                    "latency": _safe_float(row.get("latency")),
                    "vram_mb": _safe_float(row.get("peak_vram_mb")),
                    "temp_0.0_success": _safe_bool(row.get("temp_0.0_success")),
                    "temp_0.7_success": _safe_bool(row.get("temp_0.7_success")),
                    "temp_1.0_success": _safe_bool(row.get("temp_1.0_success")),
                    "error": (row.get("error") or "").strip()[:100] or None,
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {e}")

    return {
        "queue_id": queue_id,
        "model": tracked.get("model_name", queue_id),
        "csv_file": csv_path,
        "prompts": prompts,
    }


@router.post("/stop")
def stop_pipeline():
    pid = _get_benchmark_pid()
    if not pid:
        raise HTTPException(status_code=404, detail="No benchmark process found")

    if not _is_process_alive(pid):
        raise HTTPException(status_code=404, detail=f"Process {pid} is not running")

    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=10)
    except psutil.TimeoutExpired:
        proc.kill()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop process: {e}")

    return {"stopped": True, "pid": pid}


@router.get("/health")
def health_check():
    return {"status": "ok"}
