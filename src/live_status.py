"""Live benchmark status projection.

FastAPI routes delegate here so process checks, artifact reads, ETA math, and
dashboard response shaping live behind one testable interface.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from typing import Any, Callable

import psutil

from src.artifact_store import BenchmarkArtifactStore


class SystemStatusAdapter:
    def __init__(self, store: BenchmarkArtifactStore):
        self.store = store

    def get_gpu_info(self) -> dict[str, str]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
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

    def get_benchmark_pid(self) -> int | None:
        if self.store.pid_file.exists():
            try:
                return int(self.store.pid_file.read_text().strip())
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

    @staticmethod
    def is_process_alive(pid: int) -> bool:
        return psutil.pid_exists(pid)

    @staticmethod
    def stop_process(pid: int) -> None:
        proc = psutil.Process(pid)
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            proc.kill()


class LiveStatusProjection:
    def __init__(
        self,
        store: BenchmarkArtifactStore | None = None,
        system: SystemStatusAdapter | None = None,
        queue_loader: Callable[[], list[dict[str, Any]]] | None = None,
    ):
        self.store = store or BenchmarkArtifactStore()
        self.system = system or SystemStatusAdapter(self.store)
        self.queue_loader = queue_loader or self._load_model_queue

    @staticmethod
    def _load_model_queue() -> list[dict[str, Any]]:
        try:
            from src.model_queue import build_model_queue

            return build_model_queue()
        except Exception:
            return []

    def status_payload(self) -> dict[str, Any]:
        progress = self.store.load_progress()
        tracked_models = progress.get("models", {})
        queue = self.queue_loader()

        pid = self.system.get_benchmark_pid()
        alive = self.system.is_process_alive(pid) if pid else False
        gpu = self.system.get_gpu_info()
        log_tail = self.store.read_log_tail(30)

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
                completed_models.append(
                    {
                        "name": model_name,
                        "category": data.get("category", ""),
                        "prompts_completed": data.get("prompts_completed", 0),
                        "csv_file": csv_path,
                        **self.store.compute_csv_metrics(csv_path),
                        **self.store.compute_quality_metrics(model_name),
                    }
                )
            elif model_status == "failed":
                failed_models.append(
                    {
                        "name": model_name,
                        "category": data.get("category", ""),
                        "error": (data.get("error") or "Unknown")[:200],
                    }
                )
            elif model_status == "in_progress":
                current_model = model_name
                prompts_completed = data.get("prompts_completed", 0)
                total_prompts = data.get("total_prompts", 50)
                status = "in_progress"
                elapsed_time = _format_elapsed(data.get("started_at"))
                eta = _calc_eta(data.get("started_at"), prompts_completed, total_prompts)
                last_updated = data.get("last_checkpoint", last_updated)
                current_model_index = len(completed_models) + 1
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

        all_models = self._all_models(queue, tracked_models)
        pending_count = sum(1 for m in all_models if m["status"] == "pending")
        current_phase, current_phase_label = _detect_phase(log_tail)
        if current_phase == "idle" and status == "in_progress":
            current_phase = "test_a"
            current_phase_label = "Test A: Phase 1 (CSV Data)"

        difficulty_stats = {}
        if status == "in_progress":
            for key, data in tracked_models.items():
                if data.get("status") == "in_progress":
                    csv_path = data.get("csv_file", "") or self.store.find_csv_for_model(key)
                    difficulty_stats = self.store.compute_difficulty_stats(csv_path)
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
            "per_prompt_eta": _calc_per_prompt_eta(elapsed_sec, prompts_completed)
            if status == "in_progress"
            else "-",
            "total_model_eta": _calc_total_model_eta(
                current_phase, prompts_completed, total_prompts, elapsed_sec
            )
            if status == "in_progress"
            else "-",
            "phase_eta": eta,
            "difficulty_stats": difficulty_stats,
            "last_updated": last_updated,
            "completed_models": completed_models,
            "failed_models": failed_models,
            "pending_count": pending_count,
            "all_models": all_models,
            "log_tail": [line.rstrip() for line in log_tail],
        }

    def _all_models(
        self, queue: list[dict[str, Any]], tracked_models: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        all_models = []
        for model in queue:
            qid = model["queue_id"]
            name = model.get("requested_name", model.get("name", ""))
            tracked = tracked_models.get(qid, {})
            model_status = tracked.get("status", model.get("status", "pending"))
            entry = {
                "queue_id": qid,
                "name": name,
                "category": model.get("category", ""),
                "status": model_status,
                "source": model.get("source", ""),
                "resolved_runtime": model.get("resolved_runtime", ""),
                "size": model.get("size", "?"),
                "prompts_completed": tracked.get("prompts_completed", 0),
                "total_prompts": tracked.get("total_prompts", 50),
                "elapsed": _format_elapsed(tracked.get("started_at"))
                if tracked.get("started_at")
                else "-",
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
                csv_path = self.store.find_csv_for_model(qid)
            if csv_path:
                entry.update(self.store.compute_csv_metrics(csv_path))
            if model_status == "completed":
                entry.update(self.store.compute_quality_metrics(name))
            if model_status == "failed":
                entry["error"] = (tracked.get("error") or "")[:200]
            all_models.append(entry)
        return all_models

    def model_prompts_payload(self, queue_id: str) -> dict[str, Any] | None:
        return self.store.model_prompts(queue_id)

    def stop_pipeline(self) -> dict[str, Any]:
        pid = self.system.get_benchmark_pid()
        if not pid:
            raise LookupError("No benchmark process found")
        if not self.system.is_process_alive(pid):
            raise LookupError(f"Process {pid} is not running")
        self.system.stop_process(pid)
        return {"stopped": True, "pid": pid}


def _calc_eta(started_at: str, completed: int, total: int) -> str:
    if not started_at or completed <= 0:
        return "Calculating..."
    try:
        start = datetime.fromisoformat(started_at)
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed <= 0:
            return "Calculating..."
        eta_sec = (elapsed / completed) * (total - completed)
        if eta_sec < 60:
            return f"~{int(eta_sec)}s"
        if eta_sec < 3600:
            return f"~{int(eta_sec / 60)}m"
        return f"~{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
    except Exception:
        return "Calculating..."


def _format_elapsed(started_at: str) -> str:
    if not started_at:
        return "-"
    try:
        elapsed = (datetime.now() - datetime.fromisoformat(started_at)).total_seconds()
        if elapsed < 60:
            return f"{int(elapsed)}s"
        if elapsed < 3600:
            return f"{int(elapsed / 60)}m"
        return f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"
    except Exception:
        return "-"


def _detect_phase(log_tail: list[str]) -> tuple[str, str]:
    markers = [
        (">>> Downloading", "download", "Download"),
        ("--- Running Test A", "test_a", "Test A: CSV Data"),
        ("--- Running Test B", "test_b", "Test B: llama-bench"),
        ("--- Running Test C", "test_c", "Test C: promptfoo"),
        (">>> Deleting", "delete", "Delete"),
    ]
    for line in reversed(log_tail):
        stripped = line.strip()
        for marker, phase_id, phase_label in markers:
            if marker in stripped:
                return phase_id, phase_label
    return "idle", "Idle"


def _calc_per_prompt_eta(elapsed_sec: float, prompts_completed: int) -> str:
    if prompts_completed <= 0:
        return "Calculating..."
    avg = elapsed_sec / prompts_completed
    return f"~{avg:.0f}s" if avg < 60 else f"~{avg / 60:.1f}m"


def _calc_total_model_eta(
    phase: str,
    prompts_completed: int,
    total_prompts: int,
    elapsed_sec: float,
    avg_test_b: float = 120,
    avg_test_c: float = 300,
) -> str:
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
    if eta < 3600:
        return f"~{int(eta / 60)}m"
    return f"~{int(eta // 3600)}h {int((eta % 3600) // 60)}m"
