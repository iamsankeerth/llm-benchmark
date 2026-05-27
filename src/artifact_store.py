"""Read and write benchmark artifacts through one interface."""

from __future__ import annotations

import csv
import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.model_entry import as_model_entry, safe_name


class BenchmarkArtifactStore:
    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.results_dir = self.base_dir / "results"
        self.phase1_dir = self.results_dir / "phase1"
        self.perf_dir = self.results_dir / "perf"
        self.quality_dir = self.results_dir / "quality"
        self.reports_dir = self.base_dir / "reports"
        self.progress_file = self.base_dir / "test_progress.json"
        self.status_file = self.base_dir / "TEST_STATUS.md"
        self.log_file = self.base_dir / "logs" / "benchmarks.log"
        self.pid_file = self.base_dir / "logs" / "benchmark.pid"

    def load_progress(self) -> dict[str, Any]:
        if not self.progress_file.exists():
            return {"models": {}}
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"models": {}}

    def save_progress(self, data: dict[str, Any]) -> None:
        data["last_updated"] = datetime.now().isoformat()
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def read_log_tail(self, n: int = 30) -> list[str]:
        if not self.log_file.exists():
            return []
        try:
            with open(self.log_file, "r", encoding="utf-8", errors="replace") as f:
                return list(deque(f, n))
        except OSError:
            return []

    def find_csv_for_model(self, queue_id: str) -> str | None:
        if not self.phase1_dir.exists():
            return None
        safe = safe_name(queue_id)
        patterns = [f"{safe}_MegaBench_*.csv", f"{safe}_checkpoint.csv"]
        for pattern in patterns:
            matches = sorted(
                self.phase1_dir.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return str(matches[0])
        return None

    def save_unified_result(self, results_list: list[dict[str, Any]], model: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.phase1_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.phase1_dir / f"{safe_name(model)}_MegaBench_{timestamp}.csv"
        pd.DataFrame(results_list).to_csv(csv_path, index=False)
        return str(csv_path)

    def save_checkpoint_csv(self, results_list: list[dict[str, Any]], queue_id: str) -> str:
        self.phase1_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.phase1_dir / f"{safe_name(queue_id)}_checkpoint.csv"
        pd.DataFrame(results_list).to_csv(csv_path, index=False)
        return str(csv_path)

    def phase1_done(self, model_entry: dict[str, Any]) -> bool:
        entry = as_model_entry(model_entry)
        if not self.phase1_dir.exists():
            return False
        safe_id = entry.safe_queue_id
        for path in self.phase1_dir.iterdir():
            if path.name.startswith(safe_id) and (
                path.name.endswith("_checkpoint.csv") or "MegaBench" in path.name
            ):
                return True
        return False

    def llama_bench_result_path(self, queue_id: str) -> Path:
        return self.perf_dir / f"{safe_name(queue_id)}_llama_bench.json"

    def save_llama_bench_result(self, queue_id: str, result: dict[str, Any]) -> Path:
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.llama_bench_result_path(queue_id)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result_path

    def save_llama_bench_summary(
        self,
        settings: dict[str, Any],
        results: dict[str, Any],
    ) -> Path:
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.perf_dir / "llama_bench_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "settings": settings,
                    "total_models": len(results),
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return summary_path

    def load_llama_bench_result(self, queue_id: str) -> dict[str, Any] | None:
        result_path = self.llama_bench_result_path(queue_id)
        if not result_path.is_file():
            return None
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def perf_done(self, model_entry: dict[str, Any]) -> bool:
        entry = as_model_entry(model_entry)
        return self.llama_bench_result_path(entry.queue_id).is_file()

    def quality_result_path(self, model_ref: str) -> Path:
        return self.quality_dir / f"{safe_name(model_ref)}_promptfoo.json"

    def quality_raw_result_path(self, model_ref: str) -> Path:
        return self.quality_dir / f"{safe_name(model_ref)}_promptfoo_raw.json"

    def quality_done(self, model_entry: dict[str, Any]) -> bool:
        entry = as_model_entry(model_entry)
        model_ref = entry.ollama_tag or entry.resolved_model_ref
        return bool(model_ref) and self.quality_result_path(model_ref).is_file()

    def all_phases_done(self, model_entry: dict[str, Any]) -> bool:
        return (
            self.phase1_done(model_entry)
            and self.perf_done(model_entry)
            and self.quality_done(model_entry)
        )

    @staticmethod
    def load_existing_results(csv_path: str) -> list[dict[str, Any]] | None:
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path).to_dict("records")
        return None

    @staticmethod
    def _truthy_success(value: Any) -> bool:
        return str(value).strip().lower() in {"true", "1", "yes"}

    def compute_csv_metrics(self, csv_path: str) -> dict[str, Any]:
        if not csv_path or not os.path.isfile(csv_path):
            return {}
        try:
            rows: list[dict[str, str]] = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows.extend(reader)
            if not rows:
                return {}

            tps_vals = [
                float(r["tps"])
                for r in rows
                if r.get("tps") and float(r.get("tps", 0) or 0) > 0
            ]
            lat_vals = [
                float(r["latency"])
                for r in rows
                if r.get("latency") and float(r.get("latency", 0) or 0) > 0
            ]
            vram_vals = [
                float(r["peak_vram_mb"])
                for r in rows
                if r.get("peak_vram_mb") and float(r.get("peak_vram_mb", 0) or 0) > 0
            ]
            t0_vals = [r.get("temp_0.0_success", "") for r in rows]
            t0_ok = sum(1 for v in t0_vals if self._truthy_success(v))

            return {
                "tps_avg": round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0,
                "latency_avg": round(sum(lat_vals) / len(lat_vals), 2) if lat_vals else 0,
                "vram_peak": round(max(vram_vals), 0) if vram_vals else 0,
                "json_success_rate": round(t0_ok / len(rows) * 100, 1) if rows else 0,
            }
        except Exception:
            return {}

    def compute_quality_metrics(self, model_name: str) -> dict[str, Any]:
        if not self.quality_dir.exists():
            return {}
        try:
            safe_candidates = [safe_name(model_name), safe_name(model_name).replace(".", "_")]
            raw_file = self.quality_dir / f"{safe_candidates[0]}_promptfoo_raw.json"
            if not raw_file.exists():
                for candidate in self.quality_dir.glob("*_promptfoo_raw.json"):
                    folded_name = candidate.name.replace("_", "").replace(".", "")
                    if any(safe.replace("_", "").replace(".", "") in folded_name for safe in safe_candidates):
                        raw_file = candidate
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
            cats: dict[str, dict[str, int]] = {}
            for result in results:
                desc = result.get("testCase", {}).get("description", "")
                cat = desc.split(":")[0].strip() if ":" in desc else "Other"
                cats.setdefault(cat, {"passed": 0, "total": 0})
                cats[cat]["total"] += 1
                if result.get("success"):
                    cats[cat]["passed"] += 1

            def cat_rate(name: str) -> float:
                c = cats.get(name, {"passed": 0, "total": 0})
                return round(c["passed"] / c["total"] * 100, 1) if c["total"] else 0

            return {
                "pass_rate": round(passed / total * 100, 1) if total else 0,
                "coding_pass_rate": cat_rate("Coding"),
                "chat_pass_rate": cat_rate("Chat"),
                "reasoning_pass_rate": cat_rate("Reasoning"),
                "structured_pass_rate": cat_rate("Structured"),
                "total_tests": total,
                "passed": passed,
            }
        except Exception:
            return {}

    def model_prompts(self, queue_id: str) -> dict[str, Any] | None:
        tracked = self.load_progress().get("models", {}).get(queue_id, {})
        csv_path = tracked.get("csv_file", "")
        if not csv_path or not os.path.isfile(csv_path):
            csv_path = self.find_csv_for_model(queue_id)
        if not csv_path or not os.path.isfile(csv_path):
            return None

        prompts = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompts.append(
                    {
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
                    }
                )
        return {
            "queue_id": queue_id,
            "model": tracked.get("model_name", queue_id),
            "csv_file": csv_path,
            "prompts": prompts,
        }

    def compute_difficulty_stats(self, csv_path: str, total_per_diff: int = 10) -> dict[str, Any]:
        stats = {
            diff: {"count": total_per_diff, "avg_latency": 0, "completed": 0}
            for diff in ["easy", "medium", "hard", "adversarial", "long_context"]
        }
        if not csv_path or not os.path.isfile(csv_path):
            return stats
        try:
            by_diff: dict[str, list[float]] = {}
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    diff = row.get("difficulty", "")
                    lat = float(row.get("latency", 0) or 0)
                    by_diff.setdefault(diff, [])
                    if lat > 0:
                        by_diff[diff].append(lat)
            for diff, vals in by_diff.items():
                if diff in stats:
                    stats[diff]["avg_latency"] = round(sum(vals) / len(vals), 2) if vals else 0
                    stats[diff]["completed"] = len(vals)
        except Exception:
            pass
        return stats

    def load_llama_bench(self) -> dict[str, Any]:
        perf_data: dict[str, Any] = {}
        summary_file = self.perf_dir / "llama_bench_summary.json"
        if summary_file.is_file():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for qid, result in raw.get("results", {}).items():
                    key = result.get("model_name", qid)
                    perf_data[key] = {
                        "tg_tps": result.get("tg_tps", 0),
                        "pp_tps": result.get("pp_tps", 0),
                        "tg_stddev": result.get("tg_stddev", 0),
                        "pp_stddev": result.get("pp_stddev", 0),
                    }
                return perf_data
            except Exception:
                pass

        for path in self.perf_dir.glob("*_llama_bench.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                key = result.get("model_name", path.name)
                perf_data[key] = {
                    "tg_tps": result.get("tg_tps", 0),
                    "pp_tps": result.get("pp_tps", 0),
                    "tg_stddev": result.get("tg_stddev", 0),
                    "pp_stddev": result.get("pp_stddev", 0),
                }
            except Exception:
                pass
        return perf_data

    def load_promptfoo(self) -> dict[str, Any]:
        quality_data: dict[str, Any] = {}
        summary_file = self.quality_dir / "promptfoo_summary.json"
        if summary_file.is_file():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for tag, result in raw.get("results", {}).items():
                    quality_data[tag] = {
                        "pass_rate": result.get("pass_rate", 0),
                        "passed": result.get("passed", 0),
                        "total_tests": result.get("total_tests", 0),
                    }
                return quality_data
            except Exception:
                pass

        for path in self.quality_dir.glob("*_promptfoo.json"):
            if path.name.endswith("_promptfoo_raw.json"):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                tag = result.get("model", path.name)
                quality_data[tag] = {
                    "pass_rate": result.get("pass_rate", 0),
                    "passed": result.get("passed", 0),
                    "total_tests": result.get("total_tests", 0),
                }
            except Exception:
                pass
        return quality_data


def _safe_float(value: Any) -> float | None:
    try:
        return round(float(value), 2) if value and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    return BenchmarkArtifactStore._truthy_success(value) if value and str(value).strip() else None
