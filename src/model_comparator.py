import os
import glob
import json
import pandas as pd
from rich.console import Console
from config import REPORTS_DIR, RESULTS_DIR, MODEL_QUEUE
from pathlib import Path

console = Console()

# ── Paths to phase-specific result directories ────────────────────────────────
PERF_DIR    = os.path.join(RESULTS_DIR, "perf")     # llama-bench JSON files
QUALITY_DIR = os.path.join(RESULTS_DIR, "quality")  # promptfoo JSON files


class ModelComparator:
    def __init__(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self.report_path = os.path.join(REPORTS_DIR, "model_comparison_report.md")

    @staticmethod
    def _truthy_success(value) -> bool:
        """Parse a CSV cell into boolean - only explicit truthy strings count as success."""
        return str(value).strip().lower() in {"true", "1", "yes"}

    # ── Helper: load llama-bench results ────────────────────────────────────
    @staticmethod
    def _load_llama_bench() -> dict:
        """
        Load llama-bench results from results/perf/.
        Returns dict keyed by model_name → {tg_tps, pp_tps, tg_stddev, pp_stddev}.
        Gracefully returns {} if the directory or files are missing.
        """
        perf_data = {}
        summary_file = os.path.join(PERF_DIR, "llama_bench_summary.json")

        # Prefer the combined summary file
        if os.path.isfile(summary_file):
            try:
                with open(summary_file) as f:
                    raw = json.load(f)
                for qid, r in raw.get("results", {}).items():
                    key = r.get("model_name", qid)
                    perf_data[key] = {
                        "tg_tps":    r.get("tg_tps", 0),
                        "pp_tps":    r.get("pp_tps", 0),
                        "tg_stddev": r.get("tg_stddev", 0),
                        "pp_stddev": r.get("pp_stddev", 0),
                    }
                return perf_data
            except Exception:
                pass

        # Fallback: scan individual JSON files
        for jf in glob.glob(os.path.join(PERF_DIR, "*_llama_bench.json")):
            try:
                with open(jf) as f:
                    r = json.load(f)
                key = r.get("model_name", os.path.basename(jf))
                perf_data[key] = {
                    "tg_tps":    r.get("tg_tps", 0),
                    "pp_tps":    r.get("pp_tps", 0),
                    "tg_stddev": r.get("tg_stddev", 0),
                    "pp_stddev": r.get("pp_stddev", 0),
                }
            except Exception:
                pass

        return perf_data

    # ── Helper: load promptfoo results ───────────────────────────────────────
    @staticmethod
    def _load_promptfoo() -> dict:
        """
        Load promptfoo quality results from results/quality/.
        Returns dict keyed by model tag → {pass_rate, passed, total_tests}.
        Gracefully returns {} if missing.
        """
        quality_data = {}
        summary_file = os.path.join(QUALITY_DIR, "promptfoo_summary.json")

        if os.path.isfile(summary_file):
            try:
                with open(summary_file) as f:
                    raw = json.load(f)
                for tag, r in raw.get("results", {}).items():
                    quality_data[tag] = {
                        "pass_rate":   r.get("pass_rate", 0),
                        "passed":      r.get("passed", 0),
                        "total_tests": r.get("total_tests", 0),
                    }
                return quality_data
            except Exception:
                pass

        # Fallback: scan individual per-model files
        for jf in glob.glob(os.path.join(QUALITY_DIR, "*_promptfoo.json")):
            try:
                with open(jf) as f:
                    r = json.load(f)
                tag = r.get("model", os.path.basename(jf))
                quality_data[tag] = {
                    "pass_rate":   r.get("pass_rate", 0),
                    "passed":      r.get("passed", 0),
                    "total_tests": r.get("total_tests", 0),
                }
            except Exception:
                pass

        return quality_data

    # ── Main report generator ────────────────────────────────────────────────
    def run_offline_report(self):
        """
        Phase 3: Assembles a comprehensive technical report by merging:
          - Phase 1 CSV data: TTFT, latency, peak VRAM, JSON schema success rates
          - llama-bench data: tg_tps (Decode Speed), pp_tps (Prefill Speed)
          - promptfoo data:   Functional Pass Rate (%)
        """
        console.print("[bold yellow]Generating Offline Model Comparison Report...[/bold yellow]")

        # ── Load Phase 1 CSV data ────────────────────────────────────────────
        csv_files = glob.glob(os.path.join(RESULTS_DIR, "phase1", "*.csv"))
        if not csv_files:
            console.print("[red]No Phase 1 CSV records found! Run benchmarks first.[/red]")
            return

        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                console.print(f"[red]Error reading {file}: {e}[/red]")

        if not dfs:
            return

        combined_df = pd.concat(dfs, ignore_index=True)

        summary = combined_df.groupby('model').agg(
            ttft=('ttft', 'mean'),
            latency=('latency', 'mean'),
            peak_vram_mb=('peak_vram_mb', 'max'),
            total_prompts=('prompt_id', 'count'),
        ).reset_index()

        # JSON structured output success rate by temperature
        for temp_col in ['temp_0.0_success', 'temp_0.7_success', 'temp_1.0_success']:
            if temp_col in combined_df.columns:
                temp_summary = combined_df.groupby('model')[temp_col].apply(
                    lambda x: x.map(ModelComparator._truthy_success).sum() / len(x) * 100
                    if len(x) > 0 else 0
                ).reset_index()
                temp_summary.columns = ['model', f'{temp_col}_rate']
                summary = summary.merge(temp_summary, on='model', how='left')

        # Category breakdown
        cat_summary = None
        try:
            cat_summary = combined_df.pivot_table(
                values='latency', index='model', columns='category', aggfunc='mean'
            ).round(3)
        except Exception:
            pass

        # Top failures
        error_rows = combined_df[combined_df['error'].notna() & (combined_df['error'] != '')]
        top_failures = error_rows.groupby(['model', 'category']).agg(
            error_count=('error', 'count'),
            sample_prompt_ids=('prompt_id', lambda x: list(x.head(5)))
        ).reset_index()

        # ── Load llama-bench & promptfoo data ───────────────────────────────
        perf_data    = self._load_llama_bench()
        quality_data = self._load_promptfoo()

        has_perf    = len(perf_data) > 0
        has_quality = len(quality_data) > 0

        # Merge performance data into summary
        if has_perf:
            perf_rows = []
            for model_name, p in perf_data.items():
                perf_rows.append({
                    "model":      model_name,
                    "tg_tps":     p["tg_tps"],
                    "pp_tps":     p["pp_tps"],
                    "tg_stddev":  p["tg_stddev"],
                    "pp_stddev":  p["pp_stddev"],
                })
            perf_df = pd.DataFrame(perf_rows)
            # Attempt fuzzy join on model name
            summary = summary.merge(perf_df, on='model', how='left')

        # Merge quality data
        if has_quality:
            quality_rows = []
            for tag, q in quality_data.items():
                quality_rows.append({
                    "model":        tag,
                    "pass_rate":    q["pass_rate"],
                    "tests_passed": q["passed"],
                    "tests_total":  q["total_tests"],
                })
            quality_df = pd.DataFrame(quality_rows)
            summary = summary.merge(quality_df, on='model', how='left')

        # ── Determine best models ────────────────────────────────────────────
        best_decode   = None
        best_quality  = None
        best_latency  = None
        best_vram     = None

        if has_perf and 'tg_tps' in summary.columns and summary['tg_tps'].notna().any():
            best_decode = summary.loc[summary['tg_tps'].idxmax()]
        if has_quality and 'pass_rate' in summary.columns and summary['pass_rate'].notna().any():
            best_quality = summary.loc[summary['pass_rate'].idxmax()]
        if 'latency' in summary.columns:
            best_latency = summary.loc[summary['latency'].idxmin()]
        if 'peak_vram_mb' in summary.columns:
            best_vram = summary.loc[summary['peak_vram_mb'].idxmin()]

        # ── Write markdown report ────────────────────────────────────────────
        with open(self.report_path, "w", encoding="utf-8") as f:
            total_in_queue = len(MODEL_QUEUE)
            runnable_count = sum(1 for m in MODEL_QUEUE if m["status"] == "pending")
            unsupported_queue = sum(1 for m in MODEL_QUEUE if m["status"] == "provider_unsupported")
            deferred_queue = sum(1 for m in MODEL_QUEUE if m["status"] == "deferred_vision")

            progress_path = Path(__file__).parent.parent / "test_progress.json"
            tracked_models: dict = {}
            if progress_path.exists():
                try:
                    with open(progress_path, "r", encoding="utf-8") as pf:
                        tracked_models = json.load(pf).get("models", {})
                except Exception:
                    pass

            tracked_completed = sum(1 for d in tracked_models.values() if d.get("status") == "completed")
            tracked_failed    = sum(1 for d in tracked_models.values() if d.get("status") == "failed")
            tracked_skipped   = sum(1 for d in tracked_models.values()
                                    if d.get("status") in ("skipped", "provider_unsupported"))
            benchmarked_count = len(summary)

            skipped_models  = {n: d for n, d in tracked_models.items()
                               if d.get("status") in ("skipped", "provider_unsupported")}
            deferred_models = {n: d for n, d in tracked_models.items()
                               if d.get("status") == "deferred_vision"}

            f.write("# LLM Benchmark: All-Compatible Model Study\n\n")
            f.write(f"Generated on RTX 2050 (4GB VRAM). "
                    f"Source of truth: `compatible_models.py` → `config.MODEL_QUEUE`.\n\n")

            # Metric sources legend
            f.write("> **Metric Sources**\n")
            f.write("> - `tg_tps` / `pp_tps` — llama-bench C++ binary (hardware-accurate)\n")
            f.write("> - `pass_rate` — promptfoo quality evaluation (35 curated tests)\n")
            f.write("> - `ttft` / `latency` / `peak_vram_mb` — Phase 1 Ollama streaming\n\n")

            f.write("## Queue Coverage\n\n")
            f.write("| Metric | Count |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total models in queue | {total_in_queue} |\n")
            f.write(f"| Runnable models | {runnable_count} |\n")
            f.write(f"| Benchmark completed | {tracked_completed} |\n")
            f.write(f"| Benchmark failed | {tracked_failed} |\n")
            f.write(f"| Provider unsupported | {unsupported_queue + tracked_skipped} |\n")
            f.write(f"| Deferred (vision) | {deferred_queue} |\n")
            f.write(f"| llama-bench data available | {len(perf_data)} |\n")
            f.write(f"| promptfoo data available | {len(quality_data)} |\n")
            f.write(f"| CSV models in report | {benchmarked_count} |\n\n")

            qids = [m["queue_id"] for m in MODEL_QUEUE]
            dupes = len(qids) - len(set(qids))
            if dupes:
                f.write(f"**WARNING:** {dupes} duplicate queue_id(s) detected.\n\n")
            else:
                f.write("**Queue ID audit:** All queue_ids are unique.\n\n")
            f.write("---\n\n")

            # Skipped / deferred sections
            section_offset = 0
            if skipped_models:
                f.write("## 2. Skipped Models (Unsupported Provider)\n\n")
                f.write("| Model | Reason |\n|-------|--------|\n")
                for name, data in sorted(skipped_models.items()):
                    reason = (data.get("error") or "Unknown").replace("\n", " ")[:100]
                    f.write(f"| {name} | {reason} |\n")
                f.write("\n\n")
                section_offset = 1
            if deferred_models:
                sec = 2 + section_offset
                f.write(f"## {sec}. Deferred Models (Vision)\n\n")
                f.write("| Model | Reason |\n|-------|--------|\n")
                for name, data in sorted(deferred_models.items()):
                    reason = (data.get("error") or "Deferred until image assets exist").replace("\n", " ")[:100]
                    f.write(f"| {name} | {reason} |\n")
                f.write("\n\n")
                section_offset += 1

            # ── PRIMARY METRICS TABLE ────────────────────────────────────────
            sec = 2 + section_offset
            f.write(f"## {sec}. Core Performance Metrics\n\n")
            f.write("The primary metrics for assessing local LLM inference quality:\n\n")
            f.write("| Metric | Description | Why it matters |\n")
            f.write("|--------|-------------|----------------|\n")
            f.write("| `tg_tps` | **Decode Speed** (tokens/sec generated) | How fast the model streams code/text to you |\n")
            f.write("| `pp_tps` | Prefill Speed (tokens/sec processed) | How fast the model reads your long prompt/context |\n")
            f.write("| `pass_rate` | **Functional Quality** (% tests passed) | Whether generated outputs are correct |\n")
            f.write("| `ttft` | Time to First Token (seconds) | Perceived responsiveness |\n")
            f.write("| `latency` | Total Response Time (seconds) | End-to-end wait time |\n")
            f.write("| `peak_vram_mb` | Peak GPU Memory (MB) | Hardware compatibility |\n\n")

            # Decode speed table (primary)
            f.write(f"### {sec}.1 Decode Speed — `tg_tps` (Primary Metric)\n\n")
            if has_perf and 'tg_tps' in summary.columns:
                perf_cols = ['model', 'tg_tps', 'tg_stddev', 'pp_tps', 'pp_stddev']
                available = [c for c in perf_cols if c in summary.columns]
                f.write(summary[available].dropna(subset=['tg_tps'])
                        .sort_values('tg_tps', ascending=False)
                        .round(2)
                        .to_markdown(index=False))
                f.write("\n\n")
                f.write("> **`tg_tps` = Decode Speed**: This is the true inference speed — "
                        "how fast the model generates tokens after the first token. "
                        "Higher is better. Measured via llama-bench C++ binary.\n\n")
            else:
                f.write("_No llama-bench data yet. Run `python scripts/run_llama_bench.py` to generate._\n\n")

            # Quality table (primary)
            f.write(f"### {sec}.2 Functional Quality — `pass_rate` (Primary Metric)\n\n")
            if has_quality and 'pass_rate' in summary.columns:
                quality_cols = ['model', 'pass_rate', 'tests_passed', 'tests_total']
                available = [c for c in quality_cols if c in summary.columns]
                f.write(summary[available].dropna(subset=['pass_rate'])
                        .sort_values('pass_rate', ascending=False)
                        .round(1)
                        .to_markdown(index=False))
                f.write("\n\n")
                f.write("> **`pass_rate`**: Percentage of 35 curated tests passed. "
                        "Tests cover coding, reasoning, structured output, and chat. "
                        "Graded by `qwen2.5:3b-instruct` as LLM judge.\n\n")
            else:
                f.write("_No promptfoo data yet. Run `python scripts/run_promptfoo.py` to generate._\n\n")

            # Full merged summary
            f.write(f"### {sec}.3 Full Merged Summary\n\n")
            all_cols = ['model', 'tg_tps', 'pp_tps', 'pass_rate',
                        'ttft', 'latency', 'peak_vram_mb', 'total_prompts']
            available = [c for c in all_cols if c in summary.columns]
            f.write(summary[available].round(2).to_markdown(index=False))
            f.write("\n\n")

            # Temperature JSON success rates
            f.write(f"### {sec}.4 JSON Schema Success Rate by Temperature\n\n")
            f.write("Percentage of prompts where the model produced valid Pydantic-validated JSON:\n\n")
            temp_columns = ['model']
            for tc in ['temp_0.0_success_rate', 'temp_0.7_success_rate', 'temp_1.0_success_rate']:
                if tc in summary.columns:
                    temp_columns.append(tc)
            if len(temp_columns) > 1:
                f.write(summary[temp_columns].round(2).to_markdown(index=False))
            else:
                f.write("No temperature success data available.\n")
            f.write("\n\n")

            # Category breakdown
            if cat_summary is not None:
                f.write(f"### {sec}.5 Average Latency by Category\n\n")
                f.write(cat_summary.to_markdown())
                f.write("\n\n")

            # Top failures
            f.write(f"### {sec}.6 Top Failures\n\n")
            if len(top_failures) > 0:
                f.write(top_failures.to_markdown(index=False))
            else:
                f.write("No failures recorded.\n")
            f.write("\n\n")

            # Best model analysis
            f.write("## Best Model Analysis\n\n")
            if best_decode is not None:
                f.write(f"**🚀 Fastest Decode (tg_tps):** `{best_decode['model']}` "
                        f"— {best_decode['tg_tps']:.1f} tokens/sec\n\n")
            if best_quality is not None:
                f.write(f"**✅ Best Functional Quality:** `{best_quality['model']}` "
                        f"— {best_quality['pass_rate']:.1f}% pass rate\n\n")
            if best_latency is not None:
                f.write(f"**⚡ Lowest Latency:** `{best_latency['model']}` "
                        f"— {best_latency['latency']:.3f}s avg\n\n")
            if best_vram is not None:
                f.write(f"**💾 Most Memory Efficient:** `{best_vram['model']}` "
                        f"— peak {best_vram['peak_vram_mb']:.0f} MB VRAM\n\n")

            f.write("### Recommendation\n\n")
            f.write("- **Best for Coding**: Prioritize `tg_tps` (decode speed) + `pass_rate` (functional correctness).\n")
            f.write("- **Best for Chat**: Prioritize low `ttft` (perceived responsiveness) + `pass_rate`.\n")
            f.write("- **Best for Reasoning**: Prioritize `pass_rate` at temperature 0.0 (determinism).\n")
            f.write("- **Best for Constrained Hardware**: Lowest `peak_vram_mb` + acceptable `tg_tps`.\n\n")

            f.write("## Methodology Notes\n\n")
            f.write("- **Decode Speed (`tg_tps`)** measured by llama-bench C++ binary (b9159) "
                    "with `n_gen=128`, `n_gpu_layers=99`, flash attention enabled.\n")
            f.write("- **Functional Quality (`pass_rate`)** measured by promptfoo with 35 curated "
                    "tests graded by `qwen2.5:3b-instruct`.\n")
            f.write("- **VRAM** measured via `pynvml` with peak tracking during Phase 1.\n")
            f.write("- **JSON Validity** enforced via Ollama `format=schema` + Pydantic validation.\n")
            f.write("- **Ephemeral lifecycle**: each model is pulled → benchmarked → deleted "
                    "to keep disk usage minimal.\n")

        # ── Export dashboard_data.json ────────────────────────────────────────
        dashboard_data = {
            "summary": summary.fillna(0).to_dict(orient="records"),
            "has_perf_data":    has_perf,
            "has_quality_data": has_quality,
        }
        if cat_summary is not None:
            dashboard_data["category_summary"] = (
                cat_summary.reset_index().to_dict(orient="records")
            )

        dashboard_json = os.path.join(REPORTS_DIR, "dashboard_data.json")
        with open(dashboard_json, "w") as jf:
            json.dump(dashboard_data, jf, indent=2)

        console.print(f"[bold green]✓ Report: {self.report_path}[/bold green]")
        console.print(f"[bold cyan]✓ Dashboard data: {dashboard_json}[/bold cyan]")
        if has_perf:
            console.print(f"[bold green]✓ llama-bench data merged ({len(perf_data)} models)[/bold green]")
        else:
            console.print("[yellow]⚠ No llama-bench data. Run run_llama_bench.py first.[/yellow]")
        if has_quality:
            console.print(f"[bold green]✓ promptfoo data merged ({len(quality_data)} models)[/bold green]")
        else:
            console.print("[yellow]⚠ No promptfoo data. Run run_promptfoo.py first.[/yellow]")
