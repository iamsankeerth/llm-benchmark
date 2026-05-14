import os
import glob
import json
import pandas as pd
from rich.console import Console
from config import REPORTS_DIR, RESULTS_DIR, MODEL_QUEUE
from pathlib import Path

console = Console()

class ModelComparator:
    def __init__(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self.report_path = os.path.join(REPORTS_DIR, "model_comparison_report.md")

    @staticmethod
    def _truthy_success(value) -> bool:
        """Parse a CSV cell into boolean - only explicit truthy strings count as success."""
        return str(value).strip().lower() in {"true", "1", "yes"}

    def run_offline_report(self):
        """
        Phase 3: Assembles a comprehensive technical report from benchmark CSVs.
        Includes model-level metrics, temperature success rates, and written analysis.
        """
        console.print("[bold yellow]Generating Offline Model Comparison Report...[/bold yellow]")

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

        summary = combined_df.groupby('model').agg({
            'tps': 'mean',
            'ttft': 'mean',
            'latency': 'mean',
            'peak_vram_mb': 'max',
            'prompt_id': 'count'
        }).rename(columns={'prompt_id': 'total_prompts'}).reset_index()

        # JSON success rate by temperature
        for temp_col in ['temp_0.0_success', 'temp_0.7_success', 'temp_1.0_success']:
            if temp_col in combined_df.columns:
                temp_summary = combined_df.groupby('model')[temp_col].apply(
                    lambda x: x.map(ModelComparator._truthy_success).sum() / len(x) * 100 if len(x) > 0 else 0
                ).reset_index()
                temp_summary.columns = ['model', f'{temp_col}_rate']
                summary = summary.merge(temp_summary, on='model', how='left')

        # Category breakdown
        cat_summary = None
        try:
            cat_summary = combined_df.pivot_table(
                values='tps', index='model', columns='category', aggfunc='mean'
            ).round(2)
        except Exception:
            pass

        # Difficulty breakdown
        diff_summary = None
        try:
            diff_summary = combined_df.pivot_table(
                values='tps', index='model', columns='difficulty', aggfunc='mean'
            ).round(2)
        except Exception:
            pass

        # Top failures
        error_rows = combined_df[combined_df['error'].notna() & (combined_df['error'] != '')]
        top_failures = error_rows.groupby(['model', 'category']).agg({
            'error': 'count',
            'prompt_id': lambda x: list(x.head(5))
        }).rename(columns={'error': 'error_count', 'prompt_id': 'sample_prompt_ids'}).reset_index()

        # Determine best models
        if len(summary) > 0:
            best_speed = summary.loc[summary['tps'].idxmax()] if 'tps' in summary.columns else None
            best_latency = summary.loc[summary['latency'].idxmin()] if 'latency' in summary.columns else None
            best_vram = summary.loc[summary['peak_vram_mb'].idxmin()] if 'peak_vram_mb' in summary.columns else None

            success_0_col = 'temp_0.0_success_rate' if 'temp_0.0_success_rate' in summary.columns else None
            best_structured = summary.loc[summary[success_0_col].idxmax()] if success_0_col and success_0_col in summary.columns else None
        else:
            best_speed = best_latency = best_vram = best_structured = None

        with open(self.report_path, "w", encoding="utf-8") as f:
            # --- Queue / progress stats using MODEL_QUEUE as denominator ---
            total_in_queue = len(MODEL_QUEUE)
            runnable_count = sum(1 for m in MODEL_QUEUE if m["status"] == "pending")
            unsupported_queue = sum(1 for m in MODEL_QUEUE if m["status"] == "provider_unsupported")
            deferred_queue = sum(1 for m in MODEL_QUEUE if m["status"] == "deferred_vision")

            # Overlay test_progress.json statuses for actual benchmark outcomes
            progress_path = Path(__file__).parent.parent / "test_progress.json"
            tracked_models: dict = {}
            if progress_path.exists():
                try:
                    with open(progress_path, "r", encoding="utf-8") as pf:
                        tracked_models = json.load(pf).get("models", {})
                except Exception:
                    pass

            # Count tracked statuses
            tracked_completed = sum(1 for d in tracked_models.values() if d.get("status") == "completed")
            tracked_failed = sum(1 for d in tracked_models.values() if d.get("status") == "failed")
            tracked_skipped = sum(1 for d in tracked_models.values()
                                  if d.get("status") in ("skipped", "provider_unsupported"))
            tracked_deferred = sum(1 for d in tracked_models.values()
                                   if d.get("status") == "deferred_vision")

            benchmarked_count = len(summary)

            # Build skipped / deferred display lists
            skipped_models = {
                name: data for name, data in tracked_models.items()
                if data.get("status") in ("skipped", "provider_unsupported")
            }
            deferred_models = {
                name: data for name, data in tracked_models.items()
                if data.get("status") == "deferred_vision"
            }

            f.write("# LLM Benchmark: All-Compatible Model Study\n\n")
            f.write(f"Generated on RTX 2050 (4GB VRAM). "
                     f"Source of truth: `compatible_models.py` → `config.MODEL_QUEUE`.\n\n")

            # Build models with variant notes for report
            variant_models = [m for m in MODEL_QUEUE if m.get("variant_note")]

            f.write("## Queue Coverage\n\n")
            f.write(f"| Metric | Count |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total models in queue | {total_in_queue} |\n")
            f.write(f"| Runnable models | {runnable_count} |\n")
            f.write(f"| Benchmark completed | {tracked_completed} |\n")
            f.write(f"| Benchmark failed | {tracked_failed} |\n")
            f.write(f"| Provider unsupported | {unsupported_queue + tracked_skipped} |\n")
            f.write(f"| Deferred (vision) | {deferred_queue + tracked_deferred} |\n")
            f.write(f"| CSV models in report | {benchmarked_count} |\n")
            if variant_models:
                f.write(f"| Models with variant fallback | {len(variant_models)} |\n")
            f.write("\n")

            # Duplicate check
            qids = [m["queue_id"] for m in MODEL_QUEUE]
            dupes = len(qids) - len(set(qids))
            if dupes:
                f.write(f"**WARNING:** {dupes} duplicate queue_id(s) detected in MODEL_QUEUE.\n\n")
            else:
                f.write("**Queue ID audit:** All queue_ids are unique.\n\n")
            f.write("---\n\n")

            f.write("## 1. Study Overview\n\n")
            f.write("This is an **all-compatible-model study** driven by `compatible_models.py`. "
                     "Every model that fits within 4 GB VRAM (RTX 2050) is queued for its "
                     "category-matched benchmark:\n"
                     "- **Coding** models → Coding Generation prompts\n"
                     "- **Chat** models → Chat & Generation prompts\n"
                     "- **Reasoning** models → Medium Reasoning prompts\n"
                     "- **Vision** models → deferred until real image assets exist\n\n"
                     "Each model is evaluated on throughput (TPS), latency (TTFT + total), "
                     "peak VRAM usage, and structured JSON output reliability at "
                     "temperatures 0.0, 0.7, and 1.0.\n\n")

            # --- Skipped / unsupported ---
            if skipped_models:
                f.write("## 2. Skipped Models (Unsupported Provider)\n\n")
                f.write("| Model | Reason |\n")
                f.write("|-------|--------|\n")
                for name, data in sorted(skipped_models.items()):
                    reason = (data.get("error") or "Unknown").replace("\n", " ")[:100]
                    f.write(f"| {name} | {reason} |\n")
                f.write("\n\n")
                section_offset = 1
            else:
                section_offset = 0

            # --- Deferred vision ---
            if deferred_models:
                section_num = 2 + section_offset
                f.write(f"## {section_num}. Deferred Models (Multimodal/Vision)\n\n")
                f.write("These models require real image assets (`image_path`) and are deferred "
                         "until `prompts/images/` is populated.\n\n")
                f.write("| Model | Reason |\n")
                f.write("|-------|--------|\n")
                for name, data in sorted(deferred_models.items()):
                    reason = (data.get("error") or "Deferred until image assets exist").replace("\n", " ")[:100]
                    f.write(f"| {name} | {reason} |\n")
                f.write("\n\n")
                section_offset += 1

            # --- Performance metrics ---
            sec = 2 + section_offset
            f.write(f"## {sec}. Global Performance Metrics\n\n")

            f.write("### Per-Model Summary\n\n")
            f.write(summary.round(2).to_markdown(index=False))
            f.write("\n\n")

            f.write("### JSON Structured Output Success Rate\n\n")
            f.write("Percentage of prompts where the model produced valid JSON matching the expected schema "
                     "at each temperature setting.\n\n")
            temp_columns = ['model']
            for tc in ['temp_0.0_success_rate', 'temp_0.7_success_rate', 'temp_1.0_success_rate']:
                if tc in summary.columns:
                    temp_columns.append(tc)
            if len(temp_columns) > 1:
                f.write(summary[temp_columns].round(2).to_markdown(index=False))
            else:
                f.write("No temperature success data available.\n")
            f.write("\n\n")

            f.write("### Average TPS by Category\n\n")
            if cat_summary is not None:
                f.write(cat_summary.to_markdown())
            else:
                f.write("Insufficient variance to calculate pivot table.\n")
            f.write("\n\n")

            f.write("### Average TPS by Difficulty\n\n")
            if diff_summary is not None and not diff_summary.empty:
                f.write(diff_summary.to_markdown())
            else:
                f.write("Difficulty breakdown not available.\n")
            f.write("\n\n")

            f.write("### Top Failures\n\n")
            if len(top_failures) > 0:
                f.write(top_failures.to_markdown(index=False))
            else:
                f.write("No failures recorded.\n")
            f.write("\n\n")

            f.write("## Best Model Analysis\n\n")

            if best_speed is not None:
                f.write(f"**Fastest Throughput:** {best_speed['model']} "
                        f"({best_speed['tps']:.1f} tokens/sec avg)\n\n")
            if best_latency is not None:
                f.write(f"**Lowest Latency:** {best_latency['model']} "
                        f"({best_latency['latency']:.3f}s avg)\n\n")
            if best_structured is not None and success_0_col:
                f.write(f"**Best Structured Output (temp 0.0):** {best_structured['model']} "
                        f"({best_structured[success_0_col]:.1f}% success)\n\n")
            if best_vram is not None:
                f.write(f"**Most Memory Efficient:** {best_vram['model']} "
                        f"(peak {best_vram['peak_vram_mb']:.0f} MB VRAM)\n\n")

            f.write("### Recommendation\n\n")
            f.write("- **Speed:** The fastest model is recommended for real-time interactive use "
                     "where latency matters most.\n")
            f.write("- **Structured Output Reliability:** The model with highest JSON schema compliance "
                     "at temperature 0.0 is best for API responses, tool use, and structured data extraction.\n")
            f.write("- **Memory Efficiency:** The model with lowest peak VRAM is ideal for deployment "
                     "on constrained hardware or running alongside other services.\n")
            f.write("- **Overall Local Assistant:** Consider the model that balances speed, structured "
                     "output reliability, and memory efficiency. If a single model excels in most categories, "
                     "it is the best all-around local assistant.\n\n")

            f.write("## Methodology Notes\n\n")
            f.write("- Benchmarks run with `MAX_NEW_TOKENS` limit per prompt.\n")
            f.write("- GPU VRAM measured via `pynvml` with peak tracking.\n")
            f.write("- Structured output validated via Pydantic schema enforcement.\n")
            f.write("- Vision prompts require real image assets via `image_path`; "
                     "missing images are reported as errors rather than silently skipped.\n")
            f.write("- Models sourced from `compatible_models.py` (llmfit analysis). "
                     "Only models with known Ollama tags or HuggingFace GGUF repositories are attempted.\n")
            f.write("- AWQ / GPTQ / FP8 quantised repositories without corresponding "
                     "Ollama tags are marked as `provider_unsupported` and not downloaded.\n")

        # Dashboard data export
        dashboard_data = {
            "summary": summary.to_dict(orient="records"),
        }
        if cat_summary is not None:
            dashboard_data["category_summary"] = cat_summary.reset_index().to_dict(orient="records")

        with open(os.path.join(REPORTS_DIR, "dashboard_data.json"), "w") as json_file:
            json.dump(dashboard_data, json_file, indent=2)

        console.print(f"[bold green]Report exported to {self.report_path}[/bold green]")
        console.print(f"[bold cyan]Dashboard Data exported to {os.path.join(REPORTS_DIR, 'dashboard_data.json')}[/bold cyan]")
