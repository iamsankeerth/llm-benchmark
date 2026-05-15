"""
Unified Benchmark Orchestrator (Mega-Orchestrator)
====================================================
Master script that runs the full benchmark pipeline efficiently.

Lifecycle: One-Pull, Three-Tests, One-Delete
  1. Pull model from Ollama
  2. Phase 1: Latency, VRAM, JSON Schema checks (`run_benchmarks.py`)
  3. Phase 2: Hardware Performance via llama-bench (`run_llama_bench.py`)
  4. Phase 3: Quality Evaluation via promptfoo (`run_promptfoo.py`)
  5. Delete model from Ollama
  6. Generate unified dashboard report (`model_comparator.py`)

Usage:
    python scripts/run_full_benchmark.py
    python scripts/run_full_benchmark.py --model tinyllama
"""

import os
import sys
import argparse
import subprocess
from rich.console import Console

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_QUEUE
from src.model_comparator import ModelComparator

console = Console()

def log(msg, style=None):
    if style:
        console.print(f"[{style}]{msg}[/{style}]")
    else:
        console.print(msg)


def download_model(ollama_tag: str) -> bool:
    log(f"\n[bold blue]>>> Downloading {ollama_tag}...[/bold blue]")
    try:
        result = subprocess.run(
            ["ollama", "pull", ollama_tag],
            capture_output=True, encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            log(f"[green]✓ Successfully downloaded {ollama_tag}[/green]")
            return True
        else:
            log(f"[red]✗ Pull failed: {result.stderr.strip()}[/red]")
            return False
    except Exception as e:
        log(f"[red]✗ Download error: {e}[/red]")
        return False


def delete_model(ollama_tag: str):
    log(f"\n[bold blue]>>> Deleting {ollama_tag} to free up space...[/bold blue]")
    try:
        result = subprocess.run(
            ["ollama", "rm", ollama_tag],
            capture_output=True, encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            log(f"[green]✓ Deleted {ollama_tag}[/green]")
        else:
            log(f"[yellow]⚠ ollama rm failed: {result.stderr.strip()}[/yellow]")
    except Exception as e:
        log(f"[yellow]⚠ Could not delete: {e}[/yellow]")


def run_phase(script_name: str, display_name: str):
    log(f"\n[bold magenta]--- Running {display_name} ---[/bold magenta]")
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        log(f"[red]✗ Error during {display_name}: {e}[/red]")


def main(target_model=None):
    log("\n[bold cyan]" + "=" * 60 + "\n MEGA-ORCHESTRATOR: ONE-PULL, THREE-TESTS, ONE-DELETE\n" + "=" * 60 + "[/bold cyan]")

    # Identify eligible Ollama models
    eligible = []
    for m in MODEL_QUEUE:
        if m.get("status") != "pending":
            continue
        if m.get("source") != "ollama":
            continue  # Orchestrator lifecycle relies on ollama pull/rm
        if target_model and target_model.lower() not in m["requested_name"].lower():
            continue
        eligible.append(m)

    log(f"Found {len(eligible)} models to test.\n")

    # Set orchestrator flags to tell sub-scripts to skip pull/delete
    os.environ["BENCHMARK_SKIP_LIFECYCLE"] = "1"

    for idx, model in enumerate(eligible, 1):
        name = model["requested_name"]
        queue_id = model["queue_id"]
        tag = model.get("ollama_tag", "")

        if not tag:
            log(f"[yellow]Skipping {name} (no ollama_tag)[/yellow]")
            continue

        log(f"\n[bold white on blue] MODEL {idx}/{len(eligible)}: {name} ({tag}) [/bold white on blue]")

        # Restrict sub-scripts to this model only
        os.environ["BENCHMARK_SINGLE_MODEL"] = queue_id

        # 1. Download
        if not download_model(tag):
            continue

        # 2. Test A: Phase 1 (Latency, VRAM, JSON)
        run_phase("run_benchmarks.py", "Test A: Phase 1 (CSV Data)")

        # 3. Test B: Phase 2 (llama-bench tg_tps)
        run_phase("run_llama_bench.py", "Test B: llama-bench (Decode Speed)")

        # 4. Test C: Phase 3 (promptfoo pass_rate)
        run_phase("run_promptfoo.py", "Test C: promptfoo (Quality Eval)")

        # 5. Delete
        delete_model(tag)

    # 6. Generate final report
    log("\n[bold cyan]" + "=" * 60 + "\n FINAL STEP: GENERATING DASHBOARD REPORT\n" + "=" * 60 + "[/bold cyan]")
    
    # Remove single-model filter so the comparator aggregates everything
    os.environ.pop("BENCHMARK_SINGLE_MODEL", None)
    
    comparator = ModelComparator()
    comparator.run_offline_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mega-Orchestrator")
    parser.add_argument("--model", type=str, help="Filter to a specific model")
    args = parser.parse_args()
    
    main(target_model=args.model)
