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

# Force UTF-8 output on Windows to avoid UnicodeEncodeError with Rich ✓/✗ characters
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["ANSI_COLORS_DISABLED"] = "1"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import subprocess
from rich.console import Console

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_QUEUE
from src.artifact_store import BenchmarkArtifactStore
from src.lifecycle import delete_ollama_tag, pull_ollama_tag
from src.model_entry import as_model_entry
from src.model_comparator import ModelComparator

console = Console()
artifact_store = BenchmarkArtifactStore()

def log(msg, style=None):
    if style:
        console.print(f"[{style}]{msg}[/{style}]")
    else:
        console.print(msg)


def download_model(ollama_tag: str) -> bool:
    log(f"\n[bold blue]>>> Downloading {ollama_tag}...[/bold blue]")
    return pull_ollama_tag(ollama_tag, log=log)


def delete_model(ollama_tag: str):
    log(f"\n[bold blue]>>> Deleting {ollama_tag} to free up space...[/bold blue]")
    delete_ollama_tag(ollama_tag, log=log)


def run_phase(script_name: str, display_name: str):
    log(f"\n[bold magenta]--- Running {display_name} ---[/bold magenta]")
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        log(f"[red]✗ Error during {display_name}: {e}[/red]")


def all_phases_done(model_entry: dict) -> bool:
    """Check if all 3 phases have results for this model. Skips download/delete."""
    return artifact_store.all_phases_done(model_entry)


def main(target_model=None):
    log("\n[bold cyan]" + "=" * 60 + "\n MEGA-ORCHESTRATOR: ONE-PULL, THREE-TESTS, ONE-DELETE\n" + "=" * 60 + "[/bold cyan]")

    # Identify eligible Ollama models
    eligible = []
    for m in MODEL_QUEUE:
        entry = as_model_entry(m)
        if entry.status not in ("pending", "in_progress"):
            continue
        if entry.source != "ollama":
            continue  # Orchestrator lifecycle relies on ollama pull/rm
        if target_model and target_model.lower() not in entry.requested_name.lower():
            continue
        eligible.append(m)

    log(f"Found {len(eligible)} models to test.\n")

    # Set orchestrator flags to tell sub-scripts to skip pull/delete
    os.environ["BENCHMARK_SKIP_LIFECYCLE"] = "1"

    for idx, model in enumerate(eligible, 1):
        entry = as_model_entry(model)
        name = entry.requested_name
        queue_id = entry.queue_id
        tag = entry.ollama_tag

        if not tag:
            log(f"[yellow]Skipping {name} (no ollama_tag)[/yellow]")
            continue

        # Skip if all 3 phases already done — no download/delete needed
        if all_phases_done(model):
            log(f"\n[{idx}/{len(eligible)}] {name} — All phases complete, skipping")
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
