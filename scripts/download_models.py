#!/usr/bin/env python3
"""
Model Downloader Script
Downloads models from Ollama or HuggingFace based on MODEL_QUEUE.

Usage:
    python scripts/download_models.py              # Download all runnable
    python scripts/download_models.py --ollama     # Ollama only
    python scripts/download_models.py --hf         # HuggingFace only
    python scripts/download_models.py --list       # List available models
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_QUEUE, MODELS_DIR
from src.lifecycle import pull_ollama_tag
from src.model_entry import as_model_entry

console = Console()

def list_models():
    """List all models in the queue with their status."""
    table = Table(title="Model Queue — Download Status")
    table.add_column("Category", style="cyan")
    table.add_column("Model Name", style="green")
    table.add_column("Source", style="yellow")
    table.add_column("Runtime", style="magenta")
    table.add_column("Status", style="red")
    table.add_column("Ollama Tag / HF Repo", style="dim")

    for m in MODEL_QUEUE:
        entry = as_model_entry(m)
        ref = entry.ollama_tag or entry.hf_repo or entry.resolved_model_ref or "?"
        table.add_row(
            entry.category,
            entry.requested_name,
            entry.source,
            entry.resolved_runtime,
            entry.status,
            ref
        )

    console.print(table)
    runnable = sum(1 for m in MODEL_QUEUE if as_model_entry(m).is_runnable)
    console.print(f"\n[bold]Total: {len(MODEL_QUEUE)} | Runnable: {runnable}[/bold]")

def download_ollama_model(model_entry: dict) -> bool:
    """Download model from Ollama."""
    ollama_tag = model_entry.get("ollama_tag")
    if not ollama_tag:
        return False

    console.print(f"[cyan]Pulling from Ollama:[/cyan] {ollama_tag}")
    return pull_ollama_tag(ollama_tag, log=lambda msg: console.print(msg))

def download_hf_model(model_entry: dict) -> bool:
    """Download model from HuggingFace."""
    hf_repo = model_entry.get("hf_repo")

    if not hf_repo:
        return False

    local_dir_name = model_entry.get("requested_name", "").replace("/", "_").replace(":", "_")
    model_path = Path(MODELS_DIR) / local_dir_name

    if model_path.exists() and any(model_path.iterdir()):
        console.print(f"[yellow]Already exists:[/yellow] {local_dir_name}")
        return True

    console.print(f"[cyan]Downloading from HuggingFace:[/cyan] {hf_repo}")
    result = subprocess.run(
        ["huggingface-cli", "download", hf_repo, "--local-dir", str(model_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        console.print(f"[red]Failed:[/red] {result.stderr}")
        return False
    console.print(f"[green]Success:[/green] Downloaded to {model_path}")
    return True

def download_all(source_filter: str = None):
    """Download all runnable models, optionally filtered by source."""
    total = 0
    success = 0

    for m in MODEL_QUEUE:
        entry = as_model_entry(m)
        source = entry.source
        name = entry.requested_name

        # Skip non-runnable models
        if not entry.is_runnable:
            continue

        if source_filter and source != source_filter:
            continue

        total += 1
        console.print(f"\n[{source}] {name}")

        if source == "ollama":
            if download_ollama_model(m):
                success += 1
        elif source == "huggingface":
            if download_hf_model(m):
                success += 1
        else:
            console.print(f"[yellow]Skipping unsupported source: {source}[/yellow]")

    console.print(f"\n[bold green]Download Complete:[/bold green] {success}/{total} models")
    return success, total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AI models from Ollama or HuggingFace")
    parser.add_argument("--ollama", action="store_true", help="Download Ollama models only")
    parser.add_argument("--hf", action="store_true", help="Download HuggingFace models only")
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.ollama:
        download_all("ollama")
    elif args.hf:
        download_all("huggingface")
    else:
        console.print("[bold green]Downloading ALL runnable models...[/bold green]")
        download_all()
