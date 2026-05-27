# LLM Benchmark

A local benchmark lab for comparing small language models on constrained hardware.

This project is tuned for a Windows machine with an NVIDIA RTX 2050 class GPU
(4 GB VRAM). It benchmarks model speed, responsiveness, memory use, structured
JSON reliability, and functional quality across Ollama, HuggingFace, and
optional vLLM-backed runtimes.

## What It Measures

| Area | Metric | Output |
|------|--------|--------|
| Streaming speed | `tps`, `ttft`, `latency` | `results/phase1/*.csv` |
| Memory fit | `peak_vram_mb` | `results/phase1/*.csv` |
| Structured output | Pydantic schema success at temperatures `0.0`, `0.7`, `1.0` | `results/phase1/*.csv` |
| Hardware throughput | llama-bench `pp_tps`, `tg_tps` | `results/perf/*.json` |
| Functional quality | promptfoo pass rate across 35 curated tests | `results/quality/*.json` |
| Final comparison | Merged Markdown and dashboard data | `reports/` |

The benchmark prompt set contains 200 prompts: 50 each for coding, chat,
reasoning, and vision-style tasks.

## Current Scope

The queue is generated from `compatible_models.py` through `src/model_queue.py`.

| Queue slice | Count |
|-------------|------:|
| Total queue entries | 98 |
| Runnable models | 66 |
| Deferred vision models | 32 |
| Ollama runtime models | 36 |
| HuggingFace GGUF models | 8 |
| HuggingFace Transformers models | 13 |
| vLLM models | 9 |

Vision models are intentionally deferred until the benchmark has real image
assets and evaluation rules for multimodal runs.

## Quick Start

```powershell
git clone https://github.com/iamsankeerth/llm-benchmark.git
cd llm-benchmark

python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Install Ollama separately from `https://ollama.com`, then verify it is available:

```powershell
ollama --version
```

Run the fast project checks:

```powershell
python scripts\validate_model_queue.py
python scripts\validate_prompt_dataset.py
python -m unittest discover
```

## Run Benchmarks

Run the full orchestrated pipeline:

```powershell
python scripts\run_full_benchmark.py
```

Run one model by name:

```powershell
python scripts\run_full_benchmark.py --model tinyllama
```

Run only Phase 1, which captures streaming speed, VRAM, and structured-output
success:

```powershell
python scripts\run_benchmarks.py
```

Run llama-bench only:

```powershell
python scripts\run_llama_bench.py
```

Preview llama-bench coverage without downloading or running models:

```powershell
python scripts\run_llama_bench.py --dry-run
```

Generate or run promptfoo quality evaluation:

```powershell
python scripts\run_promptfoo.py --generate
python scripts\run_promptfoo.py
```

Generate the merged report from existing artifacts:

```powershell
python -c "from src.model_comparator import ModelComparator; ModelComparator().run_offline_report()"
```

## Live Dashboard And API

Start the API:

```powershell
uvicorn api.main:app --reload
```

Open:

- `http://127.0.0.1:8000/live` for the live benchmark dashboard
- `http://127.0.0.1:8000/docs` for FastAPI docs

Useful endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /models` | List the core chat, reasoning, and coding model shortcuts |
| `POST /generate` | Run a single prompt through a core model |
| `GET /api/status` | Read live benchmark progress, GPU status, ETA, and summaries |
| `GET /api/model/{queue_id}/prompts` | Inspect per-prompt results for one queued model |
| `POST /api/stop` | Stop a running benchmark process |
| `GET /api/health` | Health check |

## Architecture

The benchmark runners are intentionally thin. Shared rules live in dedicated
modules so model identity, lifecycle, artifacts, quality evaluation, and live
status projection stay consistent across CLI scripts and the API.

```text
compatible_models.py
        |
        v
src/model_queue.py ----> src/model_entry.py
        |
        v
scripts/run_full_benchmark.py
        |
        +--> scripts/run_benchmarks.py     -> results/phase1/*.csv
        +--> scripts/run_llama_bench.py    -> results/perf/*.json
        +--> scripts/run_promptfoo.py      -> results/quality/*.json
        |
        v
src/model_comparator.py -> reports/model_comparison_report.md
                         -> reports/dashboard_data.json
```

Core modules:

| Module | Responsibility |
|--------|----------------|
| `src/model_entry.py` | Normalizes queue dictionaries into a typed model identity object |
| `src/lifecycle.py` | Owns model acquisition and cleanup for Ollama, GGUF, Transformers, and vLLM |
| `src/artifact_store.py` | Owns progress files, result paths, summaries, and artifact reads/writes |
| `src/quality_eval.py` | Builds promptfoo configs, runs quality evals, and summarizes raw results |
| `src/live_status.py` | Builds dashboard/API status payloads from progress, logs, GPU state, and artifacts |
| `scripts/test_tracker.py` | Maintains checkpoint/resume progress through the artifact store |

## Project Layout

```text
api/                  FastAPI app, generation routes, live status routes
dashboard/            Static HTML dashboards
prompts/              Benchmark prompt dataset and image fixtures
promptfoo/            Generated promptfoo configs
reports/              Merged Markdown and JSON dashboard outputs
results/phase1/       Streaming, VRAM, and structured-output CSVs
results/perf/         llama-bench throughput JSON
results/quality/      promptfoo quality JSON
scripts/              CLI entry points, validators, watchdog, runners
src/                  Benchmark domain modules and runtime adapters
tests/                Unit tests for architecture boundaries
```

## Configuration

Most runtime settings live in `config.py`.

| Setting | Meaning |
|---------|---------|
| `CORE_MODELS` | Shortcuts used by the FastAPI `/generate` endpoint |
| `MODEL_QUEUE` | Generated queue from `src/model_queue.py` |
| `MAX_NEW_TOKENS` | Output token limit for benchmark generations |
| `TEMPS_TO_TEST` | Temperatures used for structured-output checks |
| `PHASE2_PROMPT_LIMIT` | Number of prompts used for structured-output checks |
| `SMOKE_RUN_PROMPTS` | Optional quick-run limit for a benchmark pass |
| `SLOW_MODEL_THRESHOLD_SECS` | Optional guardrail for models that are too slow |

Useful environment variables:

| Variable | Effect |
|----------|--------|
| `BENCHMARK_SINGLE_MODEL` | Restrict runners to one queue ID |
| `BENCHMARK_SKIP_LIFECYCLE=1` | Tell sub-runners that the orchestrator owns pull/delete |
| `LLAMA_BENCH_EXE` | Override the llama-bench executable path |

## Optional Runtimes

The default path uses Ollama. Additional runtimes are supported when their
dependencies are installed:

- HuggingFace GGUF models are downloaded and converted into local Ollama models.
- HuggingFace Transformers models require `torch`, `transformers`,
  `accelerate`, and `bitsandbytes`.
- vLLM models require `vllm` and a compatible environment.
- promptfoo quality evaluation requires `promptfoo` or `npx`.
- llama-bench requires a local `llama-bench.exe` or `LLAMA_BENCH_EXE`.

## Validation

Run these before changing benchmark logic:

```powershell
$files = @(
  "src\model_entry.py",
  "src\artifact_store.py",
  "src\lifecycle.py",
  "src\quality_eval.py",
  "src\live_status.py",
  "api\status.py",
  "scripts\run_benchmarks.py",
  "scripts\run_full_benchmark.py",
  "scripts\run_llama_bench.py",
  "scripts\run_promptfoo.py",
  "scripts\download_models.py",
  "scripts\test_tracker.py",
  "src\model_comparator.py",
  "src\model_queue.py",
  "tests\test_architecture_modules.py"
)
python -m py_compile $files
python -m unittest discover
python scripts\validate_model_queue.py
python scripts\validate_prompt_dataset.py
python scripts\run_llama_bench.py --dry-run
```

## Outputs

| Path | Description |
|------|-------------|
| `test_progress.json` | Checkpoint state for resume-safe benchmark runs |
| `TEST_STATUS.md` | Human-readable progress dashboard |
| `results/phase1/*.csv` | Per-prompt benchmark results |
| `results/perf/*.json` | llama-bench results |
| `results/quality/*.json` | promptfoo summaries and raw outputs |
| `reports/model_comparison_report.md` | Merged technical report |
| `reports/dashboard_data.json` | Dashboard-ready merged data |

## Troubleshooting

If PowerShell cannot print arrows or checkmarks, run commands from a UTF-8
terminal or set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

If Ollama models fill disk space, use the orchestrated runner. It pulls one
model, runs all phases, then deletes it before moving to the next model.

If a benchmark is interrupted, rerun the same command. Phase 1 uses
`test_progress.json` and checkpoint CSVs to resume where possible.

If promptfoo is missing:

```powershell
npm install -g promptfoo
```

If llama-bench is missing, either place it at `tools/llama-bench/bin/llama-bench.exe`
or set:

```powershell
$env:LLAMA_BENCH_EXE = "C:\path\to\llama-bench.exe"
```

## License

MIT

---

Last updated: May 2026
