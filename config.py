import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model queue – auto-generated from compatible_models.py
from src.model_queue import build_model_queue, queue_summary
MODEL_QUEUE = build_model_queue()

# Core single-model shortcuts used by the CLI and FastAPI wrapper.
CORE_MODELS = {
    "chat": "qwen2.5:3b-instruct",
    "reasoning": "phi4-mini-reasoning",
    "coding": "granite-code:3b",
}

# Execute Optimizations
BENCHMARK_RUNS = 1
WARMUP_RUNS = 0
MAX_NEW_TOKENS = 128
TEMPS_TO_TEST = [0.0, 0.7, 1.0]
PHASE2_PROMPT_LIMIT = 50

# Smoke test: set to a small number (e.g. 3) to run a quick test on one model
# before the full run. Set to 0 to disable.
SMOKE_RUN_PROMPTS = 0

# If average prompt time exceeds this (seconds), stop the pipeline.
# Protects against models too slow for current hardware. 0 = disabled.
SLOW_MODEL_THRESHOLD_SECS = 300

GPU_INDEX = 0
