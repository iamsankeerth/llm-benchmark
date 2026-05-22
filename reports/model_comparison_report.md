# LLM Benchmark: All-Compatible Model Study

Generated on RTX 2050 (4GB VRAM). Source of truth: `compatible_models.py` → `config.MODEL_QUEUE`.

> **Metric Sources**
> - `tg_tps` / `pp_tps` — llama-bench C++ binary (hardware-accurate)
> - `pass_rate` — promptfoo quality evaluation (35 curated tests)
> - `ttft` / `latency` / `peak_vram_mb` — Phase 1 Ollama streaming

## Queue Coverage

| Metric | Count |
|--------|-------|
| Total models in queue | 98 |
| Runnable models | 66 |
| Benchmark completed | 32 |
| Benchmark failed | 1 |
| Provider unsupported | 0 |
| Deferred (vision) | 32 |
| llama-bench data available | 1 |
| promptfoo data available | 1 |
| CSV models in report | 23 |

**Queue ID audit:** All queue_ids are unique.

---

## 2. Core Performance Metrics

The primary metrics for assessing local LLM inference quality:

| Metric | Description | Why it matters |
|--------|-------------|----------------|
| `tg_tps` | **Decode Speed** (tokens/sec generated) | How fast the model streams code/text to you |
| `pp_tps` | Prefill Speed (tokens/sec processed) | How fast the model reads your long prompt/context |
| `pass_rate` | **Functional Quality** (% tests passed) | Whether generated outputs are correct |
| `ttft` | Time to First Token (seconds) | Perceived responsiveness |
| `latency` | Total Response Time (seconds) | End-to-end wait time |
| `peak_vram_mb` | Peak GPU Memory (MB) | Hardware compatibility |

### 2.1 Decode Speed — `tg_tps` (Primary Metric)

| model   | tg_tps   | tg_stddev   | pp_tps   | pp_stddev   |
|---------|----------|-------------|----------|-------------|

> **`tg_tps` = Decode Speed**: This is the true inference speed — how fast the model generates tokens after the first token. Higher is better. Measured via llama-bench C++ binary.

### 2.2 Functional Quality — `pass_rate` (Primary Metric)

| model    |   pass_rate |   tests_passed |   tests_total |
|:---------|------------:|---------------:|--------------:|
| qwen3:4b |          20 |              7 |            35 |

> **`pass_rate`**: Percentage of 35 curated tests passed. Tests cover coding, reasoning, structured output, and chat. Graded by `qwen2.5:3b-instruct` as LLM judge.

### 2.3 Full Merged Summary

| model                       |   tg_tps |   pp_tps |   pass_rate |   ttft |   latency |   peak_vram_mb |   total_prompts |
|:----------------------------|---------:|---------:|------------:|-------:|----------:|---------------:|----------------:|
| deepseek-r1:1.5b            |      nan |      nan |         nan |   0.34 |      2.02 |        1539.5  |              50 |
| gemma2:2b                   |      nan |      nan |         nan |   1.05 |      2.77 |        2613.22 |             100 |
| gemma:2b                    |      nan |      nan |         nan |   0.48 |      1.8  |        2398.71 |             100 |
| granite-code:3b             |      nan |      nan |         nan |   0.19 |      3    |        2916.71 |               3 |
| llama3.2:1b                 |      nan |      nan |         nan |   0.37 |      2.09 |        1834.71 |               3 |
| llama3.2:3b                 |      nan |      nan |         nan |   0.41 |      2.02 |        2697.5  |             100 |
| phi3:mini                   |      nan |      nan |         nan |   0.11 |      2.48 |        3145.5  |              50 |
| phi4-mini-reasoning         |      nan |      nan |         nan |   0.45 |      4.73 |        3943.5  |              50 |
| qwen2.5-coder:0.5b          |      nan |      nan |         nan |   0.22 |      1.14 |        2414.45 |              50 |
| qwen2.5-coder:0.5b-instruct |      nan |      nan |         nan |   0.23 |      1.16 |        2414.45 |              50 |
| qwen2.5-coder:1.5b-instruct |      nan |      nan |         nan |   0.22 |      1.82 |        3014.45 |             100 |
| qwen2.5-coder:3b            |      nan |      nan |         nan |   0.39 |      3.21 |        2474.71 |              46 |
| qwen2.5-coder:3b-instruct   |      nan |      nan |         nan |   0.44 |      3.26 |        2474.71 |              14 |
| qwen2.5:1.5b-instruct       |      nan |      nan |         nan |   0.34 |      1.22 |        1546.71 |              50 |
| qwen2.5:3b-instruct         |      nan |      nan |         nan |   0.32 |      1.94 |        2474.71 |              22 |
| qwen2:1.5b-instruct         |      nan |      nan |         nan |   0.37 |      1.35 |        1496.71 |             100 |
| qwen3.5:0.8b                |      nan |      nan |         nan |   0.65 |      2.71 |        2168.71 |               6 |
| qwen3.5:2b                  |      nan |      nan |         nan |   0.08 |      0.46 |        3878.71 |              56 |
| qwen3:0.6b                  |      nan |      nan |         nan |   0.29 |      2.62 |         930.71 |              50 |
| qwen3:1.7b                  |      nan |      nan |         nan |   0.3  |      2.31 |        1668.71 |             100 |
| qwen3:4b                    |      nan |      nan |          20 |   0.35 |      3.88 |        3039.5  |              33 |
| starcoder2:3b               |      nan |      nan |         nan |  27.23 |     28.46 |        2783.95 |              37 |
| tinyllama                   |      nan |      nan |         nan |   0.12 |      0.97 |         946.71 |              50 |

### 2.4 JSON Schema Success Rate by Temperature

Percentage of prompts where the model produced valid Pydantic-validated JSON:

| model                       |   temp_0.0_success_rate |   temp_0.7_success_rate |   temp_1.0_success_rate |
|:----------------------------|------------------------:|------------------------:|------------------------:|
| deepseek-r1:1.5b            |                   76    |                   94    |                   78    |
| gemma2:2b                   |                  100    |                  100    |                  100    |
| gemma:2b                    |                  100    |                  100    |                  100    |
| granite-code:3b             |                  100    |                  100    |                  100    |
| llama3.2:1b                 |                  100    |                  100    |                  100    |
| llama3.2:3b                 |                  100    |                  100    |                  100    |
| phi3:mini                   |                  100    |                  100    |                  100    |
| phi4-mini-reasoning         |                  100    |                   98    |                   96    |
| qwen2.5-coder:0.5b          |                  100    |                  100    |                  100    |
| qwen2.5-coder:0.5b-instruct |                  100    |                  100    |                  100    |
| qwen2.5-coder:1.5b-instruct |                   99    |                  100    |                  100    |
| qwen2.5-coder:3b            |                   97.83 |                  100    |                  100    |
| qwen2.5-coder:3b-instruct   |                  100    |                  100    |                  100    |
| qwen2.5:1.5b-instruct       |                  100    |                  100    |                  100    |
| qwen2.5:3b-instruct         |                  100    |                  100    |                  100    |
| qwen2:1.5b-instruct         |                  100    |                  100    |                  100    |
| qwen3.5:0.8b                |                    0    |                    0    |                    0    |
| qwen3.5:2b                  |                    3.57 |                    5.36 |                    8.93 |
| qwen3:0.6b                  |                  100    |                  100    |                  100    |
| qwen3:1.7b                  |                  100    |                  100    |                  100    |
| qwen3:4b                    |                   69.7  |                   63.64 |                   60.61 |
| starcoder2:3b               |                  100    |                   94.59 |                   97.3  |
| tinyllama                   |                   98    |                   98    |                   98    |

### 2.5 Average Latency by Category

| model                       |   Chat & Generation |   Coding Generation |   Medium Reasoning |   Multimodal Vision |
|:----------------------------|--------------------:|--------------------:|-------------------:|--------------------:|
| deepseek-r1:1.5b            |             nan     |             nan     |              2.022 |                 nan |
| gemma2:2b                   |               2.768 |             nan     |            nan     |                 nan |
| gemma:2b                    |               1.796 |             nan     |            nan     |                 nan |
| granite-code:3b             |             nan     |               2.998 |            nan     |                 nan |
| llama3.2:1b                 |               2.088 |             nan     |            nan     |                 nan |
| llama3.2:3b                 |               2.018 |             nan     |            nan     |                 nan |
| phi3:mini                   |               2.479 |             nan     |            nan     |                 nan |
| phi4-mini-reasoning         |             nan     |             nan     |              4.731 |                 nan |
| qwen2.5-coder:0.5b          |             nan     |               1.141 |            nan     |                 nan |
| qwen2.5-coder:0.5b-instruct |             nan     |               1.16  |            nan     |                 nan |
| qwen2.5-coder:1.5b-instruct |             nan     |               1.821 |            nan     |                 nan |
| qwen2.5-coder:3b            |             nan     |               3.212 |            nan     |                 nan |
| qwen2.5-coder:3b-instruct   |             nan     |               3.264 |            nan     |                 nan |
| qwen2.5:1.5b-instruct       |               1.221 |             nan     |            nan     |                 nan |
| qwen2.5:3b-instruct         |               1.937 |             nan     |            nan     |                 nan |
| qwen2:1.5b-instruct         |               1.351 |             nan     |            nan     |                 nan |
| qwen3.5:0.8b                |               2.709 |             nan     |            nan     |                 nan |
| qwen3.5:2b                  |               4.281 |             nan     |            nan     |                   0 |
| qwen3:0.6b                  |               2.62  |             nan     |            nan     |                 nan |
| qwen3:1.7b                  |               2.313 |             nan     |            nan     |                 nan |
| qwen3:4b                    |               3.909 |             nan     |              3.871 |                 nan |
| starcoder2:3b               |             nan     |              28.465 |            nan     |                 nan |
| tinyllama                   |               0.975 |             nan     |            nan     |                 nan |

### 2.6 Top Failures

| model         | category          |   error_count | sample_prompt_ids         |
|:--------------|:------------------|--------------:|:--------------------------|
| qwen3.5:2b    | Multimodal Vision |            50 | [151, 152, 153, 154, 155] |
| starcoder2:3b | Coding Generation |             7 | [4, 11, 17, 19, 20]       |

## Best Model Analysis

**✅ Best Functional Quality:** `qwen3:4b` — 20.0% pass rate

**⚡ Lowest Latency:** `qwen3.5:2b` — 0.459s avg

**💾 Most Memory Efficient:** `qwen3:0.6b` — peak 931 MB VRAM

### Recommendation

- **Best for Coding**: Prioritize `tg_tps` (decode speed) + `pass_rate` (functional correctness).
- **Best for Chat**: Prioritize low `ttft` (perceived responsiveness) + `pass_rate`.
- **Best for Reasoning**: Prioritize `pass_rate` at temperature 0.0 (determinism).
- **Best for Constrained Hardware**: Lowest `peak_vram_mb` + acceptable `tg_tps`.

## Methodology Notes

- **Decode Speed (`tg_tps`)** measured by llama-bench C++ binary (b9159) with `n_gen=128`, `n_gpu_layers=99`, flash attention enabled.
- **Functional Quality (`pass_rate`)** measured by promptfoo with 35 curated tests graded by `qwen2.5:3b-instruct`.
- **VRAM** measured via `pynvml` with peak tracking during Phase 1.
- **JSON Validity** enforced via Ollama `format=schema` + Pydantic validation.
- **Ephemeral lifecycle**: each model is pulled → benchmarked → deleted to keep disk usage minimal.
