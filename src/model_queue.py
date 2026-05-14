"""
Model queue builder – uses compatible_models.py as the source of truth
to generate a flat queue of models for category-matched benchmarking.

Provider resolution (honest path):
  1. Has known Ollama tag        → source = "ollama",      runtime = "ollama"
  2. Is VL/Vision/OCR/multimodal → status = "deferred_vision"
  3. Is AWQ/GPTQ/FP8 without Ollama tag → source = "provider_unsupported"
  4. HF repo in _KNOWN_GGUF_REPOS → source = "huggingface", runtime = "huggingface_gguf"
  5. HF repo NOT in _KNOWN_GGUF_REPOS
     → source = "provider_unsupported", status = "provider_unsupported"
       (Transformers runner not implemented – cannot load these repos)

Every queue entry has a stable unique queue_id and full identity metadata
so the benchmark never silently runs the wrong model variant.
"""

from compatible_models import COMPATIBLE_MODELS, GOOD_FIT_MOE

# ---------------------------------------------------------------------------
# Known Ollama tags derived from model_commands.md Ollama section + TESTED_MODELS
# ---------------------------------------------------------------------------
_KNOWN_OLLAMA_TAGS = {
    # --- Coding ---
    "Qwen2.5-Coder-0.5B":             "qwen2.5-coder:0.5b",
    "Qwen2.5-Coder-0.5B-Instruct":    "qwen2.5-coder:0.5b-instruct",
    "Qwen2.5-Coder-1.5B-Instruct":    "qwen2.5-coder:1.5b-instruct",
    "Qwen2.5-Coder-1.5B-Instruct-AWQ":"qwen2.5-coder:1.5b-instruct",
    "Qwen2.5-Coder-3B":               "qwen2.5-coder:3b",
    "Qwen2.5-Coder-3B-Instruct":      "qwen2.5-coder:3b-instruct",
    "granite-3b-code-base-2k":        "granite-code:3b",
    # --- Chat ---
    "TinyLlama-1.1B-Chat-v1.0":       "tinyllama",
    "Llama-3.2-1B-Instruct":          "llama3.2:1b",
    "Qwen2.5-1.5B-Instruct":          "qwen2.5:1.5b-instruct",
    "Qwen2-1.5B-Instruct":            "qwen2:1.5b-instruct",
    "Qwen2.5-3B-Instruct":            "qwen2.5:3b-instruct",
    "Phi-3-mini-4k-instruct":         "phi3:mini",
    "Phi-3.5-mini-instruct":          "phi3:mini",
    "gemma-2b":                       "gemma:2b",
    "gemma-1.1-2b-it":                "gemma:2b",
    # --- Reasoning ---
    "Phi-4-mini-reasoning":           "phi4-mini-reasoning",
    "DeepSeek-R1-Distill-Qwen-1.5B":  "deepseek-r1:1.5b",
}

# ---------------------------------------------------------------------------
# HF repos that are KNOWN to contain GGUF artifacts and can be loaded.
# Until a full Transformers runner is implemented, HF repos NOT on this list
# are marked provider_unsupported.
# ---------------------------------------------------------------------------
_KNOWN_GGUF_REPOS: set[str] = set()

# ---------------------------------------------------------------------------
# Dirty check for VL / Vision / OCR models
# ---------------------------------------------------------------------------
_VL_KEYWORDS = [
    "VL-", "vl-", "VL2", "VL3", "Vision", "OCR", "ocr",
    "SmolVLM", "InternVL", "h2ovl", "moondream", "deepseek-vl",
    "paligemma", "Video", "GOT-OCR", "Typhoon-OCR", "DeepSeek-OCR",
]


def _is_vl(name: str) -> bool:
    return any(kw.lower() in name.lower() for kw in _VL_KEYWORDS)


def _is_awq_or_gptq(name: str) -> bool:
    upper = name.upper()
    return ("AWQ" in upper or "GPTQ" in upper or "-FP8" in upper)


def _make_queue_id(category: str, source: str, name: str, ref: str) -> str:
    """Build a stable unique queue_id. Safe as a JSON key."""
    ref = ref.replace("\\", "/")
    return f"{category}:{source}:{name}:{ref}"


def _resolve_identity(model: dict) -> dict:
    """Resolve provider and variant identity for one model.

    Returns a dict with:
      source, resolved_runtime, resolved_model_ref, variant_note, status, ollama_tag
    """
    name = model["name"]
    hf_repo = model.get("hf", "")

    # 1. Vision models are always deferred
    if _is_vl(name):
        return {
            "source": "provider_unsupported",
            "resolved_runtime": "deferred_vision",
            "resolved_model_ref": hf_repo or name,
            "variant_note": "",
            "status": "deferred_vision",
            "ollama_tag": "",
        }

    # 2. Has a known Ollama tag
    if name in _KNOWN_OLLAMA_TAGS:
        tag = _KNOWN_OLLAMA_TAGS[name]
        variant = ""
        if _is_awq_or_gptq(name):
            variant = (
                f"Requested AWQ/GPTQ/FP8 HF variant resolved to standard Ollama "
                f"instruct tag {tag}; results reflect the Ollama tag, not the "
                f"quantised variant."
            )
        return {
            "source": "ollama",
            "resolved_runtime": "ollama",
            "resolved_model_ref": tag,
            "variant_note": variant,
            "status": "pending",
            "ollama_tag": tag,
        }

    # 3. AWQ/GPTQ/FP8 without Ollama tag → unsupported
    if _is_awq_or_gptq(name):
        return {
            "source": "provider_unsupported",
            "resolved_runtime": "unsupported",
            "resolved_model_ref": hf_repo,
            "variant_note": "AWQ/GPTQ/FP8 quantised repo without known Ollama tag",
            "status": "provider_unsupported",
            "ollama_tag": "",
        }

    # 4. HF repo in known GGUF list → loadable
    if hf_repo in _KNOWN_GGUF_REPOS:
        return {
            "source": "huggingface",
            "resolved_runtime": "huggingface_gguf",
            "resolved_model_ref": hf_repo,
            "variant_note": "",
            "status": "pending",
            "ollama_tag": "",
        }

    # 5. HF-only repo, not in GGUF allowlist → unsupported
    return {
        "source": "provider_unsupported",
        "resolved_runtime": "unsupported",
        "resolved_model_ref": hf_repo,
        "variant_note": (
            f"HF repo {hf_repo} is not known to contain GGUF artifacts; "
            "Transformers runner not implemented."
        ) if hf_repo else "No loadable artifacts available for this model.",
        "status": "provider_unsupported",
        "ollama_tag": "",
    }


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def build_model_queue() -> list[dict]:
    """Build the benchmark model queue.

    Returns a flat list of dicts, one per model, with keys:
      queue_id, requested_name, category, source, resolved_runtime,
      resolved_model_ref, variant_note, ollama_tag, hf_repo,
      size, estimated_tps, fit_level, is_moe, status
    """
    queue: list[dict] = []
    seen_ids: set[str] = set()

    def _add(model: dict, category: str, fit_level: str,
             is_moe: bool = False):
        ident = _resolve_identity(model)
        name = model["name"]
        hf_repo = model.get("hf", "")
        tag = ident["ollama_tag"]

        queue_id = _make_queue_id(category, ident["source"], name,
                                   tag or hf_repo or name)

        if queue_id in seen_ids:
            return
        seen_ids.add(queue_id)

        entry = {
            "queue_id": queue_id,
            "requested_name": name,
            "category": category,
            "source": ident["source"],
            "resolved_runtime": ident["resolved_runtime"],
            "resolved_model_ref": ident["resolved_model_ref"],
            "variant_note": ident["variant_note"],
            "ollama_tag": tag,
            "hf_repo": hf_repo,
            "size": model.get("size", "?"),
            "estimated_tps": model.get("tps", 0),
            "fit_level": fit_level,
            "is_moe": model.get("moe", is_moe),
            "status": ident["status"],
        }
        queue.append(entry)

    # --- Coding ---
    for m in COMPATIBLE_MODELS["coding"]["perfect"]:
        _add(m, "Coding", "perfect")
    for m in COMPATIBLE_MODELS["coding"]["good"]:
        _add(m, "Coding", "good", is_moe=True)

    # --- Chat ---
    for tier in ("small", "medium", "large", "moe"):
        for m in COMPATIBLE_MODELS["chat"].get(tier, []):
            _add(m, "Chat", tier, is_moe=m.get("moe", tier == "moe"))

    # --- Reasoning ---
    for m in COMPATIBLE_MODELS["reasoning"].get("perfect", []):
        _add(m, "Reasoning", "perfect")

    # --- GOOD_FIT_MOE ---
    for m in GOOD_FIT_MOE:
        ident = _resolve_identity(m)
        if ident["status"] == "deferred_vision":
            _add(m, "Vision", "deferred_vision")
            continue

        n = m["name"]
        if "Coder" in n or "code" in n.lower():
            cat = "Coding"
        elif "Reason" in n or "Think" in n or "Math" in n:
            cat = "Reasoning"
        else:
            cat = "Chat"
        _add(m, cat, "good_fit_moe", is_moe=m.get("moe", True))

    return queue


def queue_summary(queue: list[dict]) -> str:
    """Return a human-readable summary of the queue."""
    total = len(queue)
    runnable = sum(1 for m in queue if m["status"] == "pending")
    unsupported = sum(1 for m in queue if m["status"] == "provider_unsupported")
    deferred = sum(1 for m in queue if m["status"] == "deferred_vision")

    by_cat: dict[str, int] = {}
    for m in queue:
        cat = m["category"]
        by_cat[cat] = by_cat.get(cat, 0) + 1

    by_runtime: dict[str, int] = {}
    for m in queue:
        rt = m.get("resolved_runtime", "?")
        by_runtime[rt] = by_runtime.get(rt, 0) + 1

    lines = [
        "Model Queue Summary",
        "=" * 40,
        f"Total models in queue: {total}",
        f"  Runnable (pending):  {runnable}",
        f"  Provider unsupported: {unsupported}",
        f"  Deferred vision:      {deferred}",
        "-" * 40,
        "By benchmark category:",
    ]
    for cat, cnt in sorted(by_cat.items()):
        lines.append(f"  {cat}: {cnt}")
    lines.append("-" * 40)
    lines.append("By resolved runtime:")
    for rt, cnt in sorted(by_runtime.items()):
        lines.append(f"  {rt}: {cnt}")
    return "\n".join(lines)
