#!/usr/bin/env python
"""Validate model queue integrity.

Checks:
  - queue has no duplicate queue_id
  - no provider_unsupported model has status pending
  - no deferred_vision model has status pending
  - every ollama-source model has non-empty ollama_tag
  - every huggingface-source model has non-empty hf_repo
  - every runnable huggingface model is in _KNOWN_GGUF_REPOS
  - every entry has requested_name, resolved_runtime, resolved_model_ref, variant_note
  - queue summary counts add up to total
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_queue import build_model_queue, queue_summary, _KNOWN_GGUF_REPOS


def validate(queue: list[dict]) -> int:
    errors: list[str] = []
    total = len(queue)

    print(f"Loaded {total} models from queue builder")
    print(queue_summary(queue))
    print()

    # 1. No duplicate queue_id
    ids = [m["queue_id"] for m in queue]
    seen: set[str] = set()
    dupe_ids: list[str] = []
    for qid in ids:
        if qid in seen:
            dupe_ids.append(qid)
        seen.add(qid)
    if dupe_ids:
        for did in set(dupe_ids):
            matches = [m["requested_name"] for m in queue if m["queue_id"] == did]
            errors.append(
                f"Duplicate queue_id '{did}': {matches}"
            )
    else:
        print(f"  [PASS] All {total} queue_ids are unique")

    # 2. No provider_unsupported/deferred_vision with status pending
    bad_status = []
    for m in queue:
        if m["status"] == "pending" and m.get("resolved_runtime") in ("unsupported", "deferred_vision"):
            bad_status.append(f"{m['queue_id']} (resolved_runtime={m['resolved_runtime']}, status=pending)")
    if bad_status:
        errors.append(f"Models with unsupported runtime but status 'pending': {bad_status}")
    else:
        print("  [PASS] No unsupported/deferred models incorrectly marked as pending")

    # 3. Ollama source models have non-empty ollama_tag
    missing_tag = []
    for m in queue:
        if m["source"] == "ollama" and not m.get("ollama_tag"):
            missing_tag.append(m["queue_id"])
    if missing_tag:
        errors.append(f"Ollama-source models missing ollama_tag: {missing_tag}")
    else:
        print("  [PASS] All ollama-source models have non-empty ollama_tag")

    # 4. Huggingface source models have non-empty hf_repo
    missing_hf = []
    for m in queue:
        if m["source"] == "huggingface" and not m.get("hf_repo"):
            missing_hf.append(m["queue_id"])
    if missing_hf:
        errors.append(f"Huggingface-source models missing hf_repo: {missing_hf}")
    else:
        print("  [PASS] All huggingface-source models have non-empty hf_repo")

    # 5. Runnable huggingface_gguf models are in _KNOWN_GGUF_REPOS
    bad_hf = []
    for m in queue:
        if (m["source"] == "huggingface"
            and m.get("resolved_runtime") == "huggingface_gguf"
            and m["status"] == "pending"):
            if m.get("hf_repo") not in _KNOWN_GGUF_REPOS:
                bad_hf.append(m["queue_id"])
    if bad_hf:
        errors.append(
            f"Huggingface GGUF models marked runnable but NOT in _KNOWN_GGUF_REPOS: {bad_hf}"
        )
    else:
        print("  [PASS] All runnable huggingface_gguf models are in _KNOWN_GGUF_REPOS")

    # 5b. Report runtime distribution for visibility
    hf_trans = sum(1 for m in queue if m.get("resolved_runtime") == "hf_transformers" and m["status"] == "pending")
    vllm_cnt = sum(1 for m in queue if m.get("resolved_runtime") == "vllm" and m["status"] == "pending")
    gguf_cnt = sum(1 for m in queue if m.get("resolved_runtime") == "huggingface_gguf" and m["status"] == "pending")
    ollama_cnt = sum(1 for m in queue if m.get("resolved_runtime") == "ollama" and m["status"] == "pending")
    print(f"  [INFO] Runnable by runtime: ollama={ollama_cnt}, hf_transformers={hf_trans}, vllm={vllm_cnt}, gguf={gguf_cnt}")

    # 6. Every entry has required identity fields
    required = ["requested_name", "resolved_runtime", "resolved_model_ref", "variant_note"]
    missing_fields = []
    for m in queue:
        for field in required:
            if field not in m:
                missing_fields.append(f"{m.get('queue_id', '?')}: missing '{field}'")
    if missing_fields:
        errors.extend(missing_fields)
    else:
        print("  [PASS] All entries have requested_name, resolved_runtime, resolved_model_ref, variant_note")

    # 7. Summary counts add up
    runnable = sum(1 for m in queue if m["status"] == "pending")
    unsupported = sum(1 for m in queue if m["status"] == "provider_unsupported")
    deferred = sum(1 for m in queue if m["status"] == "deferred_vision")
    other = total - runnable - unsupported - deferred
    if other != 0:
        errors.append(
            f"Status counts don't add up: pending={runnable} + "
            f"provider_unsupported={unsupported} + deferred_vision={deferred} "
            f"+ other={other} != total={total}"
        )
    else:
        print(f"  [PASS] Status counts add up: {runnable} pending + {unsupported} "
              f"unsupported + {deferred} deferred = {total}")

    # 8. Duplicate requested names are OK if queue_id differs
    name_counts: dict[str, list[str]] = {}
    for m in queue:
        n = m["requested_name"]
        name_counts.setdefault(n, []).append(m["queue_id"])
    dup_names = {n: ids for n, ids in name_counts.items() if len(ids) > 1}
    if dup_names:
        print(f"  [INFO] {len(dup_names)} duplicate requested_name(s) found (different queue_ids — OK)")
        for n, ids in list(dup_names.items())[:3]:
            print(f"         {n}: {ids}")

    print()
    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  X {e}")
        return 1

    print("ALL QUEUE VALIDATIONS PASSED")
    return 0


def main():
    queue = build_model_queue()
    sys.exit(validate(queue))


if __name__ == "__main__":
    main()
