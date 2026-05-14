#!/usr/bin/env python
"""Validate the benchmark prompt dataset against expected contracts.

Checks:
  - exactly 200 prompts
  - exactly 50 per category
  - 10 per difficulty (easy/medium/hard) per category
  - IDs continuous 1-200
  - no duplicate prompts
  - all required fields present
  - all prompts contain the JSON schema instruction string
  - non-vision prompts have image_asset_spec: null
  - vision prompts have non-null image_asset_spec
  - if image_path field exists, the file exists on disk
"""

import json
import os
import sys

CATEGORIES = [
    "Coding Generation",
    "Medium Reasoning",
    "Chat & Generation",
    "Multimodal Vision",
]

STANDARD_DIFFICULTIES = {"easy", "medium", "hard", "adversarial", "long_context"}

REQUIRED_FIELDS = [
    "id", "category", "difficulty", "skill_targets", "prompt",
    "expected_behavior", "scoring_focus", "image_asset_spec"
]

SCHEMA_REQUIRED_PHRASE = (
    "Return only valid JSON matching the schema. "
    "Do not include markdown or any text outside JSON."
)


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array.")
    return data


def validate(prompts: list[dict], dataset_path: str) -> int:
    errors = []
    warnings = []
    n = len(prompts)

    print(f"Loaded {n} prompts from {dataset_path}")

    # 1. Exactly 200 prompts
    if n != 200:
        errors.append(f"Expected 200 prompts, got {n}")
    else:
        print("  [PASS] Exactly 200 prompts")

    # 2. Exactly 50 per category
    cat_counts = {}
    for p in prompts:
        cat = p.get("category", "MISSING")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    for cat in CATEGORIES:
        cnt = cat_counts.get(cat, 0)
        if cnt != 50:
            errors.append(f"Category '{cat}' has {cnt} prompts (expected 50)")
    if all(cat_counts.get(c, 0) == 50 for c in CATEGORIES) and len(cat_counts) == 4:
        print("  [PASS] Exactly 50 per category")
    else:
        print(f"  Category counts: {cat_counts}")

    # 3. Exactly 10 per difficulty per category
    for cat in CATEGORIES:
        cat_prompts = [p for p in prompts if p.get("category") == cat]
        diff_counts = {}
        for p in cat_prompts:
            diff = p.get("difficulty", "MISSING")
            diff_counts[diff] = diff_counts.get(diff, 0) + 1

        for diff in STANDARD_DIFFICULTIES:
            cnt = diff_counts.get(diff, 0)
            if cnt != 10:
                errors.append(
                    f"Category '{cat}' difficulty '{diff}' has {cnt} prompts (expected 10)"
                )

        nonstandard = set(diff_counts.keys()) - STANDARD_DIFFICULTIES
        for ns in nonstandard:
            warnings.append(
                f"Non-standard difficulty '{ns}' found {diff_counts[ns]} time(s) "
                f"in category '{cat}' (IDs: {[p['id'] for p in cat_prompts if p.get('difficulty') == ns]})"
            )

    # 4. IDs are continuous 1-200
    ids = sorted(p.get("id") for p in prompts)
    expected_ids = list(range(1, 201))
    if ids == expected_ids:
        print("  [PASS] IDs continuous 1-200")
    else:
        missing = set(expected_ids) - set(ids)
        extra = set(ids) - set(expected_ids)
        if missing:
            errors.append(f"Missing IDs: {sorted(missing)}")
        if extra:
            errors.append(f"Extra IDs: {sorted(extra)}")

    # 5. No duplicate prompts
    prompt_texts = [p.get("prompt", "").strip() for p in prompts]
    seen = {}
    duplicates = []
    for i, pt in enumerate(prompt_texts):
        if pt in seen:
            duplicates.append((seen[pt], i))
        else:
            seen[pt] = i
    if duplicates:
        for a, b in duplicates:
            errors.append(
                f"Duplicate prompt between IDs {prompts[a]['id']} and {prompts[b]['id']}"
            )
    else:
        print("  [PASS] No duplicate prompts")

    # 6. All required fields exist
    missing_fields = []
    for p in prompts:
        pid = p.get("id", "?")
        for field in REQUIRED_FIELDS:
            if field not in p:
                missing_fields.append(f"ID {pid}: missing field '{field}'")
    if missing_fields:
        errors.extend(missing_fields)
    else:
        print("  [PASS] All required fields present")

    # 7. All prompts contain schema instruction string
    missing_instruction = []
    for p in prompts:
        if SCHEMA_REQUIRED_PHRASE not in p.get("prompt", ""):
            missing_instruction.append(p.get("id", "?"))
    if missing_instruction:
        errors.append(
            f"Prompts missing schema instruction: IDs {missing_instruction}"
        )
    else:
        print("  [PASS] All prompts contain schema instruction string")

    # 8. Non-vision prompts have image_asset_spec: null
    non_vision_with_spec = []
    for p in prompts:
        if p.get("category") != "Multimodal Vision":
            spec = p.get("image_asset_spec")
            if spec is not None:
                non_vision_with_spec.append(p.get("id"))
    if non_vision_with_spec:
        errors.append(
            f"Non-vision prompts with non-null image_asset_spec: IDs {non_vision_with_spec}"
        )
    else:
        print("  [PASS] Non-vision prompts have image_asset_spec: null")

    # 9. Vision prompts have non-null image_asset_spec
    vision_without_spec = []
    for p in prompts:
        if p.get("category") == "Multimodal Vision":
            spec = p.get("image_asset_spec")
            if spec is None:
                vision_without_spec.append(p.get("id"))
    if vision_without_spec:
        errors.append(
            f"Vision prompts with null image_asset_spec: IDs {vision_without_spec}"
        )
    else:
        print("  [PASS] Vision prompts have non-null image_asset_spec")

    # 10. If image_path exists, the file exists
    missing_images = []
    for p in prompts:
        img_path = p.get("image_path")
        if img_path:
            if not os.path.exists(img_path):
                missing_images.append((p.get("id"), img_path))
    if missing_images:
        for pid, path in missing_images:
            errors.append(f"ID {pid}: image_path '{path}' does not exist on disk")
    else:
        print("  [PASS] All image_path references exist on disk (or field absent)")

    print()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
        print()

    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  X {e}")
        return 1

    print("ALL VALIDATIONS PASSED")
    return 0


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_path = os.path.join(base_dir, "prompts", "benchmark_prompts.json")

    if not os.path.exists(prompts_path):
        print(f"ERROR: Dataset not found at {prompts_path}")
        sys.exit(1)

    prompts = load_dataset(prompts_path)
    sys.exit(validate(prompts, prompts_path))


if __name__ == "__main__":
    main()
