import os
import json
from src.ollama_client import OllamaClient
from src.gpu_monitor import GPUMonitor

class Benchmarker:
    def __init__(self):
        self.client = OllamaClient()

    @staticmethod
    def _model_identity(model_entry: dict) -> dict:
        """Extract model identity fields from a queue entry."""
        return {
            "queue_id": model_entry.get("queue_id", ""),
            "requested_name": model_entry.get("requested_name", model_entry.get("name", "")),
            "resolved_model_ref": model_entry.get("resolved_model_ref", ""),
            "resolved_runtime": model_entry.get("resolved_runtime", ""),
            "variant_note": model_entry.get("variant_note", ""),
            "fit_level": model_entry.get("fit_level", ""),
            "size": model_entry.get("size", "?"),
            "estimated_tps": model_entry.get("estimated_tps", 0),
            "is_moe": model_entry.get("is_moe", False),
        }

    def benchmark_single(self, model: str, prompt_data: dict,
                         model_entry: dict = None) -> dict:
        """
        Phase 1: Run a single prompt to measure TPS, TTFT, Latency, and Peak VRAM.
        Preserves all prompt and model identity metadata in the output row.

        model      – the actual runtime model tag/name to send to Ollama
        prompt_data – dict from the prompt dataset
        model_entry – queue entry dict with identity fields (optional for back-compat)
        """
        category = prompt_data.get("category", "")
        is_vis = (category == "Multimodal Vision")
        image_path = prompt_data.get("image_path", None)

        from config import MAX_NEW_TOKENS

        with GPUMonitor() as monitor:
            res = self.client.generate_benchmark(
                model,
                prompt_data["prompt"],
                max_tokens=MAX_NEW_TOKENS,
                is_vision=is_vis,
                image_path=image_path
            )
            peak_vram = monitor.peak_vram

        row = {
            "model": model,
            "prompt_id": prompt_data.get("id"),
            "category": category,
            "difficulty": prompt_data.get("difficulty", ""),
            "skill_targets": json.dumps(prompt_data.get("skill_targets", [])),
            "expected_behavior": prompt_data.get("expected_behavior", ""),
            "scoring_focus": json.dumps(prompt_data.get("scoring_focus", [])),
            "image_asset_spec": json.dumps(prompt_data.get("image_asset_spec")),
            "image_path": image_path or "",
            "prompt": prompt_data["prompt"],
            "output": res.content,
            "tps": res.tps,
            "ttft": res.ttft,
            "latency": res.latency,
            "peak_vram_mb": peak_vram,
            "temperature": 1.0,
            "error": res.error,
        }

        if model_entry:
            row.update(self._model_identity(model_entry))

        return row
