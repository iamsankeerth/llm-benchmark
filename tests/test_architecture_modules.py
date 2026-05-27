import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.artifact_store import BenchmarkArtifactStore
from src.lifecycle import RuntimeHandle, acquire_runtime, cleanup_runtime, pull_ollama_tag
from src.live_status import LiveStatusProjection
from src.model_entry import ModelEntry, safe_name
from src.quality_eval import QualityEvaluator, _matches_target_model


def sample_model(**overrides):
    data = {
        "queue_id": "Coding:ollama:Sample:qwen2.5:3b-instruct",
        "requested_name": "Sample",
        "category": "Coding",
        "source": "ollama",
        "resolved_runtime": "ollama",
        "resolved_model_ref": "qwen2.5:3b-instruct",
        "ollama_tag": "qwen2.5:3b-instruct",
        "status": "pending",
    }
    data.update(overrides)
    return data


class ModelEntryTests(unittest.TestCase):
    def test_model_entry_normalizes_dict_fields(self):
        entry = ModelEntry.from_dict(sample_model(estimated_tps=12.5, is_moe=True))

        self.assertTrue(entry.is_runnable)
        self.assertEqual(entry.runtime_ref, "qwen2.5:3b-instruct")
        self.assertEqual(entry.safe_queue_id, "Coding_ollama_Sample_qwen2.5_3b-instruct")
        self.assertEqual(safe_name("a:b/c\\d"), "a_b_c_d")


class ArtifactStoreTests(unittest.TestCase):
    def test_store_owns_progress_and_phase_completion_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = BenchmarkArtifactStore(tmp)
            model = sample_model()

            store.save_progress({"models": {"m": {"status": "pending"}}})
            self.assertIn("last_updated", store.load_progress())

            store.save_checkpoint_csv([{"model": "Sample", "tps": 10}], model["queue_id"])
            store.save_llama_bench_result(model["queue_id"], {"model_name": "Sample", "tg_tps": 22})
            store.quality_result_path(model["ollama_tag"]).parent.mkdir(parents=True, exist_ok=True)
            store.quality_result_path(model["ollama_tag"]).write_text("{}", encoding="utf-8")

            self.assertTrue(store.phase1_done(model))
            self.assertTrue(store.perf_done(model))
            self.assertTrue(store.quality_done(model))
            self.assertTrue(store.all_phases_done(model))


class LifecycleTests(unittest.TestCase):
    def test_existing_ollama_model_is_not_marked_for_cleanup(self):
        with patch("src.lifecycle.check_ollama_model_exists", return_value=True), patch(
            "src.lifecycle.pull_ollama_tag"
        ) as pull:
            handle = acquire_runtime(sample_model())

        self.assertIsNotNone(handle)
        self.assertFalse(handle.delete_on_cleanup)
        pull.assert_not_called()

    def test_cleanup_respects_runtime_ownership(self):
        handle = RuntimeHandle("qwen2.5:3b-instruct", None, sample_model(), delete_on_cleanup=False)

        with patch("src.lifecycle.subprocess.run") as run, patch(
            "src.lifecycle.get_free_disk_space_gb", return_value=42.0
        ):
            cleanup_runtime(handle)

        run.assert_not_called()

    def test_pull_ollama_tag_uses_shared_lifecycle_command(self):
        run_result = Mock(returncode=0, stdout="", stderr="")
        with patch(
            "src.lifecycle.check_ollama_model_exists", side_effect=[False, True]
        ), patch("src.lifecycle.subprocess.run", return_value=run_result) as run:
            self.assertTrue(pull_ollama_tag("tinyllama"))

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["ollama", "pull", "tinyllama"])


class QualityEvaluatorTests(unittest.TestCase):
    def test_quality_evaluator_preserves_queue_id_for_single_model_filtering(self):
        tests = lambda: [
            {
                "description": "Coding: sample",
                "vars": {"prompt": "Say hi"},
                "assert": [{"type": "contains", "value": "hi"}],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            evaluator = QualityEvaluator([sample_model()], Path(tmp) / "promptfoo", Path(tmp) / "results", tests)
            providers = evaluator.get_ollama_providers()

            self.assertEqual(providers[0]["queue_id"], sample_model()["queue_id"])
            self.assertTrue(
                _matches_target_model(
                    sample_model()["queue_id"],
                    sample_model()["ollama_tag"],
                    sample_model()["requested_name"],
                    sample_model()["queue_id"],
                )
            )
            config = evaluator.render_config(providers, tests(), "qwen2.5:3b-instruct")
            self.assertIn("ollama:chat:qwen2.5:3b-instruct", config)


class LiveStatusProjectionTests(unittest.TestCase):
    def test_status_projection_reads_metrics_through_artifact_store(self):
        class FakeSystem:
            def get_benchmark_pid(self):
                return None

            def is_process_alive(self, pid):
                return False

            def get_gpu_info(self):
                return {"utilization": "N/A", "vram_used": "N/A", "vram_total": "N/A"}

        with tempfile.TemporaryDirectory() as tmp:
            store = BenchmarkArtifactStore(tmp)
            csv_path = store.save_unified_result(
                [
                    {
                        "model": "Sample",
                        "prompt_id": 1,
                        "tps": 10,
                        "latency": 2,
                        "peak_vram_mb": 512,
                        "temp_0.0_success": True,
                    }
                ],
                sample_model()["queue_id"],
            )
            store.save_progress(
                {
                    "models": {
                        sample_model()["queue_id"]: {
                            "model_name": "Sample",
                            "category": "Coding",
                            "status": "completed",
                            "prompts_completed": 1,
                            "total_prompts": 1,
                            "csv_file": csv_path,
                        }
                    }
                }
            )

            projection = LiveStatusProjection(
                store=store,
                system=FakeSystem(),
                queue_loader=lambda: [sample_model()],
            )
            payload = projection.status_payload()

        self.assertEqual(payload["status"], "idle")
        self.assertEqual(payload["completed_models"][0]["tps_avg"], 10.0)
        self.assertEqual(payload["all_models"][0]["json_success_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
