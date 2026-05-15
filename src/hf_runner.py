"""
HuggingFace Transformers Runner for Local LLM Benchmarking

Provides an inference backend for models that are only available as
SafeTensors/PyTorch weights on HuggingFace (not GGUF / not in Ollama).

Key features:
  - 4-bit quantization via bitsandbytes for 4GB VRAM constraint
  - Automatic device_map for GPU/CPU split
  - Same BenchmarkResult interface as OllamaClient
  - Vision model support via AutoProcessor
  - Download-Test-Delete lifecycle for disk management

Requires: transformers, torch, accelerate, bitsandbytes
"""

import time
import os
import gc
import json
import base64
from dataclasses import dataclass
from typing import Optional

import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
        BitsAndBytesConfig,
        GenerationConfig,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import bitsandbytes
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


@dataclass
class BenchmarkResult:
    """Mirror of ollama_client.BenchmarkResult for compatibility."""
    model: str
    tps: float
    ttft: float
    latency: float
    eval_count: int
    content: str
    error: Optional[str] = None


class HFTransformersRunner:
    """
    Load and run HuggingFace models locally with 4-bit quantization.

    Usage:
        runner = HFTransformersRunner()
        runner.load_model("Qwen/Qwen3-0.6B")
        result = runner.generate_benchmark("Qwen/Qwen3-0.6B", "Hello", max_tokens=128)
        runner.unload_model()
    """

    def __init__(self, gpu_index: int = 0):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is not installed. "
                "Run: pip install transformers accelerate bitsandbytes torch"
            )
        self.gpu_index = gpu_index
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.current_model_id = None
        self._device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_quantization_config(self) -> Optional["BitsAndBytesConfig"]:
        """Get 4-bit quantization config if bitsandbytes is available."""
        if not HAS_BNB or not torch.cuda.is_available():
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def load_model(self, model_id: str, is_vision: bool = False) -> Optional[str]:
        """
        Download and load a model from HuggingFace.

        Args:
            model_id: HuggingFace repo ID (e.g. "Qwen/Qwen3-0.6B")
            is_vision: If True, load with AutoProcessor for vision models

        Returns:
            None on success, error string on failure
        """
        if self.current_model_id == model_id and self.model is not None:
            return None  # Already loaded

        # Unload any previously loaded model
        self.unload_model()

        try:
            quant_config = self._get_quantization_config()

            load_kwargs = {
                "pretrained_model_name_or_path": model_id,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if quant_config:
                load_kwargs["quantization_config"] = quant_config

            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

            if is_vision:
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        model_id, trust_remote_code=True
                    )
                    self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None
                except Exception:
                    pass

            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True
                )

            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.current_model_id = model_id
            return None

        except Exception as e:
            self.unload_model()
            return f"Failed to load {model_id}: {str(e)}"

    def unload_model(self):
        """Unload the current model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.current_model_id = None

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_benchmark(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        is_vision: bool = False,
        image_path: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Run a single inference for benchmarking.
        Mirrors the OllamaClient.generate_benchmark interface.
        """
        if self.model is None:
            return BenchmarkResult(
                model=model, tps=0.0, ttft=0.0, latency=0.0,
                eval_count=0, content="",
                error="Model not loaded. Call load_model() first."
            )

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=2048
            )
            input_ids = inputs["input_ids"].to(self.model.device)
            input_length = input_ids.shape[1]

            # Generation config
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            # Measure timing
            start_time = time.perf_counter()

            # Synchronize GPU before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                outputs = self.model.generate(input_ids, **gen_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Calculate metrics
            output_ids = outputs[0][input_length:]
            eval_count = len(output_ids)
            latency = end_time - start_time

            # TTFT approximation: for non-streaming, estimate as latency / eval_count
            ttft = latency / max(eval_count, 1) if eval_count > 0 else latency

            tps = eval_count / latency if latency > 0 else 0

            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            return BenchmarkResult(
                model=model,
                tps=tps,
                ttft=ttft,
                latency=latency,
                eval_count=eval_count,
                content=content
            )

        except Exception as e:
            return BenchmarkResult(
                model=model, tps=0.0, ttft=0.0, latency=0.0,
                eval_count=0, content="", error=str(e)
            )

    def generate_structured(
        self,
        model: str,
        prompt: str,
        schema: dict,
        temperature: float = 0.0,
        is_vision: bool = False,
        image_path: Optional[str] = None
    ) -> tuple[str, Optional[str]]:
        """
        Generate structured JSON output.
        Mirrors the OllamaClient.generate_structured interface.

        Note: Unlike Ollama's native JSON mode, this uses prompt engineering
        to request JSON output. Success rate may be lower for small models.
        """
        if self.model is None:
            return "", "Model not loaded. Call load_model() first."

        try:
            schema_str = json.dumps(schema, indent=2)
            structured_prompt = (
                f"{prompt}\n\n"
                f"Respond ONLY with valid JSON matching this schema:\n"
                f"```json\n{schema_str}\n```\n"
                f"Output JSON only, no other text:"
            )

            result = self.generate_benchmark(
                model, structured_prompt,
                max_tokens=512,
                temperature=temperature,
                is_vision=is_vision,
                image_path=image_path
            )

            if result.error:
                return "", result.error

            # Try to extract JSON from the response
            content = result.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines).strip()

            return content, None

        except Exception as e:
            return "", str(e)

    @staticmethod
    def check_dependencies() -> dict:
        """Check which optional dependencies are available."""
        return {
            "transformers": HAS_TRANSFORMERS,
            "bitsandbytes": HAS_BNB,
            "torch": True,  # Always True if we got this far
            "cuda_available": torch.cuda.is_available() if HAS_TRANSFORMERS else False,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "N/A"
            ),
            "vram_total_mb": (
                torch.cuda.get_device_properties(0).total_mem / (1024**2)
                if torch.cuda.is_available() else 0
            ),
        }


def cleanup_hf_model_cache(model_id: str, cache_dir: str = None):
    """Remove downloaded model files to free disk space."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    safe_name = model_id.replace("/", "--")
    model_cache_path = os.path.join(cache_dir, f"models--{safe_name}")

    if os.path.exists(model_cache_path):
        import shutil
        shutil.rmtree(model_cache_path, ignore_errors=True)
        return True
    return False


if __name__ == "__main__":
    """Quick self-test."""
    print("HF Transformers Runner — Dependency Check")
    print("=" * 50)

    deps = HFTransformersRunner.check_dependencies()
    for k, v in deps.items():
        status = "✅" if v else "❌"
        print(f"  {status} {k}: {v}")

    if not deps["transformers"]:
        print("\nInstall with: pip install transformers accelerate bitsandbytes torch")
