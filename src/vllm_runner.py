"""
vLLM Runner for AWQ/GPTQ/MoE Model Benchmarking

Provides an inference backend for quantized models (AWQ, GPTQ, FP8)
and Mixture-of-Experts (MoE) models that need vLLM's optimized serving.

vLLM serves an OpenAI-compatible API, so this runner starts a vLLM server
as a subprocess and communicates via HTTP.

Key features:
  - Starts vLLM server on-demand for each model
  - Automatic AWQ/GPTQ quantization detection
  - Same BenchmarkResult interface as OllamaClient
  - Memory-efficient serving for MoE models
  - Clean shutdown after benchmarking

Requires: vllm (pip install vllm)
"""

import time
import os
import json
import subprocess
import signal
import sys
import gc
import requests
from dataclasses import dataclass
from typing import Optional


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


def _detect_quantization(model_id: str) -> Optional[str]:
    """Detect quantization type from model name."""
    upper = model_id.upper()
    if "AWQ" in upper:
        return "awq"
    if "GPTQ" in upper:
        return "gptq"
    if "FP8" in upper:
        return "fp8"
    return None


class VLLMRunner:
    """
    Manages a vLLM server process and provides inference via its OpenAI API.

    Usage:
        runner = VLLMRunner()
        error = runner.start_server("cyankiwi/Qwen3-30B-A3B-Instruct-AWQ-4bit")
        if error is None:
            result = runner.generate_benchmark(model_id, "Hello", max_tokens=128)
        runner.stop_server()
    """

    def __init__(self, port: int = 8100, gpu_memory_utilization: float = 0.85):
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.base_url = f"http://localhost:{port}"
        self.server_process = None
        self.current_model_id = None

    @property
    def is_running(self) -> bool:
        return self.server_process is not None and self.server_process.poll() is None

    def start_server(
        self,
        model_id: str,
        max_model_len: int = 2048,
        quantization: Optional[str] = None,
    ) -> Optional[str]:
        """
        Start a vLLM server for the given model.

        Args:
            model_id: HuggingFace model ID
            max_model_len: Maximum context length (lower = less VRAM)
            quantization: "awq", "gptq", "fp8", or None for auto-detection

        Returns:
            None on success, error string on failure
        """
        if self.current_model_id == model_id and self.is_running:
            return None  # Already running

        self.stop_server()

        # Auto-detect quantization if not specified
        if quantization is None:
            quantization = _detect_quantization(model_id)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id,
            "--port", str(self.port),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", "float16",
            "--trust-remote-code",
        ]

        if quantization:
            cmd.extend(["--quantization", quantization])

        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
            )

            # Wait for server to be ready (up to 120 seconds)
            for attempt in range(120):
                time.sleep(1)
                if self.server_process.poll() is not None:
                    # Server exited — read stderr for error
                    _, stderr = self.server_process.communicate()
                    self.server_process = None
                    return f"vLLM server exited early: {stderr[-500:]}"

                try:
                    resp = requests.get(f"{self.base_url}/health", timeout=2)
                    if resp.status_code == 200:
                        self.current_model_id = model_id
                        return None
                except requests.ConnectionError:
                    continue

            # Timeout — kill and report
            self.stop_server()
            return f"vLLM server failed to start within 120s for {model_id}"

        except FileNotFoundError:
            return "vLLM is not installed. Run: pip install vllm"
        except Exception as e:
            self.stop_server()
            return f"Failed to start vLLM server: {str(e)}"

    def stop_server(self):
        """Stop the vLLM server process."""
        if self.server_process is not None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            except Exception:
                pass
            self.server_process = None
        self.current_model_id = None
        gc.collect()

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
        Run a single inference via vLLM's OpenAI-compatible API.
        Mirrors the OllamaClient.generate_benchmark interface.
        """
        if not self.is_running:
            return BenchmarkResult(
                model=model, tps=0.0, ttft=0.0, latency=0.0,
                eval_count=0, content="",
                error="vLLM server not running. Call start_server() first."
            )

        try:
            payload = {
                "model": self.current_model_id or model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }

            start_time = time.perf_counter()

            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=300,
            )

            end_time = time.perf_counter()

            if resp.status_code != 200:
                return BenchmarkResult(
                    model=model, tps=0.0, ttft=0.0, latency=0.0,
                    eval_count=0, content="",
                    error=f"vLLM API error {resp.status_code}: {resp.text[:200]}"
                )

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("text", "")
            usage = data.get("usage", {})
            eval_count = usage.get("completion_tokens", 0)

            latency = end_time - start_time
            tps = eval_count / latency if latency > 0 else 0

            # vLLM doesn't report TTFT in non-streaming mode;
            # estimate as prompt processing time
            prompt_tokens = usage.get("prompt_tokens", 0)
            ttft = latency * 0.1 if eval_count > 0 else latency  # rough estimate

            return BenchmarkResult(
                model=model,
                tps=tps,
                ttft=ttft,
                latency=latency,
                eval_count=eval_count,
                content=content
            )

        except requests.Timeout:
            return BenchmarkResult(
                model=model, tps=0.0, ttft=0.0, latency=0.0,
                eval_count=0, content="",
                error="vLLM inference timed out (300s)"
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
        Generate structured JSON output via vLLM.
        Uses guided decoding if available, otherwise prompt engineering.
        """
        if not self.is_running:
            return "", "vLLM server not running. Call start_server() first."

        try:
            schema_str = json.dumps(schema, indent=2)
            structured_prompt = (
                f"{prompt}\n\n"
                f"Respond ONLY with valid JSON matching this schema:\n"
                f"```json\n{schema_str}\n```\n"
                f"Output JSON only, no other text:"
            )

            payload = {
                "model": self.current_model_id or model,
                "prompt": structured_prompt,
                "max_tokens": 512,
                "temperature": temperature,
                "stream": False,
            }

            # Try guided decoding if vLLM supports it
            try:
                payload["guided_json"] = schema
            except Exception:
                pass

            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=300,
            )

            if resp.status_code != 200:
                return "", f"vLLM API error {resp.status_code}: {resp.text[:200]}"

            data = resp.json()
            content = data.get("choices", [{}])[0].get("text", "").strip()
            return content, None

        except Exception as e:
            return "", str(e)

    @staticmethod
    def check_dependencies() -> dict:
        """Check if vLLM is installed and available."""
        try:
            import vllm
            has_vllm = True
            version = getattr(vllm, "__version__", "unknown")
        except ImportError:
            has_vllm = False
            version = "not installed"

        import torch
        return {
            "vllm": has_vllm,
            "vllm_version": version,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "N/A"
            ),
        }


if __name__ == "__main__":
    """Quick self-test."""
    print("vLLM Runner — Dependency Check")
    print("=" * 50)

    deps = VLLMRunner.check_dependencies()
    for k, v in deps.items():
        status = "✅" if v else "❌"
        print(f"  {status} {k}: {v}")

    if not deps["vllm"]:
        print("\nInstall with: pip install vllm")
