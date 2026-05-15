import time
import os
import base64
import ollama
from dataclasses import dataclass
from typing import Optional

@dataclass
class BenchmarkResult:
    model: str
    tps: float
    ttft: float
    latency: float
    eval_count: int
    content: str
    error: Optional[str] = None

class OllamaClient:
    def __init__(self, host="http://localhost:11434"):
        # The ollama Python library reads from OLLAMA_HOST env var
        if host != "http://localhost:11434":
            os.environ["OLLAMA_HOST"] = host

    def _encode_image(self, image_path: str) -> str:
        """Read an image file and return a base64-encoded data URI string."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        return base64.b64encode(img_bytes).decode("utf-8")

    def _inference_options(self, extra: dict = None) -> dict:
        """Build optimized Ollama inference options.
        Merged with caller overrides (extra dict takes precedence).
        """
        opts = {
            "num_ctx": 2048,
            "num_batch": 512,
            "num_thread": 8,
            "num_gpu": 999,
        }
        if extra:
            opts.update(extra)
        return opts

    def generate_benchmark(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        is_vision: bool = False,
        image_path: Optional[str] = None
    ) -> BenchmarkResult:
        try:
            start_time = time.perf_counter()
            first_token_time = None

            message = {'role': 'user', 'content': prompt}

            if is_vision:
                if image_path and os.path.exists(image_path):
                    message['images'] = [self._encode_image(image_path)]
                else:
                    return BenchmarkResult(
                        model=model, tps=0.0, ttft=0.0, latency=0.0,
                        eval_count=0, content="",
                        error=f"Multimodal Vision prompt requires image_path. "
                               f"Provided: {repr(image_path)}. "
                               f"Real image assets must be present for vision benchmarking."
                    )

            stream = ollama.generate(
                model=model,
                prompt=prompt,
                images=message.get('images'),
                stream=True,
                keep_alive="24h",
                options=self._inference_options({
                    "num_predict": max_tokens,
                    "temperature": temperature,
                })
            )

            content_chunks = []
            final_chunk = {}  # Safety: prevent UnboundLocalError on empty stream

            for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                content_chunks.append(chunk.get('response', ''))

                if chunk.get('done'):
                    final_chunk = chunk

            end_time = time.perf_counter()

            if first_token_time is None:
                first_token_time = end_time

            latency = end_time - start_time
            ttft = first_token_time - start_time

            eval_count = final_chunk.get('eval_count', 0)
            eval_duration_ns = final_chunk.get('eval_duration', 0)

            tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0

            if tps == 0 and eval_count > 0:
                tps = eval_count / (latency - ttft) if (latency - ttft) > 0 else 0

            return BenchmarkResult(
                model=model,
                tps=tps,
                ttft=ttft,
                latency=latency,
                eval_count=eval_count,
                content="".join(content_chunks)
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
        try:
            message = {'role': 'user', 'content': prompt}

            if is_vision:
                if image_path and os.path.exists(image_path):
                    message['images'] = [self._encode_image(image_path)]
                else:
                    return "", (
                        f"Multimodal Vision prompt requires image_path. "
                        f"Provided: {repr(image_path)}. "
                        f"Real image assets must be present for vision benchmarking."
                    )

            response = ollama.chat(
                model=model,
                messages=[message],
                format=schema,
                keep_alive="24h",
                options=self._inference_options({
                    "temperature": temperature,
                })
            )
            return response.message.content, None
        except Exception as e:
            return "", str(e)
