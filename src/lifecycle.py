"""Benchmark model lifecycle module.

This module owns runtime acquisition and cleanup so benchmark scripts do not
need to duplicate provider-specific pull/load/delete rules.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.model_entry import as_model_entry

LogFn = Callable[[str], None]


@dataclass
class RuntimeHandle:
    model_ref: str
    runtime_client: Any | None
    model_entry: dict[str, Any]
    delete_on_cleanup: bool = True


def _noop_log(_: str) -> None:
    return None


def _log(log: LogFn | None, message: str) -> None:
    (log or _noop_log)(message)


def get_free_disk_space_gb() -> float:
    from config import MODELS_DIR

    target = MODELS_DIR if os.path.exists(MODELS_DIR) else Path(__file__).parent.parent
    _, _, free = shutil.disk_usage(target)
    return free / (1024**3)


def clear_huggingface_cache(log: LogFn | None = None) -> None:
    from config import MODELS_DIR

    if not os.path.exists(MODELS_DIR):
        return
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                _log(log, f"Cleared HuggingFace cache: {item}")
        except Exception as exc:
            _log(log, f"Warning: Could not clear {item}: {exc}")


def check_disk_space(required_gb: float = 5.0, log: LogFn | None = None) -> bool:
    free_gb = get_free_disk_space_gb()
    if free_gb < required_gb:
        _log(log, f"Low disk space: {free_gb:.2f} GB free (need {required_gb} GB)")
        _log(log, "Clearing HuggingFace cache...")
        clear_huggingface_cache(log)
        free_gb = get_free_disk_space_gb()
        _log(log, f"After cleanup: {free_gb:.2f} GB free")
    if free_gb < required_gb:
        _log(log, f"ERROR: Insufficient disk space ({free_gb:.2f} GB). Skipping model.")
        return False
    return True


def check_ollama_model_exists(model_tag: str) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return False
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.split()
            if not parts:
                continue
            listed_name = parts[0]
            if listed_name == model_tag or listed_name == f"{model_tag}:latest":
                return True
            if ":" not in model_tag and listed_name == f"{model_tag}:latest":
                return True
        return False
    except Exception:
        return False


def acquire_runtime(model_entry: dict[str, Any], log: LogFn | None = None) -> RuntimeHandle | None:
    from config import MODELS_DIR

    entry = as_model_entry(model_entry)

    if entry.is_provider_unsupported:
        _log(log, "  SKIP: provider_unsupported")
        return None

    if entry.resolved_runtime == "ollama":
        ollama_tag = entry.ollama_tag or entry.resolved_model_ref
        if check_ollama_model_exists(ollama_tag):
            _log(log, f"  Model {ollama_tag} already exists in Ollama")
            return RuntimeHandle(ollama_tag, None, model_entry, delete_on_cleanup=False)

        if pull_ollama_tag(ollama_tag, log=log):
            return RuntimeHandle(ollama_tag, None, model_entry)
        _log(log, f"  Failed to pull {ollama_tag}")
        return None

    if entry.resolved_runtime == "huggingface_gguf":
        hf_repo = entry.hf_repo
        ollama_tag = entry.requested_name.lower().replace(" ", "-")

        if check_ollama_model_exists(ollama_tag):
            _log(log, f"  Model {ollama_tag} already exists in Ollama (from GGUF)")
            return RuntimeHandle(ollama_tag, None, model_entry, delete_on_cleanup=False)

        if not hf_repo:
            _log(log, "  SKIP: No hf_repo for GGUF download")
            return None

        if not check_disk_space(required_gb=5.0, log=log):
            return None

        _log(log, f"  Downloading {hf_repo} from HuggingFace (GGUF)...")
        local_dir = os.path.join(MODELS_DIR, ollama_tag.replace(":", "_").replace("/", "_"))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "huggingface_hub.commands.huggingface_cli",
                "download",
                hf_repo,
                "--local-dir",
                local_dir,
                "--local-dir-use-symlinks",
                "False",
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            _log(log, f"  HuggingFace download error: {result.stderr[-500:]}")
            return None

        gguf_files = [f for f in os.listdir(local_dir) if f.endswith(".gguf")]
        if not gguf_files:
            _log(log, f"  SKIP: No GGUF file found in {hf_repo}")
            try:
                shutil.rmtree(local_dir)
            except Exception:
                pass
            return None

        modelfile_path = os.path.join(local_dir, "Modelfile")
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(f"FROM ./{gguf_files[0]}")

        _log(log, f"  Creating Ollama model from GGUF: {gguf_files[0]}")
        subprocess.run(
            ["ollama", "create", ollama_tag, "-f", modelfile_path],
            encoding="utf-8",
            errors="replace",
        )

        if check_ollama_model_exists(ollama_tag):
            _log(log, f"  Successfully loaded {ollama_tag} from GGUF")
            return RuntimeHandle(ollama_tag, None, model_entry)
        _log(log, "  Failed to create Ollama model from GGUF")
        return None

    if entry.resolved_runtime == "hf_transformers":
        try:
            from src.hf_runner import HFTransformersRunner
        except ImportError:
            _log(log, "  SKIP: HF Transformers runner not installed (pip install transformers accelerate bitsandbytes)")
            return None

        if not entry.hf_repo:
            _log(log, "  SKIP: No hf_repo for Transformers download")
            return None

        _log(log, f"  Loading {entry.hf_repo} via HuggingFace Transformers (4-bit)...")
        runner = HFTransformersRunner()
        error = runner.load_model(entry.hf_repo, is_vision=entry.category == "Vision")
        if error:
            _log(log, f"  HF load error: {error}")
            return None

        _log(log, f"  Successfully loaded {entry.hf_repo} via Transformers")
        return RuntimeHandle(entry.hf_repo, runner, model_entry)

    if entry.resolved_runtime == "vllm":
        try:
            from src.vllm_runner import VLLMRunner
        except ImportError:
            _log(log, "  SKIP: vLLM runner not installed (pip install vllm)")
            return None

        if not entry.hf_repo:
            _log(log, "  SKIP: No hf_repo for vLLM")
            return None

        _log(log, f"  Starting vLLM server for {entry.hf_repo}...")
        runner = VLLMRunner(gpu_memory_utilization=0.85)
        error = runner.start_server(entry.hf_repo, max_model_len=2048)
        if error:
            _log(log, f"  vLLM start error: {error}")
            return None

        _log(log, f"  Successfully started vLLM for {entry.hf_repo}")
        return RuntimeHandle(entry.hf_repo, runner, model_entry)

    _log(log, f"  SKIP: Unknown runtime '{entry.resolved_runtime}'")
    return None


def cleanup_runtime(handle: RuntimeHandle | None, log: LogFn | None = None) -> None:
    if handle is None:
        return

    from config import MODELS_DIR

    entry = as_model_entry(handle.model_entry)
    skip_lifecycle = os.environ.get("BENCHMARK_SKIP_LIFECYCLE") == "1"

    if handle.runtime_client is not None:
        if entry.resolved_runtime == "hf_transformers" and hasattr(handle.runtime_client, "unload_model"):
            handle.runtime_client.unload_model()
            _log(log, "  Unloaded HF Transformers model")
            if entry.hf_repo:
                try:
                    from src.hf_runner import cleanup_hf_model_cache

                    if cleanup_hf_model_cache(entry.hf_repo):
                        _log(log, f"  Cleared HF cache for {entry.hf_repo}")
                except Exception:
                    pass
        elif entry.resolved_runtime == "vllm" and hasattr(handle.runtime_client, "stop_server"):
            handle.runtime_client.stop_server()
            _log(log, "  Stopped vLLM server")
            if entry.hf_repo:
                _cleanup_hf_cache(entry.hf_repo, log)
    else:
        if handle.model_ref and handle.delete_on_cleanup and not skip_lifecycle:
            _log(log, f"  Deleting model from Ollama: {handle.model_ref}")
            try:
                result = subprocess.run(
                    ["ollama", "rm", handle.model_ref],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
                if result.returncode == 0:
                    _log(log, f"  Deleted {handle.model_ref} from Ollama")
                else:
                    _log(log, f"  Warning: ollama rm failed: {result.stderr.strip()}")
            except Exception as exc:
                _log(log, f"  Warning: Could not delete model: {exc}")

        if entry.source == "huggingface" or entry.resolved_runtime == "huggingface_gguf":
            local_dir_name = (
                handle.model_ref.replace(":", "_").replace("/", "_") if handle.model_ref else None
            )
            if local_dir_name:
                local_dir = os.path.join(MODELS_DIR, local_dir_name)
                if os.path.exists(local_dir):
                    try:
                        shutil.rmtree(local_dir)
                        _log(log, f"  Cleared HuggingFace download dir: {local_dir_name}")
                    except Exception as exc:
                        _log(log, f"  Warning: Could not clear HF dir: {exc}")

    free_gb = get_free_disk_space_gb()
    _log(log, f"  Free disk space after cleanup: {free_gb:.2f} GB")


def _cleanup_hf_cache(hf_repo: str, log: LogFn | None = None) -> None:
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    repo_dir_name = f"models--{hf_repo.replace('/', '--')}"
    repo_cache = os.path.join(hf_cache, repo_dir_name)
    if os.path.isdir(repo_cache):
        try:
            shutil.rmtree(repo_cache)
            _log(log, f"  Cleared HF cache: {repo_dir_name}")
        except Exception as exc:
            _log(log, f"  Warning: Could not clear HF cache: {exc}")


def download_model(model_entry: dict[str, Any], log: LogFn | None = None) -> tuple[str | None, Any | None]:
    handle = acquire_runtime(model_entry, log=log)
    if handle is None:
        return None, None
    return handle.model_ref, handle.runtime_client


def cleanup_model(
    model_entry: dict[str, Any],
    model_ref: str,
    runtime_client: Any | None = None,
    log: LogFn | None = None,
) -> None:
    cleanup_runtime(RuntimeHandle(model_ref, runtime_client, model_entry), log=log)


def pull_ollama_tag(ollama_tag: str, log: LogFn | None = None, timeout: int | None = None) -> bool:
    if check_ollama_model_exists(ollama_tag):
        _log(log, f"  Model {ollama_tag} already exists in Ollama")
        return True
    _log(log, f"  Pulling {ollama_tag} from Ollama...")
    try:
        kwargs: dict[str, Any] = {
            "capture_output": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        result = subprocess.run(["ollama", "pull", ollama_tag], **kwargs)
    except Exception as exc:
        _log(log, f"  Download error: {exc}")
        return False
    if result.returncode == 0:
        _log(log, f"  Successfully downloaded {ollama_tag}")
    else:
        output = result.stderr.strip() or result.stdout
        _log(log, f"  Pull output: {output[-500:]}")
    return check_ollama_model_exists(ollama_tag)


def delete_ollama_tag(ollama_tag: str, log: LogFn | None = None, timeout: int = 60) -> None:
    if not ollama_tag:
        return
    _log(log, f"  Deleting model from Ollama: {ollama_tag}")
    try:
        result = subprocess.run(
            ["ollama", "rm", ollama_tag],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode == 0:
            _log(log, f"  Deleted {ollama_tag} from Ollama")
        else:
            _log(log, f"  Warning: ollama rm failed: {result.stderr.strip()}")
    except Exception as exc:
        _log(log, f"  Warning: Could not delete model: {exc}")
