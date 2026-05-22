"""
Pipeline Watchdog
=================
Monitors the benchmark pipeline and Ollama server every 5 minutes.
If Ollama becomes unresponsive (hung subprocess holding the port),
kills the hung process and restarts the pipeline.

Usage:
    python scripts/watchdog.py
"""

import os
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import json
import socket
import subprocess
import signal
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
PID_FILE = os.path.join(LOG_DIR, "benchmark.pid")
WATCHDOG_LOG = os.path.join(LOG_DIR, "watchdog.log")
OLLAMA_PORT = 11434
CHECK_INTERVAL = 300  # 5 minutes


def log(msg: str):
    """Write timestamped message to watchdog log and stdout."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 3.0) -> bool:
    """Check if a TCP port is listening."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def is_ollama_responsive(timeout: int = 15) -> bool:
    """Send a minimal generate request to Ollama and check for response."""
    try:
        result = subprocess.run(
            [
                "powershell", "-Command",
                f"Invoke-RestMethod -Uri 'http://127.0.0.1:{OLLAMA_PORT}/api/generate' "
                f"-Method POST -ContentType 'application/json' "
                f"-Body '{{\"model\":\"tinyllama\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{{\"num_predict\":1}}}}' "
                f"-TimeoutSec {timeout}"
            ],
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=timeout + 5,
        )
        return result.returncode == 0 and "response" in (result.stdout or "")
    except (subprocess.TimeoutExpired, Exception):
        return False


def get_ollama_pid() -> int | None:
    """Get PID of the Ollama process listening on the port."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "(Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue | "
             "Where-Object { $_.State -eq 'Listen' } | "
             "Select-Object -First 1).OwningProcess"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=10,
        )
        pid_str = (result.stdout or "").strip()
        return int(pid_str) if pid_str.isdigit() else None
    except Exception:
        return None


def kill_hung_ollama_connections():
    """Kill any CloseWait/FinWait2 connections to Ollama port."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue | "
             "Where-Object { $_.State -in @('CloseWait','FinWait2') } | "
             "Select-Object -ExpandProperty OwningProcess -Unique"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=10,
        )
        pids = [int(p.strip()) for p in (result.stdout or "").strip().split("\n") if p.strip().isdigit()]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                log(f"  Killed hung connection PID {pid}")
            except (OSError, ProcessLookupError):
                pass
        return len(pids)
    except Exception:
        return 0


def get_benchmark_pid() -> int | None:
    """Read the benchmark PID from the PID file."""
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            # Verify process exists
            os.kill(pid, 0)
            return pid
    except (ValueError, OSError, ProcessLookupError):
        pass
    return None


def restart_pipeline():
    """Restart the benchmark pipeline via run_background.ps1."""
    log("  Restarting pipeline...")
    ps1 = os.path.join(BASE_DIR, "run_background.ps1")
    try:
        subprocess.Popen(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps1],
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        log("  Pipeline restart initiated.")
        time.sleep(10)
    except Exception as e:
        log(f"  ERROR: Could not restart pipeline: {e}")


def main():
    log("=" * 60)
    log("WATCHDOG STARTED — checking every 5 minutes")
    log("=" * 60)

    while True:
        time.sleep(CHECK_INTERVAL)
        log("--- Watchdog check ---")

        # 1. Is Ollama port listening?
        if not is_port_open(OLLAMA_PORT):
            log("  Ollama port NOT listening. Waiting...")
            continue

        # 2. Is benchmark pipeline running?
        bench_pid = get_benchmark_pid()
        if bench_pid:
            log(f"  Pipeline alive (PID {bench_pid})")
        else:
            log("  Pipeline NOT running. Restarting...")
            restart_pipeline()
            continue

        # 3. Is Ollama responsive?
        if is_ollama_responsive():
            log("  Ollama responsive. All OK.")
            continue

        # 4. Ollama is hung — kill hung connections
        log("  Ollama HUNG! Attempting recovery...")
        killed = kill_hung_ollama_connections()
        log(f"  Killed {killed} hung connection(s)")
        time.sleep(3)

        # 5. Check again
        if is_ollama_responsive():
            log("  Ollama recovered after killing hung connections.")
            continue

        # 6. Still hung — restart pipeline
        log("  Ollama still hung. Restarting pipeline...")
        restart_pipeline()


if __name__ == "__main__":
    main()
