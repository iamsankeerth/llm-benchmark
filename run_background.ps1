$ErrorActionPreference = "Continue"

$baseDir = "C:\Users\lenovo\Desktop\San\Fun_Projects\llmbenchamrk_v2"
$pythonExe = "$baseDir\.venv\Scripts\python.exe"
$script = "$baseDir\scripts\run_full_benchmark.py"
$logFile = "$baseDir\logs\benchmarks.log"

New-Item -ItemType Directory -Path "$baseDir\logs" -Force | Out-Null

$env:PYTHONIOENCODING = "utf-8"

# Ensure all child Python processes inherit UTF-8 encoding
$pythonEnv = @{
    "PYTHONIOENCODING" = "utf-8"
    "ANSI_COLORS_DISABLED" = "1"
}

# 1. Start dashboard API server (background)
$apiArgs = @("-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000")
$apiProcess = Start-Process -FilePath $pythonExe -ArgumentList $apiArgs -WorkingDirectory $baseDir -NoNewWindow -PassThru
Write-Host "Dashboard API started on http://localhost:8000 (PID: $($apiProcess.Id))"
Write-Host "Live Dashboard: http://localhost:8000/live"

# Give API server a moment to start
Start-Sleep -Seconds 2

# 2. Start benchmark pipeline (background)
$process = Start-Process -FilePath $pythonExe -ArgumentList $script -WorkingDirectory $baseDir -NoNewWindow -PassThru -RedirectStandardOutput $logFile -RedirectStandardError "$baseDir\logs\benchmarks_err.log" -Environment @{PYTHONIOENCODING="utf-8";ANSI_COLORS_DISABLED="1"}

# 3. Write benchmark PID for stop button
$process.Id | Set-Content "$baseDir\logs\benchmark.pid"

Write-Host ""
Write-Host "Benchmark PID: $($process.Id)"
Write-Host "Log file: $logFile"
Write-Host ""
Write-Host "Monitor: Get-Content $logFile -Tail 30"
Write-Host "Stop:    Stop-Process -Id $($process.Id)"
Write-Host "Dashboard: http://localhost:8000/live"

# 4. Start watchdog (background) — monitors Ollama + pipeline every 5 minutes
$watchdogScript = "$baseDir\scripts\watchdog.py"
$watchdogArgs = @($watchdogScript)
$watchdogProcess = Start-Process -FilePath $pythonExe -ArgumentList $watchdogArgs -WorkingDirectory $baseDir -NoNewWindow -PassThru -RedirectStandardOutput "$baseDir\logs\watchdog_stdout.log" -RedirectStandardError "$baseDir\logs\watchdog_stderr.log" -Environment @{PYTHONIOENCODING="utf-8"}
$watchdogProcess.Id | Set-Content "$baseDir\logs\watchdog.pid"
Write-Host ""
Write-Host "Watchdog PID: $($watchdogProcess.Id) (checks every 5 min)"
Write-Host "Watchdog log: $baseDir\logs\watchdog.log"
