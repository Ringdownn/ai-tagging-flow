# AI Tagging Flow - Local Stop Script (Windows PowerShell)

$ROOT = Split-Path -Parent $PSScriptRoot
$PID_DIR = "$ROOT\temp\pids"

if (-not (Test-Path $PID_DIR)) {
    Write-Host "[INFO] No running service records"
    exit 0
}

$services = @('orchestration', 'remove-bg', 'detect-crop')
foreach ($name in $services) {
    $pidFile = "$PID_DIR\$name.pid"
    if (Test-Path $pidFile) {
        $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($pidValue) {
            $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host "[STOP] $name (PID $pidValue)"
                $proc.Kill()
                $proc.WaitForExit(5000) | Out-Null
            } else {
                Write-Host "[SKIP] $name process not found"
            }
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}

# Fallback: kill processes on known ports
$ports = @(8000, 8001, 8002)
foreach ($port in $ports) {
    $listeners = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $proc = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
        if ($proc -and ($proc.Name -match 'python|uvicorn')) {
            Write-Host "[STOP] Freeing port $port (PID $($proc.Id))"
            $proc.Kill()
        }
    }
}

Write-Host "[DONE] Services stopped"
