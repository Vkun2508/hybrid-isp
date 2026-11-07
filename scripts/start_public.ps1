# Serve the Hybrid ISP app from this laptop and make it public with a free Cloudflare quick tunnel.
#
#   Double-click scripts\start_public.bat   (or: powershell -ExecutionPolicy Bypass -File scripts\start_public.ps1)
#
# Starts (if not already running) the app on http://127.0.0.1:7860 and a tunnel to it, both as background
# processes that keep running after this window closes. Prints the public https://....trycloudflare.com link
# and saves it to .cache\public\url.txt. The link changes whenever the tunnel restarts (reboot, network loss),
# so after a restart run this again and update the link in the README.
# Stop everything with scripts\stop_public.bat.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$cf = Join-Path $root ".cache\tools\cloudflared.exe"
$logs = Join-Path $root ".cache\public"
New-Item -ItemType Directory -Force $logs | Out-Null

function Test-App {
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:7860/api/health" | Out-Null; return $true }
    catch { return $false }
}

# 1. The app
if (Test-App) {
    Write-Host "App already running on http://127.0.0.1:7860"
} else {
    Write-Host "Starting the app..."
    Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", "7860" `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "app.out.log") -RedirectStandardError (Join-Path $logs "app.err.log")
    for ($i = 0; $i -lt 90 -and -not (Test-App); $i++) { Start-Sleep -Seconds 1 }
    if (-not (Test-App)) { Write-Host "The app did not start. See $logs\app.err.log"; exit 1 }
    Write-Host "App is up."
}

# 2. The tunnel
$tunnelLog = Join-Path $logs "tunnel.log"
$running = Get-Process cloudflared -ErrorAction SilentlyContinue
$url = $null
if ($running -and (Test-Path $tunnelLog)) {
    $m = Select-String -Path $tunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -Last 1
    if ($m) { $url = $m.Matches[0].Value; Write-Host "Tunnel already running." }
}
if (-not $url) {
    if ($running) { $running | Stop-Process -Force }
    Write-Host "Starting the Cloudflare tunnel..."
    if (Test-Path $tunnelLog) { Remove-Item $tunnelLog -Force }
    Start-Process -FilePath $cf -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:7860" `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "tunnel.out.log") -RedirectStandardError $tunnelLog
    for ($i = 0; $i -lt 60 -and -not $url; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $tunnelLog) {
            $m = Select-String -Path $tunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -Last 1
            if ($m) { $url = $m.Matches[0].Value }
        }
    }
    if (-not $url) { Write-Host "No tunnel link yet. See $tunnelLog"; exit 1 }
}

Set-Content -Path (Join-Path $logs "url.txt") -Value $url -Encoding ascii
Write-Host ""
Write-Host "PUBLIC LINK: $url"
Write-Host "Keep this laptop on, awake and online while people use it."
