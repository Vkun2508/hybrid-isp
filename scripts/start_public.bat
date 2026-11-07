@echo off
REM Double-click: start the Hybrid ISP app + public Cloudflare tunnel (see start_public.ps1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_public.ps1"
pause
