@echo off
REM Double-click: stop the public tunnel and the app started by start_public.bat
taskkill /IM cloudflared.exe /F >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "127.0.0.1:7860" ^| findstr LISTENING') do taskkill /PID %%p /F >nul 2>&1
echo Stopped the tunnel and the app.
pause
