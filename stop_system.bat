@echo off
echo Stopping Pairs Trading System...

REM Kill scheduler daemon via PID file
if exist logs\scheduler.pid (
    set /p PID=<logs\scheduler.pid
    echo Stopping scheduler daemon (PID %PID%)...
    taskkill /PID %PID% /F >NUL 2>&1
    del logs\scheduler.pid >NUL 2>&1
) else (
    echo No scheduler PID file found.
)

REM Kill Streamlit processes
echo Stopping Streamlit...
taskkill /IM streamlit.exe /F >NUL 2>&1

REM Kill uvicorn processes
echo Stopping API server...
taskkill /IM uvicorn.exe /F >NUL 2>&1

echo.
echo System stopped.
pause
