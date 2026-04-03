@echo off
echo ============================================
echo   Pairs Trading Alpha System - Starting...
echo ============================================
echo.

REM Start Fair Value API server
echo [1/4] Starting API server (port 8000)...
start /B .venv\Scripts\python -m uvicorn root.api_server:app --port 8000 2>NUL

REM Start Streamlit dashboard
echo [2/4] Starting dashboard (port 8501)...
start /B .venv\Scripts\streamlit run root\dashboard.py --server.port 8501 --server.headless true 2>NUL

REM Wait for servers to start
timeout /t 5 /nobreak >NUL

REM Start scheduler daemon (daily pipeline + auto-improve + data refresh)
echo [3/4] Starting scheduler daemon...
start /B .venv\Scripts\python scripts\run_scheduler_daemon.py

REM Run initial alpha pipeline (one-time on first launch)
echo [4/4] Running initial alpha pipeline...
.venv\Scripts\python scripts\run_full_alpha.py --universe all --trials 30 --min-sharpe 0.3

echo.
echo ============================================
echo   System Ready!
echo   Dashboard:  http://localhost:8501
echo   API:        http://localhost:8000/docs
echo   Scheduler:  Running (see logs\scheduler_daemon.log)
echo   Stop:       Run stop_system.bat
echo ============================================
pause
