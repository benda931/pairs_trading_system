@echo off
echo Starting Pairs Trading Alpha System...
echo.

REM Start Fair Value API server
start /B .venv\Scripts\python -m uvicorn root.api_server:app --port 8000 2>/dev/null

REM Start Streamlit dashboard
start /B .venv\Scripts\streamlit run root\dashboard.py --server.port 8501 --server.headless true

REM Wait for servers to start
timeout /t 10

REM Run initial alpha pipeline
echo Running alpha pipeline...
.venv\Scripts\python scriptsun_full_alpha.py --universe all --trials 30 --min-sharpe 0.3

REM Run auto-improvement cycle
echo Running auto-improvement...
.venv\Scripts\python scriptsun_auto_improve.py

echo.
echo System ready! Dashboard at http://localhost:8501
echo Alpha pipeline and auto-improvement complete.
pause
