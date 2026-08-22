@echo off
title Intelligent Customer Intelligence Platform - Web Launcher
echo ====================================================================
echo   Intelligent Customer Intelligence Platform - Web Application Launcher
echo ====================================================================
echo.
echo Launching Web Application Services:
echo 1. Web Frontend & FastAPI Server: http://localhost:8000 (loads index.html)
echo 2. Streamlit Business Dashboard:  http://localhost:8501
echo.
echo Starting Web Server (FastAPI + index.html)...
start "Web Frontend (index.html)" "D:\APP HNU\Anaconda\python.exe" -m uvicorn src.serving.main:app --host 0.0.0.0 --port 8000

timeout /t 3 >nul

echo Starting Streamlit Business Dashboard...
start "Streamlit Dashboard" "D:\APP HNU\Anaconda\python.exe" -m streamlit run src/dashboard/dashboard.py --server.port 8501

echo.
echo ====================================================================
echo   Web Applications Launched Successfully!
echo   - Web App (index.html): http://localhost:8000
echo   - API Documentation:   http://localhost:8000/docs
echo   - Streamlit Dashboard: http://localhost:8501
echo ====================================================================
echo.
pause
