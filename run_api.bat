@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Follow README.md setup instructions first.
  pause
  exit /b 1
)
echo Starting FastAPI backend on http://127.0.0.1:8000...
".venv\Scripts\python.exe" -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
pause
