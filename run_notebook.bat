@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Follow README.md setup instructions first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m jupyter lab
