@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please install the project Python environment first. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 "scripts\_launch_workbench.py"
if errorlevel 1 pause
