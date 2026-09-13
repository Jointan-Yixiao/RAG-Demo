@echo off
cd /d "%~dp0"
if not exist "RAG-Workbench.exe" (
  echo Launcher is missing. Please download the complete project.
  pause
  exit /b 1
)
start "" "%~dp0RAG-Workbench.exe"
