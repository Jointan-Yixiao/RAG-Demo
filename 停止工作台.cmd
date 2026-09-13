@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 "scripts\_launch_workbench.py" --stop
if errorlevel 1 pause
