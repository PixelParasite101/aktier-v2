@echo off
setlocal enabledelayedexpansion

REM One-click runner for the standard pipeline on Windows (CMD)

set WORKSPACE=%~dp0
set PYTHON=%WORKSPACE%venv\Scripts\python.exe
if not exist "%PYTHON%" (
  echo venv python not found; using system python
  set PYTHON=python
)

echo [1/2] Fetch (preset standard)...
"%PYTHON%" "%WORKSPACE%fetch_history_pro.py" --preset standard || goto :error

echo [2/2] Features (preset standard)...
"%PYTHON%" "%WORKSPACE%compute_features.py" --preset standard || goto :error

echo Done.
exit /b 0

:error
echo Pipeline failed with error %errorlevel%
exit /b %errorlevel%
