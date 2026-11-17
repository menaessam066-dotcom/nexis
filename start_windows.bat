@echo off
setlocal
cd /d "%~dp0"
title PlateGate AI Studio - Dashboard
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  set "PY=py -3"
) else (
  set "PY=python"
)
echo.
echo ===========================================================
echo   PLATEGATE AI STUDIO - 150 IMAGES + SERVO DASHBOARD
 echo ===========================================================
echo.
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Setting up Python virtual environment...
  %PY% -m venv .venv || goto :fail
)
echo [2/3] Installing dependencies (internet required first time)...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
echo [3/3] Starting local dashboard...
".venv\Scripts\python.exe" run.py || goto :fail
goto :done
:fail
echo.
echo ERROR. Review the message above, then consult README_STEP_BY_STEP.md.
pause
exit /b 1
:done
endlocal
