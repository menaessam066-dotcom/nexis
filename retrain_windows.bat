@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run start_windows.bat at least once first.
  pause
  exit /b 1
)
echo.
echo IMPORTANT: Keep 50+ SAME-PLATE photos in data\target and 100+ DIFFERENT-PLATE photos in data\other.
echo Delete old synthetic photos first if training on REAL pictures.
set /p REAL=Have you replaced BOTH folders with real photos? (Y/N): 
if /I "%REAL%"=="Y" (
    ".venv\Scripts\python.exe" train.py --real-data
) else (
    ".venv\Scripts\python.exe" train.py
)
if errorlevel 1 (
  echo TRAINING FAILED. Please read the error and README_STEP_BY_STEP.md.
) else (
  echo Model saved. Check models\training_report.json.
  echo Run evaluate.py on NEW images before connecting the Arduino.
)
pause
endlocal
