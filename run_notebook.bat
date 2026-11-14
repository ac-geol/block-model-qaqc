@echo off
rem ==========================================================================================
rem  Block model QA/QC - open the step-by-step notebook in your web browser (Windows).
rem  Keep this window open while you use the notebook; close it when you are finished.
rem ==========================================================================================
cd /d "%~dp0"
if not exist ".venv\Scripts\jupyter-lab.exe" (
  echo The tool is not set up yet. Double-click setup_windows.bat first.
  echo.
  pause
  exit /b 1
)
.venv\Scripts\jupyter-lab.exe notebooks\block_model_qaqc.ipynb
pause
