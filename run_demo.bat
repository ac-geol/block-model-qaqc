@echo off
rem ==========================================================================================
rem  Block model QA/QC - try the tool on a small made-up model (Windows). Double-click.
rem  The demo files are written to the "demo_project" folder and the report opens in your
rem  web browser.
rem ==========================================================================================
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The tool is not set up yet. Double-click setup_windows.bat first.
  echo.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m qaqc demo --open
echo.
pause
