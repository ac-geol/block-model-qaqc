@echo off
rem ==========================================================================================
rem  Block model QA/QC - run the checks (Windows).
rem  Drag your settings file (.yaml) onto this file, or double-click it and paste the path.
rem  The report opens in your web browser when the run is finished.
rem ==========================================================================================
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The tool is not set up yet. Double-click setup_windows.bat first.
  echo.
  pause
  exit /b 1
)

set "SETTINGS=%~1"
if not defined SETTINGS (
  echo Drag your settings file ^(.yaml^) into this window, or paste its full path,
  set /p "SETTINGS=then press Enter: "
)
if not defined SETTINGS goto nosettings
set "SETTINGS=%SETTINGS:"=%"
if not defined SETTINGS goto nosettings

echo.
echo Checking the block model named in:
echo   %SETTINGS%
echo This can take a few minutes for a large model...
.venv\Scripts\python.exe -m qaqc run "%SETTINGS%" --open
echo.
pause
exit /b 0

:nosettings
echo No settings file was given.
echo.
pause
exit /b 1
