@echo off
rem ==========================================================================================
rem  Block model QA/QC - one-time setup (Windows). Double-click this file.
rem  It makes a private copy of Python for this tool in the ".venv" folder and installs the
rem  packages the tool needs. Nothing else on the PC is changed. Takes a few minutes.
rem ==========================================================================================
cd /d "%~dp0"
echo.
echo Setting up the block model QA/QC tool in:
echo   %CD%
echo.

where uv >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
  ) else (
    echo The installer "uv" was not found on this PC.
    echo Install it first - see GETTING_STARTED.md, step 2 - then double-click this file again.
    echo.
    pause
    exit /b 1
  )
)

echo Step 1 of 2: making the private Python copy (.venv)...
uv venv --python 3.11 .venv
if errorlevel 1 goto failed

echo.
echo Step 2 of 2: installing the packages (needs internet, a few minutes)...
uv pip install --python .venv\Scripts\python.exe -e ".[notebook]"
if errorlevel 1 goto failed

echo.
echo ==========================================================================================
echo  Setup finished.
echo  To try the tool on a made-up model: double-click run_demo.bat
echo  To check your own model: drag your settings file (.yaml) onto run_qaqc.bat
echo ==========================================================================================
echo.
pause
exit /b 0

:failed
echo.
echo Setup did not finish. Scroll up to see the message. Common causes: no internet
echo connection, or the company network blocks downloads (ask IT to allow python.org and
echo pypi.org). Copy the messages above if you need help.
echo.
pause
exit /b 1
