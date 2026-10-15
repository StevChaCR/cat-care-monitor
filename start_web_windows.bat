@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.13 -m venv .venv
    if errorlevel 1 (
        echo Install Python 3.13 for Windows, then run this file again.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" web_app.py --open
if errorlevel 1 pause
