@echo off
echo ==========================================
echo AIRI - AI Readiness Index Application
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "airi_env" (
    echo Creating virtual environment...
    python -m venv airi_env
)

REM Activate virtual environment
echo Activating virtual environment...
call airi_env\Scripts\activate

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Run the application
echo.
echo ==========================================
echo Starting AIRI Application...
echo ==========================================
echo.
echo The app will open in your browser automatically
echo.
streamlit run airi_app.py

pause
