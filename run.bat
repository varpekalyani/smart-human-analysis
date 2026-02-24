@echo off
REM Always use the project venv so hand detection works (NumPy 1.x + OpenCV 4.9)
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Creating venv...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)
python app.py
pause
