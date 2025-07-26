@echo off
echo ========================================
echo AI Challenge HCM - Multimodal Assistant
echo ========================================
echo.

echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

echo [2/4] Checking dependencies...
python -c "import fastapi, uvicorn, faiss, torch" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo [3/4] Building indexes...
python src/build_index_fixed.py
if errorlevel 1 (
    echo WARNING: Failed to build indexes. Continuing anyway...
)

echo [4/4] Starting server...
echo.
echo Backend will be available at: http://localhost:8001
echo Frontend will be available at: http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload

pause 