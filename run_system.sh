#!/bin/bash

echo "========================================"
echo "AI Challenge HCM - Multimodal Assistant"
echo "========================================"
echo

echo "[1/4] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv venv"
    exit 1
fi

echo "[2/4] Checking dependencies..."
python -c "import fastapi, uvicorn, faiss, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "[3/4] Building indexes..."
python src/build_index_fixed.py
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to build indexes. Continuing anyway..."
fi

echo "[4/4] Starting server..."
echo
echo "Backend will be available at: http://localhost:8001"
echo "Frontend will be available at: http://localhost:3000"
echo
echo "Press Ctrl+C to stop the server"
echo

python -m uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload 