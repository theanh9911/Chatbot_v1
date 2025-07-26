@echo off
echo ========================================
echo AI Challenge HCM - Hệ thống tìm kiếm đa phương thức
echo ========================================

echo.
echo 🎯 Khởi động hệ thống...
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python không được cài đặt hoặc không có trong PATH
    pause
    exit /b 1
)

REM Kiểm tra Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js không được cài đặt hoặc không có trong PATH
    pause
    exit /b 1
)

echo ✅ Python và Node.js đã sẵn sàng
echo.

REM Tạo virtual environment nếu chưa có
if not exist "venv" (
    echo 🔧 Tạo virtual environment...
    python -m venv venv
)

REM Kích hoạt virtual environment
echo 🔧 Kích hoạt virtual environment...
call venv\Scripts\activate.bat

REM Cài đặt dependencies
echo 📦 Cài đặt Python dependencies...
pip install -r requirements.txt

REM Tạo dữ liệu mẫu và build indexes
echo 🎬 Tạo dữ liệu mẫu...
python create_sample_data.py

echo 🔨 Build indexes...
python src/build_index_fixed.py

echo.
echo 🚀 Khởi động hệ thống...
echo.

REM Khởi động backend API
echo 🔧 Khởi động Backend API (http://localhost:8001)...
start "Backend API" cmd /k "call venv\Scripts\activate.bat && python -m uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload"

REM Đợi backend khởi động
timeout /t 5 /nobreak >nul

REM Khởi động frontend
echo 🌐 Khởi động Frontend (http://localhost:3000)...
cd frontend
start "Frontend" cmd /k "npm install && npm start"
cd ..

echo.
echo 🎉 Hệ thống đã khởi động!
echo.
echo 📱 Frontend: http://localhost:3000
echo 🔧 Backend API: http://localhost:8001
echo 📚 API Documentation: http://localhost:8001/docs
echo.
echo ⏹️  Nhấn bất kỳ phím nào để mở trình duyệt...
pause >nul

REM Mở trình duyệt
start http://localhost:3000
start http://localhost:8001/docs

echo.
echo ✅ Đã mở trình duyệt!
echo.
echo 💡 Hướng dẫn sử dụng:
echo - Text Search: Nhập từ khóa để tìm kiếm trong các file text
echo - Image Search: Upload hình ảnh để tìm kiếm hình ảnh tương tự
echo - Kết quả sẽ hiển thị cả text và hình ảnh liên quan
echo.
echo 🛑 Để dừng hệ thống, đóng các cửa sổ terminal
pause 