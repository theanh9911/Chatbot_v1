#!/usr/bin/env python3
"""
Script cài đặt dependencies cho AI Challenge HCM
Chạy: python install_dependencies.py
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Chạy lệnh và hiển thị kết quả"""
    print(f"\n🔧 {description}...")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {description} thành công")
            return True
        else:
            print(f"   ❌ {description} thất bại")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return False

def main():
    print("🚀 CÀI ĐẶT DEPENDENCIES CHO AI CHALLENGE HCM")
    print("=" * 50)
    
    # Kiểm tra Python
    print(f"🐍 Python version: {sys.version}")
    
    # Kiểm tra pip
    try:
        import pip
        print(f"📦 Pip version: {pip.__version__}")
    except:
        print("❌ Không thể kiểm tra pip version")
    
    # Cài đặt dependencies
    success = True
    
    # Upgrade pip trước
    success &= run_command("python -m pip install --upgrade pip", "Upgrade pip")
    
    # Cài đặt core requirements
    success &= run_command("pip install -r requirements.txt", "Cài đặt core dependencies")
    
    # Kiểm tra cài đặt
    if success:
        print("\n" + "=" * 50)
        print("✅ CÀI ĐẶT HOÀN TẤT!")
        print("\n🎯 Bước tiếp theo:")
        print("1. Tạo dữ liệu mẫu: python create_sample_data.py")
        print("2. Build indexes: python src/build_index_fixed.py")
        print("3. Chạy backend: python -m uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload")
        print("4. Chạy frontend: cd frontend && npm install && npm start")
        
        # Chạy kiểm tra thư viện
        print("\n🔍 Kiểm tra thư viện...")
        os.system("python check_libraries.py")
    else:
        print("\n" + "=" * 50)
        print("❌ CÓ LỖI TRONG QUÁ TRÌNH CÀI ĐẶT!")
        print("\n💡 Gợi ý:")
        print("1. Kiểm tra kết nối internet")
        print("2. Thử cài từng package một")
        print("3. Kiểm tra Python version (cần 3.8+)")
        print("4. Thử: pip install --upgrade setuptools wheel")

if __name__ == "__main__":
    main() 