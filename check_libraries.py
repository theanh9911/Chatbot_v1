#!/usr/bin/env python3
"""
Script kiểm tra thư viện cho AI Challenge HCM
Kiểm tra xem tất cả dependencies đã được cài đặt chưa
"""

import sys
import subprocess
import importlib

def check_package(package_name, import_name=None):
    """Kiểm tra một package có được cài đặt không"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - CHƯA CÀI ĐẶT")
        return False

def main():
    print("🔍 KIỂM TRA THƯ VIỆN CHO AI CHALLENGE HCM")
    print("=" * 50)
    
    # Danh sách các thư viện cần kiểm tra
    packages = [
        # Xử lý tiếng Việt
        ("pyvi", "pyvi"),
        
        # Xử lý embedding văn bản, hình ảnh, audio
        ("sentence-transformers", "sentence_transformers"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("pillow", "PIL"),
        ("soundfile", "soundfile"),
        
        # Xử lý ảnh
        ("opencv-python", "cv2"),
        
        # Xử lý audio
        ("librosa", "librosa"),
        ("pydub", "pydub"),
        
        # FAISS cho vector search
        ("faiss-cpu", "faiss"),
        
        # API backend
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("python-multipart", "multipart"),
        
        # Khác
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
    ]
    
    missing_packages = []
    installed_count = 0
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            installed_count += 1
        else:
            missing_packages.append(package_name)
    
    print("\n" + "=" * 50)
    print(f"📊 KẾT QUẢ: {installed_count}/{len(packages)} thư viện đã cài đặt")
    
    if missing_packages:
        print(f"\n❌ THIẾU {len(missing_packages)} THƯ VIỆN:")
        for package in missing_packages:
            print(f"   - {package}")
        
        print(f"\n🔧 CÀI ĐẶT THIẾU:")
        print(f"pip install {' '.join(missing_packages)}")
        
        print(f"\n📝 HOẶC CÀI ĐẶT TẤT CẢ:")
        print(f"pip install -r requirements.txt")
    else:
        print("🎉 TẤT CẢ THƯ VIỆN ĐÃ SẴN SÀNG!")
        print("✅ Bạn có thể chạy hệ thống ngay bây giờ")
    
    # Kiểm tra Python version
    print(f"\n🐍 Python version: {sys.version}")
    
    # Kiểm tra pip
    try:
        import pip
        print(f"📦 Pip version: {pip.__version__}")
    except:
        print("❌ Không thể kiểm tra pip version")

if __name__ == "__main__":
    main() 