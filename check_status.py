#!/usr/bin/env python3
"""
AI Challenge HCM - Quick Status Check
Chạy từ thư mục Chatbot: python check_status.py
"""

import os
import sys
import subprocess
import requests
import time

def check_file_exists(path, description):
    """Kiểm tra file có tồn tại không"""
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "✅" if exists else "❌"
    print(f"  {status} {description}: {path}")
    if exists:
        print(f"     Size: {size/1024/1024:.2f} MB")
    return exists, size

def check_python_packages():
    """Kiểm tra các package Python cần thiết"""
    print("\n📦 Checking Python packages...")
    required_packages = [
        'fastapi', 'uvicorn', 'faiss-cpu', 'sentence-transformers',
        'transformers', 'torch', 'pyvi', 'opencv-python', 'pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_data_files():
    """Kiểm tra các file dữ liệu"""
    print("\n📁 Checking data files...")
    
    data_files = [
        ("data/text/t1.txt", "Text data 1"),
        ("data/text/t2.txt", "Text data 2"),
        ("data/vid/vid.mp4", "Video file"),
        ("vietnamese-stopwords-dash.txt", "Vietnamese stopwords")
    ]
    
    all_exist = True
    for path, desc in data_files:
        exists, size = check_file_exists(path, desc)
        if not exists:
            all_exist = False
    
    return all_exist

def check_index_files():
    """Kiểm tra các file index"""
    print("\n🔍 Checking index files...")
    
    index_files = [
        ("data/faiss_text.bin", "Text index binary"),
        ("data/faiss_text.pkl", "Text index metadata"),
        ("data/faiss_image.bin", "Image index binary"),
        ("data/faiss_image.pkl", "Image index metadata")
    ]
    
    all_exist = True
    for path, desc in index_files:
        exists, size = check_file_exists(path, desc)
        if not exists:
            all_exist = False
    
    return all_exist

def check_api_status():
    """Kiểm tra trạng thái API"""
    print("\n🌐 Checking API status...")
    
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ API is running on port 8001")
            print(f"     Text searcher: {'✅' if data.get('text_searcher') else '❌'}")
            print(f"     Image searcher: {'✅' if data.get('image_searcher') else '❌'}")
            print(f"     Text index size: {data.get('text_index_size', 0)}")
            print(f"     Image index size: {data.get('image_index_size', 0)}")
            return True
        else:
            print(f"  ❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ❌ API is not running (Connection refused)")
        return False
    except Exception as e:
        print(f"  ❌ Error checking API: {e}")
        return False

def check_frontend_status():
    """Kiểm tra trạng thái frontend"""
    print("\n🎨 Checking frontend status...")
    
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("  ✅ Frontend is running on port 3000")
            return True
        else:
            print(f"  ❌ Frontend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ❌ Frontend is not running (Connection refused)")
        return False
    except Exception as e:
        print(f"  ❌ Error checking frontend: {e}")
        return False

def check_ports():
    """Kiểm tra các port đang được sử dụng"""
    print("\n🔌 Checking port usage...")
    
    try:
        # Kiểm tra port 8001 (API)
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        port_8001_used = any('8001' in line for line in lines)
        port_3000_used = any('3000' in line for line in lines)
        
        print(f"  {'✅' if port_8001_used else '❌'} Port 8001 (API): {'In use' if port_8001_used else 'Available'}")
        print(f"  {'✅' if port_3000_used else '❌'} Port 3000 (Frontend): {'In use' if port_3000_used else 'Available'}")
        
        return port_8001_used, port_3000_used
    except Exception as e:
        print(f"  ❌ Error checking ports: {e}")
        return False, False

def main():
    print("🚀 AI Challenge HCM - System Status Check")
    print("=" * 50)
    
    # Kiểm tra Python packages
    packages_ok = check_python_packages()
    
    # Kiểm tra data files
    data_ok = check_data_files()
    
    # Kiểm tra index files
    index_ok = check_index_files()
    
    # Kiểm tra ports
    api_port_used, frontend_port_used = check_ports()
    
    # Kiểm tra API status
    api_ok = check_api_status() if api_port_used else False
    
    # Kiểm tra frontend status
    frontend_ok = check_frontend_status() if frontend_port_used else False
    
    # Tổng kết
    print("\n" + "=" * 50)
    print("📊 SYSTEM STATUS SUMMARY:")
    print(f"  Python Packages: {'✅' if packages_ok else '❌'}")
    print(f"  Data Files: {'✅' if data_ok else '❌'}")
    print(f"  Index Files: {'✅' if index_ok else '❌'}")
    print(f"  API Running: {'✅' if api_ok else '❌'}")
    print(f"  Frontend Running: {'✅' if frontend_ok else '❌'}")
    
    print("\n🎯 RECOMMENDATIONS:")
    
    if not packages_ok:
        print("  ❌ Install missing packages: pip install -r requirements.txt")
    
    if not data_ok:
        print("  ❌ Missing data files. Check data/ directory structure.")
    
    if not index_ok:
        print("  ❌ Build indexes: cd src && python build_index_fixed.py")
    
    if not api_port_used:
        print("  ❌ Start API: cd src && uvicorn api:app --host 0.0.0.0 --port 8001")
    elif not api_ok:
        print("  ❌ API is not responding properly. Check logs.")
    
    if not frontend_port_used:
        print("  ❌ Start Frontend: cd frontend && npm start")
    elif not frontend_ok:
        print("  ❌ Frontend is not responding properly. Check logs.")
    
    if packages_ok and data_ok and index_ok and api_ok and frontend_ok:
        print("  🎉 All systems are running properly!")
        print("  🌐 Access: http://localhost:3000")
        print("  📚 API Docs: http://localhost:8001/docs")

if __name__ == "__main__":
    main() 