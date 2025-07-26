 #!/usr/bin/env python3
"""
Script test nhanh cho hệ thống AI Challenge HCM
"""

import os
import sys
import requests
import json
import time

def test_backend():
    """Test backend API"""
    print("🔧 Testing Backend API...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend healthy - Text index: {data.get('text_index_size', 0)}, Image index: {data.get('image_index_size', 0)}")
            return True
        else:
            print(f"❌ Backend unhealthy - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not responding: {e}")
        return False

def test_text_search():
    """Test text search"""
    print("📝 Testing Text Search...")
    
    try:
        response = requests.post(
            "http://localhost:8001/search_text",
            json={"query": "AI", "top_k": 3},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("matched_files", [])
            print(f"✅ Text search OK - Found {len(results)} results")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result.get('file', 'Unknown')} (score: {result.get('score', 'N/A')})")
            return True
        else:
            print(f"❌ Text search failed - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Text search error: {e}")
        return False

def test_image_search():
    """Test image search"""
    print("🖼️  Testing Image Search...")
    
    # Kiểm tra file test có tồn tại không
    test_image = "data/vid/ai_ml.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    try:
        with open(test_image, "rb") as f:
            files = {"file": f}
            response = requests.post(
                "http://localhost:8001/search_image",
                files=files,
                data={"top_k": "3"},
                timeout=10
            )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("matched_files", [])
            print(f"✅ Image search OK - Found {len(results)} results")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result.get('file', 'Unknown')} (score: {result.get('score', 'N/A')})")
            return True
        else:
            print(f"❌ Image search failed - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Image search error: {e}")
        return False

def test_frontend():
    """Test frontend"""
    print("🌐 Testing Frontend...")
    
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend responding")
            return True
        else:
            print(f"❌ Frontend not responding - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend error: {e}")
        return False

def check_data_files():
    """Kiểm tra file dữ liệu"""
    print("📁 Checking data files...")
    
    required_files = [
        "data/text/t1.txt",
        "data/text/t2.txt", 
        "data/text/t3.txt",
        "data/vid/ai_ml.jpg",
        "data/vid/blockchain.jpg",
        "data/vid/iot_smartcity.jpg",
        "data/vid/data_science.jpg"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All data files present")
        return True

def check_index_files():
    """Kiểm tra file index"""
    print("🔍 Checking index files...")
    
    required_indexes = [
        "data/faiss_text.bin",
        "data/faiss_text.pkl", 
        "data/faiss_image.bin",
        "data/faiss_image.pkl"
    ]
    
    missing_indexes = []
    for file in required_indexes:
        if not os.path.exists(file):
            missing_indexes.append(file)
    
    if missing_indexes:
        print(f"❌ Missing indexes: {missing_indexes}")
        return False
    else:
        print("✅ All index files present")
        return True

def main():
    """Hàm chính"""
    print("🧪 AI Challenge HCM - Quick Test")
    print("=" * 40)
    
    tests = [
        ("Data Files", check_data_files),
        ("Index Files", check_index_files),
        ("Backend API", test_backend),
        ("Text Search", test_text_search),
        ("Image Search", test_image_search),
        ("Frontend", test_frontend)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            results.append((test_name, False))
    
    # Tổng kết
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        print("\n📱 Access points:")
        print("  Frontend: http://localhost:3000")
        print("  Backend API: http://localhost:8001")
        print("  API Docs: http://localhost:8001/docs")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        
        if not check_data_files():
            print("\n💡 To fix data issues, run:")
            print("  python create_sample_data.py")
            
        if not check_index_files():
            print("\n💡 To fix index issues, run:")
            print("  python src/build_index_fixed.py")

if __name__ == "__main__":
    main() 