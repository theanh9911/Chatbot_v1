import os
import pickle
import numpy as np
from text_pipeline import preprocess, get_embedding
from image_pipeline import get_image_embedding
from faiss_pipeline import FaissMultiModalSearch

def debug_search():
    print("🔍 Debugging AI Challenge HCM Pipeline...")
    print("=" * 50)
    
    # 1. Kiểm tra text pipeline
    print("\n📝 [1/4] Testing Text Pipeline...")
    try:
        test_text = "quoc hoi viet nam"
        processed = preprocess(test_text)
        print(f"✅ Original: '{test_text}'")
        print(f"✅ Processed: '{processed}'")
        
        emb = get_embedding(processed)
        print(f"✅ Embedding shape: {emb.shape}")
        print(f"✅ Embedding sample: {emb[:5]}")
    except Exception as e:
        print(f"❌ Text pipeline error: {e}")
    
    # 2. Kiểm tra image pipeline
    print("\n🖼️ [2/4] Testing Image Pipeline...")
    try:
        # Kiểm tra nếu có video file
        vid_path = "../data/vid/vid.mp4"
        if os.path.exists(vid_path):
            print(f"✅ Video file found: {vid_path}")
            # Test với frame đầu tiên
            import cv2
            cap = cv2.VideoCapture(vid_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
                    cv2.imwrite(tmp_img.name, frame)
                    tmp_img.flush()
                    
                    emb = get_image_embedding(tmp_img.name)
                    print(f"✅ Image embedding shape: {emb.shape}")
                    print(f"✅ Image embedding sample: {emb[:5]}")
                    
                os.remove(tmp_img.name)
            else:
                print("❌ Could not read video frame")
        else:
            print(f"⚠️ Video file not found: {vid_path}")
    except Exception as e:
        print(f"❌ Image pipeline error: {e}")
    
    # 3. Kiểm tra text index
    print("\n📚 [3/4] Testing Text Index...")
    text_meta_path = "../data/faiss_text.pkl"
    text_index_path = "../data/faiss_text.bin"
    
    if os.path.exists(text_meta_path) and os.path.exists(text_index_path):
        try:
            # Load metadata
            with open(text_meta_path, 'rb') as f:
                meta = pickle.load(f)
            print(f"✅ Text metadata loaded: {len(meta)} entries")
            
            # Load index
            text_searcher = FaissMultiModalSearch(dim=768, index_path=text_index_path, meta_path=text_meta_path)
            text_searcher.load()
            print(f"✅ Text index loaded: {text_searcher.index.ntotal} vectors")
            
            # Test search
            test_query = "quoc hoi"
            q_processed = preprocess(test_query)
            q_emb = get_embedding(q_processed)
            results = text_searcher.search(q_emb, top_k=3)
            
            print(f"✅ Search test for '{test_query}':")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result}")
                
        except Exception as e:
            print(f"❌ Text index error: {e}")
    else:
        print(f"❌ Text index files not found")
        print(f"   Meta: {text_meta_path} - {'✅' if os.path.exists(text_meta_path) else '❌'}")
        print(f"   Index: {text_index_path} - {'✅' if os.path.exists(text_index_path) else '❌'}")
    
    # 4. Kiểm tra image index
    print("\n🖼️ [4/4] Testing Image Index...")
    img_meta_path = "../data/faiss_image.pkl"
    img_index_path = "../data/faiss_image.bin"
    
    if os.path.exists(img_meta_path) and os.path.exists(img_index_path):
        try:
            # Load metadata
            with open(img_meta_path, 'rb') as f:
                meta = pickle.load(f)
            print(f"✅ Image metadata loaded: {len(meta)} entries")
            
            # Load index
            image_searcher = FaissMultiModalSearch(dim=512, index_path=img_index_path, meta_path=img_meta_path)
            image_searcher.load()
            print(f"✅ Image index loaded: {image_searcher.index.ntotal} vectors")
            
            # Test search với random vector
            test_emb = np.random.rand(512).astype('float32')
            results = image_searcher.search(test_emb, top_k=2)
            
            print(f"✅ Image search test:")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result}")
                
        except Exception as e:
            print(f"❌ Image index error: {e}")
    else:
        print(f"❌ Image index files not found")
        print(f"   Meta: {img_meta_path} - {'✅' if os.path.exists(img_meta_path) else '❌'}")
        print(f"   Index: {img_index_path} - {'✅' if os.path.exists(img_index_path) else '❌'}")
    
    # 5. Tổng kết
    print("\n" + "=" * 50)
    print("📊 PIPELINE STATUS SUMMARY:")
    
    status = {
        "Text Pipeline": "✅" if 'emb' in locals() and emb.shape[0] == 768 else "❌",
        "Image Pipeline": "✅" if 'emb' in locals() and emb.shape[0] == 512 else "⚠️",
        "Text Index": "✅" if os.path.exists(text_meta_path) and os.path.exists(text_index_path) else "❌",
        "Image Index": "✅" if os.path.exists(img_meta_path) and os.path.exists(img_index_path) else "❌"
    }
    
    for component, status_icon in status.items():
        print(f"  {component}: {status_icon}")
    
    print("\n🎯 NEXT STEPS:")
    if status["Text Index"] == "❌" or status["Image Index"] == "❌":
        print("  1. Run: python build_index_fixed.py")
    print("  2. Run: uvicorn api:app --host 0.0.0.0 --port 8001")
    print("  3. Run: cd ../frontend && npm start")

if __name__ == "__main__":
    debug_search() 