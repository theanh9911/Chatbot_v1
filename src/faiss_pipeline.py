import faiss
import numpy as np
import os
import pickle
import time

class FaissMultiModalSearch:
    def __init__(self, dim=512, index_path="data/faiss_index.bin", meta_path="data/faiss_meta.pkl", nlist=100, use_ivfpq=True, use_cosine=True):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.nlist = nlist
        self.use_ivfpq = use_ivfpq
        self.use_cosine = use_cosine
        self.trained = False
        self.meta = []
        
        # Tối ưu index type dựa trên kích thước dữ liệu
        if use_ivfpq:
            if use_cosine:
                # Sử dụng IndexFlatIP cho cosine similarity
                self.index = faiss.IndexFlatIP(dim)
            else:
                quantizer = faiss.IndexFlatL2(dim)
                # Tối ưu IVF+PQ parameters
                m = min(8, dim // 64)  # Số sub-vectors
                bits = 8  # Bits per sub-vector
                self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
            # Enable GPU nếu có
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("✅ Using GPU acceleration")
            except:
                print("⚠️ GPU not available, using CPU")
        else:
            # Sử dụng IndexFlatIP cho cosine hoặc IndexFlatL2 cho L2
            if use_cosine:
                self.index = faiss.IndexFlatIP(dim)
            else:
                self.index = faiss.IndexFlatL2(dim)

    def normalize_embedding(self, emb):
        """Normalize embedding để sử dụng cosine similarity"""
        if self.use_cosine:
            # L2 normalize cho cosine similarity
            norm = np.linalg.norm(emb)
            if norm > 0:
                return emb / norm
        return emb

    def add(self, emb, meta):
        emb = np.array(emb).reshape(1, -1).astype('float32')
        emb = self.normalize_embedding(emb)
        if self.use_ivfpq and not self.trained:
            raise RuntimeError("Index chưa được train! Hãy gọi train trước khi add.")
        self.index.add(emb)
        self.meta.append(meta)

    def add_batch(self, embs, metas):
        embs = np.array(embs).astype('float32')
        # Normalize tất cả embeddings
        if self.use_cosine:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Tránh division by zero
            embs = embs / norms
        if self.use_ivfpq and not self.trained:
            raise RuntimeError("Index chưa được train! Hãy gọi train trước khi add.")
        self.index.add(embs)
        self.meta.extend(metas)

    def train(self, embs):
        if self.use_ivfpq and not self.trained:
            embs = np.array(embs).astype('float32')
            # Normalize training data
            if self.use_cosine:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embs = embs / norms
            print(f"🔄 Training index with {len(embs)} samples...")
            start_time = time.time()
            self.index.train(embs)
            self.trained = True
            print(f"✅ Training completed in {time.time() - start_time:.2f}s")

    def search(self, emb, top_k=5):
        try:
            emb = np.array(emb).reshape(1, -1).astype('float32')
            emb = self.normalize_embedding(emb)
            start_time = time.time()
            
            # Tối ưu search parameters
            if self.use_ivfpq and not self.use_cosine:
                # Sử dụng nprobe để cân bằng tốc độ và độ chính xác
                nprobe = min(16, self.nlist // 4)
                self.index.nprobe = nprobe
            
            D, I = self.index.search(emb, top_k)
            search_time = time.time() - start_time
            
            results = []
            for i, idx in enumerate(I[0]):
                if idx < len(self.meta):
                    result = self.meta[idx].copy()
                    # Chuyển đổi distance dựa trên metric
                    if self.use_cosine:
                        # Cosine similarity: càng cao càng tốt, chuyển thành distance
                        similarity = float(D[0][i])
                        distance = 1.0 - similarity  # Chuyển thành distance (0-2)
                    else:
                        # L2 distance: càng thấp càng tốt
                        distance = float(D[0][i])
                    result['distance'] = distance
                    result['search_time_ms'] = round(search_time * 1000, 2)
                    results.append(result)
                else:
                    print(f"⚠️ Warning: Index {idx} out of range (meta length: {len(self.meta)})")
            
            return results
        except Exception as e:
            print(f"❌ Error in search: {e}")
            print(f"   Index size: {self.index.ntotal}")
            print(f"   Meta length: {len(self.meta)}")
            raise

    def save(self):
        try:
            # Lưu index
            faiss.write_index(self.index, self.index_path)
            
            # Lưu metadata với compression
            with open(self.meta_path, 'wb') as f:
                pickle.dump(self.meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"✅ Saved index to {self.index_path}")
            print(f"✅ Saved metadata to {self.meta_path}")
        except Exception as e:
            print(f"❌ Error saving: {e}")
            raise

    def load(self):
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                if self.use_ivfpq:
                    self.trained = self.index.is_trained
                print(f"✅ Loaded index from {self.index_path} (size: {self.index.ntotal})")
            else:
                print(f"❌ Index file not found: {self.index_path}")
                raise FileNotFoundError(f"Index file not found: {self.index_path}")
                
            if os.path.exists(self.meta_path):
                with open(self.meta_path, 'rb') as f:
                    self.meta = pickle.load(f)
                print(f"✅ Loaded metadata from {self.meta_path} (size: {len(self.meta)})")
            else:
                print(f"❌ Metadata file not found: {self.meta_path}")
                raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")
                
            # Kiểm tra tính nhất quán
            if self.index.ntotal != len(self.meta):
                print(f"⚠️ Warning: Index size ({self.index.ntotal}) != Meta size ({len(self.meta)})")
                
        except Exception as e:
            print(f"❌ Error loading: {e}")
            raise

    def get_stats(self):
        """Lấy thống kê về index"""
        return {
            "index_size": self.index.ntotal,
            "meta_size": len(self.meta),
            "dimension": self.dim,
            "index_type": "IVFPQ" if self.use_ivfpq else "FlatIP" if self.use_cosine else "FlatL2",
            "distance_metric": "cosine" if self.use_cosine else "L2",
            "trained": self.trained if self.use_ivfpq else True
        }

if __name__ == "__main__":
    # Ví dụ build index cho text với IVF+PQ
    from text_pipeline import preprocess, get_embedding as get_text_emb
    texts = ["Tôi yêu tiếng Việt và AI Challenge 2025.", "Trí tuệ nhân tạo đang thay đổi thế giới."]
    embs = []
    metas = []
    for t in texts:
        clean = preprocess(t)
        emb = get_text_emb(clean)
        embs.append(emb)
        metas.append(t)
    searcher = FaissMultiModalSearch(dim=768, index_path="data/faiss_text.bin", meta_path="data/faiss_text.pkl", nlist=10, use_ivfpq=True)
    searcher.train(embs)
    searcher.add_batch(embs, metas)
    searcher.save()
    # Truy vấn thử
    q = "AI và tiếng Việt"
    q_emb = get_text_emb(preprocess(q))
    print("Kết quả truy vấn:", searcher.search(q_emb, top_k=2)) 