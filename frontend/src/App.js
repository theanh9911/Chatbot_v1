import React, { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [file, setFile] = useState(null);
  const [modal, setModal] = useState("text"); // "text", "image"
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleSearch = async () => {
    setLoading(true);
    setError("");
    setSuccess("");
    setResults([]);

    try {
      if (modal === "text") {
        if (!query.trim()) {
          setError("Vui lòng nhập truy vấn");
          setLoading(false);
          return;
        }

        const res = await fetch("http://localhost:8001/search_text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query.trim(), top_k: 5 })
        });
        
        const data = await res.json();
        
        if (!res.ok) {
          throw new Error(data.detail || `HTTP error! status: ${res.status}`);
        }
        
        if (data.matched_files && data.matched_files.length > 0) {
          setResults(data.matched_files);
          setSuccess(`Tìm thấy ${data.matched_files.length} kết quả`);
        } else {
          setError("Không tìm thấy kết quả nào");
        }
      } else if (modal === "image") {
        if (!file) {
          setError("Vui lòng chọn file ảnh");
          setLoading(false);
          return;
        }

        // Kiểm tra kích thước file (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
          setError("File quá lớn (tối đa 10MB)");
          setLoading(false);
          return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("top_k", "5");
        
        const res = await fetch("http://localhost:8001/search_image", {
          method: "POST",
          body: formData
        });
        
        const data = await res.json();
        
        if (!res.ok) {
          throw new Error(data.detail || `HTTP error! status: ${res.status}`);
        }
        
        if (data.matched_files && data.matched_files.length > 0) {
          setResults(data.matched_files);
          setSuccess(`Tìm thấy ${data.matched_files.length} kết quả`);
        } else {
          setError("Không tìm thấy kết quả nào");
        }
      }
    } catch (err) {
      setError(`Lỗi: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (modal === "image" && !selectedFile.type.startsWith("image/")) {
        setError("Vui lòng chọn file ảnh hợp lệ");
        return;
      }
      setFile(selectedFile);
      setError("");
      setSuccess("");
    }
  };

  const handleModalChange = (newModal) => {
    setModal(newModal);
    setQuery("");
    setFile(null);
    setResults([]);
    setError("");
    setSuccess("");
  };

  return (
    <div style={{ 
      maxWidth: 800, 
      margin: "auto", 
      padding: 20,
      fontFamily: "Arial, sans-serif"
    }}>
      <h1 style={{ textAlign: "center", color: "#333", marginBottom: 30 }}>
        🤖 AI Challenge HCM - Trợ lý ảo đa phương tiện
      </h1>
      
      <div style={{ 
        backgroundColor: "#f5f5f5", 
        padding: 20, 
        borderRadius: 10,
        marginBottom: 20
      }}>
        <h3 style={{ marginTop: 0 }}>Chọn loại tìm kiếm:</h3>
        <div style={{ marginBottom: 15 }}>
          <label style={{ marginRight: 20, cursor: "pointer" }}>
            <input 
              type="radio" 
              checked={modal === "text"} 
              onChange={() => handleModalChange("text")} 
            /> 📝 Văn bản
          </label>
          <label style={{ cursor: "pointer" }}>
            <input 
              type="radio" 
              checked={modal === "image"} 
              onChange={() => handleModalChange("image")} 
            /> 🖼️ Hình ảnh
          </label>
        </div>

        {modal === "text" && (
          <div>
            <input
              style={{ 
                width: "100%", 
                padding: "10px",
                fontSize: "16px",
                border: "1px solid #ddd",
                borderRadius: "5px"
              }}
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Nhập truy vấn văn bản..."
              onKeyPress={e => e.key === "Enter" && handleSearch()}
            />
          </div>
        )}

        {modal === "image" && (
          <div>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleFileChange}
              style={{ 
                width: "100%", 
                padding: "10px",
                border: "1px solid #ddd",
                borderRadius: "5px"
              }}
            />
            {file && (
              <div style={{ marginTop: 10, fontSize: "14px", color: "#666" }}>
                <p>📁 Đã chọn: {file.name}</p>
                <p>📏 Kích thước: {(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            )}
          </div>
        )}

        <button 
          onClick={handleSearch} 
          disabled={loading}
          style={{ 
            marginTop: 15,
            padding: "12px 24px",
            fontSize: "16px",
            backgroundColor: loading ? "#ccc" : "#007bff",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: loading ? "not-allowed" : "pointer",
            width: "100%"
          }}
        >
          {loading ? "⏳ Đang xử lý..." : "🔍 Tìm kiếm"}
        </button>
      </div>

      {error && (
        <div style={{ 
          backgroundColor: "#f8d7da", 
          color: "#721c24", 
          padding: "10px", 
          borderRadius: "5px",
          marginBottom: 20
        }}>
          ❌ {error}
        </div>
      )}

      {success && (
        <div style={{ 
          backgroundColor: "#d4edda", 
          color: "#155724", 
          padding: "10px", 
          borderRadius: "5px",
          marginBottom: 20
        }}>
          ✅ {success}
        </div>
      )}

      {results.length > 0 && (
        <div style={{ 
          backgroundColor: "#d4edda", 
          padding: 20, 
          borderRadius: 10
        }}>
          <h3 style={{ marginTop: 0, color: "#155724" }}>
            📋 Kết quả tìm kiếm ({results.length} kết quả):
          </h3>
          <div style={{ maxHeight: "400px", overflowY: "auto" }}>
            {results.map((r, idx) => (
              <div key={idx} style={{ 
                marginBottom: 15, 
                padding: "10px", 
                backgroundColor: "white", 
                borderRadius: "5px",
                border: "1px solid #ddd"
              }}>
                {typeof r === "string" ? (
                  <p style={{ margin: 0 }}>{r}</p>
                ) : (
                  <div>
                    <p style={{ margin: "0 0 5px 0" }}>
                      <strong>📄 File:</strong> {r.file || "N/A"}
                    </p>
                    {r.line && (
                      <p style={{ margin: "0 0 5px 0" }}>
                        <strong>📝 Dòng:</strong> {r.line}
                      </p>
                    )}
                    {r.score && (
                      <p style={{ margin: "0 0 5px 0" }}>
                        <strong>⭐ Score:</strong> {r.score.toFixed(3)}
                      </p>
                    )}
                    <p style={{ margin: 0 }}>
                      <strong>📄 Nội dung:</strong> {r.description || r.text || "N/A"}
                    </p>
                    {r.image_base64 && (
                      <div style={{ marginTop: "10px" }}>
                        <img 
                          src={`data:image/jpeg;base64,${r.image_base64}`}
                          alt="Result image"
                          style={{ maxWidth: "200px", maxHeight: "150px", borderRadius: "5px" }}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ 
        textAlign: "center", 
        marginTop: 30, 
        color: "#666",
        fontSize: "14px"
      }}>
        <p>🚀 AI Challenge HCM - Hệ thống tìm kiếm thông minh đa phương tiện</p>
        <p>💡 Hỗ trợ: Văn bản tiếng Việt | Hình ảnh | Video frames</p>
        <p>🔧 API: http://localhost:8001</p>
        <p>📊 Dữ liệu: {modal === "text" ? "4,313 dòng văn bản" : "1 video frame"}</p>
      </div>
    </div>
  );
}

export default App;
