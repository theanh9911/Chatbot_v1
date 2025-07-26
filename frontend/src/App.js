import React, { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleSearch = async () => {
    setLoading(true);
    setError("");
    setSuccess("");
    setResults([]);

    try {
      let allResults = [];
      
      // Thực hiện cả text search và image search nếu có input
      if (query.trim()) {
        console.log("🔍 Performing text search for:", query.trim());
        try {
          const textRes = await fetch("http://localhost:8001/search_text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query.trim(), top_k: 10 })
          });
          
          const textData = await textRes.json();
          console.log("📝 Text search response:", textData);
          
          if (textRes.ok && textData.matched_files) {
            // Xử lý cả text và image results từ cross-modal search
            const processedResults = textData.matched_files.map(item => ({
              ...item,
              // Đảm bảo type được set đúng
              type: item.type || (item.file && item.file.includes('.txt') ? 'text' : 'static_image'),
              source: item.source || 'text_search'
            }));
            allResults.push(...processedResults);
            console.log("✅ Added", processedResults.length, "results from cross-modal search");
          }
        } catch (err) {
          console.log("❌ Text search error:", err);
        }
      }

      if (file) {
        console.log("🖼️ Performing image search for:", file.name);
        // Kiểm tra kích thước file (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
          setError("File quá lớn (tối đa 10MB)");
          setLoading(false);
          return;
        }

        try {
          const formData = new FormData();
          formData.append("file", file);
          formData.append("top_k", "10");
          
          const imageRes = await fetch("http://localhost:8001/search_image", {
            method: "POST",
            body: formData
          });
          
          const imageData = await imageRes.json();
          console.log("🖼️ Image search response:", imageData);
          
          if (imageRes.ok && imageData.matched_files) {
            const imageResults = imageData.matched_files.map(item => ({
              ...item,
              type: item.type || 'image',
              source: item.source || 'image_search'
            }));
            allResults.push(...imageResults);
            console.log("✅ Added", imageResults.length, "image results");
          } else {
            console.log("❌ Image search failed:", imageRes.status, imageData);
          }
        } catch (err) {
          console.log("❌ Image search error:", err);
        }
      }

      console.log("📊 Total results before sorting:", allResults.length);
      console.log("📋 All results:", allResults);

      // Kiểm tra xem có input nào không
      if (!query.trim() && !file) {
        setError("Vui lòng nhập truy vấn hoặc chọn file ảnh");
        setLoading(false);
        return;
      }

      if (allResults.length === 0) {
        setError("Không tìm thấy kết quả nào. Vui lòng thử lại với từ khóa hoặc ảnh khác.");
        setLoading(false);
        return;
      }

      // Sắp xếp theo distance (càng nhỏ càng tốt)
      allResults.sort((a, b) => (a.distance || 0) - (b.distance || 0));
      console.log("📊 Results after sorting by distance:", allResults);
      
      // Lấy top 10 kết quả có distance thấp nhất
      const topResults = allResults.slice(0, 10);
      console.log("🏆 Top 10 results:", topResults);
      
      setResults(topResults);
      setSuccess(`Tìm thấy ${topResults.length} kết quả (sắp xếp theo distance - càng nhỏ càng tốt)`);
      
    } catch (err) {
      console.log("❌ General error:", err);
      setError(`Lỗi: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.type.startsWith("image/")) {
        setError("Vui lòng chọn file ảnh hợp lệ");
        return;
      }
      setFile(selectedFile);
      setError("");
      setSuccess("");
    }
  };

  const clearInput = () => {
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
        <h3 style={{ marginTop: 0 }}>🔍 Tìm kiếm đa phương thức:</h3>
        <p style={{ color: "#666", marginBottom: 15 }}>
          <strong>Text Search:</strong> Nhập từ khóa để tìm kiếm văn bản VÀ ảnh liên quan<br/>
          <strong>Image Search:</strong> Upload ảnh để tìm kiếm hình ảnh tương tự
        </p>

        {/* Text Input */}
        <div style={{ marginBottom: 15 }}>
          <input
            style={{ 
              width: "100%", 
              padding: "12px",
              fontSize: "16px",
              border: "1px solid #ddd",
              borderRadius: "5px",
              marginBottom: "10px"
            }}
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Nhập từ khóa tìm kiếm..."
            onKeyPress={e => e.key === "Enter" && handleSearch()}
          />
        </div>

        {/* File Upload */}
        <div style={{ marginBottom: 15 }}>
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

        {/* Action Buttons */}
        <div style={{ display: "flex", gap: "10px" }}>
          <button 
            onClick={handleSearch} 
            disabled={loading}
            style={{ 
              flex: 1,
              padding: "12px 24px",
              fontSize: "16px",
              backgroundColor: loading ? "#ccc" : "#007bff",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: loading ? "not-allowed" : "pointer"
            }}
          >
            {loading ? "⏳ Đang xử lý..." : "🔍 Tìm kiếm"}
          </button>
          
          <button 
            onClick={clearInput}
            style={{ 
              padding: "12px 16px",
              fontSize: "16px",
              backgroundColor: "#6c757d",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer"
            }}
          >
            🗑️ Xóa
          </button>
        </div>

        {/* Status Messages */}
        {error && (
          <div style={{ 
            marginTop: 15, 
            padding: "10px", 
            backgroundColor: "#f8d7da", 
            color: "#721c24", 
            borderRadius: "5px",
            border: "1px solid #f5c6cb"
          }}>
            ❌ {error}
          </div>
        )}

        {success && (
          <div style={{ 
            marginTop: 15, 
            padding: "10px", 
            backgroundColor: "#d4edda", 
            color: "#155724", 
            borderRadius: "5px",
            border: "1px solid #c3e6cb"
          }}>
            ✅ {success}
          </div>
        )}
      </div>

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
                    <div style={{ 
                      display: "flex", 
                      justifyContent: "space-between", 
                      alignItems: "center",
                      marginBottom: "5px"
                    }}>
                      <p style={{ margin: "0 0 5px 0" }}>
                        <strong>📄 File:</strong> {r.file || "N/A"}
                      </p>
                      <span style={{ 
                        padding: "2px 8px", 
                        borderRadius: "12px", 
                        fontSize: "12px",
                        backgroundColor: r.type === "text" ? "#e3f2fd" : r.type === "static_image" ? "#fff3e0" : "#f3e5f5",
                        color: r.type === "text" ? "#1976d2" : r.type === "static_image" ? "#f57c00" : "#7b1fa2"
                      }}>
                        {r.type === "text" ? "📝 Text" : r.type === "static_image" ? "🖼️ Image" : "🎬 Video"}
                      </span>
                    </div>
                    {r.line && (
                      <p style={{ margin: "0 0 5px 0" }}>
                        <strong>📝 Dòng:</strong> {r.line}
                      </p>
                    )}
                    {r.distance && (
                      <p style={{ margin: "0 0 5px 0" }}>
                        <strong>📏 Distance:</strong> {r.distance}
                        {r.source && (
                          <span style={{ 
                            marginLeft: "10px", 
                            fontSize: "12px", 
                            color: "#666",
                            fontStyle: "italic"
                          }}>
                            ({r.source === "cross_modal_search" ? "Cross-modal" : "Direct search"})
                          </span>
                        )}
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
        <p>📊 Dữ liệu: {file ? "20 video frames" : "41 text entries"}</p>
      </div>
    </div>
  );
}

export default App;
