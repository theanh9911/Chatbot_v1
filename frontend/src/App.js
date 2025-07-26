import React, { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [file, setFile] = useState(null);
  const [modal, setModal] = useState("text"); // "text", "image", "audio"

  const handleSearch = async () => {
    if (modal === "text") {
      const res = await fetch("http://localhost:8000/search_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 })
      });
      const data = await res.json();
      setResults(data.results);
    } else if (modal === "image" || modal === "audio") {
      const formData = new FormData();
      formData.append("file", file);
      const url = modal === "image" ? "http://localhost:8000/search_image" : "http://localhost:8000/search_audio";
      const res = await fetch(url, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      setResults(data.results);
    }
  };

  return (
    <div style={{ maxWidth: 500, margin: "auto", padding: 20 }}>
      <h2>Chatbot đa phương tiện AI Challenge</h2>
      <div>
        <label>
          <input type="radio" checked={modal === "text"} onChange={() => setModal("text")} /> Văn bản
        </label>
        <label style={{ marginLeft: 10 }}>
          <input type="radio" checked={modal === "image"} onChange={() => setModal("image")} /> Ảnh
        </label>
        <label style={{ marginLeft: 10 }}>
          <input type="radio" checked={modal === "audio"} onChange={() => setModal("audio")} /> Âm thanh
        </label>
      </div>
      {modal === "text" && (
        <div>
          <input
            style={{ width: "80%" }}
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Nhập truy vấn..."
          />
        </div>
      )}
      {(modal === "image" || modal === "audio") && (
        <div>
          <input type="file" accept={modal === "image" ? 'image/*' : 'audio/*'} onChange={e => setFile(e.target.files[0])} />
        </div>
      )}
      <button onClick={handleSearch} style={{ marginTop: 10 }}>Gửi truy vấn</button>
      <div style={{ marginTop: 20 }}>
        <h4>Kết quả:</h4>
        <ul>
          {results.map((r, idx) => (
            <li key={idx}>{typeof r === "string" ? r : JSON.stringify(r)}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;
