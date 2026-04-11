"use client";
import { useState } from "react";
import Navbar from "../components/Navbar";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const Icon = ({ d, size = 20, className = "" }: { d: string; size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d={d} />
  </svg>
);

const icons = {
  clapperboard: "M4 11h16M10 22L7 11M17 22l-3-11M9 3l1 8M15 3l1 8M4 11V5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v6M4 11v9a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-9",
  alert: "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01",
  check: "M20 6L9 17l-5-5",
  arrowRight: "M5 12h14M12 5l7 7-7 7"
};

type AnalysisResult = {
  label: "DEEPFAKE" | "REAL";
  confidence: number;
  processing_time?: number;
  frame_count?: number;
};

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();

      // Normalize backend response shape
      // Expected: { label: "DEEPFAKE"|"REAL", confidence: 0-100 }
      setResult({
        label: data.label?.toUpperCase() === "REAL" ? "REAL" : "DEEPFAKE",
        confidence: typeof data.confidence === "number"
          ? Math.round(data.confidence > 1 ? data.confidence : data.confidence * 100)
          : 0,
        processing_time: data.processing_time,
        frame_count: data.frame_count,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error occurred.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const isDeepfake = result?.label === "DEEPFAKE";

  return (
    <main style={{ minHeight: "100vh", background: "transparent", color: "#fff", overflow: "hidden" }}>
      <Navbar />

      <section style={{ position: "relative", zIndex: 10, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", paddingTop: "112px", paddingBottom: "80px", paddingLeft: "16px", paddingRight: "16px" }}>

        {/* Badge */}
        <div style={{
          display: "inline-flex", alignItems: "center", gap: "8px",
          background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "9999px", padding: "6px 18px", fontSize: "0.8rem", fontWeight: "600",
          color: "#94a3b8", marginBottom: "28px", backdropFilter: "blur(10px)",
        }}>
          <span style={{ width: "7px", height: "7px", background: "#38bdf8", borderRadius: "50%", boxShadow: "0 0 6px #38bdf8" }} />
          Upload &amp; Analyze
        </div>

        {/* Heading */}
        <h1 style={{ fontSize: "clamp(2.5rem, 5vw, 3.8rem)", fontWeight: "800", letterSpacing: "-0.04em", textAlign: "center", marginBottom: "16px", lineHeight: 1.1 }}>
          Upload a{" "}
          <span style={{ background: "linear-gradient(135deg, #38bdf8, #2563eb)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Video
          </span>
        </h1>
        <p style={{ color: "rgba(255,255,255,0.45)", marginBottom: "48px", fontSize: "1.05rem", textAlign: "center" }}>
          Drag &amp; drop or click to select a video file for AI analysis
        </p>

        {/* Drop Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById("fileInput")?.click()}
          style={{
            width: "100%", maxWidth: "640px",
            border: `2px dashed ${dragging ? "#38bdf8" : "rgba(255,255,255,0.1)"}`,
            borderRadius: "20px", padding: "64px 32px", textAlign: "center", cursor: "pointer",
            background: dragging ? "rgba(56,189,248,0.05)" : "rgba(255,255,255,0.02)",
            transition: "all 0.2s ease",
            boxShadow: dragging ? "0 0 30px rgba(56,189,248,0.15)" : "none",
          }}
        >
          <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mx-auto mb-6">
            <Icon d={icons.clapperboard} size={32} className="text-blue-400" />
          </div>
          <p style={{ color: "rgba(255,255,255,0.7)", fontSize: "1.1rem", fontWeight: "600", marginBottom: "8px" }}>
            Drop your video here
          </p>
          <p style={{ color: "rgba(255,255,255,0.3)", fontSize: "0.85rem" }}>MP4, MOV, AVI supported</p>
          <input id="fileInput" type="file" accept="video/*" style={{ display: "none" }} onChange={handleFileChange} />
        </div>

        {/* File Selected */}
        {file && (
          <div style={{
            marginTop: "20px", display: "inline-flex", alignItems: "center", gap: "8px",
            background: "rgba(56,189,248,0.08)", border: "1px solid rgba(56,189,248,0.25)",
            color: "#38bdf8", padding: "10px 22px", borderRadius: "9999px", fontSize: "0.9rem", fontWeight: "500",
          }}>
            <Icon d={icons.check} size={14} />
            <span style={{ maxWidth: "320px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</span>
            <span style={{ color: "rgba(56,189,248,0.6)", fontSize: "0.8rem" }}>({(file.size / (1024 * 1024)).toFixed(1)} MB)</span>
          </div>
        )}

        {/* Analyze Button */}
        <button
          id="analyze-btn"
          onClick={handleAnalyze}
          disabled={!file || loading}
          style={{
            marginTop: "28px",
            background: file && !loading ? "linear-gradient(135deg, #0ea5e9, #2563eb)" : "rgba(255,255,255,0.06)",
            color: file && !loading ? "#fff" : "rgba(255,255,255,0.3)",
            border: `1px solid ${file && !loading ? "rgba(56,189,248,0.3)" : "rgba(255,255,255,0.08)"}`,
            padding: "14px 48px", borderRadius: "9999px", fontSize: "1rem", fontWeight: "600",
            cursor: file && !loading ? "pointer" : "not-allowed",
            transition: "all 0.25s ease",
            boxShadow: file && !loading ? "0 0 24px rgba(14,165,233,0.35)" : "none",
            display: "flex", alignItems: "center", gap: "10px"
          }}
          onMouseEnter={(e) => {
            if (file && !loading) {
              e.currentTarget.style.boxShadow = "0 0 40px rgba(14,165,233,0.55)";
              e.currentTarget.style.transform = "translateY(-2px)";
            }
          }}
          onMouseLeave={(e) => {
            if (file && !loading) {
              e.currentTarget.style.boxShadow = "0 0 24px rgba(14,165,233,0.35)";
              e.currentTarget.style.transform = "translateY(0)";
            }
          }}
        >
          {loading ? "Analyzing..." : (
            <>
              Analyze Video
              <Icon d={icons.arrowRight} size={16} />
            </>
          )}
        </button>

        {/* Loading Spinner */}
        {loading && (
          <div style={{ marginTop: "32px", display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
            <div style={{
              width: "44px", height: "44px", border: "3px solid rgba(56,189,248,0.2)",
              borderTop: "3px solid #38bdf8", borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
            }} />
            <p style={{ color: "rgba(255,255,255,0.4)", fontSize: "0.9rem" }}>Sending video to Veritas AI...</p>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        )}

        {/* Error Card */}
        {error && (
          <div style={{
            marginTop: "32px", width: "100%", maxWidth: "640px",
            borderRadius: "16px", padding: "24px 28px",
            background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.25)",
            backdropFilter: "blur(10px)",
            display: "flex", alignItems: "flex-start", gap: "16px",
          }}>
            <div className="w-10 h-10 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center shrink-0">
              <Icon d={icons.alert} size={20} className="text-red-400" />
            </div>
            <div>
              <p style={{ fontWeight: "600", color: "#f87171", marginBottom: "4px" }}>Analysis Failed</p>
              <p style={{ color: "rgba(255,255,255,0.45)", fontSize: "0.875rem", lineHeight: "1.5" }}>{error}</p>
              <p style={{ color: "rgba(255,255,255,0.25)", fontSize: "0.8rem", marginTop: "8px" }}>
                Make sure the backend server is running at <code style={{ color: "#38bdf8", background: "rgba(56,189,248,0.07)", padding: "1px 6px", borderRadius: "4px" }}>{API_BASE_URL}</code>
              </p>
            </div>
          </div>
        )}

        {/* Result Card */}
        {result && (
          <div style={{
            marginTop: "40px", width: "100%", maxWidth: "640px",
            borderRadius: "20px", padding: "36px",
            background: isDeepfake ? "rgba(239,68,68,0.06)" : "rgba(34,197,94,0.06)",
            border: `1px solid ${isDeepfake ? "rgba(239,68,68,0.3)" : "rgba(34,197,94,0.3)"}`,
            backdropFilter: "blur(10px)",
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "24px" }}>
              <h3 style={{ fontSize: "1.4rem", fontWeight: "700", letterSpacing: "-0.02em" }}>Analysis Result</h3>
              <span style={{
                padding: "6px 18px", borderRadius: "9999px", fontSize: "0.8rem", fontWeight: "700", letterSpacing: "0.08em",
                background: isDeepfake ? "#ef4444" : "#22c55e", color: "#fff",
              }}>
                {result.label}
              </span>
            </div>



            {/* Extra stats from backend */}
            {(result.processing_time !== undefined || result.frame_count !== undefined) && (
              <div style={{ display: "flex", gap: "24px", marginTop: "20px", paddingTop: "20px", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                {result.frame_count !== undefined && (
                  <div>
                    <p style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.3)", marginBottom: "2px" }}>Frames Analyzed</p>
                    <p style={{ fontSize: "1rem", fontWeight: "600", color: "#fff" }}>{result.frame_count}</p>
                  </div>
                )}
                {result.processing_time !== undefined && (
                  <div>
                    <p style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.3)", marginBottom: "2px" }}>Processing Time</p>
                    <p style={{ fontSize: "1rem", fontWeight: "600", color: "#fff" }}>{result.processing_time.toFixed(2)}s</p>
                  </div>
                )}
              </div>
            )}

            <p style={{ color: "rgba(255,255,255,0.35)", fontSize: "0.85rem", marginTop: "16px" }}>
              {isDeepfake
                ? "⚠️ This video shows signs of AI manipulation."
                : "✅ This video appears to be authentic."}
            </p>
          </div>
        )}
      </section>
    </main>
  );
}