// src/ReviewVideo.js
import React from "react";
import { useLocation, useNavigate } from "react-router-dom";

export default function ReviewVideo() {
  const location  = useLocation();
  const navigate  = useNavigate();

  const videoURL  = location.state?.videoURL;
  const videoFile = location.state?.videoFile;

  if (!videoURL) {
    return (
      <div style={{
        minHeight: "100vh", background: "#0d0d1a",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center", gap: 16,
      }}>
        <p style={{ color: "#a78bfa", fontSize: 18 }}>
          No video found. Please upload or record a video first.
        </p>
        <button onClick={() => navigate("/")}
          style={{ background:"#3b2f8f", color:"#fff", border:"none",
                   borderRadius:8, padding:"11px 26px", fontWeight:700,
                   fontSize:14, cursor:"pointer" }}>
          ← Go Back
        </button>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0d0d1a 0%, #12122a 100%)",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      padding: "32px 20px",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
    }}>
      <div style={{
        background: "#16162a", border: "2px solid #3b2f8f",
        borderRadius: 18, padding: "32px 36px",
        maxWidth: 680, width: "100%",
        boxShadow: "0 8px 40px #3b2f8f44",
      }}>
        <h2 style={{
          color: "#a78bfa",
          fontFamily: "'Playfair Display', Georgia, serif",
          fontSize: 28, fontWeight: 800, margin: "0 0 6px",
        }}>
          Video Review
        </h2>
        <p style={{ color: "#888", margin: "0 0 24px", fontSize: 14 }}>
          Check your room video before running the analysis.
        </p>

        {/* Video player */}
        <div style={{
          borderRadius: 12, overflow: "hidden",
          border: "2px solid #3b2f8f55",
          background: "#000", marginBottom: 28,
        }}>
          <video width="100%" controls style={{ display:"block", maxHeight:400 }}>
            <source src={videoURL} type="video/mp4" />
            <source src={videoURL} type="video/webm" />
            Your browser does not support the video tag.
          </video>
        </div>

        {/* File info */}
        {videoFile && (
          <div style={{
            background: "#1e1e3a", borderRadius: 8,
            padding: "10px 16px", marginBottom: 24,
            display: "flex", gap: 24, flexWrap: "wrap",
          }}>
            <span style={{ color:"#888", fontSize:13 }}>
              📄 <strong style={{ color:"#c4b5fd" }}>{videoFile.name}</strong>
            </span>
            <span style={{ color:"#888", fontSize:13 }}>
              📦 {(videoFile.size / (1024 * 1024)).toFixed(2)} MB
            </span>
          </div>
        )}

        {/* Buttons */}
        <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
          <button onClick={() => navigate("/")}
            style={{ background:"transparent", color:"#a78bfa",
                     border:"2px solid #a78bfa", borderRadius:8,
                     padding:"11px 24px", fontWeight:700,
                     fontSize:14, cursor:"pointer" }}>
            ← Re-upload Video
          </button>

          <button
            onClick={() => navigate("/select-room", {
              state: { videoURL, videoFile },
            })}
            style={{ background:"#3b2f8f", color:"#fff", border:"none",
                     borderRadius:8, padding:"11px 28px", fontWeight:700,
                     fontSize:14, cursor:"pointer",
                     boxShadow:"0 4px 14px #3b2f8f66" }}>
            Continue → Select Room
          </button>
        </div>
      </div>
    </div>
  );
}