// src/roomsetup.js
// Requirements:
//   - Step 0: Upload or capture video
//   - Step 1: Preview video
//   - Step 2: Select room type
//   - Step 3: Select essential furniture → click "Finalize" → shows loading overlay
//   - Step 4: Results — recommendations table, "View 3D Room" button opens View3DPage
//             in a new full-screen tab with 3 tabs: Current | Optimised | Comparison

import React, { useState, useRef } from "react";
import axios from "axios";
import "./main.css";

import bedroomImg   from "./assets/bedroom.jpg";
import livingImg    from "./assets/livingroom.jpg";
import kitchenImg   from "./assets/kitchen.jpg";
import bathroomImg  from "./assets/bathroom.jpg";
import studyroomImg from "./assets/studyroom.jpg";
import room1Img     from "./assets/room1.jpg";

const BASE = "http://127.0.0.1:8000";

const rooms = [
  { name: "Bedroom",     img: bedroomImg   },
  { name: "Living Room", img: livingImg    },
  { name: "Kitchen",     img: kitchenImg   },
  { name: "Bathroom",    img: bathroomImg  },
  { name: "Study Room",  img: studyroomImg },
];



// ─────────────────────────────────────────────────────────────────────────────
// Loading overlay — full-screen, shown while pipeline runs
// ─────────────────────────────────────────────────────────────────────────────
function LoadingOverlay() {
  const [dots,   setDots]   = React.useState(".");

  React.useEffect(() => {
    const d = setInterval(() => setDots(p => p.length >= 3 ? "." : p + "."), 500);
    return () => { clearInterval(d); };
  }, []);

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 9999,
      background: "rgba(10,8,30,0.97)",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      gap: "28px", fontFamily: "'Segoe UI', sans-serif",
    }}>
      {/* Spinner */}
      <div style={{
        width: "72px", height: "72px", borderRadius: "50%",
        border: "5px solid #3b2f8f",
        borderTop: "5px solid #7c6fe0",
        animation: "spin 1s linear infinite",
      }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>

      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "24px", fontWeight: "800",
                      color: "#e2e8f0" }}>
          Analysing video{dots}
        </div>
      </div>

      <div style={{ color: "#444", fontSize: "12px", maxWidth: "320px",
                    textAlign: "center", lineHeight: "1.7" }}>
        This may take 1–3 minutes depending on video length.<br />
        Please don't close this tab.
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Table styles
// ─────────────────────────────────────────────────────────────────────────────
const ts = {
  table:    { width: "100%", borderCollapse: "collapse", borderRadius: "12px",
              overflow: "hidden", boxShadow: "0 4px 16px rgba(44,31,107,0.18)",
              marginTop: "12px" },
  thead:    { backgroundColor: "#3b2f8f" },
  th:       { padding: "12px 16px", color: "#fff", textAlign: "left",
              fontSize: "14px", fontWeight: "700" },
  rowEven:  { backgroundColor: "#ede9f8" },
  rowOdd:   { backgroundColor: "#f7f5fd" },
  rowHover: { backgroundColor: "#d6ccf5" },
  td:       { padding: "10px 16px", color: "#2c1f6b", fontSize: "13px",
              borderBottom: "1px solid #cfc8f0" },
  tdYes:    { padding: "10px 16px", fontSize: "13px", fontWeight: "700",
              color: "#b71c1c", borderBottom: "1px solid #cfc8f0" },
  tdNo:     { padding: "10px 16px", fontSize: "13px", fontWeight: "700",
              color: "#2e7d32", borderBottom: "1px solid #cfc8f0" },
};

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export default function RoomSetup() {
  const [step,              setStep]              = useState(0);
  const [videoFile,         setVideoFile]         = useState(null);
  const [videoURL,          setVideoURL]          = useState(null);
  const [roomType,          setRoomType]          = useState("");
  const [furniture,         setFurniture]         = useState({});
  const [detectedFurniture, setDetectedFurniture] = useState({});
  const [analysisResult,    setAnalysisResult]    = useState(null);
  const [loading,           setLoading]           = useState(false);
  const [isRecording,       setIsRecording]       = useState(false);
  const [hoveredRow,        setHoveredRow]        = useState(null);
  const [showThankYou,      setShowThankYou]      = useState(false);

  const videoRef         = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef        = useRef([]);

  // ── API call ──────────────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!videoFile) return alert("Please upload or record a video first!");
    if (!roomType)  return alert("Please select a room type!");

    const formData = new FormData();
    formData.append("video",          videoFile);
    formData.append("room_type",      roomType);
    formData.append("furniture_data", JSON.stringify(furniture));

    setLoading(true);
    try {
      const res = await axios.post(
        `${process.env.REACT_APP_API_URL}/run-full-pipeline/`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      const fullResult = res.data.result;
      console.log("FULL RESULT:", fullResult);
      setDetectedFurniture(fullResult.detected_objects || {});
      setAnalysisResult(fullResult);
      setVideoURL(`${BASE}${res.data.video_url}`);
      setStep(4);
    } catch (err) {
      alert(err.response?.data?.error || "Pipeline failed.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // ── Camera recording ──────────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      videoRef.current.srcObject = stream;
      mediaRecorderRef.current   = new MediaRecorder(stream);
      chunksRef.current          = [];
      mediaRecorderRef.current.ondataavailable = e => chunksRef.current.push(e.data);
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        const file = new File([blob], "room_video.webm", { type: "video/webm" });
        setVideoFile(file);
        setVideoURL(URL.createObjectURL(file));
        stream.getTracks().forEach(t => t.stop());
        setIsRecording(false);
        setStep(1);
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch {
      alert("Camera access denied!");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) mediaRecorderRef.current.stop();
  };

  // ── Furniture number input ────────────────────────────────────────────────
  const renderInput = (label, key) => (
    <label style={{ display: "block", margin: "10px 0" }}>
      {label}:
      <input
        type="number" min="0"
        value={furniture[key] || 0}
        onChange={e => setFurniture({ ...furniture, [key]: parseInt(e.target.value) || 0 })}
        style={{ marginLeft: "10px", padding: "5px", width: "60px" }}
      />
    </label>
  );


  // ─────────────────────────────────────────────────────────────────────────
  // STEP 0 — Upload / Capture
  // ─────────────────────────────────────────────────────────────────────────
  if (step === 0) return (
    <>
      {loading && <LoadingOverlay />}
      <div className="inner-box">
        <h2>Upload or Capture Video</h2>
        <div className="video-box upload-box">
          <h3>Upload Video</h3>
          <input
            type="file" accept="video/*" disabled={loading}
            onChange={e => {
              const f = e.target.files[0];
              setVideoFile(f);
              setVideoURL(URL.createObjectURL(f));
              setStep(1);
            }}
          />
        </div>
        <div className="video-box capture-box">
          <h3>Capture Video</h3>
          <div className="capture-container">
            <video ref={videoRef} autoPlay muted className="capture-video" />
            {!isRecording
              ? <button className="btn start-btn" onClick={startRecording} disabled={loading}>
                  🎥 Start Recording
                </button>
              : <button className="btn stop-btn" onClick={stopRecording}>
                  ⏹ Stop Recording
                </button>
            }
          </div>
        </div>
      </div>
    </>
  );

  // ─────────────────────────────────────────────────────────────────────────
  // STEP 1 — Preview
  // ─────────────────────────────────────────────────────────────────────────
  if (step === 1 && videoURL) return (
    <div className="inner-box">
      <h2>Video Preview</h2>
      <video src={videoURL} controls width="600" />
      <br />
      <button onClick={() => setStep(2)}>Next</button>
    </div>
  );

  // ─────────────────────────────────────────────────────────────────────────
  // STEP 2 — Room selection
  // ─────────────────────────────────────────────────────────────────────────
  if (step === 2) return (
    <div className="page">
      <div className="background" style={{ backgroundImage: `url(${room1Img})` }} />
      <div className="content">
        {rooms.map((room, i) => (
          <div key={i} className={`corner-room corner-${i + 1}`}>
            <img src={room.img} alt={room.name} />
            <p>{room.name}</p>
            <button className="btn"
              onClick={() => { setRoomType(room.name); setStep(3); }}>
              Continue
            </button>
          </div>
        ))}
      </div>
    </div>
  );

  // ─────────────────────────────────────────────────────────────────────────
  // STEP 3 — Furniture selection + Finalize
  // ─────────────────────────────────────────────────────────────────────────
  if (step === 3) {
    const bgImage = rooms.find(r => r.name === roomType)?.img;
    return (
      <>
        {loading && <LoadingOverlay />}
        <div className="furniture-page" style={{ backgroundImage: `url(${bgImage})` }}>
          <div className="furniture-overlay">
            <h2>{roomType} – Essential Furniture</h2>

            {Object.keys(detectedFurniture).length > 0 && (
              <div>
                <h3>Detected Furniture:</h3>
                <pre>{JSON.stringify(detectedFurniture, null, 2)}</pre>
              </div>
            )}

            <div className="furniture-group">
              {roomType === "Bedroom" && (<>
                {renderInput("Beds",     "beds")}
                {renderInput("Wardrobe", "wardrobe")}
                {renderInput("Tables",   "tables")}
                {renderInput("TV",       "tv")}
                {renderInput("Dustbin",  "dustbin")}
                {renderInput("Chair",    "chair")}
                {renderInput("Sofa",     "sofa")}
              </>)}
              {roomType === "Living Room" && (<>
                {renderInput("Sofas",        "sofas")}
                {renderInput("TV",           "tv")}
                {renderInput("Dustbin",      "dustbin")}
                {renderInput("Painting",     "painting")}
                {renderInput("Plants",       "plants")}
                {renderInput("Center Table", "centerTable")}
              </>)}
              {roomType === "Kitchen" && (<>
                {renderInput("Refrigerator",  "refrigerator")}
                {renderInput("Microwave",     "microwave")}
                {renderInput("Sink",          "sink")}
                {renderInput("Stove",         "stove")}
                {renderInput("Dining Table",  "diningTable")}
                {renderInput("Coffee Machine","coffeeMachine")}
                {renderInput("Dustbin",       "dustbin")}
              </>)}
              {roomType === "Bathroom" && (<>
                {renderInput("Toilet",    "toilet")}
                {renderInput("Washbasin", "washbasin")}
                {renderInput("Mirror",    "mirror")}
                {renderInput("Waste Bin", "wasteBin")}
                {renderInput("Bathtub",   "bathtub")}
                {renderInput("Tap",       "tap")}
              </>)}
              {roomType === "Study Room" && (<>
                {renderInput("Study Table", "studyTable")}
                {renderInput("Chair",       "chair")}
                {renderInput("Table Lamp",  "tableLamp")}
                {renderInput("Computer",    "computer")}
                {renderInput("Book Shelf",  "bookShelf")}
              </>)}
            </div>

            <button
              className="btn"
              style={{ marginTop: "20px" }}
              onClick={handleUpload}
            >
              Finalize &amp; Run Genetic + Vastu
            </button>
          </div>
        </div>
      </>
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // ── Thank You Page ──────────────────────────────────────────────────────────
  if (showThankYou) {
    // Auto-redirect to home after 4 seconds
    setTimeout(() => {
      setShowThankYou(false);
      setStep(0);
      setAnalysisResult(null);
      setVideoFile(null);
      setVideoURL(null);
      setRoomType("");
      setFurniture({});
      setDetectedFurniture({});
    }, 3000);

    return (
      <div style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg,#0d0d1a,#1a1a3a)",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        fontFamily: "'Segoe UI', sans-serif",
        gap: "24px", padding: "40px",
      }}>
        {/* Animated checkmark */}
        <div style={{
          width: "90px", height: "90px", borderRadius: "50%",
          background: "linear-gradient(135deg,#2e7d32,#00ffd0)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "40px",
          boxShadow: "0 0 40px rgba(0,255,208,0.4)",
          animation: "popIn 0.5s ease-out",
        }}>
          ✓
        </div>
        <style>{`
          @keyframes popIn {
            0%   { transform: scale(0); opacity: 0; }
            70%  { transform: scale(1.15); opacity: 1; }
            100% { transform: scale(1); }
          }
          @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
          }
        `}</style>

        <div style={{ textAlign: "center", animation: "fadeUp 0.6s ease-out 0.3s both" }}>
          <h1 style={{
            color: "#00ffd0", fontSize: "36px", fontWeight: "900",
            margin: "0 0 10px",
          }}>
            Thank You!
          </h1>
          <p style={{ color: "#aac4ff", fontSize: "16px", margin: "0 0 6px" }}>
            Your Vastu analysis is complete.
          </p>
          <p style={{ color: "#555", fontSize: "13px" }}>
            Redirecting to home in 3 seconds...
          </p>
        </div>

        {/* Progress bar */}
        <div style={{
          width: "260px", height: "4px",
          background: "#1e1e3a", borderRadius: "2px",
          overflow: "hidden",
          animation: "fadeUp 0.6s ease-out 0.5s both",
        }}>
          <div style={{
            height: "100%",
            background: "linear-gradient(90deg,#3b2f8f,#00ffd0)",
            borderRadius: "2px",
            animation: "progress 3s linear forwards",
          }} />
          <style>{`
            @keyframes progress {
              from { width: 0%; }
              to   { width: 100%; }
            }
          `}</style>
        </div>

        {/* Go Now button */}
        <button
          onClick={() => {
            setShowThankYou(false);
            setStep(0);
            setAnalysisResult(null);
            setVideoFile(null);
            setVideoURL(null);
            setRoomType("");
            setFurniture({});
            setDetectedFurniture({});
          }}
          style={{
            background: "transparent",
            color: "#7c6fe0",
            border: "1px solid #3b2f8f",
            borderRadius: "8px",
            padding: "10px 28px",
            fontWeight: "700",
            fontSize: "14px",
            cursor: "pointer",
            animation: "fadeUp 0.6s ease-out 0.7s both",
          }}
        >
          ← Go to Home Now
        </button>
      </div>
    );
  }


  // STEP 4 — Results
  if (step === 4 && analysisResult) {
    // Save result so View3DPage can read it from localStorage
    localStorage.setItem("vastuResult", JSON.stringify(analysisResult));

    const cmp = analysisResult["3d_models"]?.comparison || {};

    return (
      <div className="inner-box">
        <h2>Vastu Analysis Result</h2>
        <p style={{ color:"#888", fontSize:"13px", marginBottom:"8px" }}>
          Room: <b style={{ color:"#aac4ff" }}>{analysisResult.video_info?.room_type || roomType}</b>
          &nbsp;·&nbsp;
          Final Compliance: <b style={{ color:"#2e7d32" }}>{analysisResult.final_compliance_pct}%</b>
        </p>

        {/* Room dimensions */}
        <div style={{ display:"flex", gap:"20px", flexWrap:"wrap",
                      fontSize:"13px", color:"#555", marginBottom:"24px" }}>
          <span>📏 Width: <b>{analysisResult.room_dimensions?.width_m?.toFixed(2)} m</b></span>
          <span>📏 Depth: <b>{analysisResult.room_dimensions?.depth_m?.toFixed(2)} m</b></span>
          <span>📏 Height: <b>{analysisResult.room_dimensions?.height_m} m</b></span>
        </div>

        {/* ── VIEW 3D BUTTON ── opens full page with both layouts + comparison ── */}
        <div style={{ marginBottom:"28px" }}>
          <button
            onClick={() => {
              const htmlUrl = analysisResult["3d_models"]?.interactive_html;
              if (htmlUrl) {
                window.open(`${BASE}${htmlUrl}`, "_blank", "noopener,noreferrer");
              } else {
                alert("3D viewer not ready yet. Please wait a moment and try again.");
              }
            }}
            style={{
              background: "linear-gradient(135deg,#3b2f8f,#1565c0)",
              color:"#fff", border:"2px solid #7c6fe0",
              borderRadius:"12px", padding:"15px 40px",
              fontWeight:"800", fontSize:"17px", cursor:"pointer",
              boxShadow:"0 4px 20px rgba(59,47,143,0.45)",
              display:"inline-flex", alignItems:"center", gap:"12px",
            }}
          >
            🏠 View 3D Room &amp; Comparison
          </button>
          <p style={{ color:"#666", fontSize:"11px", marginTop:"8px" }}>
            ↗ Opens in a new browser tab — Current Layout | Vastu-Optimised | Comparison stats.
            Both 3D scenes are fully rotatable by dragging. No install needed.
          </p>

          {/* Quick compliance summary */}
          {cmp.compliance_before != null && (
            <div style={{
              display:"inline-flex", gap:"16px", marginTop:"12px",
              background:"#12122a", border:"1px solid #3b2f8f",
              borderRadius:"10px", padding:"10px 20px",
              fontSize:"13px", flexWrap:"wrap", alignItems:"center",
            }}>
              <span style={{ color:"#e65100" }}>Before: <b>{cmp.compliance_before}%</b></span>
              <span style={{ color:"#555" }}>→</span>
              <span style={{ color:"#2e7d32" }}>After: <b>{cmp.compliance_after}%</b></span>
              <span style={{ fontWeight:"800",
                color: cmp.improvement >= 0 ? "#00ffd0" : "#ff4444" }}>
                {cmp.improvement >= 0 ? "↑ +" : "↓ "}{cmp.improvement}%
              </span>
            </div>
          )}
        </div>

        {/* ── Download GLB ── */}
        <div style={{ marginBottom:"20px", display:"flex", gap:"12px", flexWrap:"wrap" }}>
          {analysisResult["3d_models"]?.current_glb && (
            <a
              href={`${BASE}${analysisResult["3d_models"].current_glb}`}
              download="current_layout.glb"
              style={{
                display: "inline-flex", alignItems: "center", gap: "8px",
                background: "#1a1a3a", color: "#7c6fe0",
                border: "1.5px solid #3b2f8f", borderRadius: "10px",
                padding: "10px 20px", fontWeight: "700", fontSize: "14px",
                textDecoration: "none", cursor: "pointer",
                boxShadow: "0 2px 10px rgba(59,47,143,0.3)",
              }}
            >
              ⬇ Download Current Layout (.glb)
            </a>
          )}
          {analysisResult["3d_models"]?.optimised_glb && (
            <a
              href={`${BASE}${analysisResult["3d_models"].optimised_glb}`}
              download="optimised_layout.glb"
              style={{
                display: "inline-flex", alignItems: "center", gap: "8px",
                background: "#0d2010", color: "#00ffd0",
                border: "1.5px solid #2e7d32", borderRadius: "10px",
                padding: "10px 20px", fontWeight: "700", fontSize: "14px",
                textDecoration: "none", cursor: "pointer",
                boxShadow: "0 2px 10px rgba(46,125,50,0.3)",
              }}
            >
              ⬇ Download Optimised Layout (.glb)
            </a>
          )}
        </div>

        {/* ── Done button ── */}
        <div style={{ margin:"0 0 20px" }}>
          <button
            onClick={() => setShowThankYou(true)}
            style={{
              background: "linear-gradient(135deg,#2e7d32,#1b5e20)",
              color: "#fff", border: "2px solid #4caf50",
              borderRadius: "10px", padding: "12px 32px",
              fontWeight: "700", fontSize: "15px", cursor: "pointer",
              boxShadow: "0 4px 14px rgba(46,125,50,0.4)",
              display: "inline-flex", alignItems: "center", gap: "8px",
            }}
          >
            ✅ Done
          </button>
          <span style={{ marginLeft:"12px", color:"#555", fontSize:"12px" }}>
            Click when finished to return to home
          </span>
        </div>

        {/* ── Recommendations table ── */}
        <h3>Furniture Recommendations</h3>
        <table style={ts.table}>
          <thead style={ts.thead}>
            <tr>
              <th style={ts.th}>Object</th>
              <th style={ts.th}>Current Zone</th>
              <th style={ts.th}>Recommended Zone</th>
              <th style={ts.th}>Move Needed</th>
            </tr>
          </thead>
          <tbody>
            {analysisResult.recommendations?.map((item, i) => (
              <tr key={i}
                style={hoveredRow===i ? ts.rowHover : i%2===0 ? ts.rowEven : ts.rowOdd}
                onMouseEnter={() => setHoveredRow(i)}
                onMouseLeave={() => setHoveredRow(null)}>
                <td style={ts.td}>
                  {item.object}
                  {item.is_essential && (
                    <span style={{ marginLeft:"6px", background:"#3b2f8f",
                      color:"#aac4ff", borderRadius:"4px",
                      padding:"1px 6px", fontSize:"10px", fontWeight:"700" }}>
                      Essential
                    </span>
                  )}
                </td>
                <td style={ts.td}>{item.current_zone}</td>
                <td style={ts.td}>{item.recommended_zone}</td>
                <td style={item.action_needed ? ts.tdYes : ts.tdNo}>
                  {item.action_needed ? "Yes" : "No"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Report */}
        <h3 style={{ marginTop:"24px" }}>Report</h3>
        <button
          onClick={() => window.open(`${BASE}${analysisResult.public_urls?.html_report}`, "_blank")}
          style={{ background:"#2e7d32", color:"#fff", border:"none",
                   borderRadius:"8px", padding:"12px 24px", fontWeight:"700",
                   fontSize:"15px", cursor:"pointer",
                   boxShadow:"0 4px 14px rgba(46,125,50,0.4)", marginBottom:"24px" }}
        >
          📄 View Full Vastu Report
        </button>

        {/* Compliance visualizations */}
        <h3>Compliance Visualizations</h3>
        <div style={{ display:"flex", gap:"16px", flexWrap:"wrap", marginBottom:"8px" }}>
          <img src={`${BASE}${analysisResult.public_urls?.compliance_radar}`}
            width="300" alt="Radar"
            style={{ borderRadius:"8px", border:"1px solid #3b2f8f" }} />
          <img src={`${BASE}${analysisResult.public_urls?.layout_comparison}`}
            width="300" alt="Layout"
            style={{ borderRadius:"8px", border:"1px solid #3b2f8f" }} />
        </div>


      </div>
    );
  }


  // Fallback
  return (
    <div>
      <h2>Upload video &amp; select room</h2>
      <input
        type="file" accept="video/*"
        onChange={e => {
          const f = e.target.files[0];
          setVideoFile(f);
          setVideoURL(URL.createObjectURL(f));
        }}
      />
      {renderInput("Bed", "bed")}
      <input
        type="text" placeholder="Room Type"
        onChange={e => setRoomType(e.target.value)}
      />
      <button onClick={handleUpload}>
        {loading ? "Processing..." : "Run Vastu"}
      </button>
    </div>
  );
}