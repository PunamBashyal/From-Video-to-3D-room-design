// src/components/Room3DPage.js
import React, { useState, useEffect } from "react";

const BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";
const abs      = rel => (rel ? `${BASE_URL}${rel}` : "");

function useModelViewer() {
  useEffect(() => {
    if (document.querySelector("script[data-mv]")) return;
    const s = document.createElement("script");
    s.type = "module"; s.dataset.mv = "1";
    s.src  = "https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js";
    document.head.appendChild(s);
  }, []);
}

// ── Single 3D layout panel ────────────────────────────────────────────────────
function Room3DPanel({ title, accent, renderPng, glbUrl, indexPng, stats }) {
  const [openGlb, setOpenGlb]     = useState(false);
  const [showIndex, setShowIndex] = useState(true);   // direction index open by default

  return (
    <div style={{
      background: "#16162a",
      border: `2px solid ${accent}`,
      borderRadius: 16,
      overflow: "hidden",
      boxShadow: `0 8px 32px ${accent}44`,
      marginBottom: 32,
    }}>
      {/* Header */}
      <div style={{ background:accent, color:"#fff",
                    padding:"14px 22px", fontWeight:800, fontSize:16,
                    display:"flex", justifyContent:"space-between",
                    alignItems:"center" }}>
        <span>{title}</span>
        <span style={{ fontSize:12, fontWeight:400, opacity:0.85 }}>
          {stats && Object.entries(stats).map(([k,v]) => `${k}: ${v}`).join('  ·  ')}
        </span>
      </div>

      {/* Static 3D render PNG */}
      {renderPng ? (
        <div style={{ background:"#12122a", position:"relative" }}>
          <img src={renderPng} alt={title}
               style={{ width:"100%", display:"block",
                        maxHeight:400, objectFit:"contain" }} />
          <div style={{ position:"absolute", bottom:8, left:8,
                        background:"rgba(0,0,0,0.72)", color:"#ccc",
                        borderRadius:6, padding:"3px 12px", fontSize:11 }}>
            ★ Gold = Essential &nbsp;|&nbsp; ⚠ Red = Violation &nbsp;|&nbsp;
            Coloured floor = Vastu zone
          </div>
        </div>
      ) : (
        <div style={{ padding:40, color:"#555", textAlign:"center",
                      background:"#12122a" }}>
          3D render not available
        </div>
      )}

      {/* Interactive GLB viewer */}
      {openGlb && glbUrl && (
        <div style={{ borderTop:`2px solid ${accent}44` }}>
          {/* eslint-disable-next-line */}
          <model-viewer src={glbUrl} alt={title}
            auto-rotate camera-controls
            shadow-intensity="1" environment-image="neutral"
            style={{ width:"100%", height:480,
                     background:"#1a1a2e", display:"block" }} />
          <p style={{ textAlign:"center", color:"#888",
                      fontSize:12, margin:"6px 0" }}>
            🖱 Drag to rotate · Scroll to zoom · Right-click to pan
          </p>
        </div>
      )}

      {/* Action buttons */}
      <div style={{ display:"flex", gap:10, padding:"12px 16px",
                    background:"#1a1a2e",
                    borderTop:`1px solid ${accent}22`,
                    flexWrap:"wrap" }}>
        {glbUrl && (
          <button onClick={() => setOpenGlb(v => !v)}
            style={{ background: openGlb ? "#444" : accent,
                     color:"#fff", border:"none", borderRadius:8,
                     padding:"10px 20px", fontWeight:700,
                     cursor:"pointer", fontSize:13 }}>
            {openGlb ? "✕ Close 3D Viewer" : "🧊 Open Interactive 3D Viewer"}
          </button>
        )}
        {glbUrl && (
          <a href={glbUrl} download
             style={{ background:"transparent", color:accent,
                      border:`1.5px solid ${accent}`, borderRadius:8,
                      padding:"10px 16px", fontWeight:700,
                      textDecoration:"none", fontSize:13 }}>
            ⬇ Download GLB
          </a>
        )}
        {indexPng && (
          <button onClick={() => setShowIndex(v => !v)}
            style={{ background: showIndex ? "#1e3a2a" : "#1e1e3a",
                     color: showIndex ? "#4ade80" : "#a78bfa",
                     border:`1.5px solid ${showIndex ? "#4ade80" : "#a78bfa"}`,
                     borderRadius:8, padding:"10px 20px",
                     fontWeight:700, cursor:"pointer", fontSize:13 }}>
            {showIndex ? "▲ Hide Direction Index" : "▼ Show Direction Index"}
          </button>
        )}
      </div>

      {/* Direction index table image */}
      {showIndex && indexPng && (
        <div style={{ background:"#0d0d1a",
                      borderTop:`2px solid ${accent}44`,
                      padding:"16px" }}>
          <h3 style={{ color:"#a78bfa", margin:"0 0 10px",
                       fontFamily:"'Georgia', serif", fontSize:16 }}>
            📍 Direction Index — Which Item is in Which Zone
          </h3>
          <p style={{ color:"#666", fontSize:12, margin:"0 0 10px" }}>
            Each furniture item with its Vastu zone, compass direction,
            and compliance status.
          </p>
          <img src={abs(indexPng)} alt="Direction index"
               style={{ width:"100%", borderRadius:10,
                        border:`1px solid ${accent}44`,
                        display:"block" }} />
        </div>
      )}
    </div>
  );
}

// ── Compliance numbers ────────────────────────────────────────────────────────
function ComplianceNumbers({ result }) {
  const pctB = result.initial_compliance_pct || 0;
  const pctA = result.final_compliance_pct   || 0;
  const gain = Math.round(pctA - pctB);
  const m3d  = result["3d_models"] || {};

  return (
    <div style={{ marginTop:40 }}>
      <h2 style={{ color:"#a78bfa", marginBottom:8,
                   fontFamily:"'Georgia',serif" }}>
        📊 Compliance Ratio Analysis
      </h2>

      {/* Big numbers */}
      <div style={{ display:"flex", gap:14, flexWrap:"wrap", marginBottom:24 }}>
        {[
          { label:"Total Furniture",   value:result.recommendations?.length||0, color:"#a78bfa" },
          { label:"Violations Before", value:result.initial_violations,          color:"#f87171" },
          { label:"Violations After",  value:result.optimized_violations,        color:"#34d399" },
          { label:"Compliance Before", value:`${pctB}%`,                         color:"#fbbf24" },
          { label:"Compliance After",  value:`${pctA}%`,                         color:"#34d399" },
          { label:"Improvement",       value:`+${gain}%`,                        color:"#60a5fa" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{
            background:"#1e1e3a", borderRadius:12,
            padding:"14px 20px", textAlign:"center",
            border:`1px solid ${color}44`,
            minWidth:110,
          }}>
            <div style={{ fontSize:28, fontWeight:800, color }}>{value}</div>
            <div style={{ fontSize:11, color:"#999", marginTop:4 }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Compliance chart */}
      {m3d.compliance_chart && (
        <img src={abs(m3d.compliance_chart)} alt="Compliance chart"
             style={{ width:"100%", borderRadius:12,
                      border:"2px solid #3b2f8f44", display:"block" }} />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
export default function Room3DPage() {
  useModelViewer();
  const [result, setResult] = useState(null);

  useEffect(() => {
    const raw = localStorage.getItem("vastuResult");
    if (raw) {
      try { setResult(JSON.parse(raw)); }
      catch (e) { console.error("Parse error:", e); }
    }
  }, []);

  if (!result) return (
    <div style={{ minHeight:"100vh", background:"#0d0d1a",
                  display:"flex", flexDirection:"column",
                  alignItems:"center", justifyContent:"center", gap:14 }}>
      <p style={{ color:"#a78bfa", fontSize:18 }}>
        Loading 3D analysis…
      </p>
      <p style={{ color:"#555", fontSize:13 }}>
        If this persists, go back and click "View 3D Room" again.
      </p>
      <button onClick={() => window.close()}
        style={{ background:"transparent", color:"#a78bfa",
                 border:"1.5px solid #a78bfa", borderRadius:7,
                 padding:"8px 20px", cursor:"pointer", fontSize:13 }}>
        ← Close Tab
      </button>
    </div>
  );

  const m3d   = result["3d_models"] || {};
  const rdims = result.room_dimensions || {};

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg,#0d0d1a 0%,#12122a 60%,#0d0d1a 100%)",
      color: "#e2e8f0",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      padding: "32px 24px",
      maxWidth: 1300,
      margin: "0 auto",
    }}>

      {/* Page header */}
      <div style={{ marginBottom:28 }}>
        <h1 style={{ fontFamily:"'Georgia',serif", fontSize:32,
                     fontWeight:800, color:"#a78bfa", margin:0 }}>
          3D Room Analysis
        </h1>
        <p style={{ color:"#888", marginTop:6, fontSize:14 }}>
          Room: <strong style={{ color:"#c4b5fd" }}>
            {result.video_info?.room_type || "—"}
          </strong>
          &nbsp;·&nbsp;
          {rdims.width_m?.toFixed(1)} × {rdims.depth_m?.toFixed(1)} × {rdims.height_m} m
          &nbsp;·&nbsp;
          Final compliance:{" "}
          <strong style={{ color:"#4ade80" }}>{result.final_compliance_pct}%</strong>
        </p>
        <div style={{ marginTop:10, display:"flex", gap:10, flexWrap:"wrap" }}>
          <button onClick={() => window.close()}
            style={{ background:"transparent", color:"#a78bfa",
                     border:"1.5px solid #a78bfa", borderRadius:7,
                     padding:"7px 18px", cursor:"pointer", fontSize:13,
                     fontWeight:600 }}>
            ← Close Tab
          </button>
        </div>
      </div>

      {/* Legend strip */}
      <div style={{ display:"flex", gap:18, flexWrap:"wrap",
                    marginBottom:20, fontSize:13,
                    background:"#1e1e3a", borderRadius:8,
                    padding:"10px 16px" }}>
        <span style={{ color:"#FFD700", fontWeight:700 }}>★ Gold border = Essential / user-selected</span>
        <span style={{ color:"#f87171", fontWeight:700 }}>⚠ Red border = Vastu violation</span>
        <span style={{ color:"#60a5fa" }}>🟦 Blue label = Detected by YOLO</span>
        <span style={{ color:"#888" }}>Coloured floor tiles = Vastu zones</span>
      </div>

      {/* BEFORE panel */}
      <Room3DPanel
        title="🏠 Before — Current Room Layout"
        accent="#6d28d9"
        renderPng={m3d.current_render   ? abs(m3d.current_render)   : null}
        glbUrl={m3d.current_glb         ? abs(m3d.current_glb)      : null}
        indexPng={m3d.current_index     || null}
        stats={{
          "Compliance":  `${result.initial_compliance_pct}%`,
          "Violations":  result.initial_violations,
          "Items":       result.recommendations?.length || 0,
        }}
         
      />
      

      {/* AFTER panel */}
      <Room3DPanel
        title="✨ After — Vastu-Optimised Layout"
        accent="#059669"
        renderPng={m3d.optimised_render  ? abs(m3d.optimised_render) : null}
        glbUrl={m3d.optimised_glb        ? abs(m3d.optimised_glb)    : null}
        indexPng={m3d.optimised_index    || null}
        stats={{
          "Compliance":  `${result.final_compliance_pct}%`,
          "Violations":  result.optimized_violations,
          "Improvement": `+${Math.round(result.final_compliance_pct - result.initial_compliance_pct)}%`,
        }}
      />

      {/* Compliance ratio section */}
      <ComplianceNumbers result={result} />

      <div style={{ marginTop:40, paddingTop:20,
                    borderTop:"1px solid #333",
                    display:"flex", gap:12 }}>
        <button onClick={() => window.close()}
          style={{ background:"#1e1e3a", color:"#a78bfa",
                   border:"1.5px solid #a78bfa", borderRadius:8,
                   padding:"11px 22px", fontWeight:700,
                   cursor:"pointer", fontSize:14 }}>
          ← Close Tab
        </button>
      </div>
    </div>
  );
}