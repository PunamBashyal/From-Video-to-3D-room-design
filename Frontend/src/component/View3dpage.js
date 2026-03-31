// src/View3DPage.js
// Full-screen page that opens when user clicks "View 3D Room & Comparison".
// Reads vastuResult from localStorage (saved by roomsetup.js Step 4).
// Shows 3 tabs: Current Layout | Vastu-Optimised | Comparison

import React, { useState, useEffect } from "react";

const BASE = "http://127.0.0.1:8000";

export default function View3DPage() {
  const [result,    setResult]    = useState(null);
  const [activeTab, setActiveTab] = useState("current");

  useEffect(() => {
    try {
      const stored = localStorage.getItem("vastuResult");
      if (stored) setResult(JSON.parse(stored));
    } catch (e) {
      console.error("Could not load result:", e);
    }
  }, []);

  // ── No data state ──────────────────────────────────────────────────────────
  if (!result) return (
    <div style={{
      minHeight: "100vh", background: "#0d0d1a",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      gap: "16px", fontFamily: "'Segoe UI', sans-serif",
    }}>
      <div style={{ fontSize: "56px" }}>🏠</div>
      <div style={{ fontSize: "18px", color: "#555" }}>No 3D data found.</div>
      <div style={{ fontSize: "12px", color: "#333" }}>
        Please run the Vastu analysis first.
      </div>
      <button onClick={() => window.close()} style={{
        marginTop: "12px", background: "#3b2f8f", color: "#fff",
        border: "none", borderRadius: "8px", padding: "10px 28px",
        cursor: "pointer", fontWeight: "700", fontSize: "14px",
      }}>
        ← Close
      </button>
    </div>
  );

  const models      = result["3d_models"] || {};
  const cmp         = models.comparison   || {};
  const curHtml     = models.current_html_content   || "";
  const optHtml     = models.optimised_html_content  || "";
  const curUrl      = models.current_render   ? `${BASE}${models.current_render}`   : null;
  const optUrl      = models.optimised_render ? `${BASE}${models.optimised_render}` : null;
  const curVideo    = models.current_video    ? `${BASE}${models.current_video}`    : null;
  const optVideo    = models.optimised_video  ? `${BASE}${models.optimised_video}`  : null;
  const pos         = (cmp.improvement || 0) >= 0;

  const TABS = [
    { key: "current",   label: "🏠 Current Layout",
      badge: `${cmp.detected_count || 0} detected`, color: "#7c6fe0" },
    { key: "optimised", label: "✨ Vastu-Optimised",
      badge: `${cmp.total_count || 0} items`,       color: "#00b8a5" },
    { key: "compare",   label: "📊 Comparison",
      badge: `${pos ? "+" : ""}${cmp.improvement || 0}%`,
      color: pos ? "#00ffd0" : "#ff4444" },
  ];

  return (
    <div style={{
      height: "100vh", display: "flex", flexDirection: "column",
      background: "#0d0d1a", fontFamily: "'Segoe UI', sans-serif",
      overflow: "hidden",
    }}>

      {/* ── Header ── */}
      <div style={{
        background: "linear-gradient(135deg,#1a1a3a,#12122a)",
        borderBottom: "2px solid #3b2f8f",
        padding: "10px 24px",
        display: "flex", alignItems: "center", gap: "14px", flexShrink: 0,
      }}>
        <span style={{ fontSize: "22px" }}>🏠</span>
        <div>
          <div style={{ color: "#e2e8f0", fontWeight: "800", fontSize: "16px" }}>
            Vastu 3D Room Viewer
          </div>
          <div style={{ color: "#555", fontSize: "11px" }}>
            Room: <b style={{ color:"#aac4ff" }}>
              {result.video_info?.room_type || "—"}
            </b>
            &nbsp;·&nbsp;
            Compliance: <b style={{ color:"#2e7d32" }}>
              {result.final_compliance_pct}%
            </b>
          </div>
        </div>

        {/* Quick stats */}
        {cmp.compliance_before != null && (
          <div style={{
            marginLeft: "16px", display: "flex", gap: "12px",
            alignItems: "center", flexWrap: "wrap",
          }}>
            <span style={{ fontSize: "12px", color: "#e65100" }}>
              Before: <b>{cmp.compliance_before}%</b>
            </span>
            <span style={{ color: "#444" }}>→</span>
            <span style={{ fontSize: "12px", color: "#2e7d32" }}>
              After: <b>{cmp.compliance_after}%</b>
            </span>
            <span style={{
              fontSize: "13px", fontWeight: "800",
              color: pos ? "#00ffd0" : "#ff4444",
            }}>
              {pos ? "↑ +" : "↓ "}{cmp.improvement}%
            </span>
          </div>
        )}

        <button onClick={() => window.close()} style={{
          marginLeft: "auto", background: "#1e1e3a", color: "#7c6fe0",
          border: "1px solid #3b2f8f", borderRadius: "8px",
          padding: "8px 20px", cursor: "pointer",
          fontWeight: "700", fontSize: "13px",
        }}>
          ✕ Close
        </button>
      </div>

      {/* ── Tab bar ── */}
      <div style={{
        background: "#0a0a1e",
        borderBottom: "2px solid #3b2f8f",
        display: "flex", padding: "0 20px", gap: "4px", flexShrink: 0,
      }}>
        {TABS.map(tab => {
          const active = activeTab === tab.key;
          return (
            <button key={tab.key} onClick={() => setActiveTab(tab.key)} style={{
              padding: "11px 22px",
              background:   active ? "#1a1a3a"   : "transparent",
              color:        active ? tab.color   : "#555",
              border:       "none",
              borderBottom: active
                ? `3px solid ${tab.color}` : "3px solid transparent",
              borderRadius: "8px 8px 0 0",
              fontWeight: "700", fontSize: "14px", cursor: "pointer",
              display: "flex", alignItems: "center", gap: "8px",
              transition: "all .15s",
            }}>
              {tab.label}
              <span style={{
                background: active ? `${tab.color}33` : "#ffffff11",
                color:      active ? tab.color : "#444",
                borderRadius: "10px", padding: "2px 8px",
                fontSize: "10px", fontWeight: "700",
              }}>
                {tab.badge}
              </span>
            </button>
          );
        })}
      </div>

      {/* ── Tab content — fills remaining height ── */}
      <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>

        {/* Current layout tab */}
        <div style={{
          display: activeTab === "current" ? "flex" : "none",
          flexDirection: "column", height: "100%",
        }}>
          <div style={{
            background: "#0a0a1e", padding: "7px 20px",
            display: "flex", alignItems: "center", gap: "12px", flexShrink: 0,
            borderBottom: "1px solid #3b2f8f33",
          }}>
            <span style={{ color: "#555", fontSize: "11px", flex: 1 }}>
              Furniture detected by YOLO at current positions.
              🖱 Drag to rotate · Scroll to zoom · Right-drag to pan ·
              Click sidebar item to highlight in 3D.
            </span>
            {curUrl && (
              <a href={curUrl} target="_blank" rel="noopener noreferrer"
                style={{ color: "#7c6fe0", fontSize: "11px", fontWeight: "700",
                         border: "1px solid #7c6fe044", borderRadius: "6px",
                         padding: "3px 10px", textDecoration: "none",
                         whiteSpace: "nowrap" }}>
                ↗ Standalone
              </a>
            )}
          </div>

          {/* 3D iframe — fills all remaining space */}
          <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
            {curHtml
              ? <iframe
                  srcDoc={curHtml}
                  title="Current 3D"
                  sandbox="allow-scripts"
                  style={{ width: "100%", height: "100%",
                           border: "none", display: "block",
                           background: "#0d0d1a" }}
                />
              : <div style={{
                  width: "100%", height: "100%", display: "flex",
                  alignItems: "center", justifyContent: "center",
                  color: "#555", fontSize: "15px",
                }}>
                  3D viewer not available — run the pipeline first.
                </div>
            }
          </div>

          {/* Turntable video */}
          {curVideo && (
            <div style={{
              background: "#08080f", borderTop: "1px solid #3b2f8f22",
              padding: "12px 20px", flexShrink: 0,
            }}>
              <div style={{ color:"#555", fontSize:"10px",
                            marginBottom:"6px", fontWeight:"700" }}>
                🎬 360° Turntable Preview:
              </div>
              <video src={curVideo} autoPlay loop muted playsInline controls
                style={{ width:"100%", maxWidth:"640px", borderRadius:"8px",
                         border:"1px solid #3b2f8f", background:"#0d0d1a",
                         display:"block" }} />
            </div>
          )}

          <div style={{
            background: "#080818", padding: "5px 20px",
            fontSize: "10px", color: "#333", textAlign: "center", flexShrink: 0,
          }}>
            🖱 Drag: rotate · Scroll: zoom · Right-drag: pan · Click sidebar to highlight
          </div>
        </div>

        {/* Optimised layout tab */}
        <div style={{
          display: activeTab === "optimised" ? "flex" : "none",
          flexDirection: "column", height: "100%",
        }}>
          <div style={{
            background: "#0a1a15", padding: "7px 20px",
            display: "flex", alignItems: "center", gap: "12px", flexShrink: 0,
            borderBottom: "1px solid #00b8a533",
          }}>
            <span style={{ color: "#555", fontSize: "11px", flex: 1 }}>
              All furniture repositioned to Vastu-correct zones by the genetic algorithm.
              🖱 Drag to rotate · Scroll to zoom.
            </span>
            {optUrl && (
              <a href={optUrl} target="_blank" rel="noopener noreferrer"
                style={{ color: "#00ffd0", fontSize: "11px", fontWeight: "700",
                         border: "1px solid #00ffd044", borderRadius: "6px",
                         padding: "3px 10px", textDecoration: "none",
                         whiteSpace: "nowrap" }}>
                ↗ Standalone
              </a>
            )}
          </div>

          <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
            {optHtml
              ? <iframe
                  srcDoc={optHtml}
                  title="Optimised 3D"
                  sandbox="allow-scripts"
                  style={{ width: "100%", height: "100%",
                           border: "none", display: "block",
                           background: "#0d0d1a" }}
                />
              : <div style={{
                  width: "100%", height: "100%", display: "flex",
                  alignItems: "center", justifyContent: "center",
                  color: "#555", fontSize: "15px",
                }}>
                  3D viewer not available.
                </div>
            }
          </div>

          {optVideo && (
            <div style={{
              background: "#08080f", borderTop: "1px solid #00b8a522",
              padding: "12px 20px", flexShrink: 0,
            }}>
              <div style={{ color:"#555", fontSize:"10px",
                            marginBottom:"6px", fontWeight:"700" }}>
                🎬 360° Turntable Preview:
              </div>
              <video src={optVideo} autoPlay loop muted playsInline controls
                style={{ width:"100%", maxWidth:"640px", borderRadius:"8px",
                         border:"1px solid #00b8a5", background:"#0d0d1a",
                         display:"block" }} />
            </div>
          )}

          <div style={{
            background: "#080818", padding: "5px 20px",
            fontSize: "10px", color: "#333", textAlign: "center", flexShrink: 0,
          }}>
            🖱 Drag: rotate · Scroll: zoom · Right-drag: pan · Click sidebar to highlight
          </div>
        </div>

        {/* Comparison tab — scrollable */}
        <div style={{
          display:    activeTab === "compare" ? "block" : "none",
          height:     "100%", overflowY: "auto",
          background: "linear-gradient(135deg,#12122a,#0d1a10)",
        }}>
          <div style={{ padding: "36px", maxWidth: "900px", margin: "0 auto" }}>
            <h2 style={{ color:"#a0e080", marginBottom:"32px",
                         fontWeight:"800", fontSize:"20px" }}>
              📊 Vastu Compliance Comparison
            </h2>

            {/* Big ratio cards */}
            <div style={{ display:"flex", gap:"16px", marginBottom:"32px",
                          alignItems:"center", flexWrap:"wrap" }}>
              <div style={{ flex:1, minWidth:"140px", background:"#1e1e3a",
                            borderRadius:"14px", padding:"24px", textAlign:"center",
                            border:"1.5px solid #3b2f8f" }}>
                <div style={{ fontSize:"11px", color:"#555", marginBottom:"8px",
                              textTransform:"uppercase", letterSpacing:"1.5px" }}>
                  Before Optimisation
                </div>
                <div style={{ fontSize:"60px", fontWeight:"900",
                              color:"#e65100", lineHeight:1 }}>
                  {cmp.compliance_before}%
                </div>
                <div style={{ fontSize:"12px", color:"#555", marginTop:"8px" }}>
                  Vastu Compliant
                </div>
              </div>

              <div style={{ fontSize:"36px", color:"#3b2f8f" }}>→</div>

              <div style={{ flex:1, minWidth:"140px", background:"#0d2010",
                            borderRadius:"14px", padding:"24px", textAlign:"center",
                            border:"1.5px solid #2e7d32" }}>
                <div style={{ fontSize:"11px", color:"#555", marginBottom:"8px",
                              textTransform:"uppercase", letterSpacing:"1.5px" }}>
                  After Optimisation
                </div>
                <div style={{ fontSize:"60px", fontWeight:"900",
                              color:"#00ffd0", lineHeight:1 }}>
                  {cmp.compliance_after}%
                </div>
                <div style={{ fontSize:"12px", color:"#555", marginTop:"8px" }}>
                  Vastu Compliant
                </div>
              </div>

              <div style={{ fontSize:"36px",
                            color: pos ? "#00ffd0" : "#ff4444" }}>
                {pos ? "↑" : "↓"}
              </div>

              <div style={{ flex:1, minWidth:"140px",
                            background: pos ? "#0d2010" : "#2a0a0a",
                            borderRadius:"14px", padding:"24px", textAlign:"center",
                            border:`1.5px solid ${pos?"#00ffd0":"#ff4444"}` }}>
                <div style={{ fontSize:"11px", color:"#555", marginBottom:"8px",
                              textTransform:"uppercase", letterSpacing:"1.5px" }}>
                  Improvement
                </div>
                <div style={{ fontSize:"60px", fontWeight:"900", lineHeight:1,
                              color: pos ? "#00ffd0" : "#ff4444" }}>
                  {pos ? "+" : ""}{cmp.improvement}%
                </div>
                <div style={{ fontSize:"12px", color:"#555", marginTop:"8px" }}>
                  {pos ? "Better compliance" : "Needs review"}
                </div>
              </div>
            </div>

            {/* Stat grid */}
            <div style={{ display:"grid",
                          gridTemplateColumns:"repeat(auto-fit,minmax(150px,1fr))",
                          gap:"14px", marginBottom:"28px" }}>
              {[
                { label:"Detected by YOLO",    value:cmp.detected_count,  color:"#7c6fe0", icon:"🔍" },
                { label:"Total (+ essential)", value:cmp.total_count,     color:"#1565c0", icon:"📦" },
                { label:"Compliant zones",     value:cmp.compliant_count, color:"#2e7d32", icon:"✅" },
                { label:"Zone violations",     value:cmp.violation_count, color:"#b71c1c", icon:"⚠️" },
              ].map((s, i) => (
                <div key={i} style={{ background:"#1e1e3a", borderRadius:"12px",
                                      padding:"18px", textAlign:"center",
                                      border:`1.5px solid ${s.color}44` }}>
                  <div style={{ fontSize:"28px", marginBottom:"8px" }}>{s.icon}</div>
                  <div style={{ fontSize:"34px", fontWeight:"800", color:s.color }}>
                    {s.value}
                  </div>
                  <div style={{ fontSize:"11px", color:"#555", marginTop:"6px" }}>
                    {s.label}
                  </div>
                </div>
              ))}
            </div>

            {/* Progress bar */}
            <div style={{ background:"#1e1e3a", borderRadius:"12px", padding:"22px" }}>
              <div style={{ display:"flex", justifyContent:"space-between",
                            fontSize:"13px", color:"#666", marginBottom:"10px" }}>
                <span>Before: <b style={{ color:"#e65100" }}>{cmp.compliance_before}%</b></span>
                <span>After: <b style={{ color:"#00ffd0" }}>{cmp.compliance_after}%</b></span>
              </div>
              <div style={{ position:"relative", height:"18px",
                            background:"#0d0d1a", borderRadius:"9px",
                            overflow:"hidden" }}>
                <div style={{ position:"absolute", left:0, top:0, height:"100%",
                              width:`${cmp.compliance_before}%`,
                              background:"#e65100", opacity:.6, borderRadius:"9px" }} />
                <div style={{ position:"absolute", left:0, top:0, height:"100%",
                              width:`${cmp.compliance_after}%`,
                              background:"linear-gradient(90deg,#2e7d32,#00ffd0)",
                              borderRadius:"9px" }} />
              </div>
              <div style={{ textAlign:"center", fontSize:"13px", marginTop:"10px",
                            fontWeight:"700",
                            color: pos ? "#00ffd0" : "#ff4444" }}>
                {pos
                  ? `Vastu compliance improved by ${cmp.improvement}% after optimisation`
                  : `Compliance decreased — review recommendations`}
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}