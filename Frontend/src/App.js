import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Home from "./home";
import About from "./about";
import Contact from "./contact";
import RoomSetup from "./roomsetup";
import "./App.css";
import UploadVideo from "./UploadVideo";
import ReviewPage from "./ReviewPage";
import Room3DPage from "./component/Room3DPage"; // ← ADDED
import View3DPage from "./component/View3dpage";


function App() {
  const [showWelcome, setShowWelcome] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setShowWelcome(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  if (showWelcome) {
    return (
      <div className="app-container">
        <div className="inner-box">
          <h1 style={{ fontSize: "3.5rem" }}>Welcome to Smart Room Designer</h1>
          <p style={{ fontSize: "1.5rem", marginTop: "1rem" }}>
            Creating your dream space with AI
          </p>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <Routes>

        {/* ── /3d-view renders STANDALONE — no header, no app-container ── */}
        <Route path="/3d-view" element={<Room3DPage />} />

        {/* ── All other routes keep the existing header + layout ── */}
        <Route path="/*" element={
          <div className="app-container">
            <header className="header">
              <div className="header-left">
                <Link to="/" className="btn">Home</Link>
                <Link to="/about" className="btn">About</Link>
                <Link to="/contact" className="btn">Contact</Link>
              </div>
              <div className="header-right">
                <Link to="/roomsetup" className="btn">Room Setup</Link>
              </div>
            </header>

            <div className="page-content">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/about" element={<About />} />
                <Route path="/contact" element={<Contact />} />
                <Route path="/roomsetup" element={<RoomSetup />} />
                <Route path="/upload-video" element={<UploadVideo />} />
                <Route path="/review" element={<ReviewPage />} />
                <Route path="/3d-view" element={<View3DPage />} />
              </Routes>
            </div>
          </div>
        } />

      </Routes>
    </Router>
  );
}

export default App;


