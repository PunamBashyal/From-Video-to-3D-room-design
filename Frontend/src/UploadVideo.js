import React, { useState, useRef } from "react";

export default function UploadVideo() {
  const [videoData, setVideoData] = useState(null);
  const [loading, setLoading] = useState(false);
  const videoRef = useRef(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("video", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/upload-video/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        alert(errData.error || "Upload failed");
        setLoading(false);
        return;
      }

      const data = await res.json();
      setVideoData(data); // store the returned API data
      setLoading(false);

      // Set video src for preview
      if (videoRef.current) {
        videoRef.current.src = data.video_url;
        videoRef.current.load();
      }

    } catch (err) {
      alert("Upload failed: " + err.message);
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Upload or Capture Video</h2>
      <input type="file" accept="video/*" onChange={handleUpload} disabled={loading} />
      {loading && <p>Uploading and processing video... ⏳</p>}

      {/* Video Preview */}
      {videoData && (
        <div>
          <h3>Video Preview</h3>
          <video ref={videoRef} controls width="400" />
        </div>
      )}

      {/* Genetic + Vastu Results */}
      {videoData?.genetic_vastu_analysis && (
        <div className="vastu-result">
          <h3>Vastu & Layout Analysis</h3>
          <p>Vastu Score: {videoData.genetic_vastu_analysis.vastu.vastu_score}</p>
          <h4>Optimized Layout:</h4>
          <ul>
            {Object.entries(videoData.genetic_vastu_analysis.optimized_layout).map(
              ([item, pos]) => (
                <li key={item}>{item}: {pos}</li>
              )
            )}
          </ul>
        </div>
      )}
    </div>
  );
}