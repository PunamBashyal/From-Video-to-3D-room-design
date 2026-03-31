// frontend/src/components/RunPipeline.js
import React, { useState } from "react";

export default function RunPipeline() {
  const [videoFile, setVideoFile] = useState(null);
  const [roomType, setRoomType] = useState("");
  const [furnitureData, setFurnitureData] = useState([
    { name: "bed", x: 100, y: 200 },
    { name: "table", x: 300, y: 400 },
  ]);

  const handleSubmit = async () => {
    if (!videoFile || !roomType) {
      alert("Please provide video and room type");
      return;
    }

    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("room_type", roomType);
    formData.append("furniture_data", JSON.stringify(furnitureData));

    try {
      const response = await fetch("/api/run-full-pipeline/", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("Pipeline result:", data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <h2>Run Full Pipeline</h2>
      <input type="file" onChange={(e) => setVideoFile(e.target.files[0])} />
      <input
        type="text"
        placeholder="Room type"
        value={roomType}
        onChange={(e) => setRoomType(e.target.value)}
      />
      <button onClick={handleSubmit}>Run Pipeline</button>
    </div>
  );
}