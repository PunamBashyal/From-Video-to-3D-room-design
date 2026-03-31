import React from "react";
import bedroom from "./assets/bedroom.jpg";
import livingroom from "./assets/livingroom.jpg";
import studyroom from "./assets/studyroom.jpg";
import kitchen from "./assets/kitchen.jpg";


import room2 from "./assets/room2.jpg";
import room3 from "./assets/room3.jpg";
import "./App.css";


function Home() {
  return (
    <div className="app-container">
      <div className="center-images right">
        <img src={room2} alt="center decoration 1" />
        <img src={room3} alt="center decoration 2" />
      </div>
      {/* SINGLE INNER BOX */}
        <div className="background-images">
          <img src={bedroom} alt="Bedroom" />
          <img src={livingroom} alt="Living Room" />
          <img src={studyroom} alt="Study Room" />
          <img src={kitchen} alt="Kitchen" />
          
        </div>
        <div className="foreground-box">
        <h1>Design your room with style and glee, <br /> Your dream space as fun as can be!</h1>
        <p>Welcome to your smart room design assistant</p>
        {/* FULL INNER BOX (can be empty or decorative) */}
        

        

        
      </div>
    </div>
       
  );
}


export default Home;