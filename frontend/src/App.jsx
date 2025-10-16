import React from "react";
import MapView from "./components/MapView";
import "./App.css";

export default function App() {
  return (
    <div className="app-container">
      <h1 className="app-title">🌍 Air Quality Dashboard (Live AQI + AOD + Predictions)</h1>
      <MapView />
    </div>
  );
}
