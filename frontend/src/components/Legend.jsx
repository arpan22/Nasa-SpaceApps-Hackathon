import React from "react";

export default function Legend({ dataset }) {
  const legends = {
    aqi: [
      { color: "#00e400", label: "Good (0–50)" },
      { color: "#ffff00", label: "Moderate (51–100)" },
      { color: "#ff7e00", label: "Unhealthy for Sensitive (101–150)" },
      { color: "#ff0000", label: "Unhealthy (151–200)" },
      { color: "#8f3f97", label: "Very Unhealthy (201–300)" },
      { color: "#7e0023", label: "Hazardous (301+)" },
    ],
    aod: [
      { color: "#a8dadc", label: "Low AOD (≤0.1)" },
      { color: "#457b9d", label: "Moderate AOD (0.1–0.2)" },
      { color: "#1d3557", label: "High AOD (≥0.3)" },
    ],
    predictions: [
      { color: "#00e400", label: "Good (0–50)" },
      { color: "#ffff00", label: "Moderate (51–100)" },
      { color: "#ff7e00", label: "Unhealthy for Sensitive (101–150)" },
      { color: "#ff0000", label: "Unhealthy (151–200)" },
      { color: "#8f3f97", label: "Very Unhealthy (201–300)" },
      { color: "#7e0023", label: "Hazardous (301+)" },
    ],
  };

  return (
    <div
      style={{
        position: "absolute",
        bottom: "20px",
        right: "20px",
        backgroundColor: "white",
        padding: "12px",
        borderRadius: "8px",
        boxShadow: "0 0 10px rgba(0,0,0,0.2)",
        zIndex: 1000,
      }}
    >
      <strong style={{ display: "block", marginBottom: "5px" }}>
        {dataset.toUpperCase()} Legend
      </strong>
      {legends[dataset].map((item, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", marginBottom: "3px" }}>
          <div
            style={{
              width: "15px",
              height: "15px",
              backgroundColor: item.color,
              marginRight: "6px",
              border: "1px solid #444",
            }}
          ></div>
          <span style={{ fontSize: "13px" }}>{item.label}</span>
        </div>
      ))}
    </div>
  );
}
