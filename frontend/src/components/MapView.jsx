import React, { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Tooltip,
  useMap,
} from "react-leaflet";
import axios from "axios";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet.heat";
import "./MapView.css";

const BACKEND_URL = "http://127.0.0.1:5001/api";

/* ================================================== */
/* 🔹 Helper: Auto Fit Map Bounds                     */
/* ================================================== */
function FitMapBounds({ points }) {
  const map = useMap();

  useEffect(() => {
    if (!map || !points?.length) return;
    const bounds = L.latLngBounds(points.map((p) => [p.lat, p.lon]));
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [60, 60], maxZoom: 6, animate: true });
    } else {
      map.setView([39.5, -98.35], 4);
    }
  }, [map, points]);

  return null;
}

/* ================================================== */
/* 🔹 NASA-Style AOD Heatmap Layer                    */
/* ================================================== */
function AODHeatmapLayer({ points }) {
  const map = useMap();

  useEffect(() => {
    if (!map || !points?.length) return;

    const heatPoints = points.map((p) => [p.lat, p.lon, Math.min(p.aod * 5, 1)]);
    const gradient = {
      0.0: "#a8dadc",
      0.3: "#457b9d",
      0.6: "#1d3557",
    };

    const layer = L.heatLayer(heatPoints, {
      radius: 60,
      blur: 45,
      maxZoom: 7,
      minOpacity: 0.35,
      gradient,
    }).addTo(map);

    return () => map.removeLayer(layer);
  }, [map, points]);

  return null;
}

/* ================================================== */
/* 🔸 Prediction Heatmap Layer (AQI Gradient)         */
/* ================================================== */
function PredictionHeatmapLayer({ points }) {
  const map = useMap();

  useEffect(() => {
    if (!map || !points?.length) return;

    const heatPoints = points.map((p) => [
      p.lat,
      p.lon,
      Math.min(p.predicted_aqi / 300, 1),
    ]);

    const gradient = {
      0.0: "#00E400",
      0.25: "#FFFF00",
      0.5: "#FF7E00",
      0.75: "#FF0000",
      1.0: "#8F3F97",
    };

    const layer = L.heatLayer(heatPoints, {
      radius: 55,
      blur: 40,
      maxZoom: 7,
      minOpacity: 0.4,
      gradient,
    }).addTo(map);

    return () => map.removeLayer(layer);
  }, [map, points]);

  return null;
}

/* ================================================== */
/* 🗺️ Main Map Component                              */
/* ================================================== */
export default function MapView() {
  const [data, setData] = useState([]);
  const [dataset, setDataset] = useState("aqi");
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setError(null);
      const res = await axios.get(`${BACKEND_URL}/${dataset}`);
      // Some endpoints return plain lists; handle both formats
      const incoming = Array.isArray(res.data)
        ? res.data
        : res.data.data || [];
      setData(incoming);
    } catch (err) {
      console.error("❌ Fetch failed:", err);
      setError(`Failed to fetch ${dataset.toUpperCase()} data.`);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10 * 60 * 1000); // refresh every 10 min
    return () => clearInterval(interval);
  }, [dataset]);

  const getAqiColor = (value) => {
    if (value <= 50) return "#00E400";
    if (value <= 100) return "#FFFF00";
    if (value <= 150) return "#FF7E00";
    if (value <= 200) return "#FF0000";
    return "#8F3F97";
  };

  const renderLegend = () => (
    <div className="legend">
      {dataset === "aod" ? (
        <>
          <h4>AOD Legend</h4>
          <div>
            <span style={{ background: "#a8dadc" }}></span> Low (≤ 0.1)
          </div>
          <div>
            <span style={{ background: "#457b9d" }}></span> Moderate (0.1–0.2)
          </div>
          <div>
            <span style={{ background: "#1d3557" }}></span> High (≥ 0.3)
          </div>
        </>
      ) : (
        <>
          <h4>
            {dataset === "predictions"
              ? "Predicted AQI"
              : "Air Quality Index (AQI)"}
          </h4>
          <div>
            <span style={{ background: "#00E400" }}></span> Good (0–50)
          </div>
          <div>
            <span style={{ background: "#FFFF00" }}></span> Moderate (51–100)
          </div>
          <div>
            <span style={{ background: "#FF7E00" }}></span> Unhealthy Sensitive
            (101–150)
          </div>
          <div>
            <span style={{ background: "#FF0000" }}></span> Unhealthy (151–200)
          </div>
          <div>
            <span style={{ background: "#8F3F97" }}></span> Very Unhealthy
            (201–300)
          </div>
        </>
      )}
    </div>
  );

  return (
    <div className="map-wrapper">
      <div className="controls">
        <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
          <option value="aqi">Live AQI</option>
          <option value="aod">AOD</option>
          <option value="predictions">Predicted AQI</option>
        </select>
      </div>

      {error ? (
        <div className="error-banner">{error}</div>
      ) : (
        <MapContainer
          center={[39.5, -98.35]}
          zoom={4}
          className="map-container"
          scrollWheelZoom={true}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="&copy; OpenStreetMap contributors"
          />

          <FitMapBounds points={data} />

          {dataset === "aod" && <AODHeatmapLayer points={data} />}
          {dataset === "predictions" && (
            <PredictionHeatmapLayer points={data} />
          )}

          {dataset === "aqi" &&
            data.map((item, i) => (
              <CircleMarker
                key={i}
                center={[item.lat, item.lon]}
                pathOptions={{
                  color: getAqiColor(item.aqi),
                  fillColor: getAqiColor(item.aqi),
                  fillOpacity: 0.6,
                }}
                radius={10}
              >
                <Tooltip direction="top">
                  <b>{item.city || "Unknown"}</b>
                  <br />
                  AQI: {item.aqi}
                </Tooltip>
              </CircleMarker>
            ))}
        </MapContainer>
      )}

      {renderLegend()}
    </div>
  );
}
