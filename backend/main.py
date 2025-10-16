import sys, os, random
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.predict_aqi import predict_aqi

app = FastAPI()

# Enable CORS for frontend (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ROUTES ===

@app.get("/api/aqi")
def get_live_aqi():
    """
    Returns simulated real-time AQI data (based on predictions).
    You can later replace this with actual AirNow API data.
    """
    aqi_data = predict_aqi()
    # Convert predicted data to "live" style format
    for entry in aqi_data:
        entry["aqi"] = entry.pop("predicted_aqi")
    return aqi_data


@app.get("/api/predictions")
def get_predictions():
    """
    Returns AQI predictions for various US locations.
    """
    return predict_aqi()


@app.get("/api/aod")
def get_aod_mock():
    """
    Returns mock Aerosol Optical Depth (AOD) data for NASA-style heatmap visualization.
    You can later replace this with NASA GIBS or MODIS data.
    """
    aod_data = []
    # Generate 100 points with lat/lon distributed over the US
    for _ in range(100):
        aod_data.append({
            "lat": random.uniform(25, 49),
            "lon": random.uniform(-125, -66),
            "aod": round(random.uniform(0.05, 0.35), 3)
        })
    return aod_data


@app.get("/predict")
def redirect_to_prediction():
    """Alias for compatibility with earlier routes"""
    return predict_aqi()
