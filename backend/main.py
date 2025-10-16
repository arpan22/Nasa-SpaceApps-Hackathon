from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fetch_aqi import fetch_aqi_data
from fetch_aod import fetch_aod_data
from predict_aqi import predict_aqi

app = FastAPI(title="Environmental Data API", version="3.0")

# ✅ Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "🌎 Environmental API running successfully!"}

@app.get("/api/aqi")
def get_aqi():
    data = fetch_aqi_data()
    return {"count": len(data), "data": data}

@app.get("/api/aod")
def get_aod():
    data = fetch_aod_data()
    return {"count": len(data), "data": data}

@app.get("/api/predictions")
def get_predictions():
    data = predict_aqi()
    return {"count": len(data), "data": data}

@app.get("/api/combined")
def get_combined():
    aqi = fetch_aqi_data()
    aod = fetch_aod_data()
    preds = predict_aqi()

    combined = []
    for p in preds:
        closest_aqi = min(aqi, key=lambda x: abs(x["lat"] - p["lat"]) + abs(x["lon"] - p["lon"]))
        closest_aod = min(aod, key=lambda x: abs(x["lat"] - p["lat"]) + abs(x["lon"] - p["lon"]))
        combined.append({
            "lat": p["lat"],
            "lon": p["lon"],
            "predicted_aqi": p["predicted_aqi"],
            "aqi_category": p["category"],
            "actual_aqi": closest_aqi["aqi"],
            "city": closest_aqi["city"],
            "aod": closest_aod["aod"]
        })
    return {"count": len(combined), "data": combined}
