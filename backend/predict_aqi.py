"""
aqi_predictor.py — Fetches AQI data, trains predictive model, and returns structured AQI predictions.
"""

import aiohttp
import asyncio
import json
import ssl
import math
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

API_KEY = "19F42E70-7F2F-4F4E-98DA-BA7572BE37AD"
URL = "https://www.airnowapi.org/aq/observation/latLong/current/"
CACHE_FILE = Path("aqi_data.json")
MODEL_FILE = Path("aqi_model.pkl")

COORDS = [
    (47.61, -122.33), (34.05, -118.24), (36.16, -115.15),
    (40.76, -111.89), (39.74, -104.99), (41.88, -87.63),
    (29.76, -95.37), (32.78, -96.80), (38.63, -90.20),
    (40.71, -74.00), (42.36, -71.06), (33.75, -84.39),
    (25.76, -80.19), (44.95, -93.09), (39.95, -75.16),
    (45.52, -122.68), (35.22, -80.84), (33.45, -112.07),
    (37.77, -122.42), (30.26, -97.74), (46.87, -96.78),
    (43.04, -87.91), (35.15, -90.05), (36.17, -86.78),
    (27.95, -82.46), (41.26, -95.93)
]

# ---------------------------------------------------------
# FETCHING SECTION
# ---------------------------------------------------------

async def fetch_station(session, lat, lon):
    params = {
        "format": "application/json",
        "latitude": lat,
        "longitude": lon,
        "distance": 150,
        "API_KEY": API_KEY,
    }

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    try:
        async with session.get(URL, params=params, timeout=15, ssl=ssl_ctx) as r:
            if r.status == 200:
                data = await r.json()
                stations = []
                for d in data:
                    if not d.get("AQI"):
                        continue
                    stations.append({
                        "lat": d.get("Latitude"),
                        "lon": d.get("Longitude"),
                        "aqi": d.get("AQI"),
                        "category": d.get("Category", {}).get("Name", "Unknown"),
                        "time": d.get("DateObserved", ""),
                    })
                return stations
            else:
                print(f"⚠️ AirNow error {r.status} for ({lat},{lon})")
                return []
    except Exception as e:
        print(f"❌ Error fetching ({lat},{lon}): {e}")
        return []


async def fetch_all_aqi():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_station(session, lat, lon) for lat, lon in COORDS]
        results = await asyncio.gather(*tasks)
        all_data = [s for region in results for s in region]
        seen = set()
        unique = []
        for s in all_data:
            key = (round(s["lat"], 3), round(s["lon"], 3))
            if key not in seen:
                seen.add(key)
                unique.append(s)
        print(f"✅ Retrieved {len(unique)} AQI stations.")
        return unique


def fetch_aqi_data():
    if CACHE_FILE.exists():
        mtime = CACHE_FILE.stat().st_mtime
        age = (datetime.now().timestamp() - mtime) / 3600
        if age < 2:
            print("♻️ Using cached AQI data.")
            return json.load(open(CACHE_FILE))
    print("🌎 Fetching fresh AQI data...")
    data = asyncio.run(fetch_all_aqi())
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data

# ---------------------------------------------------------
# MODEL TRAINING + PREDICTION
# ---------------------------------------------------------

def train_model(data):
    df = pd.DataFrame(data)
    df["category_num"] = df["category"].astype("category").cat.codes
    df["hour"] = 12
    df["month"] = datetime.now().month

    X = df[["lat", "lon", "hour", "month", "category_num"]]
    y = df["aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"🧠 Model trained. MAE: {mae:.2f}")

    joblib.dump(model, MODEL_FILE)
    print(f"💾 Model saved to {MODEL_FILE}")
    return model


def load_or_train_model():
    if MODEL_FILE.exists():
        print("♻️ Using cached model.")
        return joblib.load(MODEL_FILE)
    data = fetch_aqi_data()
    return train_model(data)

# ---------------------------------------------------------
# MAIN PREDICT FUNCTION (MATCHES YOUR RANDOM VERSION)
# ---------------------------------------------------------

def predict_aqi():
    """Predict AQI values for 30 random U.S. locations"""
    model = load_or_train_model()
    predictions = []

    for _ in range(30):
        lat = random.uniform(25, 49)
        lon = random.uniform(-125, -66)
        df = pd.DataFrame([{
            "lat": lat,
            "lon": lon,
            "hour": 12,
            "month": datetime.now().month,
            "category_num": 0
        }])
        pred = float(model.predict(df)[0])

        # Cap and categorize
        pred = int(min(max(pred, 0), 200))
        if pred <= 50:
            cat = "Good"
        elif pred <= 100:
            cat = "Moderate"
        elif pred <= 150:
            cat = "Unhealthy for Sensitive Groups"
        else:
            cat = "Unhealthy"

        predictions.append({
            "lat": lat,
            "lon": lon,
            "predicted_aqi": pred,
            "category": cat
        })

    return predictions

# ---------------------------------------------------------
# TEST / ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    results = predict_aqi()
    print(json.dumps(results[:5], indent=2))
    print(f"✅ Generated {len(results)} AQI predictions.")
