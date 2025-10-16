"""
fetch_aqi.py — Live AQI fetcher + smoother for U.S. coverage
"""

import aiohttp
import asyncio
import json
import ssl
import math
from datetime import datetime
from pathlib import Path

API_KEY = "19F42E70-7F2F-4F4E-98DA-BA7572BE37AD"
URL = "https://www.airnowapi.org/aq/observation/latLong/current/"
CACHE_FILE = Path("aqi_data.json")

# Major U.S. sample grid centers (spread for full coverage)
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


async def fetch_station(session, lat, lon):
    """Fetch real AQI data from AirNow API"""
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
                    if not d.get("AQI"):  # skip invalid
                        continue
                    stations.append({
                        "city": d.get("ReportingArea", "Unknown"),
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
    """Fetch AQI data for multiple regions across the U.S."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_station(session, lat, lon) for lat, lon in COORDS]
        results = await asyncio.gather(*tasks)
        all_data = [s for region in results for s in region]

        # Deduplicate
        seen = set()
        unique = []
        for s in all_data:
            key = (round(s["lat"], 3), round(s["lon"], 3))
            if key not in seen:
                seen.add(key)
                unique.append(s)

        print(f"✅ Retrieved {len(unique)} AQI stations before smoothing.")
        return unique


def haversine(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon points (km)"""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def smooth_aqi_points(points, radius_km=150):
    """
    Aggregate nearby points and lightly interpolate missing regions.
    Returns smoothed AQI data ideal for heatmaps.
    """
    if not points:
        return []

    # Step 1: group nearby stations
    smoothed = []
    used = set()

    for i, p in enumerate(points):
        if i in used:
            continue
        cluster = [p]
        for j, q in enumerate(points):
            if j != i and haversine(p["lat"], p["lon"], q["lat"], q["lon"]) < radius_km:
                cluster.append(q)
                used.add(j)

        avg_aqi = sum(c["aqi"] for c in cluster) / len(cluster)
        smoothed.append({
            "lat": sum(c["lat"] for c in cluster) / len(cluster),
            "lon": sum(c["lon"] for c in cluster) / len(cluster),
            "aqi": round(avg_aqi),
            "category": "Interpolated" if len(cluster) > 1 else cluster[0]["category"],
        })

    # Step 2: fill grid with light interpolation (approx 2° spacing)
    lat_range = range(25, 50, 2)
    lon_range = range(-125, -65, 2)
    filled = []

    for lat in lat_range:
        for lon in lon_range:
            nearest = sorted(points, key=lambda p: haversine(lat, lon, p["lat"], p["lon"]))[:5]
            if nearest:
                weights = [1 / (haversine(lat, lon, n["lat"], n["lon"]) + 1) for n in nearest]
                avg = sum(n["aqi"] * w for n, w in zip(nearest, weights)) / sum(weights)
                filled.append({
                    "lat": lat + 0.5,
                    "lon": lon + 0.5,
                    "aqi": round(avg),
                    "category": "Interpolated",
                })

    print(f"🌐 Smoothed into {len(filled)} interpolated AQI points.")
    return smoothed + filled


def fetch_aqi_data():
    """Cached fetcher for AQI data"""
    if CACHE_FILE.exists():
        mtime = CACHE_FILE.stat().st_mtime
        age = (datetime.now().timestamp() - mtime) / 3600
        if age < 2:
            print("♻️ Using cached AQI data.")
            return json.load(open(CACHE_FILE))

    print("🌎 Fetching fresh AQI data from AirNow...")
    data = asyncio.run(fetch_all_aqi())
    data = smooth_aqi_points(data)

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    results = fetch_aqi_data()
    print(json.dumps(results[:10], indent=2))
    print(f"Total processed: {len(results)}")
