import random
import numpy as np

def predict_aqi():
    """Simulated AQI predictions"""
    predictions = []
    for _ in range(30):
        lat = random.uniform(25, 49)
        lon = random.uniform(-125, -66)
        history = [random.randint(20, 180) for _ in range(7)]
        avg = np.mean(history)
        pred = int(min(max(avg + random.uniform(-10, 10), 0), 200))
        predictions.append({
            "lat": lat,
            "lon": lon,
            "predicted_aqi": pred,
            "category": (
                "Good" if pred <= 50 else
                "Moderate" if pred <= 100 else
                "Unhealthy for Sensitive Groups" if pred <= 150 else
                "Unhealthy"
            )
        })
    return predictions
