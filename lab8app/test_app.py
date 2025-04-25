import requests
import json

test_data = {
    "Age": 50,
    "Gender": 1,
    "BloodPressure": 120,
    "Cholesterol": 200,
    "HeartRate": 80,
    "QuantumPatternFeature": 0.5
}

response = requests.post(
    "http://localhost:8000/predict",
    headers={"Content-Type": "application/json"},
    json=test_data  # Changed from data=json.dumps() to json=
)

print("Status code:", response.status_code)
print("Response body:", response.json())