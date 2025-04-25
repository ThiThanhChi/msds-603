from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Define the correct input model
class PatientData(BaseModel):
    Age: float
    Gender: float
    BloodPressure: float
    Cholesterol: float
    HeartRate: float
    QuantumPatternFeature: float

app = FastAPI()

@app.post("/predict")
async def predict(data: PatientData):
    input_data = np.array([
        data.Age,
        data.Gender,
        data.BloodPressure,
        data.Cholesterol,
        data.HeartRate,
        data.QuantumPatternFeature
    ]).reshape(1, -1)
    
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
