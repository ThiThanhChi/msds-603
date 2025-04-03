import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    
    # Standardize numerical features
    num_features = ['Age', 'BloodPressure', 'Cholesterol', 'HeartRate', 'QuantumPatternFeature']
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data("data/heart_dataset.csv", "data/processed_heart_dataset.csv")
