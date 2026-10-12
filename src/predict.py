 
from pathlib import Path

import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
models_dir = PROJECT_ROOT / "models"

# Load saved model
model = joblib.load(models_dir / "house_price_model.pkl")
scaler = joblib.load(models_dir / "scaler.pkl")
selector = joblib.load(models_dir / "selector.pkl")

print("=" * 60)
print("HOUSE PRICE PREDICTION SYSTEM")
print("=" * 60)

# Get input from user
transaction_date = float(input("\nEnter transaction date (Example: 2013.5): "))
house_age = float(input("Enter house age in years: "))
mrt_distance = float(input("Enter distance to nearest MRT station: "))
convenience_stores = int(input("Enter number of convenience stores: "))
latitude = float(input("Enter latitude: "))
longitude = float(input("Enter longitude: "))

# Create input DataFrame
new_property = pd.DataFrame({
    "X1 transaction date": [transaction_date],
    "X2 house age": [house_age],
    "X3 distance to the nearest MRT station": [mrt_distance],
    "X4 number of convenience stores": [convenience_stores],
    "X5 latitude": [latitude],
    "X6 longitude": [longitude]
})

# Scale input
new_property_scaled = scaler.transform(new_property)

# Select important features
new_property_selected = selector.transform(new_property_scaled)

# Predict
predicted_price = model.predict(new_property_selected)

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("\nProperty Details:")
print(f"Transaction Date       : {transaction_date}")
print(f"House Age              : {house_age} years")
print(f"MRT Distance           : {mrt_distance}")
print(f"Convenience Stores     : {convenience_stores}")
print(f"Latitude               : {latitude}")
print(f"Longitude              : {longitude}")

print("\nPredicted House Price per Unit Area:")
print(f"{predicted_price[0]:.2f}")

print("\nPrediction completed successfully!")
 