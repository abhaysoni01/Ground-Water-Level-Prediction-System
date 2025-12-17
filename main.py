import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import load_and_preprocess
from src.synthetic_data import generate_synthetic_data
from src.model import train_model
from src.threshold import adaptive_threshold
from src.evaluation import evaluate

# Load dataset
df = load_and_preprocess(
    "data/Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv"
)
df['synthetic'] = 0

# Generate synthetic data (NOVELTY-3)
synthetic_df = generate_synthetic_data(df)
final_df = pd.concat([df, synthetic_df])

# Train model
model, X_test, y_test = train_model(final_df)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae, rmse = evaluate(y_test, predictions)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Adaptive threshold (NOVELTY-4)
threshold = adaptive_threshold(y_test.values, predictions)
print("Adaptive Critical Threshold:", round(threshold, 2))

# Alert detection
alerts = predictions < threshold

# Visualization
plt.figure(figsize=(12,5))
plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.axhline(threshold, color='red', linestyle='--', label="Adaptive Threshold")
plt.legend()
plt.title("Groundwater Level Prediction (Atal Jal Dataset)")
plt.xlabel("Time")
plt.ylabel("Groundwater Level (mbgl)")
plt.show()
