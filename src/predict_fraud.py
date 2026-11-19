print("Name: Arpita")
print("URN: 2581127")

import pandas as pd
import joblib

# Load test dataset
test_data = pd.read_csv("data/test.csv")

# Remove ID column before prediction
X_test = test_data.drop(columns=["id"])

# Load trained model and preprocessor
model = joblib.load("results/logistic_model.pkl")
_, _, _, _, preprocessor = joblib.load(
    "results/processed_data.pkl"
)

# Apply preprocessing
X_test_processed = preprocessor.transform(X_test)

# Predict fraud probabilities
fraud_probabilities = model.predict_proba(
    X_test_processed
)[:, 1]

# Create submission file
submission = pd.DataFrame({
    "id": test_data["id"],
    "is_fraud": fraud_probabilities
})

submission.to_csv(
    "results/submission.csv",
    index=False
)

print("\nTest predictions completed!")
print("Total predictions:", len(submission))
print("\nFirst five predictions:")
print(submission.head())

print("\nSubmission file saved in results/submission.csv")