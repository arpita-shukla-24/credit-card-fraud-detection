print("Name: Arpita")
print("URN: 2581127")

import joblib
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

# Load validation data
_, X_valid, _, y_valid, _ = joblib.load(
    "results/processed_data.pkl"
)

# Load trained models
models = {
    "Logistic Regression": joblib.load("results/logistic_model.pkl"),
    "Random Forest": joblib.load("results/random_forest_model.pkl")
}

thresholds = np.arange(0.05, 0.51, 0.05)

best_thresholds = {}

for name, model in models.items():

    probabilities = model.predict_proba(X_valid)[:, 1]

    print("\n", name)
    print("-" * 65)
    print("Threshold\tPrecision\tRecall\t\tF1 Score")

    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_valid, predictions, zero_division=0
        )

        recall = recall_score(
            y_valid, predictions, zero_division=0
        )

        f1 = f1_score(
            y_valid, predictions, zero_division=0
        )

        print(
            f"{threshold:.2f}\t\t"
            f"{precision:.4f}\t\t"
            f"{recall:.4f}\t\t"
            f"{f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    best_thresholds[name] = float(best_threshold)

    print("\nBest Threshold:", best_threshold)
    print("Best F1 Score:", round(best_f1, 4))

# Save selected thresholds
joblib.dump(best_thresholds, "results/best_thresholds.pkl")

print("\nThreshold tuning completed!")