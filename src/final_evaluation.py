print("Name: Arpita")
print("URN: 2581127")

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load validation data
_, X_valid, _, y_valid, _ = joblib.load(
    "results/processed_data.pkl"
)

# Load Logistic Regression model
model = joblib.load("results/logistic_model.pkl")

# Use improved threshold
threshold = 0.20

probabilities = model.predict_proba(X_valid)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print("\nFINAL MODEL EVALUATION")
print("Threshold:", threshold)

print("\nAccuracy:", accuracy_score(y_valid, predictions))
print("Precision:", precision_score(y_valid, predictions))
print("Recall:", recall_score(y_valid, predictions))
print("F1 Score:", f1_score(y_valid, predictions))
print("ROC-AUC:", roc_auc_score(y_valid, probabilities))

print("\nClassification Report:")
print(classification_report(y_valid, predictions))

# Generate confusion matrix
cm = confusion_matrix(y_valid, predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Fraud"],
    yticklabels=["Genuine", "Fraud"]
)

plt.title("Final Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("results/final_confusion_matrix.png")
plt.show()

print("\nFinal evaluation completed!")