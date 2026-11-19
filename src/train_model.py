print("Name: Arpita")
print("URN: 2581127")

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Load preprocessed data
X_train, X_valid, y_train, y_valid, preprocessor = joblib.load(
    "results/processed_data.pkl"
)

# Create and train model
model = LogisticRegression(max_iter=1000)

print("\nTraining Logistic Regression model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_valid)
y_prob = model.predict_proba(X_valid)[:, 1]

# Evaluate model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_valid, y_pred))
print("Precision:", precision_score(y_valid, y_pred, zero_division=0))
print("Recall:", recall_score(y_valid, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_valid, y_pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_valid, y_prob))

print("\nClassification Report:")
print(classification_report(y_valid, y_pred, zero_division=0))

# Save model
joblib.dump(model, "results/logistic_model.pkl")

print("\nModel saved successfully!")