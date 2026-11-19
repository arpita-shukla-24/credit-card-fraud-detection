print("Name: Arpita")
print("URN: 2581127")

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve

# Load validation data
X_train, X_valid, y_train, y_valid, preprocessor = joblib.load(
    "results/processed_data.pkl"
)

# Load trained models
logistic_model = joblib.load("results/logistic_model.pkl")
random_forest_model = joblib.load("results/random_forest_model.pkl")

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

# Generate confusion matrices
for name, model in models.items():

    predictions = model.predict(X_valid)

    cm = confusion_matrix(y_valid, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Fraud"],
        yticklabels=["Genuine", "Fraud"]
    )

    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    filename = name.lower().replace(" ", "_")
    plt.savefig("results/" + filename + "_confusion_matrix.png")
    plt.show()

# Generate ROC curves
plt.figure(figsize=(8, 6))

for name, model in models.items():

    probabilities = model.predict_proba(X_valid)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_valid, probabilities)

    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.savefig("results/roc_curve_comparison.png")
plt.show()

print("\nEvaluation graphs saved successfully!")

print("\nCONFUSION MATRICES AND ROC-AUC")

for name, model in models.items():
    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    print("\n", name)
    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, predictions))

    print("ROC-AUC:", roc_auc_score(y_valid, probabilities))
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score    