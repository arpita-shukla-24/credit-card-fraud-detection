print("Name: Arpita")
print("URN: 2581127")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset
data = pd.read_csv("data/train.csv")

# Separate features and target
X = data.drop(columns=["is_fraud", "id"])
y = data["is_fraud"]

# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["object"]).columns

# Split dataset
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# Fit on training data and transform validation data
X_train_processed = preprocessor.fit_transform(X_train)
X_valid_processed = preprocessor.transform(X_valid)

# Save processed data
joblib.dump(
    (X_train_processed, X_valid_processed, y_train, y_valid, preprocessor),
    "results/processed_data.pkl"
)

print("\nData preprocessing completed successfully!")
print("Training records:", X_train_processed.shape[0])
print("Validation records:", X_valid_processed.shape[0])
print("Number of processed features:", X_train_processed.shape[1])