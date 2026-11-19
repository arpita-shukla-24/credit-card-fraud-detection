print("Name: Arpita")
print("URN: 2581127")

import pandas as pd

# Load the training dataset
data = pd.read_csv("data/train.csv")

# Display the first five rows
print("\nFirst five rows:")
print(data.head())

# Display dataset information
print("\nDataset information:")
print(data.info())

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check the number of fraud and genuine transactions
print("\nTransaction distribution:")
print(data["is_fraud"].value_counts())