print("Name: Arpita")
print("URN: 2581127")

import pandas as pd

# Load both files
submission = pd.read_csv("results/submission.csv")
sample = pd.read_csv("data/sample_submission.csv")
test = pd.read_csv("data/test.csv")

print("\nSample Submission Columns:")
print(sample.columns.tolist())

print("\nGenerated Submission Columns:")
print(submission.columns.tolist())

print("\nNumber of test records:", len(test))
print("Number of submission records:", len(submission))

# Verify submission format
if submission.columns.tolist() == sample.columns.tolist():
    print("\nColumn format: Correct")
else:
    print("\nColumn format: Incorrect")

# Verify IDs
if submission["id"].equals(test["id"]):
    print("Transaction IDs: Correct")
else:
    print("Transaction IDs: Incorrect")

# Check prediction values
if submission["is_fraud"].between(0, 1).all():
    print("Prediction probabilities: Valid")
else:
    print("Prediction probabilities: Invalid")

print("\nSubmission verification completed!")
