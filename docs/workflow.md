# Project Execution Workflow

The Credit Card Fraud Detection project follows a systematic machine learning workflow.

## Workflow Steps

### 1. Dataset Input
The training and testing datasets are obtained from the Kaggle Playground Series competition.

### 2. Data Exploration
The dataset is examined to understand its structure, features, class distribution, and potential patterns in fraudulent transactions.

### 3. Data Preprocessing
The transaction data is prepared for model training. Input features and the target variable (`is_fraud`) are separated.

### 4. Model Training
Two machine learning models are implemented:
- Logistic Regression
- Random Forest Classifier

### 5. Model Evaluation
Both models are evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. Confusion matrices and ROC curves are also used for comparison.

### 6. Threshold Optimization
Different classification thresholds are tested to identify a better balance between precision and recall.

### 7. Final Model Selection
Logistic Regression is selected based on its best tested F1-score of 35.72% at a threshold of 0.20.

### 8. Prediction Generation
The selected model generates fraud probabilities for the test dataset.

### 9. Kaggle Submission
The predictions are exported to a CSV file and submitted to Kaggle. The submission achieves a public leaderboard score of 0.91078.

## Workflow Summary

Dataset Collection → Data Exploration → Preprocessing → Model Training → Evaluation → Threshold Tuning → Final Model Selection → Prediction → Kaggle Submission