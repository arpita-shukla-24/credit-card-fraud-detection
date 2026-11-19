# Machine Learning Methodology

## 1. Dataset Collection
The project uses the Credit Card Fraud Detection dataset from the Kaggle Playground Series competition. The training dataset contains 699,960 transaction records, while the test dataset contains 299,984 records.

## 2. Exploratory Data Analysis
Exploratory Data Analysis (EDA) was performed to understand the dataset, examine transaction features, identify class imbalance, and visualize the distribution of genuine and fraudulent transactions.

The training dataset contains:
- Genuine transactions: 688,271
- Fraudulent transactions: 11,689
- Fraud percentage: Approximately 1.67%

## 3. Data Preprocessing
The dataset was prepared for machine learning by processing transaction features and separating input features from the target variable, `is_fraud`.

## 4. Model Training
Two machine learning algorithms were implemented:

- Logistic Regression
- Random Forest Classifier

Both models were trained using the prepared transaction data.

## 5. Model Evaluation
The models were evaluated using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.

Because the dataset is imbalanced, precision, recall, and F1-score were also considered when comparing model performance.

## 6. Threshold Tuning
Different classification thresholds were tested to improve the balance between precision and recall.

Logistic Regression achieved its best tested F1-score at a threshold of 0.20.

## 7. Final Model Selection
Logistic Regression was selected as the final model based on its best tested F1-score of 35.72% at the selected threshold.

Its validation ROC-AUC was approximately 90.02%.

## 8. Kaggle Submission
The selected model generated fraud probabilities for 299,984 test transactions. The predictions were saved in CSV format and submitted to the Kaggle competition.

The submission achieved a public leaderboard ROC-AUC score of 0.91078.