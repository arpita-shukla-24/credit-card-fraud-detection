# Model Evaluation Results

## 1. Model Comparison

Two machine learning algorithms were implemented and evaluated:

- Logistic Regression
- Random Forest Classifier

Both models were evaluated on the validation dataset.

## 2. Logistic Regression

The Logistic Regression model achieved the following results at a classification threshold of 0.20:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 98.03% |
| Precision | 39.19% |
| Recall | 32.81% |
| F1-Score | 35.72% |
| ROC-AUC | 90.02% |

## 3. Random Forest

At the default classification threshold of 0.50, Random Forest achieved:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 89.42% |
| Precision | 10.38% |
| Recall | 69.89% |
| F1-Score | 18.07% |
| ROC-AUC | 89.81% |

## 4. Threshold Optimization

Different classification thresholds were tested for both models.

For Logistic Regression, a threshold of 0.20 produced the highest F1-score among the tested thresholds.

This threshold provided a better balance between precision and recall compared with the default threshold of 0.50.

## 5. Final Model Selection

Logistic Regression was selected as the final model because it achieved the higher F1-score after threshold tuning.

The final validation F1-score was 35.72%, with a ROC-AUC of approximately 90.02%.

## 6. Kaggle Competition Result

The final model generated predictions for 299,984 test transactions.

The submission achieved a public leaderboard ROC-AUC score of **0.91078** in the Kaggle Playground Series competition.