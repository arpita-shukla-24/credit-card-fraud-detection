# Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It uses transaction-related features to classify transactions and identify potentially fraudulent activity.

## Dataset
The dataset is from the Kaggle Playground Series competition.

- Training records: 699,960
- Testing records: 299,984
- Fraud cases in training data: 11,689
- Target variable: `is_fraud`

Dataset link: https://www.kaggle.com/competitions/playground-series-s6e11

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Git and GitHub

## Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier

Logistic Regression was selected as the final model after comparing F1-scores and tuning the classification threshold.

## Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 98.03% |
| Precision | 39.19% |
| Recall | 32.81% |
| F1-Score | 35.72% |
| ROC-AUC | 90.02% |

Kaggle Public Leaderboard Score: **0.91078**

## Project Structure

- `data/` – Dataset files (download separately from Kaggle)
- `src/` – Python source code
- `notebooks/` – Jupyter notebooks
- `results/` – Graphs and evaluation results
- `requirements.txt` – Project dependencies

## How to Run

1. Clone this repository.
2. Install the required dependencies:

   `pip install -r requirements.txt`

3. Download the competition dataset from Kaggle.
4. Place the CSV files inside the `data/` folder.
5. Run the Python scripts from the `src/` folder.

## Version Control
Git and GitHub are used to track project development, maintain commit history, and manage project versions.