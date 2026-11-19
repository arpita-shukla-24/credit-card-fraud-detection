# Project Information

## Project Title
Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project focuses on identifying potentially fraudulent credit card transactions using machine learning techniques. It uses transaction-related information to train classification models that distinguish between genuine and fraudulent transactions.

The project is based on the Kaggle Playground Series competition and uses the provided training and testing datasets.

## Project Objectives
1. Explore and understand the credit card transaction dataset.
2. Perform data preprocessing and prepare the data for model training.
3. Train and evaluate Logistic Regression and Random Forest classifiers.
4. Compare model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
5. Tune classification thresholds to improve fraud detection performance.
6. Generate fraud probability predictions for the test dataset and submit them to Kaggle.
7. Use Git and GitHub to maintain version control, track changes, and document project development.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Development Environment:** Visual Studio Code
- **Version Control:** Git
- **Repository Hosting:** GitHub
- **Dataset Platform:** Kaggle

## Machine Learning Models
The project implements two classification algorithms:

- **Logistic Regression:** Used as a baseline classification model. Different classification thresholds were evaluated, and a threshold of 0.20 achieved the highest F1-score among the tested thresholds.
- **Random Forest:** Used as a second model for comparison with Logistic Regression.

## Project Outcome
The Logistic Regression model with a threshold of 0.20 was selected based on its F1-score of approximately 35.72% on the validation data. The generated test predictions were submitted successfully to the Kaggle competition, achieving a public leaderboard ROC-AUC score of **0.91078**.

## Version Control Implementation
Git was used to initialize the project repository, track project files, create meaningful commits, and manage development branches. GitHub was used to host the repository and maintain the project history.