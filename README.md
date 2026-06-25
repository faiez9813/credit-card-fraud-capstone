# Credit Card Fraud Detection — Capstone Project

A capstone project that brings together the full classical ML workflow: handling extreme class imbalance with SMOTE, building a proper preprocessing-and-modeling pipeline, and systematically comparing four different algorithms on a large, real-world dataset.

## Overview

This project detects fraudulent credit card transactions, using anonymized transaction data where the original features have been transformed via PCA for privacy. It was built as a capstone to demonstrate end-to-end pipeline design and model comparison rather than focusing on a single algorithm.

## Dataset

- Rows: 284,807 transactions
- Features: 30 (Time, Amount, and 28 PCA-transformed features V1-V28)
- Target: Class (1 = Fraud, 0 = Normal)
- Class balance: 99.83% Normal, 0.17% Fraud — one of the most extreme imbalances encountered in this portfolio

## Workflow

1. Exploratory Data Analysis
   - Confirmed the extreme class imbalance, visualized on a log scale since the fraud class would otherwise be nearly invisible
   - Found that fraud transactions have a lower median amount than normal transactions ($9.25 vs $22.00) despite a higher mean, suggesting fraudulent activity includes both small "test" transactions and occasional large ones

2. Baseline Model
   - Trained Logistic Regression with no imbalance handling
   - A convergence warning revealed that Time and Amount, being on a very different scale from the PCA features, were affecting optimization; adding StandardScaler resolved this

3. SMOTE Tuning
   - Testing a full 50/50 SMOTE balance produced strong recall (0.92) but precision collapsed to 0.06, making the model impractical due to excessive false alarms
   - Several intermediate sampling ratios were tested systematically; a sampling_strategy of 0.05 (roughly a 1:20 minority-to-majority ratio) gave the best precision-recall balance

4. Pipeline Construction
   - Combined StandardScaler, SMOTE, and the classifier into a single imbalanced-learn Pipeline
   - This ensures scaling is always fit only on training data and that SMOTE is automatically skipped at prediction time, removing the risk of manually misapplying either step

5. Model Comparison
   - Compared Logistic Regression, Decision Tree, Random Forest, and XGBoost under identical pipeline conditions

## Results

| Model | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.570 | 0.878 | 0.691 | 0.967 |
| Decision Tree | 0.575 | 0.857 | 0.689 | 0.945 |
| **Random Forest** | **0.761** | 0.878 | **0.815** | 0.971 |
| XGBoost | 0.621 | 0.837 | 0.713 | 0.974 |

**Random Forest was selected as the best model.** XGBoost had a marginally higher AUC, but Random Forest had a substantially better F1-score and precision with comparable recall, making it more practically useful by producing far fewer false alarms without meaningfully sacrificing fraud detection.

## Key Takeaways

- Fully balancing an extremely imbalanced dataset with SMOTE is not necessarily the right choice; the sampling ratio should be tuned to the actual precision-recall trade-off needed, not maximized by default
- The single metric with the highest value (AUC, in this case) is not automatically the best choice for model selection; F1-score and the precision-recall balance gave a more complete picture for this problem
- Wrapping scaling, resampling, and modeling in a single pipeline removes an entire category of data leakage mistakes that are easy to make when each step is applied manually

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, StandardScaler, Pipeline)
- XGBoost
- imbalanced-learn (SMOTE, Pipeline)
- Matplotlib, Seaborn

## Files

- credit_card-fraud_capstone.ipynb: full analysis, pipeline construction, and model comparison notebook

## How to Run

1. Clone this repository
2. Download the Credit Card Fraud Detection dataset from Kaggle (Machine Learning Group - ULB)
3. Open the notebook in Jupyter Notebook or Google Colab and update the dataset path
4. Run all cells
