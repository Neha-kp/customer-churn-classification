# Predicting Customer Churn for a SaaS Product

## Project Overview
This project uses machine learning to predict whether a customer is likely to churn (cancel their SaaS subscription) based on account information, product usage, and support activity.

The goal is to help customer-success teams identify potentially at-risk customers and take preventive retention actions.

## Business Problem
Customer churn can reduce recurring revenue and increase customer acquisition costs. Identifying customers who may leave can help businesses prioritize support and retention efforts.

## Dataset
The project uses five datasets:
- `account_master.csv`
- `usage_logs.csv`
- `support_tickets.csv`
- `churned_labeled.csv`
- `churned_to_predict.csv`

The labelled dataset contains 378 customers:
- **No churn:** 270 (71.4%)
- **Churn:** 108 (28.6%)

## Tools and Libraries
- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

## Methodology
1. Cleaned the data and removed invalid negative login values.
2. Examined and capped upper outliers using the IQR method.
3. Handled missing values and represented missing categories as `Unknown`.
4. Aggregated usage and support records into customer-level features.
5. Encoded categorical features and scaled numerical features.
6. Split the data into training and testing sets using stratification.
7. Compared Logistic Regression, Decision Tree, Random Forest, and SVM models.
8. Used GridSearchCV with 5-fold cross-validation to tune model parameters, using the F1-score for the churn class.

## Final Model and Results
The tuned Random Forest was selected based on its test F1-score and ROC-AUC among the tuned models.

| Metric | Test result |
|---|---:|
| Accuracy | 64.5% |
| Precision (Churn) | 42.9% |
| Recall (Churn) | 68.2% |
| F1-score (Churn) | 52.6% |
| ROC-AUC | 69.1% |

The model correctly identified 15 of the 22 actual churn customers in the test set. However, it also incorrectly flagged 20 non-churning customers as potential churners.

## Key Learning
Accuracy alone can be misleading when the target classes are imbalanced. Precision, recall, and F1-score help evaluate how well a model identifies customers who actually churn.

## Limitations
- The labelled dataset is relatively small, with 378 customers.
- Churn precision is relatively low, meaning some flagged customers will not actually churn.
- Results are based on one held-out test set and need further validation before production use.

## Future Improvements
- Collect more labelled customer data.
- Engineer features that capture changes in customer engagement over time.
- Explore probability calibration and decision-threshold selection.
- Validate the model on additional or time-based data.

## Author
Neha KP

