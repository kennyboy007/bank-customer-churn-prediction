# Bank Customer Churn Prediction

## 📌 Project Overview

Customer churn is one of the biggest revenue risks in retail banking.  
This project builds an end-to-end machine learning pipeline to predict customer churn and translate model outputs into actionable retention strategies.

Rather than focusing only on accuracy, this project emphasizes:

- Class imbalance handling
- Model comparison
- ROC-AUC evaluation
- Threshold optimization
- Business interpretation of results

The objective is not just to predict churn — but to support smarter decision-making.

---

## 📊 Dataset

The dataset contains 10,000 bank customers with features such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Activity Status
- Estimated Salary
- Churn Status (Target Variable)

Target Variable:
- `Exited = 1` → Customer churned  
- `Exited = 0` → Customer retained  

Approximately 20% of customers in the dataset churned, creating a class imbalance problem.

---

## 🔍 Key Insights from EDA

- Customers in Germany show significantly higher churn rates.
- Inactive customers are substantially more likely to churn.
- Customers holding exactly 2 products have the lowest churn rate.
- Older customers display higher churn tendency.
- Higher account balances are associated with elevated churn risk.

These findings guided modeling and business recommendations.

---

## ⚙️ Models Implemented

Three classification models were trained and compared:

1. Logistic Regression (Baseline)
2. Random Forest
3. XGBoost (Final Model)

Class imbalance was addressed using:
- `class_weight='balanced'` in Logistic Regression

Models were evaluated using:
- Accuracy
- Precision
- Recall (Churn)
- F1-Score
- ROC-AUC

---

## 🏆 Final Model: XGBoost

XGBoost delivered the strongest overall performance.

**Performance:**
- Accuracy ≈ 87%
- ROC-AUC ≈ 0.87
- Improved recall for churn detection

To improve churn capture, probability threshold was adjusted from 0.5 to 0.3, increasing recall while maintaining reasonable precision.

This reflects a business-driven optimization where missing a churned customer is more costly than a false alarm.

---

## 📈 Feature Importance

Top drivers of churn:

- Number of Products
- Activity Status
- Age
- Geography (Germany)
- Gender

These insights provide direct guidance for retention strategies.

---

## 🧠 Business Recommendations

Based on model findings:

- Launch engagement campaigns targeting inactive customers.
- Prioritize churn prevention in Germany.
- Focus retention efforts on customers with only 1 product.
- Offer age-tailored financial products for customers above 40.
- Deploy probability-based targeting using optimized threshold scoring.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

---

## 🚀 What This Project Demonstrates

- End-to-end ML workflow implementation
- Handling imbalanced classification problems
- Comparative model evaluation
- Threshold optimization for business impact
- Translating model outputs into strategy

---

## 📂 Project Structure

```
Bank_Churn_Project/
│
├── churn_analysis.ipynb
├── data/
├── images/
└── README.md
```

---

## 📬 Contact

If you would like to discuss this project or potential opportunities, feel free to connect with me.
