# 🛒 Purchase Value Prediction – Zero-Inflated Regression (Kaggle Competition)

This project aims to predict the **purchase value per transaction** in a dataset with a **high proportion of zero values (zero-inflated distribution)**. The competition focuses on modeling customer purchase behavior where many users make no purchase at all, and a few generate significantly high values.

---

## 📊 Project Overview

The dataset contains transaction-level features for customers, including demographics, behavioral attributes, and historical engagement data.  
The goal is to predict the **continuous purchase value**, which is **zero-inflated** — i.e., many observations have a value of 0 while others follow a skewed positive distribution.

To effectively model this, we explored **three key approaches** and compared their performance:

1. **CatBoost Regressor** – Baseline and tuned gradient boosting model.
2. **Classification + Regression Hybrid** – First classifies zero vs non-zero values, then predicts non-zero purchase values using regression.
3. **Zero-Inflated Regressor** – Explicitly accounts for the zero-inflation structure in the data.
4. **Stacked Ensemble Model** – Combines predictions from the above models for improved accuracy and robustness.

---

## 🧠 Methodology

### 1. Data Preprocessing
- Handled missing values and outliers.
- Log-transformation applied on the target variable:  
  \[
  y_{\text{log}} = \log1p(y)
  \]
- Applied **Target-Guided Ordinal Encoding** for categorical variables.
- Implemented **K-Fold Smoothed Target Encoding** to reduce leakage while preserving predictive power.
- Feature scaling applied selectively (on continuous features).

### 2. Model Architecture
#### 🔹 CatBoost Regressor
- Chosen for its ability to handle categorical variables natively and manage imbalanced data.
- Tuned using Bayesian Optimization for depth, learning rate, and iterations.

#### 🔹 Classification + Regression Hybrid
- Stage 1: Binary classifier predicts zero vs non-zero transactions.
- Stage 2: Regressor predicts purchase amount only for non-zero samples.
- Combines outputs into final predictions.

#### 🔹 Zero-Inflated Regressor
- Models two processes:
  - Zero-generating mechanism (logistic model)
  - Positive value-generating mechanism (regression model)
- Captures real-world data generation structure better than standard regressors.

#### 🔹 Ensemble Stacking
- Base models: CatBoost, RandomForest, LightGBM, and Hybrid model.
- Meta-model: Linear regression (trained on out-of-fold predictions).
- Combines model strengths to improve generalization.

---

## 🧩 Evaluation Strategy

### Metrics:
- **R² Score**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

Performance was evaluated both on:
- Log-transformed scale (`y_log`)
- Original scale (`y_orig = expm1(y_log_pred)`)

### Example:
```python
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

r2 = r2_score(y_true, np.expm1(y_pred_log))
rmse = np.sqrt(mean_squared_error(y_true, np.expm1(y_pred_log)))
