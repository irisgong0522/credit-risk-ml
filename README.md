# Credit Risk Prediction Using Machine Learning

A comparative machine learning study of credit default prediction using five supervised classification models, with a particular focus on XGBoost optimisation, model evaluation, and interpretability.

## 1. Project Overview

This project investigates the application of machine learning to credit risk assessment using the Home Credit Default Risk dataset.

Five classification algorithms are implemented and evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Following initial model comparison, XGBoost is selected for further hyperparameter optimisation using Optuna, followed by early stopping and evaluation on a held-out test dataset.

The project also examines class imbalance, classification threshold selection, feature importance, and SHAP-based model interpretability.

## 2. Dataset

**Dataset:** Home Credit Default Risk

The dataset contains **307,511 labelled loan applications**, including financial, demographic, and credit-related features.

The binary target variable indicates whether an applicant experiences repayment difficulties:

- `TARGET = 0`: No repayment difficulties
- `TARGET = 1`: Repayment difficulties

The dataset is substantially imbalanced, with approximately **8.07% positive observations**.

A stratified split is used to preserve class proportions:

| Dataset | Proportion |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

## 3. Data Preprocessing

A preprocessing pipeline is constructed to handle numerical and categorical features.

**Numerical features**
- Median imputation for missing values
- Feature standardisation for Logistic Regression

**Categorical features**
- Missing-value imputation using a dedicated category
- One-hot encoding
- Handling of previously unseen categories

Preprocessing parameters are fitted exclusively on the relevant training partition to prevent data leakage.

## 4. Machine Learning Models

Five supervised classification models are trained and evaluated using ROC-AUC.

### Initial Model Comparison

| Model | Training ROC-AUC | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.7480 | 0.7433 |
| Random Forest | 0.7961 | 0.7358 |
| XGBoost | 0.7829 | 0.7573 |
| CatBoost | 0.7704 | 0.7561 |
| LightGBM | 0.8183 | **0.7582** |

LightGBM achieved the highest initial validation ROC-AUC, while Random Forest demonstrated a relatively large training–validation gap.

XGBoost was subsequently selected for further optimisation.

## 5. XGBoost Hyperparameter Optimisation

XGBoost hyperparameters are optimised using **Optuna's Tree-structured Parzen Estimator (TPE)** with five-fold stratified cross-validation.

The tuning process explores:

- Maximum tree depth
- Learning rate
- Minimum child weight
- Row and column subsampling
- L1 and L2 regularisation
- Number of boosting iterations

The subsequent 20-trial optimisation study achieved a best mean cross-validation ROC-AUC of **0.7566**.

### Early Stopping

An internal stratified fitting and monitoring split is used for early stopping.

The procedure selects **685 boosting rounds**, after which the final model is refitted on the full training partition.

The resulting validation ROC-AUC increases to **0.7605**.

## 6. Model Evaluation

The final XGBoost model is evaluated using:

- ROC-AUC
- Precision–Recall curves and Average Precision
- Precision, Recall, F1-score and Accuracy
- Confusion matrices
- Classification threshold analysis

### Final XGBoost Performance

| Metric | Validation | Test |
|---|---:|---:|
| ROC-AUC | **0.7605** | **0.7602** |
| Average Precision | 0.2447 | 0.2414 |
| Precision (threshold = 0.2) | 0.2853 | 0.2834 |
| Recall (threshold = 0.2) | 0.2986 | 0.2925 |
| F1-score (threshold = 0.2) | 0.2918 | 0.2879 |

The similarity between validation and test ROC-AUC suggests stable predictive performance under the recorded random-split evaluation.

However, classification performance remains limited by the substantial class imbalance and the trade-off between false positives and false negatives.

### Classification Threshold Analysis

Three probability thresholds are compared: **0.1, 0.2 and 0.5**.

Lower thresholds improve recall but increase false positives, whereas higher thresholds improve precision at the expense of detecting fewer positive cases.

Among the three evaluated thresholds, **0.2 achieves the highest validation F1-score**.

This comparison is descriptive and does not constitute global threshold optimisation.

## 7. Feature Importance and SHAP Analysis

Model interpretability is investigated using two approaches.

**XGBoost Feature Importance**

Gain-based feature importance identifies predictors that contribute strongly to improvements in the model's training objective.

**SHAP Analysis**

SHAP values are calculated for a sample of 1,000 validation observations using TreeExplainer.

The analysis identifies external credit risk scores, particularly `EXT_SOURCE_2` and `EXT_SOURCE_3`, as influential predictors.

Higher external risk scores generally correspond to lower predicted repayment-risk log-odds.

These findings describe predictive relationships within the fitted model and should not be interpreted as causal effects.

## 8. Limitations

Several limitations remain:

- Significant class imbalance and substantial classification errors
- Unequal optimisation budgets across the five machine learning models
- Limited classification threshold exploration
- Missing uncertainty estimates for model comparisons
- Differences between historical tuning outputs and current source code that constrain exact reproducibility
- Random-split evaluation rather than temporal or external validation

Consequently, the recorded results do not establish the statistical superiority of XGBoost over all alternative algorithms.

## 9. Future Improvements

Potential extensions include:

- More systematic threshold optimisation using application-specific error costs
- Probability calibration and cost-sensitive learning
- Broader and more consistent hyperparameter optimisation across models
- Evaluation with repeated cross-validation and uncertainty estimates
- Temporal or external validation
- Further investigation of model interpretability and fairness

## 10. Technologies

**Language:** Python

**Machine Learning:** scikit-learn, XGBoost, LightGBM, CatBoost

**Optimisation:** Optuna

**Data Processing:** pandas, NumPy

**Interpretability:** SHAP

**Visualisation:** Matplotlib, Seaborn

---

**Author:** Iris Gong

**Project Report:** *Credit Default Prediction using Gradient Boosting Models* (November 2026)
