# Privacy-Preserving Fraud Detection on Highly Imbalanced Tabular Data

Author: Shadi Nayyeri

## 📌 Project Overview

This repository contains a complete machine learning pipeline designed to detect fraudulent bank account opening applications while managing the severe class imbalance inherent to financial data. The project explores the critical trade-off between maximizing fraud detection (Recall) and minimizing the operational costs of false alarms (False Positive Rate) in a realistic banking scenario, and includes an algorithmic fairness audit of the resulting model.

## 📊 Dataset Information

The data used in this project is the Bank Account Fraud (BAF) Dataset Suite, created by researchers at Feedzai and Universidade do Porto.

The suite was created to evaluate machine learning methods and fair ML interventions under dynamic conditions and extreme scenarios.
It consists of 6 dataset variants, each containing 1 million synthetic, feature-engineered tabular instances.
The data was generated using a Conditional Generative Adversarial Network (CTGAN) trained on an anonymized, real-world bank account opening fraud dataset.
To ensure strict differential privacy, several interventions were applied before GAN training, including the addition of Laplacian noise and the categorization of continuous personal information like age and income.
The target variable is `fraud_bool`, where a positive value (1) represents a fraudulent application and a negative value (0) represents a legitimate one.
The dataset spans eight months of applications; this project uses the standard evaluation split: the first six months for training and the final two months for testing.

Disclaimer: As noted by the dataset creators, models trained on this data should be used exclusively for ML experimentation and not deployed directly for real-world bank account opening fraud detection, as real-world patterns are highly dynamic.

## 🧠 Model Architecture & Methodology

The core predictive engine is built using LightGBM, a highly efficient gradient boosting framework. Because the dataset features a strict 1:99 fraud-to-legitimate ratio and mathematical noise introduced for privacy preservation, traditional default hyperparameters fail to capture the underlying patterns without severe overfitting.

To overcome this, Optuna was used for Bayesian hyperparameter optimization via 3-fold stratified cross-validation, maximizing ROC-AUC. The search space was constrained (e.g., `max_depth` limited to 3–10) to extract strong predictive power from the noisy data without memorizing the training set.

## 📈 Key Findings & Business Impact

The evaluation phase highlighted the difference between raw mathematical optimization and applied business logic. By adjusting the decision threshold, the model demonstrates clear flexibility depending on the institution's risk appetite:

- **The Aggressive Strategy (FPR = 5%):** Aligning with the BAF paper's own evaluation metric, the threshold was set to 0.0442 to hold the False Positive Rate at 5%. At this threshold, the model achieves **54.79% Recall** — catching more than half of all fraudulent applications while flagging 5% of legitimate ones for review.
- **The Balanced Strategy (Optimal F1-Score):** Optimizing for F1-Score instead produces a stricter threshold of 0.1226. This reduces the False Positive Rate to **1.25%**, substantially lowering customer friction and manual-review load, while still catching **30.02% Recall** of fraud.

## ⚖️ Fairness & Governance Audit

Beyond raw performance, the model was audited for disparate impact across three protected attributes defined by the BAF paper — age, income, and employment stability — using **Predictive Equality** (the ratio of False Positive Rates between groups) at the 5% global FPR threshold:

| Protected Attribute | FPR (Group=1) | FPR (Group=0) | FPR Ratio |
|---|---|---|---|
| Age ≥ 50 (`is_older`) | 0.0978 | 0.0405 | **2.41** |
| Income < 0.5 (`is_low_income`) | 0.0208 | 0.0611 | 0.34 |
| Unstable employment (`is_unemployed`) | 0.0055 | 0.0539 | 0.10 |

**The headline finding:** applicants aged 50+ are falsely flagged as fraudulent at roughly **2.4x the rate** of younger applicants — a meaningful disparate impact that would warrant mitigation (e.g., group-specific thresholds or reweighting) before any real-world use.

*Methodology note:* the employment-stability attribute required mapping the dataset's anonymized categories to the paper's definition. An initial unverified assumption about category meaning produced a misleading result; this was corrected by inspecting the actual category values directly and aligning the mapping with the paper's stated rule, which reversed the finding's direction. This is a useful reminder that fairness audits are only as trustworthy as the definitions behind them.

## ⚖️ License and Acknowledgements

The BAF Dataset Suite used in this project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Credit to Feedzai and Universidade do Porto (Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS 2022) for publishing this resource. Used here for non-commercial, educational purposes; the raw dataset itself is not redistributed in this repository.
