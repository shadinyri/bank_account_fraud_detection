# Privacy-Preserving Fraud Detection on Highly Imbalanced Tabular Data
Author: Shadi Nayyeri 

## 📌 Project Overview
This repository contains a complete machine learning pipeline designed to detect fraudulent bank account opening applications while managing the severe class imbalance inherent to financial data. The project explores the critical trade-off between maximizing fraud detection (Recall) and minimizing the operational costs of false alarms (False Positive Rate) in a realistic banking scenario.

## 📊 Dataset Information

The data used in this project is the Bank Account Fraud (BAF) Dataset Suite, created by researchers at Feedzai and Universidade do Porto.   

The suite was created to evaluate machine learning methods and fair ML interventions under dynamic conditions and extreme scenarios.   
It consists of 6 dataset variants, each containing 1 million synthetic, feature-engineered tabular instances.   
The data was generated using a Conditional Generative Adversarial Network (CTGAN) trained on an anonymized, real-world bank account opening fraud dataset.   
To ensure strict differential privacy, several interventions were applied before GAN training, including the addition of Laplacian noise and the categorization of continuous personal information like age and income.   
The target variable is fraud_bool, where a positive value (1) represents a fraudulent application and a negative value (0) represents a legitimate one.   
The dataset spans eight months of applications; the standard evaluation split utilizes the first six months for training and the final two months for validation and testing.   
Disclaimer: As noted by the dataset creators, models trained on this data should be used exclusively for ML experimentation and not deployed directly for real-world bank account opening fraud detection, as real-world patterns are highly dynamic.   

## 🧠 Model Architecture & Methodology

The core predictive engine is built using LightGBM, a highly efficient gradient boosting framework. Because the dataset features a strict 1:99 fraud-to-legitimate ratio and mathematical noise introduced for privacy preservation, traditional default hyperparameters fail to capture the underlying patterns without severe overfitting.
To overcome this, Optuna was utilized for Bayesian hyperparameter optimization. The search space was strictly controlled (e.g., constraining max_depth to 3 and optimizing n_estimators) to extract the absolute maximum predictive power from the noisy data without memorizing the training set.

## 📈 Key Findings & Business Impact

The evaluation phase highlighted the critical difference between raw mathematical optimization and applied business logic. By adjusting the decision threshold, the model demonstrates high flexibility depending on the institution's risk appetite:
The Aggressive Strategy (FPR = 5%): Aligning with the BAF paper's evaluation metric, the threshold was set to 0.0437 to allow a 5% False Positive Rate. At this threshold, the model achieved a 54.45% Recall, placing it in the top tier of competitive models for this specific dataset. While this captures more than half of all fraudulent activity, it requires reviewing 5% of all legitimate customers.

The Balanced Strategy (Optimal F1-Score): Optimizing purely for the F1-Score resulted in a stricter threshold of 0.1030. This constrained the False Positive Rate to a mere 1.57%, significantly reducing customer friction and call center load, while still maintaining a robust 33.25% Recall.

## ⚖️ License and Acknowledgements
The BAF Dataset Suite utilized in this project is licensed under the Creative Commons CC BY-NC-ND 4.0 license. Credit to Feedzai and the Universidade do Porto for publishing this essential resource for algorithmic fairness and robust machine learning research.
