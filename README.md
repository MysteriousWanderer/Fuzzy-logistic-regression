# Fuzzy-logistic-regression

Official implementation of the paper **"A fuzzy logistic-based ensemble framework with heterogeneous weighting effects for credit risk evaluation"** (Applied Soft Computing, 2026).

## 📌 Overview

Credit scoring faces three critical challenges: **class imbalance**, **data fuzziness** (subjectivity, uncertainty, fuzzy variable relationships), and **model interpretability**. This repository provides a novel ensemble model – **Fuzzy logistic‑FWE** – that integrates **fuzzy logistic regression** with a **fuzzy heterogeneous weighting** strategy to address all three issues simultaneously.

## ✨ Key Features

- 🔹 **Fuzzy Logistic Classifier** – crisp inputs, fuzzy coefficients, and triangular‑fuzzy outputs.
- 🔹 **Ensemble with SMOTE‑DSR** – creates diverse sub‑datasets to tackle class imbalance.
- 🔹 **Dynamic Fuzzy Heterogeneous Weighting** – each sub‑model’s vote is weighted by either Sensitivity or (1‑Specificity) depending on the prediction tendency.
- 🔹 **Strong Interpretability** – linear traceability inherited from logistic regression, enhanced with SHAP.
- 🔹 **Statistically Validated** – Wilcoxon signed‑rank tests confirm significant improvement in Sensitivity and F1‑Score.
- 🔹 **Reproducible** – all experiments run with 3 random seeds × 5‑fold cross‑validation.

### Workflow

1. **Data Preprocessing**  
   - Impute missing values (mean for continuous, mode for discrete).  
   - Min‑Max normalization.  
   - Fuzzify binary labels into triangular fuzzy numbers (TFNs) using the defined fuzzification rule.

2. **Training Phase**  
   - Apply **SMOTE‑DSR** to generate `m` sub‑training sets with increasing minority‑class ratios.  
   - Train a **fuzzy logistic classifier** on each subset (gradient descent minimisation of Diamond’s distance).

3. **Validation Phase**  
   - Evaluate Sensitivity and Specificity of each sub‑model on an independent validation set.

4. **Testing Phase**  
   - Each sub‑model outputs a fuzzy default possibility (TFN).  
   - **Fuzzy heterogeneous weighting** assigns weights based on whether the vertex exceeds a threshold.  
   - Aggregate all weighted fuzzy predictions and defuzzify (centre of gravity) to obtain final crisp possibility.
