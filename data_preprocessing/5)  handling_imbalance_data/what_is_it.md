# Module 04: Handling Imbalanced Data

## 1. What is Class Imbalance & Why Does It Break Models?
Class imbalance occurs when one class (the majority class, e.g., legitimate transactions) drastically outnumbers the other (the minority class, e.g., fraud, rare diseases, customer churn).

When trained on severely imbalanced datasets (e.g., 99% Class 0 vs. 1% Class 1):
- The Accuracy Paradox: A naive model that predicts 0 for every single instance achieves 99% accuracy, yet fails completely at detecting the actual target class.
- Loss Function Bias: Standard loss functions (like binary cross-entropy) minimize the total overall error. Since the majority class dominates the loss calculation, gradient updates heavily favor predicting the majority class.
- Metric Failure: Standard accuracy becomes meaningless; models must instead be evaluated using Precision-Recall curves, F1-Score, PR-AUC, and Confusion Matrices.

## 2. Folder structure
```
04_imbalanced_data/
├── README.md                                         <-- Root module guide & decision matrix
├── 1) class_weight_tuning/
│   └── cost_sensitive_learning.ipynb                 <-- Notebook 1: class_weight='balanced'
├── 2) resampling_techniques/
│   └── random_under_over_sampling.ipynb              <-- Notebook 2: Random Under/Over-sampling
├── 3) synthetic_sampling/
│   └── smote_and_adasyn.ipynb                        <-- Notebook 3: SMOTE & ADASYN (imblearn)
└── 4) imbalanced_pipelines/
    └── imblearn_pipeline_evaluation.ipynb            <-- Notebook 4: End-to-End Leak-Free Pipeline
```

## 3. High-Level Sub-Concept Breakdown

### A. Cost-Sensitive Learning (class_weight='balanced') — The First Line of Defense
- Modifies the loss function rather than changing the data.
- Heavily penalizes misclassifications on the minority class by assigning weights inversely proportional to class frequencies:

$$w_j = \frac{N}{K \cdot n_j}$$

- Zero data distortion, zero added memory overhead.

### B. Resampling Methods (Under-Sampling vs. Over-Sampling)
- Random Under-Sampling: Deletes majority class instances until balance is achieved. (Fast, but discards potentially valuable training information).
- Random Over-Sampling: Duplicates minority class rows with replacement. (Retains all data, but increases risk of overfitting exact duplicates).

### C. Advanced Synthetic Generation (SMOTE & ADASYN)
- SMOTE (Synthetic Minority Over-sampling Technique): Creates synthetic points along the line segments connecting minority class $k$-nearest neighbors instead of duplicating rows.
- ADASYN (Adaptive Synthetic): Focuses synthetic generation specifically on minority samples that are harder to learn (near decision boundaries).

### D. Imbalance in Pipelines & Proper Metric Evaluation
- Using imblearn.pipeline.Pipeline to ensure resampling is only applied during .fit() on training folds, never on validation/test data.
- Evaluating with Precision, Recall, ROC-AUC vs. PR-AUC for heavy skew.
