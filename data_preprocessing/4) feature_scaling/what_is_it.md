# Module 03: Feature Scaling (Standardization & Normalization)

## 1. What is Feature Scaling?
Feature Scaling is a data preprocessing technique used to standardize the independent numeric variables (features) of a dataset into a comparable magnitude or common range. 

In raw data, numerical columns often exist on radically different scales:
* **Age:** 18 to 65 (scale of tens)
* **Annual Salary:** $25,000 to $200,000 (scale of thousands)
* **Transaction Amount:** $5 to $10,000,000 (scale of millions)

Feature scaling transforms these raw numbers so that algorithms treat features based on their true statistical variance rather than their arbitrary measurement units.

---

## 2. Why Do We Scale Features?

### A. Preventing Distance Distortion (KNN, SVM, K-Means)
Algorithms that rely on Euclidean distance:
$$d(p, q) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$
If one feature is `Salary` ($\Delta = 50,000$) and another is `Age` ($\Delta = 10$), the squared salary difference completely overwhelms the age difference, effectively blinding the model to age.

### B. Accelerating Gradient Descent Convergence (Neural Nets, Logistic Regression)
When features have vastly different scales, the loss surface resembles an elongated, steep valley. Gradient Descent bounces back and forth inefficiently. Scaling creates symmetric, spherical contours, enabling direct and fast convergence toward the global minimum.

### C. Fair Regularization Penalties (Ridge, Lasso, ElasticNet)
$L_1$ and $L_2$ penalties penalize large weight coefficients ($w$). If features are unscaled, smaller-magnitude features naturally require much larger weights to impact predictions, causing regularization to penalize them disproportionately and unfairly.

---

## 3. Which Models Require Scaling vs. Which Ignore It?

```text
Do I Need to Scale My Numerical Features?
│
├── Distance-Based Models (KNN, K-Means, SVM, PCA)
│     └──> REQUIRED (Prevents large-scale features from dominating distances)
│
├── Gradient-Based & Linear Models (Linear/Logistic Regression, Neural Nets, Ridge/Lasso)
│     └──> REQUIRED (Ensures stable optimization and fair regularization)
│
└── Tree-Based Models (Decision Trees, Random Forest, XGBoost, LightGBM, CatBoost)
      └──> NOT REQUIRED / INVARIANT
           Reason: Trees split one feature at a time via step-functions (e.g., Age > 30).
           Monotonic scaling does not change the order or location of optimal split thresholds.
```

## 4) feature_scaling/

```
├── README.md                                  <-- You are here
├── 1) standardization/
│   └── standard_scaler.ipynb                 <-- Z-score scaling (mean=0, std=1)
├── 2) normalization_and_minmax/
│   └── minmax_and_robust.ipynb               <-- Fixed bounds [0, 1] & Outlier-safe IQR scaling
├── 3) impact_on_models/
│   └── linear_vs_tree_scaling.ipynb          <-- Benchmark proof: Linear vs Tree models
└── 4) scaling_in_production_pipelines/
    └── full_preprocessing_pipeline.ipynb     <-- Categorical + Numerical Production Pipeline
```