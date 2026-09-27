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


---
> Note: This section covers the explanations of the sub concepts which may need some more explanation to let you understand if you are new to this thing or can't understand the short explanation in the notebook. As also because, I didn't get that in the first place.

### 1. I got in the from the very first notebook [1)cost_sensitive_learning] is not about the calculation we are doing there, as it can be understandable at a point, but what way are we treating that imbalance, there are also multiple loss functions, how does calculation happen there, as the notebook didn't cover which loss function is getting reffered there. 


You are asking the exact right engineering question: **Where does this weight $w$ actually plug into the loss function, and how does that force the model to pay attention to the minority class?**

### 1. The Concrete Example: Binary Cross-Entropy (Logistic Loss)

In standard Logistic Regression (and binary classification in Deep Learning), the default loss function is **Binary Cross-Entropy (Log Loss)**.

For a single data point $i$, the standard unweighted loss is:

$$\mathcal{L}_i = - \Big[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \Big]$$

Where:

* $y_i \in \{0, 1\}$ is the ground truth (e.g., $1 = \text{Fraud}$, $0 = \text{Legitimate}$).
* $p_i$ is the model's predicted probability of fraud.

#### The Total Unweighted Loss Over the Entire Dataset:

$$\text{Total Loss} = \sum_{i=1}^{N} \mathcal{L}_i$$

### 2. The Root Cause of Failure on Imbalanced Data

Suppose your dataset has **760 Legit rows ($y=0$)** and only **40 Fraud rows ($y=1$)**:

$$\text{Total Loss} = \underbrace{\sum_{y=0}^{760} \mathcal{L}_i}_{\text{Huge contribution (760 terms)}} + \underbrace{\sum_{y=1}^{40} \mathcal{L}_i}_{\text{Tiny contribution (40 terms)}}$$

When the optimization algorithm (Gradient Descent or L-BFGS) runs, it updates the weights to minimize the **Total Loss**:

* If the model gets **all 40 Fraud cases wrong**, the total loss only increases by a tiny amount because there are only 40 terms.
* If the model gets even a fraction of the 760 Legit cases wrong, the total loss blows up.
* **The Result:** The optimizer takes the lazy path: it pushes all predictions toward $0$ (Legit) to easily minimize the dominant 760 terms, virtually ignoring the 40 fraud cases.

### 3. The "Tweak": Where $w$ Plugs In (Weighted Cross-Entropy)

When you set `class_weight='balanced'`, we introduce the class multiplier $w_{y_i}$ **directly inside the summation**:

$$\text{Weighted Total Loss} = \sum_{i=1}^{N} \mathbf{w_{y_i}} \cdot \mathcal{L}_i$$

From our calculation:

* For any Legit row ($y_i = 0$), $w_0 = 0.526$
* For any Fraud row ($y_i = 1$), $w_1 = 10.0$

Let's look at the expanded formula now:

$$\text{Weighted Total Loss} = \mathbf{0.526} \times \sum_{y=0}^{760} \mathcal{L}_i \;+\; \mathbf{10.0} \times \sum_{y=1}^{40} \mathcal{L}_i$$

### 4. What Happens Under the Hood During Training?

1. **Error Amplification (The Penalty):**
* If the model misclassifies **1 Legit row**, it incurs a loss of $0.526 \times \text{error}$.
* If the model misclassifies **1 Fraud row**, it incurs a loss of $10.0 \times \text{error}$ (**19 times larger penalty!**).


2. **Gradient Updates (Forcing the Optimizer to Care):**
The gradient tells the model weights ($w$) how to update:

$$\frac{\partial (\text{Total Loss})}{\partial \mathbf{W}} = \sum_{i=1}^N \mathbf{w_{y_i}} \cdot \frac{\partial \mathcal{L}_i}{\partial \mathbf{W}}$$

Because the gradient for each fraud case is multiplied by **10.0**, each minority sample produces a massive push on the weights, forcing the decision boundary to move and wrap around the minority class points.

### 5. How Does This Same Principle Apply Across Other Algorithms?

This weighting mechanism is universal across ML and Deep Learning:

* **In Tree-Based Models (Random Forest / Decision Trees):**
* Weighted Gini Impurity: When deciding where to split a node, misclassifying a minority sample adds $10.0$ to the impurity score, forcing the tree to split specifically to isolate the fraud cases.

* **In Gradient Boosting (XGBoost / LightGBM):**
* `scale_pos_weight = (count of negative samples) / (count of positive samples)`. It directly multiplies the gradients ($g_i$) and hessians ($h_i$) of the positive class samples during residual fitting.

* **In Deep Learning (PyTorch / TensorFlow):**
* `torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))` or `torch.nn.CrossEntropyLoss(weight=class_weights_tensor)`. It scales the backpropagation loss tensor before calling `loss.backward()`.

### In Summary

* We do **not** create or delete any rows in the dataset.
* We **multiply the individual sample loss** by its class weight ($w_{y_i}$).
* The optimizer now sees 1 missed fraud case as **19 times more painful** than 1 missed legit case, pulling the decision boundary directly toward capturing the minority class.
- A simple and short answer in also in the notebook.
