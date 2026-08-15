# Missing Data Handling in simple words
A process in which you work the cells of your data, which are empty. There are different way you can do that, either fill them with something if you find you there are only some missing in that row or column, or delete that row or column, as its no use of using that, even if added, the quality of that data will never be the same as if it would have real values. 

## 1. What is Missing Data (Real Meaning)

Missing data = absence of values in features, but more importantly:

👉 It can carry information, not just be a problem.

Example:

- Salary missing → person might not want to disclose (important signal)
- Medical test missing → test not required (also signal)

## 2. Why Handling Missing Data is Critical

If you ignore it:
Most ML models (Linear Regression, SVM, NN) → ❌ crash
Some models (XGBoost, LightGBM) → handle internally, but not always optimally

More importantly:
- Wrong handling → bias introduced
- Can destroy relationships in data
- Leads to data leakage if done incorrectly

## 3. Types of Missing Data (Three types)

### 1. MCAR: Missing Completely At Random
- **The Concept**: The missingness has zero relationship with any feature in the dataset—observed or unobserved. It happens purely by chance or random accident.
- **Real-World Example**: A lab technician accidentally drops a test tube, or a sensor randomly loses connection for a second due to a transient power flicker.
- **Impact & Imputation**: Safe to delete rows or use basic imputation (SimpleImputer with mean/median/mode) because removing or imputing them will not introduce systematic bias.

### 2. MAR: Missing At Random
- **The Concept**: The missingness is related to other observed features, but not related to the missing value itself.
- **Real-World Example**: In a health survey, older people are less likely to report their annual income. The missingness of `Income` depends on `Age` (which is recorded), not on whether their income is high or low.
- **Impact & Imputation**: Dropping rows causes bias. Advanced multivariate methods like `KNNImputer` or `IterativeImputer (MICE)` work best here because they reconstruct the missing values using the related observed features (e.g., using `Age` to predict `Income`).


You don’t need heavy theory, just this:

|Type |	Meaning	| Real-world intuition|
|:--  | :-- | :-- | 
|MCAR |	Completely random | sensor failed randomly |
|MAR  |	Depends on other features | salary missing depends on job role|
|MNAR |	Depends on itself | rich people hide salary |

👉 Why this matters:

- MCAR → simple imputation works
- MAR/MNAR → need smarter handling (or feature engineering)

## 4. First Decision: What Should You Do?

Before touching any function, ALWAYS decide:

### ✅ Option 1: Drop Data

Use when:

- Missing % is very high (>40–60%)
- Column is not important
```python 
df.drop(columns=['col_name'])
df.dropna()
```
### ✅ Option 2: Impute (Most Common in Real World)

Fill missing values with something meaningful.

### ✅ Option 3: Keep Missing as Signal

VERY IMPORTANT (often ignored)
```python
df['col_missing_flag'] = df['col'].isnull().astype(int)
```
👉 Used in production systems.

## 5. Why Imputers Exist (Important Concept)

Earlier:
```python 
df['age'].fillna(df['age'].mean())
```
Problem:

- Not reusable
- Train/test mismatch
- No pipeline integration
- Risk of leakage

👉 Solution: Imputers (sklearn)

They:

Learn from train data only
Apply same logic to test data
Work inside pipelines

## 6. Imputation Techniques (WHAT is used WHEN)

### 1. Simple Imputer (MOST USED)
Explained individually in the folder: [data_preprocessing\1) missing_data_handling\1) simple_imputer]

### 2. KNN Imputer
Explained individually in the folder: [data_preprocessing\1) missing_data_handling\2) knn_imputer]

### 3. Iterative Imputer (Advanced)
Explained individually in the folder: [data_preprocessing\1) missing_data_handling\2) iterative_imputer]

### 4. Model-based Handling (Very Practical)
Explained individually in the folder: [data_preprocessing\1) missing_data_handling\2) model_based]

## 7. Functions You MUST Remember (Important)

### Pandas (Basic bu important)
```python 
df.isnull()
df.isnull().sum()
df.dropna()
df.fillna()
```

### Sklearn core
```python
from sklearn.impute SimpleImputer
from skelarn.impute KNNImputer
from skelarn.impute IterativeImputer
```

### Pipeline
```python
from skelarn.pipeline import Pipeline
```

## 8. Common mistakes (very important)
- using **mean** on **skewed data**
- fitting Imputer on full dataset (data leakage)
- Dropping too much data blindly
- Ignoring missing as a feature
- Using knn on huge datasets

To avoid these mistakes, make sure that you understand all the 4 techniques mentioned very clearly.

