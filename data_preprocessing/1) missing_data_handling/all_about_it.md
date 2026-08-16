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

### 3. MNAR: Missing Not At Random
- **The Concept**: The missingness is directly related to the unobserved value itself. The fact that it is missing carries specific meaning.
- **Real-World Example**: People with extremely high salaries or people with severe depression choose not to fill out the `Salary` or `Mental Health Score` questions on a survey because of the value itself.
- **Impact & Imputation**: Imputing with mean or KNN will distort reality because the missing values come from a specific extreme subgroup. The best approach is Missing-Indicator imputation (adding a binary flag `is_missing`) or using Model-Based tree methods (like XGBoost/LightGBM) to learn from the missing state directly.

### Comparison of Missing Data Mechanisms:

| **Mechanism** | **Cause of Missingness**                              | **Example**                                                       | **Best Handling Strategy**                                        |
| ------------- | ----------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| **MCAR**      | Completely random noise; independent of all features. | Dropped sample, random equipment glitch.                          | `SimpleImputer` (Mean / Median) or Listwise deletion.             |
| **MAR**       | Dependent on other known, observed features.          | Men skipping depression score; Older individuals skipping salary. | `KNNImputer`, `IterativeImputer` (MICE).                          |
| **MNAR**      | Dependent on the actual missing value itself.         | High-earners hiding income; failing students skipping survey.     | Missing-Indicator flag + Model-based handling (XGBoost/LightGBM). |

**You don’t need heavy theory, just this:**

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
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer
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

## 9. Missing-Indicator Features (Preserving Missingness Signal)

Imputing a value (replacing `NaN` with a mean or median) destroys the information that the value was absent. A **Missing Indicator** adds an explicit binary column (`1` = was missing, `0` = was present) alongside the imputed feature.

### Why It Matters:
- Essential for **MNAR** scenarios where missingness is non-random (e.g., users skipping sensitive survey questions).
- Gives tree models and linear classifiers direct access to the missingness pattern.

```python
from sklearn.impute import SimpleImputer

# Imputes missing values AND appends a binary indicator column automatically
imputer = SimpleImputer(strategy='median', add_indicator=True)
X_imputed = imputer.fit_transform(X)
```
## 10. Handling Missing Data in Time-Series & Sequential Data

Standard imputation assumes rows are independent. In time-series data, **row order represents time**, meaning past events lead to present events, which lead to future events.

> **Note on Scope:** This section covers the core fundamentals required for general ML workflows. For specialized roles (e.g., Quantitative Finance, IoT/Sensor analytics, or Supply Chain forecasting), more advanced domain-specific techniques like Kalman Filters, spline interpolations, or state-space models may be applied.

### The Fundamental Rule: Avoid Lookahead Leakage
- **Never use global mean, median, or backward fill (`bfill`)** across a time-series column. 
- Using future timestamps to fill past gaps causes **Lookahead Data Leakage**—the model trains on information that would never exist at that point in time during real-world inference.

### Core Practical Techniques

**1. Forward Fill (`ffill` / Last Observation Carried Forward)**
- **Concept:** Propagates the last known valid observation forward until a new data point arrives.
- **Best For:** Discrete state changes, stock prices, account balances, and step-wise data.

```python
# Carries the last observed value forward
df['Stock_Price'] = df['Stock_Price'].ffill()
```

**2. Linear / Time Interpolation**
- Concept: Connects the points immediately before and after a missing gap with a straight line, estimating smooth intermediate values.
- Best For: Continuous physical signals that transition gradually (e.g., temperature, vehicle speed, pressure).

```python 
# Estimates continuous values based on line slope
df['Temperature'] = df['Temperature'].interpolate(method='linear')
```

**3. Rolling Lag Average (Historical Window)**
- Concept: Replaces a missing value with the average of the past $N$ observed time steps strictly looking backward.
- Best For: Data with periodic or cyclical patterns (e.g., daily website traffic, hourly power load).

### Quick comparison table

| **Technique**              | **How It Works**                              | **Best For**                              | **Leakage Safe?**                                           |
| -------------------------- | --------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| **`ffill` (Forward Fill)** | Carries previous known value forward          | Flat states, stock prices, exchange rates | **Yes** (uses only past data)                               |
| **Linear Interpolation**   | Draws a straight line between $t-1$ and $t+1$ | Continuous sensor/weather signals         | **Yes** for historical analysis; use with care in streaming |
| **Rolling Lag Mean**       | Averages the last $K$ past time steps         | Cyclical signals (daily/weekly peaks)     | **Yes** (when using strictly historical windows)            |
| **Global Mean / `bfill`**  | Uses full dataset average or future values    | Standard tabular data only                | **No** (Causes temporal data leakage in time-series)        |

> note: this concept may not be that important for the ml side learners, but for other roles related to finance this is very important, for ml siders just knowing this is enough.

## 11. Data Leakage During Imputation (The #1 Interview Trap)

Data Leakage happens when information from your test/validation split leaks into the training step.

❌ The Incorrect Way (Causes Leakage):

```python
# Calculating mean/median across the entire dataset BEFORE splitting
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y)
```

✅ The Correct Way (Strict Split First):

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
# 1. Learn the mean strictly from training data
X_train_imputed = imputer.fit_transform(X_train)

# 2. Apply the learned training statistics to test data (transform only!)
X_test_imputed = imputer.transform(X_test)
```

## 12. Imputation Inside a Proper Scikit-Learn Pipeline

In production ML, we chain imputation and models together to ensure zero leakage and clean cross-validation.

```python 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Define the unified Pipeline
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('model', RandomForestClassifier(random_state=42))
])

# Fit computes statistics strictly on X_train and fits the model
pipeline.fit(X_train, y_train)

# Predict applies learned training statistics directly to raw input
predictions = pipeline.predict(X_test)
```