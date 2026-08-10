### Missing Data Handling in simple words
A process in which you work the cells of your data, which are empty. There are different way you can do that, either fill them with something if you find you there are only some missing in that row or column, or delete that row or column, as its no use of using that, even if added, the quality of that data will never be the same as if it would have real values. 

### 1) What is Missing Data (Real Meaning)

Missing data = absence of values in features, but more importantly:

👉 It can carry information, not just be a problem.

Example:

- Salary missing → person might not want to disclose (important signal)
- Medical test missing → test not required (also signal)

#### 2) Why Handling Missing Data is Critical

If you ignore it:
Most ML models (Linear Regression, SVM, NN) → ❌ crash
Some models (XGBoost, LightGBM) → handle internally, but not always optimally

More importantly:
- Wrong handling → bias introduced
- Can destroy relationships in data
- Leads to data leakage if done incorrectly

#### 3) Types of Missing Data (Practical View Only)

You don’t need heavy theory, just this:

|Type |	Meaning	| Real-world intuition|
|:--  | :-- | :-- | 
|MCAR |	Completely random | sensor failed randomly |
|MAR  |	Depends on other features | salary missing depends on job role|
|MNAR |	Depends on itself | rich people hide salary |

👉 Why this matters:

- MCAR → simple imputation works
- MAR/MNAR → need smarter handling (or feature engineering)

#### 4. First Decision: What Should You Do?

Before touching any function, ALWAYS decide:

##### ✅ Option 1: Drop Data

Use when:

- Missing % is very high (>40–60%)
- Column is not important
```python 
df.drop(columns=['col_name'])
df.dropna()
```
##### ✅ Option 2: Impute (Most Common in Real World)

Fill missing values with something meaningful.

##### ✅ Option 3: Keep Missing as Signal

VERY IMPORTANT (often ignored)
```python
df['col_missing_flag'] = df['col'].isnull().astype(int)
```
👉 Used in production systems.