# Data Cleaning & Quality Assurance

Data cleaning is the process of detecting and correcting (or removing) corrupt, inaccurate, inconsistent, or improperly formatted records from a dataset before feeding it to an ML pipeline.

---

## 1. What We Cover in Data Cleaning

| Step | Topic | Key Focus |
| :--- | :--- | :--- |
| **1** | **Duplicate Data Handling** | Exact vs. subset/partial duplicates, primary keys, deduplication strategies. |
| **2** | **Data Types & Safe Casting** | Fixing numeric/dates stored as strings, `pd.to_numeric(errors='coerce')`, memory optimization. |
| **3** | **String & Category Normalization** | Whitespace stripping, case folding, typo resolution, regex standardizations (`"India"`, `"IND"`). |
| **4** | **Invalid Values & Domain Constraints** | Impossible values (Age: -5 or 300), negative prices, rule-based masking. |
| **5** | **Date/Time Cleaning & Parsing** | Format reconciliation (`YYYY-MM-DD` vs `DD/MM/YYYY`), time zone alignment, datetime feature extraction. |
| **6** | **Outlier Detection & Treatment** | IQR rule, Z-score / Modified Z-score, Isolation Forest, Capping/Winsorization vs. Removal. |
| **7** | **Data Leakage in Cleaning** | Avoiding global statistics leakage during type cleaning, outlier thresholds, and capping. |
| **8** | **Validation & Automated Checks** | Schema validation, boundary checks, assert statements, drift detection overview. |
| **9** | **Practical Implementation & Workflow** | End-to-end Python cleaning script, core Pandas functions, and interview FAQ. |

---
## 2. Practical Workflow (How to Approach a Raw Dataset)

```
Raw Corrupted Data
       │
       ▼
1. Drop True Exact Duplicates
       │
       ▼
2. Correct Structural Types (Strings -> Numeric / Datetime with safe error coercion)
       │
       ▼
3. Standardize Categorical Text (Trim spaces, lower-case, normalize labels)
       │
       ▼
4. Enforce Domain Rules & Bounds (Turn impossible values into NaN or valid bounds)
       │
       ▼
5. Train / Test Split  <--- (Crucial step to prevent Data Leakage)
       │
       ▼
6. Compute Outlier Bounds & Imputation Parameters (On Train only -> Apply to Test)
       │
       ▼
Clean, Leak-Free Data for ML Pipelines
```

## Different types of duplicates we can face
### 1. Exact duplicate
Every column identical
```
101 | Rahul | 22 | Hyderabad
101 | Rahul | 22 | Hyderabad
```
This is the simple one.
```python
import pandas as pd
df = pd.DataFrame(data)
df.duplicated()
```
This function will return something like:
```python
False # the respected row is doesn't have any duplicate.
False
True # this row has a identical duplicate in the data, same to same.
False
# I'll also show you this with examples
```
And:
```python
df.drop_duplicates()
```
removes them.

Functions to remember:
```python 
df.duplicated()
df.drop_duplicates()
```
and also:
```python 
df.duplicated().sum()
```
This function gives a number, which tells how many duplicates are present in the data. 

### 2. Partially duplicates
now consider:
```
| customer_id | name  | age | city      |
| ----------- | ----- | --- | --------- |
| 101         | Rahul |  22 | Hyderabad |
| 102         | Priya |  24 | Mumbai    |
| 101         | Rahul |  23 | Hyderabad |
```
See here, except the `customer_id` everything is same for `rahul` in another row. This is called partial duplicates. In this cases you need to understand what you need to do, in real world id is considered more important, so if we take that case mainly, here's *how you can only get the number of duplicates in data based on specific column*.
```python 
df.duplicated(subset=['customer_id'])
```
to remove duplicates based on a single column would be:
```python 
df.drop_duplicates(subset=['customer_id'])
```
### 3. The dangerous part: which duplicate you need to keep.
In some cases, you may want to keep the first occurrence of that duplicate. 
```
| customer_id | name  | age | updated_at |
| ----------- | ----- | --: | ---------- |
| 101         | Rahul |  22 | Jan 1      |
| 101         | Rahul |  23 | Feb 1      |
```
here if we take example of real world perspective, feb is the latest data right, so we can do something like:
```python 
df.drop_duplicates(
       subset=['customer_id'],
       keep='last
)
```
you can also use:
```python
keep='first'
keep='last'
keep=False
```
```
| Option  | Meaning                       |
| ------- | ----------------------------- |
| `first` | Keep first occurrence         |
| `last`  | Keep last occurrence          |
| `False` | Remove all duplicated records |
```
### 4. Duplicates caused by formatting
```
"Rahul"
"rahul"
" Rahul "
"RAHUL"
``` 
Here in this case, as a person we can understand all those are same names, but for machine all those are different right, so in this type of cases, you need to get logic to them into another form, lets take we need to get them to lower case without any spaces infront or end"
```python
df['name'] = df['name'].str.strip().str.lower()
```
This is siple one, but you need to consider different situations, where there might be some numbers or punctuations or something else attached with the word, so getting some good logic is better. 

### 5. Why blindly removing duplicates can be dangerous
imagine a e-commerce database
```
| customer | product | date  |
| -------- | ------- | ----- |
| A        | Laptop  | Aug 1 |
| A        | Laptop  | Aug 1 |
```
Is that a duplicate there?, maybe.

But the customer also bought two laptops right, deleting one can destroy some data. 

So ***A duplicate row is not necessarily to be a duplicate event***

Everything comes to understanding the situation. what should you do?, what is that data about?.

> Now with this you might logically understand how duplicate data can cause major issues, especially when sent to train model. 

## 3. Core Functions to Master

### Duplicate Management
```python 
df.duplicated(subset=['id'], keep='first')
df.drop_duplicates(subset=['id'], keep='first', inplace=True)
```
### Type Conversion & Coercion
```python 
pd.to_numeric(df['price'], errors='coerce')
pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
df['category_col'] = df['category_col'].astype('category')
```
### String Normalization
```python 
df['city'] = df['city'].str.strip().str.lower()
df['phone'] = df['phone'].str.replace(r'\D+', '', regex=True)
```
### Outlier Bounds Calculation (Train Set Only)
```python 
# IQR Method
Q1 = df_train['salary'].quantile(0.25)
Q3 = df_train['salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping (Winsorization)
df_train['salary'] = df_train['salary'].clip(lower=lower_bound, upper=upper_bound)
```

## All the functions I got encountered in the sub concept
- 
