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

