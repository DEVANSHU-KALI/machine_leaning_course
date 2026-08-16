Lets first know, what is data pre-processing and why do we need to learn about it in the initial stages of learning machine learning. Also include what all will we learn about this concept and its depth with codes and all. 

## Data Preprocessing
It's a process in which we care only about data, see if data is good to use, is there any modification needed before passing it forward, do we need to add anything, all these thing get covered in this concept.

### Why learning now?
As we know machine learning is all about getting a model trained on data, to make it useful in real world. So to get a good model to work with, we need data to be clean and useful mainly, which we why we learn this concept here initially. We need to have basic understanding about data to work with model later in time.

### What all concepts do we cover in this!
| **Module**                                   | **Core Topics**                                                                                                                                       | **Key Focus**                                                                                                     |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **1. Missing Data Handling**                 | MCAR/MAR/MNAR, Drop vs Impute, `SimpleImputer`, `KNNImputer`, `IterativeImputer` (MICE), Native Tree Handling (XGBoost/LightGBM), Missing Indicators. | Avoiding data leakage during imputation; knowing when to use Median vs. MICE vs. Native handling.                           |
| **2. Data Cleaning & Outliers**              | Duplicates, Inconsistent/Invalid formats, Outlier detection (IQR, Z-Score, Isolation Forest), DateTime parsing.                                       | Robust outlier handling without indiscriminately deleting valid extreme data.                                               |
| **3. Categorical Encoding**                  | Nominal vs. Ordinal, One-Hot, Ordinal, Target Encoding, Frequency Encoding, High Cardinality handling, Unseen Categories.                             | Preventing target leakage with Out-of-Fold (OOF) Target Encoding; One-Hot vs. Target Encoding tradeoffs.                    |
| **4. Feature Scaling**                       | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`.                                                                                     | Knowing exactly which algorithms require scaling (KNN, SVM, Neural Nets, Ridge/Lasso) and which do not (Tree-based models). |
| **5. Feature Engineering & Transformations** | Log/Power transforms, Polynomial features, Interaction terms, Binning, Domain-specific signals.                                                       | Normalizing heavily skewed features using Log or Box-Cox transforms to help linear/gradient models.                         |
| **6. Handling Imbalanced Data**              | Resampling (SMOTE, ADASYN, Undersampling), Algorithm-level (`class_weight='balanced'`), Evaluation Metric shifts (PR-AUC, F1).                        | Knowing when NOT to balance data and avoiding SMOTE data leakage before train-test splits.                                  |
| **7. Production Preprocessing Pipelines**    | `Pipeline`, `ColumnTransformer`, Reproducibility, Train/Test mismatch prevention.                                                                     | Writing modular, leak-free Scikit-Learn transformers that export cleanly to production.                                     |

