# Example dataset

import numpy as np
import pandas as pd

# Create a sample dataset with correlated features and missing values
data = {
    'Experience_Years': [1, 2, 3, 10, 11, 12, 20, 22],
    'Age': [22, 24, 25, 33, 35, 36, 48, 52],
    'Salary': [30000, 35000, 38000, 85000, np.nan, 95000, 150000, 160000] # Missing senior-level salary
}

df = pd.DataFrame(data)

print("=== ORIGINAL DATASET WITH MISSING VALUES ===")
print(df)

#Knn imputer code
# instead of copy, we can also replace the original column with the transformed values if wanted.

from sklearn.impute import KNNImputer

# Create a copy so original df stays untouched
df_knn = df.copy()

# Initialize KNNImputer looking at 3 nearest neighbors (k=3)
knn_imp = KNNImputer(n_neighbors=3)

# Fit and transform across all correlated columns
df_knn_filled = pd.DataFrame(knn_imp.fit_transform(df_knn), columns=df_knn.columns)

# to get the original column replaced, you can change the line like: df_knn = pd.DataFrame(knn_imp.fit_transform(df_knn), columns=df_knn.columns) and also delete that copy line above.

print("=== KNN IMPUTER RESULT (k=3) ===")
print(df_knn_filled)