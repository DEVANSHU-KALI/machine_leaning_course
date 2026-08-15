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

from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb

# Features (X) and Target (y)
X = df[['Age', 'Salary']] # Salary contains a NaN
y = df['Experience_Years']

# -------------------------------------------------------------
# Method A: Scikit-Learn's HistGradientBoosting (Native NaN support)
# -------------------------------------------------------------
hgb_model = HistGradientBoostingRegressor(random_state=42)
# Fits directly with NaNs present — NO SimpleImputer needed!
hgb_model.fit(X, y)
print("HistGradientBoosting Predictions:", hgb_model.predict(X))

hgb_model_df=pd.DataFrame({
    'Age': X['Age'],
    'Salary': X['Salary'],
    'Predicted_Experience_Years': hgb_model.predict(X)
})
print("\n=== HistGradientBoosting Predictions ===")
print(hgb_model_df)
