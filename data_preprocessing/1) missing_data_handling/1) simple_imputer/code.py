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


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Check Skewness using Pandas .skew()
print("=== SKEWNESS VALUES ===")
print(df.skew())
print("\n")

# 2. Plot Box Plots for each feature
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

columns = ['Experience_Years', 'Age', 'Salary']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, col in enumerate(columns):
    sns.boxplot(x=df[col], ax=axes[i], color=colors[i])
    axes[i].set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
