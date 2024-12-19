# ======================================================
# 1. Import required modules and set up the environment
# ======================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the system path to access custom utilities.
sys.path.append("..")
import utility.plot_settings  # Import custom plot settings

# ======================================================
# 2. Load datasets
# ======================================================
# Load processed dataset from interim folder.
data = pd.read_pickle("../../data/interim/housing.pkl")
# Load raw dataset for comparison.
data2 = pd.read_csv("../../data/raw/Housing.csv")

# Convert datasets to pandas DataFrames for further analysis.
df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)

# ======================================================
# 3. Calculate correlation matrices
# ======================================================
# Correlation matrix for the processed dataset.
corr_matrix = df.corr()
# Correlation matrix for integer columns in the raw dataset.
df2_int = df2.select_dtypes(include=["int64"])
corr_matrix2 = df2_int.corr()

# ======================================================
# 4. Generate correlation heatmaps
# ======================================================
# Create a figure with two subplots for side-by-side comparison.
fig, ax = plt.subplots(1, 2, figsize=(20, 16))

# Heatmap for the processed dataset.
sns.heatmap(
    corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax[0]
)
ax[0].set_title("Korelasyon Haritası 1")

# Heatmap for integer columns in the raw dataset.
sns.heatmap(
    corr_matrix2, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax[1]
)
ax[1].set_title("Korelasyon Haritası 2")

# Adjust layout to prevent overlapping elements.
plt.tight_layout()

# ======================================================
# 5. Display the plots
# ======================================================
plt.show()
