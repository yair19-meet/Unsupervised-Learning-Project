import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier 



df = pd.read_csv("./data/customer_info.csv", sep = ',')

pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(df.head())


print(df.info())
print(df.describe())

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler


df["customer_birthdate"] = pd.to_datetime(df["customer_birthdate"])
df=df.drop(["loyalty_card_number", "latitude", "longitude"], axis=1)

# Selecting numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
# Exclude customer_id if it's not a feature
features_to_scale = [col for col in numerical_cols if col not in ['customer_id']]

# 1. Display Outliers based on 1.5*IQR Rule
print("\n Outlier Analysis (1.5*IQR Rule)") 
Q1 = df[features_to_scale].quantile(0.25)
Q3 = df[features_to_scale].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers per column
outlier_counts = ((df[features_to_scale] < (Q1 - 1.5 * IQR)) | (df[features_to_scale] > (Q3 + 1.5 * IQR))).sum()
print("Number of outliers per feature:")
print(outlier_counts[outlier_counts > 0]) # Show only columns with outliers

# 2. Standardize using RobustScaler (Better for data with outliers)
# It uses Median and IQR instead of Mean and Std Dev
scaler = RobustScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# 3. Perform KNN Imputation for missing values
imputer = KNNImputer(n_neighbors=5)
df[features_to_scale] = imputer.fit_transform(df[features_to_scale])

print("\nMissing values after imputation:")
print(df[features_to_scale].isnull().sum())

df.to_csv("./data/customer_info_cleaned.csv", index=False)

# 4. Plot Correlation Heatmap
print("\nPlotting correlation heatmap...")
plt.style.use('seaborn-v0_8-white')
plt.figure(figsize=(14, 12), dpi=120)
corr_matrix = df[features_to_scale].corr()

sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap="coolwarm", 
    fmt=".2f", 
    linewidths=0.5, 
    annot_kws={"size": 8},
    square=True
)

plt.title("Correlation Heatmap of Customer Features", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.show()
