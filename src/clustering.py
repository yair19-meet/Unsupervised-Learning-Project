import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv("./data/customer_info_cleaned.csv")

df_numeric = df.drop(columns=["customer_id"]).select_dtypes(include=[np.number])


# let's run kmeans on the clean data 30 times on each amount of clusters from 2 to 20.


# lets plot the silouhette score for each amount of clusters as well as the elbow chart. 


k_inertia_scores = []
silouhette_scores = []
clusters = []
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=30)
    kmeans.fit(df_numeric)
    k_inertia_scores.append(kmeans.inertia_)
    silouhette_scores.append(silhouette_score(df_numeric, kmeans.labels_))
    clusters.append(kmeans.labels_) 

# Create a figure with dual Y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Elbow (Inertia) on the left Y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Elbow)', color=color1)
ax1.plot(range(2, 21), k_inertia_scores, marker='o', color=color1, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color1)

# Create a second Y-axis for Silhouette Score
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color2)
ax2.plot(range(2, 21), silouhette_scores, marker='o', color=color2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color2)

# Set discrete xticks
plt.xticks(range(2, 21))
ax1.set_title("K-Means Clustering: Elbow Method vs Silhouette Score")

# Optional: Add a grid and layout adjustment
ax1.grid(True, linestyle='--', alpha=0.6)
fig.tight_layout()

plt.show()