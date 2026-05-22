import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from clustering import KmeansClustering
from som import SOM
from umap import UMAP


def compare_clusters():
    # 1. Load Data
    print("Loading and preparing data...")
    data = pd.read_csv("data/customer_info_cleaned.csv")
    
    # Selecting the same columns as in other scripts
    data_for_clustering = data.iloc[:, 4:].copy()
    feature_names = data_for_clustering.columns
    
    # Scaling data (highly recommended for K-Means and SOM)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_clustering)
    
    # 2. Run K-Means
    print("Running K-Means...")
    # Using k=9 as seen in clustering.py
    k_kmeans = 9
    km_algo = KmeansClustering(min_k=2, max_k=15, data=data_scaled, random_seed=7)
    kmeans_labels, _, _ = km_algo.cluster(k_kmeans, 50) # reduced epochs for speed 
    
    # 3. Run SOM 
    print("Running SOM...")
    # Using k=9 (3x3 grid) as seen in som.py
    k_som = 9
    som_algo = SOM(sigma=1, alpha=0.5, dimensions=len(feature_names), k=k_som, 
                   low_range=-1, high_range=1, epochs=500, random_seed=7) 
    _ = som_algo.algorithm(data_scaled, feature_names)
    som_labels = som_algo.get_labels(data_scaled)
    
    # Shift labels to start from 1 for better readability if preferred, 
    # but KmeansClustering.cluster already shifts them! 
    # Let's check som_labels.
    som_labels = som_labels + 1 # Aligning with K-Means labels which are shifted in clustering.py
    
    # 4. Create Comparison DataFrame
    df_compare = pd.DataFrame({
        'K-Means Cluster': kmeans_labels,
        'SOM Cluster': som_labels
    })
    
    # 5. Generate Contingency Table (Crosstab)
    # K-Means on Y, SOM on X
    contingency_matrix = pd.crosstab(df_compare['K-Means Cluster'], 
                                     df_compare['SOM Cluster'],
                                     dropna=False)
    
    print("\nContingency Table (K-Means vs SOM):")
    print(contingency_matrix)
    
    # 6. Visualization
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(12, 10), dpi=120)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs("output", exist_ok=True)
    
    sns.heatmap(contingency_matrix, annot=True, fmt='d', cmap='YlGnBu', 
                linewidths=.5, cbar_kws={'label': 'Number of Datapoints'})
    
    plt.title('Clustering Agreement: K-Means vs SOM', fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('SOM Cluster ID', fontsize=14, fontweight='bold')
    plt.ylabel('K-Means Cluster ID', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "output/kmeans_vs_som_comparison.png"
    plt.savefig(output_path)
    print(f"\nComparison heatmap saved to {output_path}")
    
    # Print the table in a more readable way
    print("\n" + "="*50)
    print("FINAL COMPARISON TABLE")
    print("="*50)
    print(contingency_matrix)
    print("="*50)
    
    return contingency_matrix

def umap_visualization():
    # 1. Load Data
    print("Loading and preparing data...")
    data = pd.read_csv("data/customer_info_cleaned.csv")
    
    # Selecting the same columns as in other scripts
    data_for_clustering = data.iloc[:, 4:].copy()
    feature_names = data_for_clustering.columns
    
    # Scaling data (highly recommended for K-Means and SOM)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_clustering)

    # 2. Obtain cluster labels (prefer existing labels file)
    labels_path = "output/cluster_labels.csv"
    kmeans_labels = None
    if os.path.exists(labels_path):
        try:
            df_labels = pd.read_csv(labels_path)
            cols = [c for c in df_labels.columns if 'cluster' in c.lower()]
            if cols:
                labels = df_labels[cols[0]].values
            else:
                labels = df_labels.iloc[:, 0].values
            labels = np.array(labels, dtype=int)
            if labels.shape[0] != data_scaled.shape[0]:
                raise ValueError(f"labels length {labels.shape[0]} != data length {data_scaled.shape[0]}")
            kmeans_labels = labels
            print(f"Loaded cluster labels from {labels_path}")
        except Exception as e:
            print(f"Could not use {labels_path}: {e}")
            kmeans_labels = None

    if kmeans_labels is None:
        print("Running K-Means (fallback)...")
        # Using k=9 as seen in clustering.py
        k_kmeans = 9
        km_algo = KmeansClustering(min_k=2, max_k=15, data=data_scaled, random_seed=7)
        kmeans_labels, _, _ = km_algo.cluster(k_kmeans, 50) # reduced epochs for speed
        kmeans_labels = np.array(kmeans_labels, dtype=int)

    os.makedirs("output", exist_ok=True)

    print("Applying UMAP for dimensionality reduction...")

    # Use n_components=2 for a 2D visualization
    umap_obj = UMAP(n_components=2, 
                  random_state=7, 
                  n_neighbors=30, 
                  min_dist=0.1, 
                  metric='euclidean')

    # Fit and transform the scaled data
    embedding = umap_obj.fit_transform(data_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
    embedding[:, 0], embedding[:, 1],
    c=kmeans_labels,          # ← this is what colors the dots
    cmap='tab10',             # or 'Set1', 'Paired', 'Spectral', etc.
    s=5, alpha=0.7
    )
    plt.colorbar(scatter, label='K-Means Cluster')
    plt.title('UMAP colored by K-Means Cluster')
    plt.savefig("output/umap_kmeans.png")


if __name__ == "__main__":
    umap_visualization()
