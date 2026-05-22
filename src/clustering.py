import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import RobustScaler


class KmeansClustering:
    def __init__(self, min_k, max_k, data, random_seed):
        """
        Class of Kmeans operations. 
        Parameters:
        min_k - the minimum amount of centroids to test the algorithm
        max_k - maximum amount of centroids to test the algorithm
        data - the data to cluster
        random seed - for code reproducability
        """
        self.min_k = min_k
        self.max_k = max_k
        self.data = data
        self.random_seed = random_seed

    def elbow_silhoette_plot(self, epochs):
        k_inertia_scores = []
        silouhette_scores = []
        clusters = []
        
        k_range = range(self.min_k, self.max_k) 
        
        print("Calculating elbow and silhouette scores... (This may take a few seconds)")
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=epochs)
            kmeans.fit(self.data)
            k_inertia_scores.append(kmeans.inertia_)
            
            # Using sample_size 
            score = silhouette_score(self.data, kmeans.labels_, sample_size=5000, random_state=self.random_seed)
            silouhette_scores.append(score)
            
            clusters.append(kmeans.labels_) 
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Elbow)', color=color1)
        
        ax1.plot(k_range, k_inertia_scores, marker='o', color=color1, label='Inertia')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Silhouette Score', color=color2)
        
        ax2.plot(k_range, silouhette_scores, marker='o', color=color2, label='Silhouette Score')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.xticks(k_range)
        ax1.set_title("K-Means Clustering: Elbow Method vs Silhouette Score")

        ax1.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        plt.show()

    def cluster(self, k, epochs):
        kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=epochs)
        kmeans.fit(self.data)
        # shift labels to start from 1 for readability
        shifted_labels = kmeans.labels_ + 1

        # Create a DataFrame of per-cluster averages using the assigned labels
        df = self.data.copy()
        df['Cluster'] = shifted_labels
        cluster_averages = df.groupby('Cluster').mean()

        return shifted_labels, kmeans.inertia_, kmeans.cluster_centers_, cluster_averages
        
    
    def plot_cluster_profiles(self, centroids, feature_names):
        """Generates a heatmap of the cluster centroids for customer profiling."""
        df_centroids = pd.DataFrame(centroids, columns=feature_names)
        
        df_centroids.index = np.arange(1, len(centroids) + 1)
        df_centroids.index.name = 'Cluster (Segment)'
        
        plt.style.use('seaborn-v0_8-white')
        fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
        
        sns.heatmap(df_centroids, cmap="coolwarm", annot=True, fmt=".2f", 
                    linewidths=1, ax=ax, cbar_kws={'label': 'Average Feature Value'})
        
        ax.set_title("Customer Segment Profiles (K-Means Centroids)", fontsize=16, fontweight='bold', pad=15)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=11)
        
        plt.tight_layout()
        # Save to repository visuals folder (use absolute path so script cwd won't matter)
        project_root = Path(__file__).resolve().parent.parent
        visuals_dir = project_root / 'visuals'
        visuals_dir.mkdir(parents=True, exist_ok=True)
        save_path = visuals_dir / 'kmeans_centroids.png'
        plt.savefig(save_path)
        print(f"Saved centroid heatmap to: {save_path}")
        plt.show()

if __name__ == '__main__':
    
    data = pd.read_csv("data/customer_info_cleaned.csv")
    data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])

    data_for_clustering = data.iloc[:,4:].copy()

    algorithm = KmeansClustering(min_k=2, max_k=15, data=data_for_clustering, random_seed=7)

    labels, inertia, centroids, cluster_averages = algorithm.cluster(9, 20)

    # Persist per-customer cluster labels so other scripts can reuse them
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        labels_df = pd.DataFrame({
            'customer_id': data['customer_id'],
            'cluster_label': labels,
        })
        labels_path = output_dir / 'cluster_labels.csv'
        labels_df.to_csv(labels_path, index=False)
        print(f"Wrote cluster labels to: {labels_path}")
    except Exception as e:
        print(f"Could not write cluster labels: {e}")

    algorithm.plot_cluster_profiles(centroids, data_for_clustering.columns)

    print(centroids)
    print('\nCluster averages (per-feature mean by cluster):')
    print(cluster_averages)

    # Try to reconstruct the scaler from the original raw data so we can inverse-transform
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / 'data' / 'customer_info.csv'

    output_dir = project_root / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if raw_path.exists():
            raw = pd.read_csv(raw_path)
            # ensure same preprocessing of date column if present
            if 'customer_birthdate' in raw.columns:
                raw['customer_birthdate'] = pd.to_datetime(raw['customer_birthdate'])

            # select the same feature columns used for clustering
            raw_for_clustering = raw.iloc[:, 4:].copy()

            scaler = RobustScaler()
            scaler.fit(raw_for_clustering.values)

            unscaled_avgs = pd.DataFrame(
                scaler.inverse_transform(cluster_averages.values),
                index=cluster_averages.index,
                columns=cluster_averages.columns,
            )

            out_path = output_dir / 'average_cluster_unscaled.csv'
            unscaled_avgs.to_csv(out_path)
            print(f"Wrote unscaled cluster averages to: {out_path}")
        else:
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    except Exception as e:
        # fallback: save the scaled averages and report the error
        out_path = output_dir / 'average_cluster_scaled.csv'
        cluster_averages.to_csv(out_path)
        print(f"Could not unscale cluster averages ({e}). Wrote scaled averages to: {out_path}")
