import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


class KmeansClustering:
    def __init__(self, min_k, max_k, data):
        self.min_k = min_k
        self.max_k = max_k
        self.data = data

    def elbow_silhoette_plot(self, epochs):
        k_inertia_scores = []
        silouhette_scores = []
        clusters = []
        for k in range(self.min_k, self.max_k):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=epochs)
            kmeans.fit(self.data)
            k_inertia_scores.append(kmeans.inertia_)
            silouhette_scores.append(silhouette_score(self.data, kmeans.labels_))
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

    def cluster(self, k, epochs):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=epochs)
        kmeans.fit(self.data)
        return kmeans.labels_, kmeans.inertia_, kmeans.cluster_centers_
        
    
    def plot_cluster_profiles(self, centroids, feature_names):
        """Generates a heatmap of the cluster centroids for customer profiling."""
        # Convert centroids to a DataFrame for automatic labeling
        df_centroids = pd.DataFrame(centroids, columns=feature_names)
        df_centroids.index.name = 'Cluster (Segment)'
        
        plt.style.use('seaborn-v0_8-white')
        fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
        
        # Use a diverging colormap (coolwarm) to highlight highs (red) and lows (blue)
        sns.heatmap(df_centroids, cmap="coolwarm", annot=True, fmt=".2f", 
                    linewidths=1, ax=ax, cbar_kws={'label': 'Average Feature Value'})
        
        ax.set_title("Customer Segment Profiles (K-Means Centroids)", fontsize=16, fontweight='bold', pad=15)
        
        # Angle the feature names so they don't overlap
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=11)
        
        plt.tight_layout()
        plt.show()

    def plot_cluster_sizes(self, labels):
        """Plots a bar chart showing the number of customers in each cluster."""
        # Count how many customers fall into each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.style.use('seaborn-v0_8-white')
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        
        bars = ax.bar(unique_labels, counts, color='#3498db', edgecolor='black', alpha=0.8)
        
        ax.set_title("Customer Population per Segment", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Cluster ID", fontsize=12, fontweight='bold')
        ax.set_ylabel("Number of Customers", fontsize=12, fontweight='bold')
        ax.set_xticks(unique_labels)
        
        # Add the exact count on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (max(counts) * 0.02), 
                    int(yval), ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        # Clean borders
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    
    data = pd.read_csv("data/customer_info_cleaned.csv")
    data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])

    data_for_clustering = data.iloc[:,4:].copy()

    algorithm = KmeansClustering(2, 21, data_for_clustering)

    labels, inertia, centroids = algorithm.cluster(9, 20)

    algorithm.plot_cluster_profiles(centroids, data_for_clustering.columns)
    algorithm.plot_cluster_sizes(labels)

    print(centroids)
