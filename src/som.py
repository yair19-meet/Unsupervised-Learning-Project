import random
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


class SOM():
    """
    Self-Organizing Map (SOM) implementation.
    """
    def __init__(self, sigma, alpha, dimensions, k, low_range, high_range, epochs, random_seed=42):
        self.sigma0 = sigma  # How wide the 'neighborhood' influence starts
        self.alpha0 = alpha  # How fast we learn initially
        self.dimensions = dimensions  # Number of features (e.g., 21 customer attributes)
        self.k = k  # Total number of neurons (e.g., 9 for a 3x3 grid)
        self.low_range = low_range  # Initial weight range (min)
        self.high_range = high_range  # Initial weight range (max)
        self.units = np.array([])
        self.grid_coords = np.array([]) # The fixed physical locations on our 2D map
        self.epochs = epochs  # How many times we show the data to the map
        self.grid_size = int(np.sqrt(self.k))
        self.feature_names = None
        self.clusters = None
        self.random_seed = random_seed

    def initialise(self):
        """
        Set up our neurons. We give them random initial values 
        and fix their physical positions on the 2D map.
        """
        if self.random_seed is not None:       
            random.seed(self.random_seed)         
            np.random.seed(self.random_seed)

        # Start neurons with random values within the data's range
        self.units = np.random.uniform(self.low_range, self.high_range, (self.k, self.dimensions))
        
        # Create the grid - every neuron gets a fixed (x, y) coordinate
        x_coords, y_coords = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.grid_coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)

    
    def get_bmu(self, x):
        """
        The 'Competition' step: 
        Find which neuron is the Best Matching Unit (BMU) for a specific customer.
        """
        # Calculate Euclidean distance between the customer 'x' and all neurons
        distances = np.linalg.norm(self.units - x, axis=1)
        return np.argmin(distances)
    
    def get_labels(self, inputs):
        """Helper to get the winning neuron index for every input in a batch."""
        return np.array([self.get_bmu(x) for x in inputs])

    def cluster_inputs(self, inputs):
        """
        Group the raw data points into lists based on which neuron they 'won'.
        This lets us see exactly which customers belong to which cluster.
        """
        inputs = np.array(inputs)
        self.clusters = [[] for _ in range(self.k)]
        for x in inputs:
            bmu = self.get_bmu(x)
            self.clusters[bmu].append(x)

    def algorithm(self, inputs, columns):
        """
        The core training loop. We iterate through the data many times, 
        slowly adjust the values of neurons toward the dense areas of our data.
        """
        inputs = np.array(inputs)
        self.initialise()
        self.feature_names = columns
        history = []

        # This constant helps sigma decay smoothly so it reaches roughly 1.0 by the end
        lambda_val = self.epochs / np.log(self.sigma0) if self.sigma0 > 1 else self.epochs

        for t in range(self.epochs):
            # As time goes on, we become less 'flexible'.
            # We move neurons less (alpha) and affect fewer neighbors (sigma).
            alpha = self.alpha0 * (1 - t / self.epochs)
            sigma = self.sigma0 * np.exp(-t / lambda_val)

            # Shuffling prevents the map from 'favoring' the first few rows of data
            indices = np.random.permutation(len(inputs))
            
            for idx in indices:
                x = inputs[idx]
                
                # Find the Best Matching Unit
                bmu_idx = self.get_bmu(x)
                bmu_coord = self.grid_coords[bmu_idx]
                
                # Neurons close to the BMU on the 2D grid get pulled along with it.
                grid_distances = np.linalg.norm(self.grid_coords - bmu_coord, axis=1)
                influence = np.exp(-(grid_distances**2) / (2 * (sigma**2)))
                
                # Pull the neurons toward the winning customer's data   
                self.units += alpha * influence[:, np.newaxis] * (x - self.units)

            # Keep track of how the map evolved
            history.append(self.units.copy())

        # After training, categorize all our customers one last time
        self.cluster_inputs(inputs)
        return history
    

    def get_grid(self):
        """Reshapes the flat list of units back into a 2D grid for plotting."""
        return self.units.reshape(self.grid_size, self.grid_size, -1)
    
    def u_matrix(self):
        """
        Calculates the distance between each neuron and its neighbors.
        Small distance = similar neurons (cluster center).
        Large distance = boundary/gap between clusters.
        """
        n = int(np.sqrt(self.k))
        grid = self.units.reshape(n, n, -1)

        U = np.zeros((2 * n - 1, 2 * n - 1))

        # Fill the gaps with Euclidean distances between neighboring weights
        for i in range(n):
            for j in range(n - 1):
                U[2 * i, 2 * j + 1] = np.linalg.norm(grid[i, j] - grid[i, j + 1])

        for i in range(n - 1):
            for j in range(n):
                U[2 * i + 1, 2 * j] = np.linalg.norm(grid[i, j] - grid[i + 1, j])

        # Fill the centers of the gaps by averaging the 4 surrounding distances
        for i in range(1, 2 * n - 2, 2):
            for j in range(1, 2 * n - 2, 2):
                U[i, j] = np.mean([U[i - 1, j], U[i + 1, j], U[i, j - 1], U[i, j + 1]])

        return U

    def plot_u_matrix(self, save_path=None):
        """
        Visualizes the cluster boundaries. Darker areas are 'valleys' where 
        customers are very similar. Lighter 'ridges' are transitions between groups.
        """
        U = self.u_matrix()
        
        plt.style.use('seaborn-v0_8-white')
        fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
        
        im = ax.imshow(U, cmap="viridis", interpolation="nearest")
        
        # Draw a clean grid to separate the cells
        ax.set_xticks(np.arange(-.5, U.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, U.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
        
        # Mark the actual neuron locations with dots
        n = int(np.sqrt(self.k))
        for i in range(n):
            for j in range(n):
                ax.plot(2 * j, 2 * i, marker='o', color='white', markersize=5, markeredgecolor='black')
        
        ax.set_title("SOM U-Matrix (Cluster Boundaries)", fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Distance (Ridge = Boundary)', rotation=-90, va="bottom", fontsize=11, labelpad=15)
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved -> {save_path}")
            plt.close(fig)
        else:
            plt.show()


    def component_planes(self):
        """Returns a list of 2D grids, one for each feature (Age, Spend, etc.)."""
        n = self.grid_size
        grid = self.units.reshape(n, n, -1)
        return [grid[:, :, d] for d in range(self.dimensions)]

    def plot_component_planes(self, save_path=None):
        """
        Shows 'Heat Maps' for every customer attribute. 
        Great for saying: 'This top-left cluster is high-income but low-spending.'
        """
        planes = self.component_planes()
        n_features = len(planes)

        cols = min(4, n_features)
        rows = int(np.ceil(n_features / cols))

        plt.style.use("seaborn-v0_8-white")
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=120, constrained_layout=True)
        
        if n_features > 1:
            axes = np.array(axes).reshape(-1)
        else:
            axes = [axes]

        for i, plane in enumerate(planes):
            ax = axes[i]
            im = ax.imshow(plane, cmap="coolwarm", interpolation="bilinear")

            title = self.feature_names[i] if self.feature_names is not None else f"Feature {i+1}"
            ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
            ax.set_xticks([]); ax.set_yticks([])

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.outline.set_visible(False)

        # Clean up empty subplots
        for j in range(n_features, len(axes)):
            axes[j].axis("off")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved -> {save_path}")
            plt.close(fig)
        else:
            plt.show()


    def plot_hit_map(self, save_path=None):
        """
        Shows where the customers 'land' on the map. 
        Highlights which neurons are the 'hubs' for your customer base.
        """
        if not self.clusters:
            print("Clusters are empty. Run the algorithm first.")
            return

        counts = np.array([len(cluster) for cluster in self.clusters])
        hit_map = counts.reshape(self.grid_size, self.grid_size)

        plt.style.use('seaborn-v0_8-white')
        fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
        
        im = ax.imshow(hit_map, cmap="Blues", interpolation="nearest")
        
        # Add population numbers directly inside the map
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                ax.text(j, i, int(hit_map[i, j]), ha="center", va="center", 
                        color="white" if hit_map[i, j] > np.max(hit_map)/2 else "black",
                        fontweight='bold')

        ax.set_title("SOM Hit Map (Where Customers Land)", fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(np.arange(self.grid_size)); ax.set_yticks(np.arange(self.grid_size))
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Customer Count', rotation=-90, va="bottom", fontsize=11, labelpad=15)
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved -> {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def plot_cluster_sizes(self, save_path=None):
        """Simple bar chart view of cluster populations."""
        if not self.clusters:
            print("Clusters are empty. Run the algorithm first.")
            return
            
        sizes = [len(cluster) for cluster in self.clusters]
        cluster_labels = [f"Unit {i + 1}" for i in range(self.k)]
        
        plt.style.use('seaborn-v0_8-white')
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        
        bars = ax.bar(cluster_labels, sizes, color='#3498db', edgecolor='black', alpha=0.8)
        ax.set_title("Population per SOM Unit", fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel("Number of Customers", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (max(sizes) * 0.02), 
                    int(yval), ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved -> {save_path}")
            plt.close(fig)
        else:
            plt.show()


# if __name__ == '__main__':
#     # Load and prep the customer data
#     data = pd.read_csv("data/customer_info_cleaned.csv")
#     data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])

#     # We only cluster on numeric/behavioral features (excluding ID, Name, etc.)
#     data_for_clustering = data.iloc[:,4:].copy()

#     print(f"Clustering on {len(data_for_clustering.columns)} features: {list(data_for_clustering.columns)}")

#     # Initialize SOM: 3x3 grid (9 units) for 150 epochs
#     som = SOM(sigma=1, alpha=0.5, dimensions=21, k=9, low_range=-1, high_range=1, epochs=150, random_seed=7)
#     history = som.algorithm(data_for_clustering.values, data_for_clustering.columns)

#     # Let's see the results!
#     som.plot_u_matrix()
#     som.plot_component_planes()
#     som.plot_hit_map()
#     som.plot_cluster_sizes()