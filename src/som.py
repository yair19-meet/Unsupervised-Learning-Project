import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


class SOM():
    def __init__(self, sigma, alpha, dimensions, k, low_range, high_range, epochs, random_seed=42):
        self.sigma = sigma
        self.alpha = alpha
        self.dimensions = dimensions
        self.k = k
        self.low_range = low_range
        self.high_range = high_range
        self.units = np.array([])
        self.epochs = epochs
        self.grid_size = int(np.sqrt(self.k))
        self.feature_names = None
        self.clusters = None
        self.random_seed = random_seed

    def initialise(self):
        """initializing units randomly - values are in a range [self.low_range, self.high_range]"""
        if self.random_seed is not None:       
            random.seed(self.random_seed)         
            np.random.seed(self.random_seed)

        self.units = np.array(
                    [[random.uniform(self.low_range, self.high_range) for _ in range(self.dimensions)] for _ in range(self.k)]
                            )

    
    def distance(self, input, unit):
        """Euclidean distance calculation"""
        return np.linalg.norm(input - unit)
    
    def how_far(self, unit1, unit2):
        """the "how far" equation - neighborhood influence"""
        return np.exp(-(np.linalg.norm(unit1 - unit2) ** 2) / (2 * (self.sigma ** 2)))
    

    def get_bmu(self, x):
        """get the Best Matching Unit for a given input vector"""
        distances = [np.linalg.norm(x - w) for w in self.units]
        return np.argmin(distances)
    
    def cluster_inputs(self, inputs):
        """assign the Best Matching Unit for each input vector"""
        inputs = np.array(inputs)
        self.clusters = [[] for _ in range(self.k)]
        for x in inputs:
            bmu = self.get_bmu(x)
            self.clusters[bmu].append(x)

    def algorithm(self, inputs, columns):
        """the SOM algorithm"""
        #initialising the units randomly
        self.initialise()
        self.feature_names = columns
        history = []
        for _ in range(self.epochs):
            for x in inputs:
                # calculate distance of each unit to the corresponding input vector
                distances = [self.distance(x, unit) for unit in self.units]
                # find the BMU
                best_matching_unit = self.units[np.argmin(distances)]
                for i in range(len(self.units)):
                    # update the position of each unit based on the neighborhood function as well as the distance
                    # of the input vector from the corresponding unit.
                    unit = self.units[i]
                    influence = self.how_far(unit, best_matching_unit)
                    self.units[i] = unit + self.alpha * influence * (x - unit)

            # record the current positions of the units.
            history.append(self.units.copy())

        # at this point self.units are  finalised.
        # we label the clusters.
        self.cluster_inputs(inputs)
        # we return the history (history[-1] is the final units' positions)
        return history
    

    def get_grid(self):
        """auxiliary funciton for plotting"""
        return self.units.reshape(self.grid_size, self.grid_size, -1)
    
    def u_matrix(self):
        n = int(np.sqrt(self.k))  # grid size (n x n)
        grid = self.units.reshape(n, n, -1)

        # U-matrix is (2n-1 x 2n-1)
        U = np.zeros((2 * n - 1, 2 * n - 1))

        # neuron positions
        for i in range(n):
            for j in range(n):
                U[2 * i, 2 * j] = 0  # neuron itself

        # horizontal distances
        for i in range(n):
            for j in range(n - 1):
                U[2 * i, 2 * j + 1] = np.linalg.norm(
                    grid[i, j] - grid[i, j + 1]
                )

        # vertical distances
        for i in range(n - 1):
            for j in range(n):
                U[2 * i + 1, 2 * j] = np.linalg.norm(
                    grid[i, j] - grid[i + 1, j]
                )

        # smoothing (fills gaps)
        for i in range(1, 2 * n - 2, 2):
            for j in range(1, 2 * n - 2, 2):
                U[i, j] = np.mean([
                    U[i - 1, j],
                    U[i + 1, j],
                    U[i, j - 1],
                    U[i, j + 1]
                ])

        return U

    def plot_u_matrix(self):
        """Generates a professional U-Matrix plot."""
        U = self.u_matrix()
        
        plt.style.use('seaborn-v0_8-white')
        fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
        
        # 1. Change 'bilinear' to 'nearest' so the boxes aren't blurry
        im = ax.imshow(U, cmap="viridis", interpolation="nearest")
        
        # --- ADD THIS NEW BLOCK ---
        # 2. Draw a precise white grid to bound every individual cell
        ax.set_xticks(np.arange(-.5, U.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, U.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
        
        # 3. Mark the actual SOM units (neurons) with a distinct dot
        n = int(np.sqrt(self.k))
        for i in range(n):
            for j in range(n):
                # In the U-matrix, neurons are always at the even coordinates
                ax.plot(2 * j, 2 * i, marker='o', color='white', markersize=5, markeredgecolor='black')
        
        ax.set_title("SOM U-Matrix (Distance Map)", fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Professional colorbar formatting
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Euclidean Distance', rotation=-90, va="bottom", fontsize=11, labelpad=15)
        cbar.outline.set_visible(False)
        
        # Add a subtle border around the grid
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1.5)
            
        plt.tight_layout()
        plt.show()


    def component_planes(self):
        """ this function return a list of planes - each plane represent the values of one feature.
            To be more precise, each plane represents the learned weights of a single feature across all SOM units."""
        n = self.grid_size
        grid = self.units.reshape(n, n, -1)

        planes = []

        for d in range(self.dimensions):
            plane = grid[:, :, d]
            planes.append(plane)

        return planes

    def plot_component_planes(self):
        """Generates professional component planes with individual scales."""
        planes = self.component_planes()
        n_features = len(planes)

        cols = min(4, n_features)
        rows = int(np.ceil(n_features / cols))

        plt.style.use("seaborn-v0_8-white")
        
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=120, constrained_layout=True)
        
        # Ensure axes is always iterable
        if n_features > 1:
            axes = np.array(axes).reshape(-1)
        else:
            axes = [axes]

        for i, plane in enumerate(planes):
            ax = axes[i]

            im = ax.imshow(
                plane,
                cmap="coolwarm",   
                interpolation="bilinear"
            )

            title = self.feature_names[i] if self.feature_names is not None else f"Feature {i+1}"
            
            # Increased pad to 15 to give the title a bit more breathing room from its own map
            ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#eeeeee')
                spine.set_linewidth(1)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=9)

        # Hide any empty subplots in the grid
        for j in range(n_features, len(axes)):
            axes[j].axis("off")

        plt.show()


    def plot_cluster_sizes(self):
        """Plots a bar chart showing the number of items assigned to each SOM unit."""
        if not self.clusters:
            print("Clusters are empty. Run the algorithm first.")
            return
            
        # Count how many data points are in each cluster list
        sizes = [len(cluster) for cluster in self.clusters]
        cluster_labels = [f"Unit {i + 1}" for i in range(self.k)]
        
        plt.style.use('seaborn-v0_8-white')
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        
        bars = ax.bar(cluster_labels, sizes, color='#3498db', edgecolor='black', alpha=0.8)
        
        ax.set_title("Population per SOM Unit", fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel("Number of Customers", fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        
        # Add the exact count on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (max(sizes) * 0.02), 
                    int(yval), ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    
    data = pd.read_csv("data/customer_info_cleaned.csv")
    data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])

    data_for_clustering = data.iloc[:,4:].copy()

    print(data_for_clustering.columns)

    som = SOM(sigma=1, alpha=0.5, dimensions=21, k=9, low_range=-1, high_range=1, epochs=150, random_seed=7)
    history = som.algorithm(data_for_clustering.values, data_for_clustering.columns)

    som.plot_u_matrix()
    som.plot_component_planes()
    som.plot_cluster_sizes()