import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


class SOM():
    def __init__(self, sigma, alpha, dimensions, k, low_range, high_range, epochs):
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


    def initialise(self):
        self.units = np.array(
                    [[random.uniform(self.low_range, self.high_range) for _ in range(self.dimensions)] for _ in range(self.k)]
                            )

    def distance(self, input, unit):
        return np.linalg.norm(input - unit)
    
    def how_far(self, unit1, unit2):
        return np.exp(-(np.linalg.norm(unit1 - unit2) ** 2) / (2 * (self.sigma ** 2)))
    

    def get_bmu(self, x):
        distances = [np.linalg.norm(x - w) for w in self.units]
        return np.argmin(distances)
    
    def cluster_inputs(self, inputs):
        inputs = np.array(inputs)
        clusters = [[] for _ in range(self.k)]
        for x in inputs:
            bmu = self.get_bmu(x)
            clusters[bmu].append(x)
        return clusters

    def algorithm(self, inputs, columns):
        self.initialise()
        self.feature_names = columns
        history = []
        for _ in range(self.epochs):
            #print('entered iter')
            for x in inputs:
                distances = [self.distance(x, unit) for unit in self.units]
                best_matching_unit = self.units[np.argmin(distances)]
                for i in range(len(self.units)):
                    unit = self.units[i]
                    influence = self.how_far(unit, best_matching_unit)
                    self.units[i] = unit + self.alpha * influence * (x - unit)

            history.append(self.units.copy())

        return history, self.cluster_inputs(inputs)
    

    def get_grid(self):
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
        
        # 'viridis' or 'bone' are standard, perceptually uniform maps for U-matrices
        im = ax.imshow(U, cmap="viridis", interpolation="bilinear")
        
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
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=120)
        
        # Ensure axes is always iterable even if there's only 1 feature
        if n_features > 1:
            axes = np.array(axes).reshape(-1)
        else:
            axes = [axes]

        for i, plane in enumerate(planes):
            ax = axes[i]

            # Bilinear interpolation makes the grid transitions smoother
            im = ax.imshow(
                plane,
                cmap="coolwarm", # 'coolwarm' or 'viridis' look great for features   
                interpolation="bilinear"
            )

            # Safely get feature names
            title = self.feature_names[i] if self.feature_names is not None else f"Feature {i+1}"
            
            ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Clean borders
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#eeeeee')
                spine.set_linewidth(1)

            # Individual colorbars for each plane
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=9)

        # Hide any empty subplots in the grid
        for j in range(n_features, len(axes)):
            axes[j].axis("off")

        # Add a main title to the figure
        fig.suptitle("SOM Component Planes", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()   

if __name__ == '__main__':
    
    data = pd.read_csv("data/customer_info_cleaned.csv")
    data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])

    data_for_clustering = data.iloc[:,4:].copy()

    print(data_for_clustering.columns)

    som = SOM(sigma=1, alpha=1.5, dimensions=21, k=16, low_range=-1, high_range=1, epochs=100)
    history, clusters = som.algorithm(data_for_clustering.values, data_for_clustering.columns)

    som.plot_u_matrix()
    som.plot_component_planes()