"""
hierachichal.py
===============
Hierarchical (agglomerative) clustering utilities.

Provides
--------
build_linkage_and_plot  -  reconstruct a scipy linkage matrix from a fitted
                           AgglomerativeClustering model and render it.
plot_dendrogram         -  fit Ward-linkage hierarchical clustering on the
                           supplied data and save a dendrogram with a cut line.

Called from main.py; can also be run standalone.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, cophenet
from scipy.spatial.distance import pdist

# Allow importing sibling module clustering.py
sys.path.insert(0, os.path.dirname(__file__))


# ── Helper: build linkage matrix and render dendrogram ───────────────────────
def build_linkage_and_plot(model, **kwargs):
    """Reconstruct scipy linkage matrix from AgglomerativeClustering and plot."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix


def plot_dendrogram(data, k=8, save_path=None):
    """
    Fit Ward-linkage hierarchical clustering, plot the dendrogram with a
    cut line at *k* clusters, and optionally save the figure.

    Parameters
    ----------
    data : array-like
        Feature matrix (already cleaned / scaled with RobustScaler).
        An additional StandardScaler is applied internally to match the
        original hierachichal.py behaviour.
    k : int
        Number of clusters to mark on the dendrogram.
    save_path : str or None
        If provided, save the figure there and close it.
        Otherwise call plt.show().
    """
    # StandardScaler on top of the RobustScaled data (matches original script)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Fit full tree (distance_threshold=0 keeps all merge distances)
    clustering = AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, linkage="ward"
    )
    clustering.fit(data_scaled)

    # Cut line: midpoint between the two merge distances around the chosen k
    cut_distance = (
        clustering.distances_[-k] + clustering.distances_[-(k - 1)]
    ) / 2

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(
        "Hierarchical Clustering Dendrogram  (Ward linkage)",
        fontsize=16, fontweight="bold", pad=15,
    )

    lm = build_linkage_and_plot(
        clustering,
        truncate_mode="level",
        p=5,
        color_threshold=cut_distance,
        ax=ax,
    )

    ax.axhline(
        y=cut_distance, color="red", linestyle="--", linewidth=1.5,
        label=f"Chosen cut  (k = {k})",
    )
    ax.set_xlabel("Sample index (or cluster size)", fontsize=12)
    ax.set_ylabel("Ward distance", fontsize=12)
    ax.legend(fontsize=11)

    # Cophenetic correlation
    c, _ = cophenet(lm, pdist(data_scaled))
    ax.text(
        0.98, 0.97,
        f"Cophenetic r = {c:.4f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color="dimgray",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    print(f"  Cophenetic correlation: {c:.4f}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ── Standalone entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    from kmeans import KmeansClustering

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "customer_info_cleaned.csv")

    data = pd.read_csv(data_path)
    data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])
    data_for_clustering = data.iloc[:, 4:].copy()

    K = 8
    output_dir = os.path.join(current_dir, "..", "output", "hierarchical")
    os.makedirs(output_dir, exist_ok=True)

    plot_dendrogram(
        data_for_clustering, k=K,
        save_path=os.path.join(output_dir, "dendrogram.png"),
    )
