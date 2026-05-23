"""
compare_clusters.py
===================
Utility functions that **receive pre-computed labels** and produce:
  • A contingency-table heatmap  (K-Means vs SOM)
  • A 2-D UMAP scatter coloured by cluster labels

These are called from main.py which handles data loading and training.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP


# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)


def compare_clusters(kmeans_labels: np.ndarray, som_labels: np.ndarray):
    """
    Build and save a contingency-table heatmap from two sets of
    pre-computed cluster labels (K-Means vs SOM).
    """
    df_compare = pd.DataFrame({
        "K-Means Cluster": kmeans_labels,
        "SOM Cluster":     som_labels,
    })

    contingency = pd.crosstab(
        df_compare["K-Means Cluster"],
        df_compare["SOM Cluster"],
        dropna=False,
    )

    print("\n  Contingency Table (K-Means vs SOM):")
    print(contingency)

    # ── heatmap ──────────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(12, 10), dpi=120)

    sns.heatmap(
        contingency, annot=True, fmt="d", cmap="YlGnBu",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Number of Datapoints"},
    )
    ax.set_title("Clustering Agreement: K-Means vs SOM",
                  fontsize=18, fontweight="bold", pad=25)
    ax.set_xlabel("SOM Cluster ID",     fontsize=14, fontweight="bold")
    ax.set_ylabel("K-Means Cluster ID", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(VISUALS_DIR, "kmeans_vs_som_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Heatmap saved -> {save_path}")
    plt.close(fig)

    return contingency


def umap_visualization(data: pd.DataFrame, kmeans_labels: np.ndarray,
                       random_seed: int = 7, cluster_names: list = None):
    """
    Compute a 2-D UMAP embedding from *data* and colour it with the
    pre-computed *kmeans_labels*, labelled with descriptive *cluster_names*.

    Parameters
    ----------
    cluster_names : list[str], optional
        Descriptive names in the same order as the label values (1-indexed).
        If provided, a categorical legend is drawn instead of a numeric colorbar.
    """
    print("  Computing UMAP embedding ...")

    reducer = UMAP(
        n_components=2,
        random_state=random_seed,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
    )
    embedding = reducer.fit_transform(data.values)

    # ── scatter ──────────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(12, 9), dpi=120)

    unique_labels = np.unique(kmeans_labels)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = kmeans_labels == label
        # Use descriptive name if available (labels are 1-indexed)
        if cluster_names is not None and (label - 1) < len(cluster_names):
            name = cluster_names[label - 1]
        else:
            name = f"Cluster {label}"

        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[cmap(i)], s=5, alpha=0.7, label=name,
        )

    ax.legend(
        title="Cluster", title_fontsize=11,
        fontsize=9, markerscale=3,
        loc="upper right", framealpha=0.85,
    )
    ax.set_title("UMAP coloured by K-Means Cluster",
                  fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()

    save_path = os.path.join(VISUALS_DIR, "umap_kmeans.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  UMAP saved -> {save_path}")
    plt.close(fig)
