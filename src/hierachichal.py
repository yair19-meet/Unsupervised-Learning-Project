import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, cophenet
from scipy.spatial.distance import pdist

# Allow importing sibling module clustering.py
sys.path.insert(0, os.path.dirname(__file__))
from clustering import KmeansClustering

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "hierarchical")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load & scale data ─────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "customer_info_cleaned.csv")
data = pd.read_csv(data_path)
data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])

data_for_clustering = data.iloc[:, 4:].copy()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_clustering)

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

# ── Fit hierarchical model ────────────────────────────────────────────────────
clustering = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage="ward"
)
clustering.fit(data_scaled)

# ── Chosen k (from visual inspection of dendrogram + silhouette score) ───────
# The elbow/silhouette plots technically peak at k=2, but the dendrogram shows
# a meaningful gap around k=8 and silhouette also improves there — use that.
K = 8

# Midpoint between the two distances that surround the chosen cut → cut line
cut_distance = (
    clustering.distances_[-K] + clustering.distances_[-(K - 1)]
) / 2

# ── Dendrogram plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_title(
    "Hierarchical Clustering Dendrogram  (Ward linkage)",
    fontsize=16, fontweight="bold", pad=15,
)

lm = build_linkage_and_plot(
    clustering,
    truncate_mode="level",
    p=5,
    color_threshold=cut_distance,  # colours branches below the cut
    ax=ax,
)

ax.axhline(
    y=cut_distance, color="red", linestyle="--", linewidth=1.5,
    label=f"Chosen cut  (k = {K})",
)
ax.set_xlabel("Sample index (or cluster size)", fontsize=12)
ax.set_ylabel("Ward distance", fontsize=12)
ax.legend(fontsize=11)

# Cophenetic correlation – quality metric for how well the dendrogram preserves distances
c, _ = cophenet(lm, pdist(data_scaled))
ax.text(
    0.98, 0.97,
    f"Cophenetic r = {c:.4f}",
    transform=ax.transAxes, ha="right", va="top",
    fontsize=10, color="dimgray",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
)
print(f"Cophenetic correlation: {c:.4f}")

plt.tight_layout()
dendrogram_path = os.path.join(OUTPUT_DIR, "dendrogram.png")
fig.savefig(dendrogram_path, dpi=150, bbox_inches="tight")
print(f"Dendrogram saved → {dendrogram_path}")
plt.show()
plt.close(fig)

# ── Hand off k to the existing KmeansClustering class ───────────────────────
print(f"\nRunning KmeansClustering with k = {K} ...")

# Use scaled data so both algorithms operate on the same feature space
km = KmeansClustering(
    min_k=2, max_k=K + 3,
    data=data_scaled,
    random_seed=42,
)

labels, inertia, centroids = km.cluster(k=K, epochs=20)

print(f"Inertia: {inertia:.2f}")
print("\nCluster distribution:")
unique, counts = np.unique(labels, return_counts=True)
for seg, cnt in zip(unique, counts):
    print(f"  Segment {seg}: {cnt} customers")

# ── Reuse existing visualisation methods ──────────────────────────────────────
km.plot_cluster_profiles(centroids, data_for_clustering.columns)
km.plot_cluster_sizes(labels)
