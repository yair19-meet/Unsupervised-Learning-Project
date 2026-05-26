"""
main.py  --  Single entry-point for the clustering pipeline
============================================================
Loads data once, trains both models, assigns descriptive cluster names,
then calls existing visualisation functions and the recommendation pipeline.

All plots saved to visuals/.  Recommendation CSVs to output/recommendations/.

Usage
-----
    cd <project_root>
    python src/main.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ── allow importing sibling modules ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kmeans import KmeansClustering
from som import SOM
from compare_clusters import compare_clusters, umap_visualization
from hierachichal import plot_dendrogram
from cluster_profiles import *
from cluster_recommendation import run_recommendations

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data", "customer_info_cleaned.csv")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)

# ── config ───────────────────────────────────────────────────────────────────
K             = 8
RANDOM_SEED   = 7
KMEANS_EPOCHS = 20     # n_init passed to sklearn KMeans
SOM_EPOCHS    = 500    # SOM training iterations
SOM_K         = 9      # 3x3 grid
Run_recomendations = False


def main():
    # =====================================================================
    #  1. LOAD DATA
    # =====================================================================
    print("=" * 60)
    print("  Step 1 -- Loading data")
    print("=" * 60)
    data = pd.read_csv(DATA_PATH)
    data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])
    data_for_clustering = data.iloc[:, 4:].copy()
    feature_names = list(data_for_clustering.columns)
    print(f"  {len(data):,} customers  x  {len(feature_names)} features")

    # =====================================================================
    #  2. TRAIN K-MEANS + ASSIGN DESCRIPTIVE NAMES
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"  Step 2 -- Training K-Means  (k={K})")
    print("=" * 60)
    km = KmeansClustering(min_k=2, max_k=K + 3,
                          data=data_for_clustering,
                          random_seed=RANDOM_SEED)
    kmeans_labels, inertia, centroids, cluster_avgs = km.cluster(K, KMEANS_EPOCHS)
    print(f"  Inertia: {inertia:,.2f}")

    # Match centroids to descriptive names
    cluster_names = assign_labels_by_heuristics(new_centroids=centroids, feature_names=feature_names)
    print("  Cluster names assigned:")
    unique, counts = np.unique(kmeans_labels, return_counts=True)
    for seg, cnt in zip(unique, counts):
        print(f"    {seg} = {cluster_names[seg - 1]}: {cnt:,} customers")

    # Build labels DataFrame for downstream use
    labels_df = pd.DataFrame({
        "customer_id": data["customer_id"],
        "cluster_name": [cluster_names[l - 1] for l in kmeans_labels],
    })

    # =====================================================================
    #  3. TRAIN SOM
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"  Step 3 -- Training SOM  (3x3, {SOM_EPOCHS} epochs)")
    print("=" * 60)
    som = SOM(sigma=1, alpha=0.5,
              dimensions=len(feature_names), k=SOM_K,
              low_range=-1, high_range=1,
              epochs=SOM_EPOCHS,
              random_seed=RANDOM_SEED)
    som.algorithm(data_for_clustering.values, feature_names)
    som_labels = som.get_labels(data_for_clustering.values) + 1
    unique, counts = np.unique(som_labels, return_counts=True)
    for seg, cnt in zip(unique, counts):
        print(f"    Cluster {seg}: {cnt:,} customers")

    # =====================================================================
    #  4. K-MEANS CENTROID HEATMAP  (with descriptive names!)
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Step 4 -- K-Means centroid heatmap")
    print("=" * 60)
    km.plot_cluster_profiles(
        centroids, feature_names,
        save_path=os.path.join(VISUALS_DIR, "kmeans_centroids.png"),
        cluster_names=cluster_names,
    )

    som.plot_cluster_profiles(som.units, feature_names,
        save_path=os.path.join(VISUALS_DIR, "som_centroids.png"),
    )


    # =====================================================================
    #  5. SOM U-MATRIX
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Step 5 -- SOM U-Matrix")
    print("=" * 60)
    som.plot_u_matrix(
        save_path=os.path.join(VISUALS_DIR, "Umatrix.png"),
    )

    # =====================================================================
    #  6. SOM COMPONENT PLANES
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Step 6 -- SOM Component Planes")
    print("=" * 60)
    som.plot_component_planes(
        save_path=os.path.join(VISUALS_DIR, "component_planes.png"),
    )

    # =====================================================================
    #  7. SOM HIT MAP
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Step 7 -- SOM Hit Map")
    print("=" * 60)
    som.plot_hit_map(
        save_path=os.path.join(VISUALS_DIR, "where_costumers_land_som.png"),
    )

    # =====================================================================
    #  8. SOM CLUSTER SIZES
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Step 8 -- SOM Cluster Sizes")
    print("=" * 60)
    som.plot_cluster_sizes(
        save_path=os.path.join(VISUALS_DIR, "som_cluster_sizes.png"),
    )

    # =====================================================================
    #  9. DENDROGRAM
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"  Step 9 -- Dendrogram  (k={K})")
    print("=" * 60)
    plot_dendrogram(
        data_for_clustering, k=K,
        save_path=os.path.join(VISUALS_DIR, "dendrogram.png"),
    )

    # =====================================================================
    #  10. K-MEANS vs SOM COMPARISON HEATMAP
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Step 10 -- K-Means vs SOM comparison")
    print("=" * 60)
    compare_clusters(kmeans_labels, som_labels)

    # =====================================================================
    #  11. UMAP COLOURED BY K-MEANS
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Step 11 -- UMAP visualisation")
    print("=" * 60)
    umap_visualization(data_for_clustering, kmeans_labels,
                       random_seed=RANDOM_SEED,
                       cluster_names=cluster_names)
    
    umap_visualization(data_for_clustering, som_labels,
                       random_seed=RANDOM_SEED,
                       cluster_names=som_labels, kmeans=False)



    # =====================================================================
    #  12. RECOMMENDATIONS (Phase A: product lift, Phase B: Apriori)
    # =====================================================================
    print("\n")
    if Run_recomendations:
        print("running apriori")
        run_recommendations(labels_df)
    else:
        print("apriori wasnt run")

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
