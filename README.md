# Customer Segmentation & Clustering Analysis

## Overview
This repository contains a complete, end-to-end pipeline for data-driven customer segmentation. It leverages exploratory data analysis (EDA), an *a priori* RFM (Recency, Frequency, Monetary) segmentation, and advanced machine learning clustering algorithms. The pipeline processes raw data, evaluates multiple models to find the optimal customer groupings, and automatically generates visualizations and business recommendations.

## Project Structure

```text
├── data/
│   ├── customer_basket.csv        # Raw input data
│   ├── customer_info.csv          # Raw input data
│   └── customer_info_cleaned.csv  # Cleaned data outputted by EDA.py
├── recommendations/               # Generated recommendation outputs (CSV/TXT)
├── visuals/                       # Visualizations, umap plots, and heatmaps
└── src/
    ├── main.py                    # Main execution script controlling algorithms, visualizations, and recommendations
    ├── EDA.py                     # Data preprocessing and feature correlation heatmap generation
    ├── kmeans.py                  # K-Means clustering model 
    ├── hierarchical.py            # Hierarchical clustering model
    ├── som.py                     # Self-Organizing Map (SOM) 
    ├── clustering_profiles.py     # automated centroids labeling
    ├── compare_clusters.py        # Functions for algorithm comparison, UMAP, and cluster coherence heatmaps
    ├── cluster_recommendation.py  # Recommendation generation using lift and apriori algorithm
    └── rfm/                       # 5-5-5 RFM segmentation logic and grouping
```

## Prerequisites & Dependencies

This project relies on a few standard data science libraries. Note that the Self-Organizing Map (SOM) algorithm is a custom, in-house implementation, so it does not require any external SOM libraries.

To install the required dependencies, run the following command in your terminal:

```bash
pip install numpy pandas scikit-learn seaborn umap-learn scipy
```

## Usage & Execution
The repository already includes the preprocessed_data.csv in the data/ folder, meaning you can immediately start training the models and generating outputs without needing to run the EDA script first.

To run the clustering pipeline:

1) Open src/main.py.

2) Locate the configuration variables at the top of the file to set your desired number of clusters:

    k_kmeans: Set to your desired number of K-Means clusters (e.g., k_kmeans = 5).

    k_som: Set to your desired number of SOM clusters. Important: This value must be a perfect square (e.g., 4, 9, 16, 25) due to our design choice to enforce a square grid with identical dimensions.

3) run main.py 
