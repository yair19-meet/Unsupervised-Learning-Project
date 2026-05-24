"""
cluster_recommendation.py
=========================
Simplified recommendation pipeline (~250 lines instead of 693).

Phase A -- Product Lift Analysis
    For each cluster, compute how much more popular each product is vs the
    overall population.  Surfaces the products that CHARACTERISE each cluster.

Phase B -- Apriori Association Rules
    Mine "buy A -> try B" rules per cluster using mlxtend.

This file receives pre-computed labels (with descriptive names) as input.
It does NOT train K-Means itself.

Usage
-----
    # From main.py:
    from cluster_recommendation import run_recommendations
    results = run_recommendations(labels_df)

    # Standalone:
    python src/cluster_recommendation.py
"""

import os
import sys
import ast
from collections import Counter

import pandas as pd
import numpy as np

# ── allow importing sibling modules ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── mlxtend for Apriori ──────────────────────────────────────────────────────
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    raise ImportError("mlxtend is required.  Install with:  pip install mlxtend")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASKET_PATH = os.path.join(BASE_DIR, "data", "customer_basket.csv")
OUT_DIR     = os.path.join(BASE_DIR, "recommendations")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
#  1. BUILD SHOPPING BAGS
# =============================================================================

def build_shopping_bags(basket_df):
    """
    Parse basket transactions and aggregate per customer.

    Returns
    -------
    dict : {customer_id: set of product strings}
    """
    print("  Building shopping bags ...")

    def parse_goods(val):
        if isinstance(val, float):  # NaN
            return []
        try:
            return ast.literal_eval(val)
        except Exception:
            return []

    bags = {}
    for cid, group in basket_df.groupby("customer_id"):
        products = set()
        for goods_str in group["list_of_goods"]:
            for product in parse_goods(goods_str):
                products.add(str(product).strip())
        bags[cid] = products

    print(f"  {len(bags):,} customers with shopping bags")
    return bags


# =============================================================================
#  2. PHASE A -- PRODUCT LIFT ANALYSIS
# =============================================================================

def analyze_cluster_products(shopping_bags, labels_df, top_n=15):
    """
    For each cluster, compute product lift:
        lift = (% of cluster buying product) / (% of ALL customers buying product)

    Products with lift >> 1 are disproportionately popular in that cluster.

    Parameters
    ----------
    shopping_bags : dict {customer_id: set of products}
    labels_df     : DataFrame with columns 'customer_id' and 'cluster_name'
    top_n         : how many top products to keep per cluster

    Returns
    -------
    DataFrame with columns: cluster_name, product, cluster_pct, overall_pct, lift
    """
    print("  Computing product lift per cluster ...")

    # Overall product frequency (across ALL customers)
    total_customers = len(shopping_bags)
    overall_counts = Counter()
    for products in shopping_bags.values():
        overall_counts.update(products)

    # Merge labels with bags
    merged = labels_df[["customer_id", "cluster_name"]].copy()

    rows = []
    for cluster_name in sorted(merged["cluster_name"].unique()):
        cluster_ids = set(
            merged.loc[merged["cluster_name"] == cluster_name, "customer_id"]
        )
        cluster_size = len(cluster_ids)
        if cluster_size == 0:
            continue

        # Product frequency within cluster
        cluster_counts = Counter()
        for cid in cluster_ids:
            if cid in shopping_bags:
                cluster_counts.update(shopping_bags[cid])

        # Compute lift for each product
        for product, c_count in cluster_counts.items():
            cluster_pct = c_count / cluster_size
            overall_pct = overall_counts[product] / total_customers
            if overall_pct > 0:
                lift = cluster_pct / overall_pct
                rows.append({
                    "cluster_name": cluster_name,
                    "product": product,
                    "cluster_pct": round(cluster_pct, 4),
                    "overall_pct": round(overall_pct, 4),
                    "lift": round(lift, 4),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep only top_n per cluster (by lift)
    df = (
        df.sort_values(["cluster_name", "lift"], ascending=[True, False])
          .groupby("cluster_name")
          .head(top_n)
          .reset_index(drop=True)
    )
    return df


# =============================================================================
#  3. PHASE B -- APRIORI + BEST PAIRS
# =============================================================================

def run_apriori_per_cluster(shopping_bags, labels_df,
                            min_support=0.02, min_confidence=0.30, min_lift=1.3):
    """
    Run Apriori association rules on each cluster's shopping bags.

    Returns
    -------
    dict : {cluster_name: rules_df or None}
    """
    print("  Running Apriori per cluster ...")

    merged = labels_df[["customer_id", "cluster_name"]].copy()
    all_rules = {}

    for cluster_name in sorted(merged["cluster_name"].unique()):
        cluster_ids = set(
            merged.loc[merged["cluster_name"] == cluster_name, "customer_id"]
        )

        # Build transaction list
        transactions = [
            list(shopping_bags[cid])
            for cid in cluster_ids
            if cid in shopping_bags and len(shopping_bags[cid]) > 0
        ]

        if len(transactions) < 10:
            print(f"    {cluster_name}: too few transactions ({len(transactions)}), skipping")
            all_rules[cluster_name] = None
            continue

        # One-hot encode
        te = TransactionEncoder()
        te_arr = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_arr, columns=te.columns_)

        # Frequent itemsets
        try:
            freq = apriori(basket_df, min_support=min_support,
                           use_colnames=True, max_len=2)
        except Exception as e:
            print(f"    {cluster_name}: apriori failed ({e})")
            all_rules[cluster_name] = None
            continue

        if freq.empty:
            print(f"    {cluster_name}: no frequent itemsets at min_support={min_support}")
            all_rules[cluster_name] = None
            continue

        # Association rules
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
        rules = rules[rules["confidence"] >= min_confidence].copy()

        if rules.empty:
            print(f"    {cluster_name}: no rules survived filters")
            all_rules[cluster_name] = None
            continue

        rules["antecedents"] = rules["antecedents"].apply(list)
        rules["consequents"] = rules["consequents"].apply(list)
        rules.sort_values("lift", ascending=False, inplace=True)
        rules.reset_index(drop=True, inplace=True)

        all_rules[cluster_name] = rules
        print(f"    {cluster_name}: {len(rules):,} rules")

    return all_rules


def build_best_pairs(rules_df, top_n=30):
    """
    From association rules, keep only 1->1 rules for clean
    'buy A -> try B' product pairs.  Sorted by lift.
    """
    if rules_df is None or rules_df.empty:
        return pd.DataFrame(
            columns=["antecedent", "consequent", "support", "confidence", "lift"]
        )

    mask = (rules_df["antecedents"].apply(len) == 1) & \
           (rules_df["consequents"].apply(len) == 1)
    pairs = rules_df.loc[mask].copy()

    if pairs.empty:
        return pd.DataFrame(
            columns=["antecedent", "consequent", "support", "confidence", "lift"]
        )

    pairs["antecedent"] = pairs["antecedents"].apply(lambda x: x[0])
    pairs["consequent"] = pairs["consequents"].apply(lambda x: x[0])

    pairs = (
        pairs[["antecedent", "consequent", "support", "confidence", "lift"]]
        .sort_values("lift", ascending=False)
        .drop_duplicates(subset=["antecedent", "consequent"])
        .head(top_n)
        .reset_index(drop=True)
    )
    pairs["support"]    = pairs["support"].round(4)
    pairs["confidence"] = pairs["confidence"].round(4)
    pairs["lift"]       = pairs["lift"].round(4)
    return pairs


# =============================================================================
#  4. SAVE OUTPUTS
# =============================================================================

def save_outputs(cluster_products, all_rules, all_pairs):
    """Save all recommendation CSVs to output/recommendations/."""
    # Phase A: distinctive products
    if not cluster_products.empty:
        path = os.path.join(OUT_DIR, "cluster_distinctive_products.csv")
        cluster_products.to_csv(path, index=False)
        print(f"  Saved -> {path}")

    # Phase B: per-cluster rules and pairs
    for name in all_rules:
        safe_name = name.lower().replace(" ", "_")

        rules = all_rules[name]
        if rules is not None and not rules.empty:
            path = os.path.join(OUT_DIR, f"{safe_name}_rules.csv")
            rules.to_csv(path, index=False)

        pairs = all_pairs.get(name, pd.DataFrame())
        if not pairs.empty:
            path = os.path.join(OUT_DIR, f"{safe_name}_best_pairs.csv")
            pairs.to_csv(path, index=False)

    print(f"  All outputs saved to: {OUT_DIR}")


# =============================================================================
#  5. MAIN ORCHESTRATOR
# =============================================================================

def run_recommendations(labels_df):
    """
    Full recommendation pipeline.

    Parameters
    ----------
    labels_df : DataFrame with columns 'customer_id' and 'cluster_name'

    Returns
    -------
    dict with keys: 'cluster_products', 'all_rules', 'all_pairs'
    """
    print("=" * 60)
    print("  RECOMMENDATION PIPELINE")
    print("=" * 60)

    cluster_names = sorted(labels_df["cluster_name"].unique())
    print(f"  Clusters: {cluster_names}")

    # 1. Load basket + build shopping bags
    print("\n  Loading basket data ...")
    basket = pd.read_csv(BASKET_PATH)
    print(f"  {len(basket):,} transactions loaded")

    shopping_bags = build_shopping_bags(basket)

    # 2. Phase A: product lift
    print()
    cluster_products = analyze_cluster_products(shopping_bags, labels_df)

    # Print summary
    print("\n  DISTINCTIVE PRODUCTS (top 5 per cluster by lift):")
    print("  " + "-" * 56)
    for name in cluster_names:
        subset = cluster_products[cluster_products["cluster_name"] == name]
        if subset.empty:
            print(f"  {name}: (no data)")
            continue
        top5 = subset.head(5)
        products = ", ".join(
            f"{r['product']} ({r['lift']:.1f}x)" for _, r in top5.iterrows()
        )
        print(f"  {name}: {products}")

    # 3. Phase B: Apriori + best pairs
    print()
    all_rules = run_apriori_per_cluster(shopping_bags, labels_df)

    all_pairs = {}
    for name, rules in all_rules.items():
        all_pairs[name] = build_best_pairs(rules)

    # Print pairs summary
    print("\n  BEST PRODUCT PAIRS (top 3 per cluster by lift):")
    print("  " + "-" * 56)
    for name in cluster_names:
        pairs = all_pairs.get(name, pd.DataFrame())
        if pairs.empty:
            print(f"  {name}: (no pairs)")
        else:
            top3 = pairs.head(3)
            for _, r in top3.iterrows():
                print(f"  {name}: {r['antecedent']} -> {r['consequent']}  "
                      f"(lift={r['lift']:.2f}, conf={r['confidence']:.0%})")

    # 4. Save
    print()
    save_outputs(cluster_products, all_rules, all_pairs)

    print("\n" + "=" * 60)
    print("  Recommendations done!")
    print("=" * 60)

    return {
        "cluster_products": cluster_products,
        "all_rules": all_rules,
        "all_pairs": all_pairs,
    }


# =============================================================================
#  STANDALONE
# =============================================================================

if __name__ == "__main__":
    # Standalone mode: load labels from saved file or run K-Means
    from kmeans import KmeansClustering
    from cluster_profiles import get_cluster_names

    info_path = os.path.join(BASE_DIR, "data", "customer_info_cleaned.csv")
    data = pd.read_csv(info_path)
    data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])
    data_for_clustering = data.iloc[:, 4:].copy()

    km = KmeansClustering(min_k=2, max_k=11, data=data_for_clustering, random_seed=7)
    labels, _, centroids, _ = km.cluster(8, 20)
    names = get_cluster_names(centroids)

    labels_df = pd.DataFrame({
        "customer_id": data["customer_id"],
        "cluster_name": [names[l - 1] for l in labels],  # labels are 1-indexed
    })

    run_recommendations(labels_df)
