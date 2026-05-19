"""
cluster_recommendation.py
==========================
Replicates the Merged_Dataset_Notebook pipeline end-to-end as a standalone
Python script.

What it does
------------
1.  Loads customer_info_cleaned.csv and customer_basket.csv.
2.  Merges them and builds each customer's unique shopping bag.
3.  Scales the numeric features and runs K-Means (k = 8) exactly as in the
    notebook (same seed, same feature slice, same epoch count).
4.  Produces 8 DataFrames – one per cluster – each containing:
        customer_id  |  shopping_bag  (list of unique products)
5.  Runs the Apriori algorithm on every cluster's transaction list to mine
    frequent itemsets and association rules.
6.  Derives product RECOMMENDATIONS for each cluster:
        • "consequents" that appear most frequently as rule conclusions and
          are NOT already among the cluster's overall top-N products.
7.  Saves results to  output/recommendations/
        cluster_<k>_customers.csv   – the per-cluster customer+bag DataFrames
        cluster_<k>_rules.csv       – the association rules
        cluster_<k>_recommendations.csv – ranked new-product suggestions

Dependencies
------------
    pip install mlxtend
(pandas, numpy, scikit-learn are already used by the rest of the project)

Usage
-----
    cd <project_root>
    python src/cluster_recommendation.py
"""

import os
import sys
import ast
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── allow importing sibling modules (clustering.py / hierachichal.py) ─────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clustering import KmeansClustering

# ── mlxtend for Apriori ────────────────────────────────────────────────────────
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    raise ImportError(
        "mlxtend is required.  Install it with:  pip install mlxtend"
    )

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INFO_PATH   = os.path.join(BASE_DIR, "data", "customer_info_cleaned.csv")
BASKET_PATH = os.path.join(BASE_DIR, "data", "customer_basket.csv")
OUT_DIR     = os.path.join(BASE_DIR, "output", "recommendations")
os.makedirs(OUT_DIR, exist_ok=True)

K                = 9       # number of K-Means clusters (matches notebook)
RANDOM_SEED      = 7      # same seed as notebook
KMEANS_EPOCHS    = 100     # n_init passed to KMeans (same as notebook)

# Apriori thresholds – tweak if runtimes are too long or rules too sparse
MIN_SUPPORT      = 0.02    # item must appear in ≥ 5 % of cluster transactions
MIN_CONFIDENCE   = 0.30    # rule must have ≥ 30 % confidence
MIN_LIFT         = 1.3     # only keep rules that beat random co-occurrence
TOP_N_RECS       = 20      # how many new-product recommendations to keep per cluster

# ==============================================================================
# STEP 1 – LOAD DATA
# ==============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading customer_info_cleaned.csv ...")
    info = pd.read_csv(INFO_PATH)
    print(f"  >> {len(info):,} customers loaded.")

    print("Loading customer_basket.csv ...")
    basket = pd.read_csv(BASKET_PATH)
    print(f"  >> {len(basket):,} transactions loaded.")

    return info, basket


# ==============================================================================
# STEP 2 – BUILD SHOPPING BAGS (replicates notebook cells)
# ==============================================================================

def safe_parse(val) -> list:
    """Convert a string representation of a list into an actual list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return []


def parse_goods(val) -> list:
    """Flatten nested / string-encoded product lists."""
    if isinstance(val, float):       # NaN
        return []
    if isinstance(val, list):
        flat = []
        for item in val:
            if isinstance(item, list):
                flat.extend([str(p).strip() for p in item])
            else:
                flat.append(str(item).strip())
        return flat
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        try:
            parsed = ast.literal_eval(val)
            flat = []
            for item in parsed:
                if isinstance(item, list):
                    flat.extend([str(p).strip() for p in item])
                else:
                    flat.append(str(item).strip())
            return flat
        except Exception as e:
            print(f"  [WARN] Failed to parse: {repr(val[:60])} → {e}")
            return []
    return []


def build_shopping_bags(info: pd.DataFrame, basket: pd.DataFrame) -> pd.DataFrame:
    """
    Merge customer info with basket data and derive each customer's unique
    shopping bag (union of all products bought across all transactions).
    Returns the enriched merged DataFrame.
    """
    print("\nBuilding per-customer shopping bags ...")

    # Aggregate all transactions per customer → unique product list
    basket_cleaned = (
        basket
        .groupby("customer_id")["list_of_goods"]
        .apply(lambda x: list(set(
            product
            for transaction in x
            for product in safe_parse(transaction)
        )))
        .reset_index()
    )
    basket_cleaned.columns = ["customer_id", "list_of_goods"]

    # Left-join so every customer is retained
    merged = pd.merge(info, basket_cleaned, on="customer_id", how="left")

    # Flatten and deduplicate the product lists
    merged["all_products"]    = merged["list_of_goods"].apply(parse_goods)
    merged["unique_products"] = merged["all_products"].apply(
        lambda products: list(set(products))
    )
    merged["product_counts"]  = merged["all_products"].apply(
        lambda products: dict(Counter(products))
    )
    merged.drop(columns=["all_products"], inplace=True)

    print(f"  >> {len(merged):,} rows in merged DataFrame.")
    return merged


# ==============================================================================
# STEP 3 – CLUSTER (replicates notebook cells)
# ==============================================================================

def run_kmeans(merged: pd.DataFrame) -> np.ndarray:
    """
    Scale the same feature slice used in the notebook (iloc[:, 4:-3]) and
    run K-Means with k = K.  Returns the 1-indexed cluster labels array.
    """
    print(f"\nRunning K-Means (k = {K}, seed = {RANDOM_SEED}, epochs = {KMEANS_EPOCHS}) ...")

    # The notebook slices: merged_df.iloc[:, 4:-3]
    # After merge that is columns 4 to the third-from-last
    data_for_clustering = merged.iloc[:, 4:-3].copy()

    scaler      = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_clustering)

    km = KmeansClustering(
        min_k=2,
        max_k=K + 3,
        data=data_scaled,
        random_seed=RANDOM_SEED,
    )

    labels, inertia, centroids = km.cluster(k=K, epochs=KMEANS_EPOCHS)

    print(f"  >> Inertia: {inertia:,.2f}")
    unique, counts = np.unique(labels, return_counts=True)
    for seg, cnt in zip(unique, counts):
        print(f"     Cluster {seg:2d}: {cnt:,} customers")

    return labels


# ==============================================================================
# STEP 4 – SPLIT INTO 8 PER-CLUSTER DataFrames
# ==============================================================================

def build_cluster_dataframes(
    merged: pd.DataFrame, labels: np.ndarray
) -> dict[int, pd.DataFrame]:
    """
    Attach the cluster labels and return a dict mapping cluster_id (1-8) to a
    DataFrame with columns: customer_id, shopping_bag (list of unique products).
    """
    merged = merged.copy()
    merged["cluster_label"] = labels

    cluster_dfs: dict[int, pd.DataFrame] = {}
    for cluster_id in range(1, K + 1):
        mask = merged["cluster_label"] == cluster_id
        df   = (
            merged.loc[mask, ["customer_id", "unique_products"]]
            .rename(columns={"unique_products": "shopping_bag"})
            .reset_index(drop=True)
        )
        cluster_dfs[cluster_id] = df
        print(f"  Cluster {cluster_id}: {len(df):,} customers  "
              f"({mask.sum() / len(merged) * 100:.1f} %)")

    return cluster_dfs


# ==============================================================================
# STEP 5 – APRIORI + ASSOCIATION RULES (per cluster)
# ==============================================================================

def run_apriori_for_cluster(
    cluster_df: pd.DataFrame,
    cluster_id: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Mine frequent itemsets and association rules from the shopping bags of one
    cluster.  Returns (rules_df, frequent_itemsets) or (None, None) if not
    enough data.
    """
    # Build the transaction list (list of lists)
    transactions = [
        bag for bag in cluster_df["shopping_bag"] if isinstance(bag, list) and len(bag) > 0
    ]

    if len(transactions) < 10:
        print(f"  [SKIP] Cluster {cluster_id}: too few transactions ({len(transactions)}).")
        return None, None

    # One-hot encode via TransactionEncoder
    te     = TransactionEncoder()
    te_arr = te.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(te_arr, columns=te.columns_)

    # Frequent itemsets
    try:
        frequent_items = apriori(
            basket_df,
            min_support=MIN_SUPPORT,
            use_colnames=True,
            max_len=4,          # itemsets up to 4 products (keeps it tractable)
        )
    except Exception as e:
        print(f"  [ERR]  Cluster {cluster_id} apriori failed: {e}")
        return None, None

    if frequent_items.empty:
        print(f"  [INFO] Cluster {cluster_id}: no frequent itemsets found at "
              f"min_support = {MIN_SUPPORT}.")
        return None, None

    # Association rules
    rules = association_rules(
        frequent_items,
        metric="lift",
        min_threshold=MIN_LIFT,
    )
    rules = rules[rules["confidence"] >= MIN_CONFIDENCE].copy()

    if rules.empty:
        print(f"  [INFO] Cluster {cluster_id}: no rules survived confidence filter.")
        return None, frequent_items

    rules["antecedents"] = rules["antecedents"].apply(list)
    rules["consequents"] = rules["consequents"].apply(list)
    rules.sort_values("lift", ascending=False, inplace=True)
    rules.reset_index(drop=True, inplace=True)

    return rules, frequent_items


# ==============================================================================
# STEP 6 – DERIVE RECOMMENDATIONS
# ==============================================================================

def derive_recommendations(
    cluster_df: pd.DataFrame,
    rules: pd.DataFrame,
    cluster_id: int,
    top_n: int = TOP_N_RECS,
) -> pd.DataFrame:
    """
    From the association rules select the most-recommended NEW products –
    i.e. consequents that are NOT already in the cluster's top-50 bestsellers.
    Products are ranked by (weighted_lift = sum of lift across all rules where
    the product appears as a consequent).
    """
    # Cluster's own top-50 bestsellers (products already well known in cluster)
    all_products: Counter = Counter()
    for bag in cluster_df["shopping_bag"]:
        if isinstance(bag, list):
            all_products.update(bag)
    top_50 = {item for item, _ in all_products.most_common(50)}

    # Accumulate lift for every consequent product
    rec_lift: dict[str, float] = {}
    rec_conf: dict[str, float] = {}
    rec_supp: dict[str, float] = {}

    for _, row in rules.iterrows():
        for product in row["consequents"]:
            if product not in top_50:
                rec_lift[product]  = rec_lift.get(product, 0.0)  + row["lift"]
                rec_conf[product]  = max(rec_conf.get(product, 0.0), row["confidence"])
                rec_supp[product]  = max(rec_supp.get(product, 0.0), row["support"])

    if not rec_lift:
        return pd.DataFrame(columns=["product", "total_lift", "max_confidence", "max_support"])

    rec_df = pd.DataFrame({
        "product":        list(rec_lift.keys()),
        "total_lift":     list(rec_lift.values()),
        "max_confidence": [rec_conf[p] for p in rec_lift],
        "max_support":    [rec_supp[p] for p in rec_lift],
    })
    rec_df.sort_values("total_lift", ascending=False, inplace=True)
    rec_df = rec_df.head(top_n).reset_index(drop=True)
    return rec_df


# ==============================================================================
# STEP 7 – BEST PRODUCT PAIRS BY LIFT  (new – what you asked for!)
# ==============================================================================

def build_best_pairs_df(rules: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    From the raw association rules DataFrame build a clean, human-readable
    table of the *best product pairs* sorted by lift.

    Only keeps 1-antecedent → 1-consequent rules so every row is a clean
    "Product A  →  Product B" pair (easier to read and act on).

    Columns returned
    ----------------
    antecedent   : the product the customer already bought
    consequent   : the product to recommend
    support      : share of transactions containing both
    confidence   : P(buy B | bought A)
    lift         : how much more likely than random co-occurrence (higher = better)
    """
    if rules is None or rules.empty:
        return pd.DataFrame(
            columns=["antecedent", "consequent", "support", "confidence", "lift"]
        )

    # Keep only 1→1 rules for clarity
    mask = rules["antecedents"].apply(len) == 1
    mask &= rules["consequents"].apply(len) == 1
    pairs = rules.loc[mask].copy()

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


# ==============================================================================
# STEP 8 – PER-CUSTOMER RECOMMENDATIONS
# ==============================================================================

def recommend_for_customer(
    customer_id,
    cluster_dfs: dict[int, pd.DataFrame],
    all_rules:   dict[int, pd.DataFrame | None],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Given a customer_id, find every association rule where:
      • the customer HAS bought all antecedent products, AND
      • the customer has NOT yet bought the consequent product.

    Returns a DataFrame sorted by lift:

    customer_id | cluster | antecedent (bought) | consequent (recommend) | confidence | lift

    Usage
    -----
        recs = recommend_for_customer("CUS-00042", cluster_dfs, all_rules)
        print(recs.to_string(index=False))
    """
    # ── find which cluster the customer belongs to ─────────────────────────────
    customer_cluster = None
    customer_bag     = set()
    for cid, df in cluster_dfs.items():
        row = df[df["customer_id"] == customer_id]
        if not row.empty:
            customer_cluster = cid
            bag = row.iloc[0]["shopping_bag"]
            customer_bag = set(bag) if isinstance(bag, list) else set()
            break

    if customer_cluster is None:
        print(f"  [WARN] customer_id '{customer_id}' not found in any cluster.")
        return pd.DataFrame(
            columns=["customer_id", "cluster", "antecedent", "consequent", "confidence", "lift"]
        )

    rules = all_rules.get(customer_cluster)
    if rules is None or rules.empty:
        print(f"  [INFO] No rules available for cluster {customer_cluster}.")
        return pd.DataFrame(
            columns=["customer_id", "cluster", "antecedent", "consequent", "confidence", "lift"]
        )

    # ── filter rules: bought all antecedents, missing consequent ──────────────
    rows = []
    for _, rule in rules.iterrows():
        antecedents = rule["antecedents"]
        consequents = rule["consequents"]

        # Customer must own ALL antecedent products
        if not set(antecedents).issubset(customer_bag):
            continue

        # Recommend only products the customer does NOT already have
        missing = [p for p in consequents if p not in customer_bag]
        for product in missing:
            rows.append({
                "customer_id":  customer_id,
                "cluster":      customer_cluster,
                "antecedent":   ", ".join(antecedents),   # readable string
                "consequent":   product,
                "support":      round(rule["support"],    4),
                "confidence":   round(rule["confidence"], 4),
                "lift":         round(rule["lift"],       4),
            })

    if not rows:
        print(f"  [INFO] No new recommendations for customer '{customer_id}' "
              f"(cluster {customer_cluster}).")
        return pd.DataFrame(
            columns=["customer_id", "cluster", "antecedent", "consequent",
                     "support", "confidence", "lift"]
        )

    rec_df = (
        pd.DataFrame(rows)
        .sort_values("lift", ascending=False)
        .drop_duplicates(subset=["consequent"])
        .head(top_n)
        .reset_index(drop=True)
    )
    return rec_df


# ==============================================================================
# STEP 9 – SAVE OUTPUTS
# ==============================================================================

def save_outputs(
    cluster_dfs:  dict[int, pd.DataFrame],
    all_rules:    dict[int, pd.DataFrame | None],
    all_recs:     dict[int, pd.DataFrame],
    all_pairs:    dict[int, pd.DataFrame],
) -> None:
    """Persist all per-cluster CSVs to output/recommendations/."""
    for cluster_id in range(1, K + 1):
        # Customer + shopping bag
        cust_path = os.path.join(OUT_DIR, f"cluster_{cluster_id}_customers.csv")
        df_out = cluster_dfs[cluster_id].copy()
        df_out["shopping_bag"] = df_out["shopping_bag"].apply(
            lambda bag: str(bag) if isinstance(bag, list) else "[]"
        )
        df_out.to_csv(cust_path, index=False)

        # Association rules (may be None)
        rules = all_rules.get(cluster_id)
        if rules is not None and not rules.empty:
            rules_path = os.path.join(OUT_DIR, f"cluster_{cluster_id}_rules.csv")
            rules.to_csv(rules_path, index=False)

        # Best product pairs by lift
        pairs = all_pairs.get(cluster_id)
        if pairs is not None and not pairs.empty:
            pairs_path = os.path.join(OUT_DIR, f"cluster_{cluster_id}_best_pairs.csv")
            pairs.to_csv(pairs_path, index=False)

        # Cluster-level recommendations
        recs = all_recs.get(cluster_id)
        if recs is not None and not recs.empty:
            recs_path = os.path.join(OUT_DIR, f"cluster_{cluster_id}_recommendations.csv")
            recs.to_csv(recs_path, index=False)

    print(f"\nAll outputs saved to: {OUT_DIR}")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> dict:
    """
    Full pipeline.  Returns a results dict so the function can also be
    imported and called from a notebook:

        from cluster_recommendation import main, recommend_for_customer
        results = main()

        results["cluster_dfs"]   # dict of 8 customer DataFrames
        results["all_rules"]     # dict of 8 raw rules DataFrames
        results["all_pairs"]     # dict of 8 best-pairs DataFrames (lift-sorted)
        results["all_recs"]      # dict of 8 cluster-level recommendation DFs

        # Per-customer recommendation:
        recs = recommend_for_customer("CUS-00042",
                                      results["cluster_dfs"],
                                      results["all_rules"])
    """
    print("=" * 65)
    print("  CLUSTER RECOMMENDATION PIPELINE")
    print("=" * 65)

    # ── 1. Load ────────────────────────────────────────────────────
    info, basket = load_data()

    # ── 2. Build shopping bags ─────────────────────────────────────
    merged = build_shopping_bags(info, basket)

    # ── 3. K-Means clustering ──────────────────────────────────────
    labels = run_kmeans(merged)

    # ── 4. Split into cluster DataFrames ──────────────────────────
    print(f"\nSplitting into {K} cluster DataFrames ...")
    cluster_dfs = build_cluster_dataframes(merged, labels)

    # -- 5. Apriori per cluster -------------------------------------
    all_rules: dict[int, pd.DataFrame | None] = {}
    all_pairs: dict[int, pd.DataFrame]        = {}   # <- best pairs by lift
    all_recs:  dict[int, pd.DataFrame]        = {}   # <- cluster-level new-product recs

    print(f"\nRunning Apriori on each cluster "
          f"(min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}) ...")

    for cluster_id in range(1, K + 1):
        print(f"\n  -- Cluster {cluster_id} ----------------------------------")
        cdf = cluster_dfs[cluster_id]

        rules, _ = run_apriori_for_cluster(cdf, cluster_id)
        all_rules[cluster_id] = rules

        if rules is not None and not rules.empty:
            # Best product pairs by lift ----------------------------
            pairs = build_best_pairs_df(rules, top_n=30)
            all_pairs[cluster_id] = pairs

            # Cluster-level new-product recommendations -------------
            recs = derive_recommendations(cdf, rules, cluster_id)
            all_recs[cluster_id] = recs

            print(f"  >> {len(rules):,} rules mined  |  "
                  f"{len(pairs)} best pairs  |  "
                  f"{len(recs)} new-product recommendations")
        else:
            all_pairs[cluster_id] = pd.DataFrame(
                columns=["antecedent", "consequent", "support", "confidence", "lift"]
            )
            all_recs[cluster_id] = pd.DataFrame()
            print(f"  >> No recommendations could be generated.")

    # -- 6. Save outputs --------------------------------------------
    save_outputs(cluster_dfs, all_rules, all_recs, all_pairs)

    # -- 7. Print best-pairs preview (top-5 per cluster) -----------
    print("\n" + "=" * 65)
    print("  BEST PRODUCT PAIRS BY LIFT  (top 5 per cluster)")
    print("=" * 65)
    for cluster_id in range(1, K + 1):
        pairs = all_pairs.get(cluster_id, pd.DataFrame())
        print(f"\n  Cluster {cluster_id}  ({len(cluster_dfs[cluster_id]):,} customers)")
        if pairs.empty:
            print("    (no pairs found)")
        else:
            print(pairs.head(5).to_string(index=False))

    # -- 8. Cluster-level recommendation summary --------------------
    print("\n" + "=" * 65)
    print("  CLUSTER-LEVEL NEW-PRODUCT RECOMMENDATIONS (top 3)")
    print("=" * 65)
    for cluster_id in range(1, K + 1):
        recs   = all_recs.get(cluster_id, pd.DataFrame())
        n_cust = len(cluster_dfs[cluster_id])
        if recs.empty:
            print(f"  Cluster {cluster_id:2d}  ({n_cust:,} customers)  ->  no recommendations")
        else:
            top3 = ", ".join(recs["product"].head(3).tolist())
            print(f"  Cluster {cluster_id:2d}  ({n_cust:,} customers)  ->  top-3: {top3}")

    print("\nDone.")
    return {
        "cluster_dfs": cluster_dfs,
        "all_rules":   all_rules,
        "all_pairs":   all_pairs,
        "all_recs":    all_recs,
    }


if __name__ == "__main__":
    results = main()

    cluster_dfs = results["cluster_dfs"]
    all_rules   = results["all_rules"]
    all_pairs   = results["all_pairs"]

    # -- Show cluster 1 customer DataFrame -------------------------
    print("\n--- Cluster 1: customer shopping bags (first 10 rows) ---")
    print(cluster_dfs[1].head(10).to_string())

    # -- Show best product pairs for cluster 1 ---------------------
    print("\n--- Cluster 1: best product pairs by LIFT (top 10) ---")
    print(all_pairs[1].head(10).to_string(index=False))

    # -- Per-customer recommendation example -----------------------
    # Replace the ID below with any real customer_id from your data.
    # The function finds rules where the customer bought the antecedent
    # but is MISSING the consequent, then ranks by lift.
    example_id = cluster_dfs[1].iloc[0]["customer_id"]   # first customer in cluster 1
    print(f"\n--- Personalised recommendations for customer '{example_id}' ---")
    personal_recs = recommend_for_customer(example_id, cluster_dfs, all_rules)
    if personal_recs.empty:
        print("  No targeted recommendations found (customer may own all recommended products).")
    else:
        print(personal_recs.to_string(index=False))
