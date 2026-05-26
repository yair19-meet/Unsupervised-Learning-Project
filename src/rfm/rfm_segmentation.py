"""
RFM Segmentation -- Recency * Frequency * Monetary  (5-5-5 scoring)
===================================================================
Assigns every customer a score from 1 (lowest) to 5 (highest) on each of
three dimensions derived from two datasets, then combines them into a
three-digit segment code (e.g. "555").

Dimensions
----------
R  - Recency   : max(invoice_id) per customer from customer_basket.csv
                 invoice_id is sequential, so the highest value = most recent
                 transaction. Customers not in the basket file receive NaN
                 and are scored 0 (excluded from score-based analysis but
                 retained in the output with segment = "---").

F  - Frequency : count(invoice_id) per customer from customer_basket.csv
                 = number of distinct shopping trips recorded.

M  - Monetary  : sum of all lifetime_spend_* columns from
                 customer_info.csv (total lifetime monetary value).

The resulting enriched customer table is saved to src/rfm/rfm_scores.csv
so it can be joined back with customer_basket.csv for association-rules.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
INFO_PATH     = os.path.join(BASE_DIR, "data", "customer_info.csv")
BASKET_PATH   = os.path.join(BASE_DIR, "data", "customer_basket.csv")
OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Spend columns (Monetary dimension) ───────────────────────────────────────
SPEND_COLS = [
    "lifetime_spend_groceries",
    "lifetime_spend_electronics",
    "lifetime_spend_vegetables",
    "lifetime_spend_nonalcohol_drinks",
    "lifetime_spend_alcohol_drinks",
    "lifetime_spend_meat",
    "lifetime_spend_fish",
    "lifetime_spend_hygiene",
    "lifetime_spend_videogames",
    "lifetime_spend_petfood",
]

# ── Scoring helper ────────────────────────────────────────────────────────────
def rank_score(series: pd.Series, n: int = 5) -> pd.Series:
    """
    Score a numeric series into n equal-population buckets (1=lowest, n=highest).
    Uses rank(method='first') to avoid duplicate-bin-edge errors on
    low-cardinality / tied columns.
    NaN values are preserved as NaN.
    """
    ranked = series.rank(method="first", ascending=True, na_option="keep")
    return pd.qcut(ranked, q=n, labels=list(range(1, n + 1))).astype("Int64")


# -- Main build function -------------------------------------------------------
def build_rfm(n_quantiles: int = 5) -> pd.DataFrame:
    """
    Load both datasets, compute RFM metrics, quintile-score each dimension,
    combine into a segment code, and return the enriched DataFrame.
    """
    print("Loading customer_info.csv ...")
    info = pd.read_csv(INFO_PATH)

    print("Loading customer_basket.csv ...")
    basket = pd.read_csv(BASKET_PATH)

    # -- 1. Derive basket-level metrics per customer ---------------------------
    basket_agg = basket.groupby("customer_id")["invoice_id"].agg(
        recency_invoice =("max"),   # highest invoice_id = most recent trip
        frequency       =("count"), # number of shopping trips
    ).reset_index()

    # -- 2. Compute total spending from customer_info --------------------------
    info["total_spending"] = info[SPEND_COLS].sum(axis=1)

    # -- 3. Merge -- left join keeps all customers -----------------------------
    df = info.merge(basket_agg, on="customer_id", how="left")

    n_no_basket = df["recency_invoice"].isna().sum()
    print(f"  >> {len(df):,} total customers | "
          f"{df['recency_invoice'].notna().sum():,} with basket history | "
          f"{n_no_basket:,} without")

    # -- 4. Quintile scoring ---------------------------------------------------
    # Score only on customers who have basket data; NaN -> stays NaN
    df["R_score"] = rank_score(df["recency_invoice"], n_quantiles)
    df["F_score"] = rank_score(df["frequency"],       n_quantiles)
    df["M_score"] = rank_score(df["total_spending"],  n_quantiles)

    # -- 5. Segment code -------------------------------------------------------
    # Customers without basket history get code "---"
    df["RFM_code"] = df.apply(
        lambda row: (
            "---"
            if pd.isna(row["R_score"]) or pd.isna(row["F_score"])
            else f"{int(row['R_score'])}{int(row['F_score'])}{int(row['M_score'])}"
        ),
        axis=1,
    )
    df["RFM_total"] = df[["R_score", "F_score", "M_score"]].sum(axis=1)

    return df


# -- Console summary -----------------------------------------------------------
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("RFM SCORE DISTRIBUTION  (basket customers only)")
    print("=" * 60)
    has_basket = df[df["RFM_code"] != "---"]

    for dim, col in [("Recency (max invoice_id)", "R_score"),
                     ("Frequency (n invoices)",    "F_score"),
                     ("Monetary (lifetime total)", "M_score")]:
        counts = has_basket[col].value_counts().sort_index()
        print(f"\n{dim}  -- 1 = lowest ... 5 = highest:")
        for score, cnt in counts.items():
            print(f"  {score}: {cnt:>6,} customers  ({cnt/len(has_basket)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("SEGMENT SIZES  (all customers)")
    print("=" * 60)
    for seg, cnt in df["RFM_code"].value_counts().head(20).items():
        print(f"  {seg:<22} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_segment_sizes(df: pd.DataFrame):
    """Horizontal bar chart of top RFM codes."""
    # Exclude the "---" segment (no basket history) from the plot
    counts = df[df["RFM_code"] != "---"]["RFM_code"].value_counts()
    
    # Take top 20 for readability if there are many segments
    top_counts = counts.head(20)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=120)
    bars = ax.barh(top_counts.index[::-1], top_counts.values[::-1],
                   color="#3498db", edgecolor="white", linewidth=1.2, alpha=0.9)
    ax.set_title("Customer Segments (Top 20 RFM Codes)",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Customers", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar in bars:
        w = bar.get_width()
        ax.text(w + (max(top_counts.values) * 0.01), bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "segment_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_rfm(n_quantiles=5)

    print_summary(df)

    # ── Save enriched table for use in association-rules analysis ─────────────
    export_cols = [
        "customer_id",
        "recency_invoice", "frequency", "total_spending",
        "R_score", "F_score", "M_score",
        "RFM_code", "RFM_total"
    ]
    out_path = os.path.join(OUTPUT_DIR, "rfm_scores.csv")
    df[export_cols].to_csv(out_path, index=False)
    print(f"\nRFM scores saved -> {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_segment_sizes(df)
