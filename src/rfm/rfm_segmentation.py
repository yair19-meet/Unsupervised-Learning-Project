"""
RFS Segmentation -- Recency * Frequency * Spending  (5-5-5 scoring)
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
                 retained in the output with segment = "No Basket History").

F  - Frequency : count(invoice_id) per customer from customer_basket.csv
                 = number of distinct shopping trips recorded.

S  - Spending  : sum of all lifetime_spend_* columns from
                 customer_info_cleaned.csv (total lifetime monetary value).

The resulting enriched customer table is saved to output/rsv/rfs_scores.csv
so it can be joined back with customer_basket.csv for association-rules.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
INFO_PATH     = os.path.join(BASE_DIR, "data", "customer_info.csv")
BASKET_PATH   = os.path.join(BASE_DIR, "data", "customer_basket.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output", "rsv")
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

# ── Segment label lookup ──────────────────────────────────────────────────────
def label_segment(row) -> str:
    """Map (R, F, S) scores -> a human-readable segment name."""
    r, f, s = row["R_score"], row["F_score"], row["S_score"]

    # Customers with no basket history get their own segment
    if pd.isna(r) or pd.isna(f):
        return "No Basket History"

    total = r + f + s

    if total >= 13:
        return "Champions"          # recent, frequent, big spenders
    elif total >= 10:
        if s >= 4:
            return "High Spenders"  # high monetary, moderate activity
        else:
            return "Loyal Visitors" # frequent & recent, moderate spend
    elif total >= 7:
        if r >= 4:
            return "Promising"      # recently active, building habit
        elif f <= 2:
            return "At Risk"        # used to spend but infrequent now
        else:
            return "Need Attention" # moderate everything
    else:
        return "Hibernating"        # low on all dimensions


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
def build_rfs(n_quantiles: int = 5) -> pd.DataFrame:
    """
    Load both datasets, compute RFS metrics, quintile-score each dimension,
    combine into a segment code, and return the enriched DataFrame.
    """
    print("Loading customer_info_cleaned.csv ...")
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

    # -- 3. Merge -- left join keeps all 33,038 customers ----------------------
    df = info.merge(basket_agg, on="customer_id", how="left")

    n_no_basket = df["recency_invoice"].isna().sum()
    print(f"  >> {len(df):,} total customers | "
          f"{df['recency_invoice'].notna().sum():,} with basket history | "
          f"{n_no_basket:,} without")

    # -- 4. Quintile scoring ---------------------------------------------------
    # Score only on customers who have basket data; NaN -> stays NaN
    df["R_score"] = rank_score(df["recency_invoice"], n_quantiles)
    df["F_score"] = rank_score(df["frequency"],       n_quantiles)
    df["S_score"] = rank_score(df["total_spending"],  n_quantiles)

    # -- 5. Segment code -------------------------------------------------------
    # Customers without basket history get code "---"
    df["RFS_code"] = df.apply(
        lambda row: (
            "---"
            if pd.isna(row["R_score"]) or pd.isna(row["F_score"])
            else f"{int(row['R_score'])}{int(row['F_score'])}{int(row['S_score'])}"
        ),
        axis=1,
    )
    df["RFS_total"]   = df[["R_score", "F_score", "S_score"]].sum(axis=1)
    df["RFS_segment"] = df.apply(label_segment, axis=1)

    return df


# -- Console summary -----------------------------------------------------------
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("RFS SCORE DISTRIBUTION  (basket customers only)")
    print("=" * 60)
    has_basket = df[df["RFS_segment"] != "No Basket History"]

    for dim, col in [("Recency (max invoice_id)", "R_score"),
                     ("Frequency (n invoices)",    "F_score"),
                     ("Spending (lifetime total)", "S_score")]:
        counts = has_basket[col].value_counts().sort_index()
        print(f"\n{dim}  -- 1 = lowest ... 5 = highest:")
        for score, cnt in counts.items():
            print(f"  {score}: {cnt:>6,} customers  ({cnt/len(has_basket)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("SEGMENT SIZES  (all customers)")
    print("=" * 60)
    for seg, cnt in df["RFS_segment"].value_counts().items():
        print(f"  {seg:<22} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_score_distributions(df: pd.DataFrame):
    """Bar charts of R, F, S score distributions (basket customers only)."""
    has_basket = df[df["RFS_segment"] != "No Basket History"]

    dims   = ["R_score", "F_score", "S_score"]
    titles = ["Recency Score\n(max invoice_id)", 
              "Frequency Score\n(# invoices)", 
              "Spending Score\n(lifetime total)"]
    colors = ["#4C9BE8", "#9B4CE8", "#E8874C"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=120)
    fig.suptitle("RFS Score Distributions -- Basket Customers Only  (1 = lowest * 5 = highest)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, col, title, color in zip(axes, dims, titles, colors):
        counts = has_basket[col].value_counts().sort_index()
        bars = ax.bar(counts.index.astype(int), counts.values,
                      color=color, edgecolor="white", linewidth=1.2, alpha=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Score", fontsize=11)
        ax.set_ylabel("Customers", fontsize=11)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "score_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.show()
    plt.close(fig)


def plot_segment_sizes(df: pd.DataFrame):
    """Horizontal bar chart of named segment sizes."""
    order = ["Champions", "High Spenders", "Loyal Visitors",
             "Promising", "Need Attention", "At Risk",
             "Hibernating", "No Basket History"]
    counts = df["RFS_segment"].value_counts().reindex(order, fill_value=0)

    palette = ["#2ECC71", "#27AE60", "#1ABC9C",
               "#F39C12", "#E67E22", "#E74C3C", "#95A5A6", "#BDC3C7"]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=palette[::-1], edgecolor="white", linewidth=1.2, alpha=0.9)
    ax.set_title("Customer Segments -- RFS 5-5-5 Scoring",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Customers", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 30, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "segment_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.show()
    plt.close(fig)


def plot_rfs_heatmap(df: pd.DataFrame):
    """Mean R, F, S scores per segment -- the segment profile card."""
    has_basket = df[df["RFS_segment"] != "No Basket History"].copy()
    # Cast nullable Int64 -> float64 so seaborn can render the heatmap
    for col in ["R_score", "F_score", "S_score"]:
        has_basket[col] = has_basket[col].astype(float)
    pivot = (has_basket
             .groupby("RFS_segment")[["R_score", "F_score", "S_score"]]
             .mean()
             .round(2))
    pivot.columns = ["Recency", "Frequency", "Spending"]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).drop(columns="total")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.8, ax=ax,
                cbar_kws={"label": "Average Score (1-5)"})
    ax.set_title("Average RFS Scores per Segment",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "segment_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.show()
    plt.close(fig)


def plot_rfs_bubble(df: pd.DataFrame):
    """
    Bubble chart: mean frequency vs mean spending per segment.
    Bubble size = number of customers. Color = mean recency score.
    """
    has_basket = df[df["RFS_segment"] != "No Basket History"].copy()
    for col in ["R_score", "F_score", "S_score"]:
        has_basket[col] = has_basket[col].astype(float)
    summary = has_basket.groupby("RFS_segment").agg(
        mean_spending  =("total_spending",  "mean"),
        mean_frequency =("frequency",       "mean"),
        mean_recency   =("R_score",         "mean"),
        count          =("customer_id",     "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(11, 7), dpi=120)
    scatter = ax.scatter(
        summary["mean_frequency"],
        summary["mean_spending"],
        s=summary["count"] / 5,
        c=summary["mean_recency"],
        cmap="RdYlGn",
        vmin=1, vmax=5,
        alpha=0.85,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )
    for _, row in summary.iterrows():
        ax.annotate(
            row["RFS_segment"],
            (row["mean_frequency"], row["mean_spending"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=10,
            fontweight="bold",
        )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mean Recency Score (1=oldest * 5=newest)", fontsize=10)
    ax.set_xlabel("Mean Number of Invoices (Frequency)", fontsize=12)
    ax.set_ylabel("Mean Total Lifetime Spending", fontsize=12)
    ax.set_title(
        "Customer Segments: Spending vs Frequency\n"
        "(bubble size = segment population  *  color = recency)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "segment_bubble.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_rfs(n_quantiles=5)

    print_summary(df)

    # ── Save enriched table for use in association-rules analysis ─────────────
    export_cols = [
        "customer_id",
        "recency_invoice", "frequency", "total_spending",
        "R_score", "F_score", "S_score",
        "RFS_code", "RFS_total", "RFS_segment",
    ]
    out_path = os.path.join(OUTPUT_DIR, "rfs_scores.csv")
    df[export_cols].to_csv(out_path, index=False)
    print(f"\nRFS scores saved -> {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_score_distributions(df)
    plot_segment_sizes(df)
    plot_rfs_heatmap(df)
    plot_rfs_bubble(df)
    print(df)
    
