"""
cluster_profiles.py
===================
Hardcoded reference centroids from K-Means (k=8, seed=7, epochs=20)
and descriptive cluster names assigned by domain analysis.

The key function `get_cluster_names(centroids)` matches new centroids
to these references by Euclidean distance, so the descriptive labels
are always correct regardless of K-Means label shuffling.
"""

import numpy as np

# ── Reference centroids (8 clusters x 18 features) ──────────────────────────
# Order: Vegetarians, Gamers, Grocery Guys, Big Families,
#        Body Conscious, Karens, Average Nice Joes, Promo Hunters
REFERENCE_CENTROIDS = np.array([
    [ 0.032099, -0.101492, -0.237348,  0.098260, -0.194863,  0.175936,  0.267106,  1.788425, -0.013872, -0.089404, -0.978171, -0.558100,  0.026742, -0.057908, -0.321673, -0.281546, -0.403883, -0.015825],
    [-0.668126, -0.620958,  0.062818, -0.672969, -0.297781,  2.457651,  0.615417, -0.257888,  0.307645,  0.789245,  0.228216,  0.415272, -0.474997,  5.336679, -0.080267, -0.087727,  0.076174,  0.295056],
    [ 0.036461,  0.006439,  0.055608,  0.424115,  1.681738,  0.713860,  0.142761, -0.078485,  0.727259,  0.497878,  0.522082,  0.667126,  0.607675,  0.442460,  0.138156,  0.953739,  0.383312, -0.320134],
    [ 2.724882,  1.825792, -0.082623,  0.097887,  0.970883,  1.135192, -0.240565,  0.438347,  1.001963,  1.542515,  1.172943,  1.087152,  0.302545,  0.755204,  0.170251,  1.255524, -0.093033, -0.241998],
    [ 0.048836, -0.134806, -0.445970,  0.301940, -0.071301, -0.198637, -0.162082,  1.205838,  0.481504, -0.447524, -0.837846, -0.511316,  1.737306, -0.382307,  0.413837,  0.001088, -0.247527,  0.216333],
    [ 0.840690, -0.265402,  1.362586, -0.360690, -0.145113, -0.121299, -0.179634, -0.191927, -0.410391,  0.129636, -0.171570, -0.001120, -0.271120, -0.346254, -0.096891, -0.340519,  0.615473,  0.275328],
    [-0.420779, -0.170365, -0.694015, -0.600000,  0.374232, -0.144560, -0.468856,  0.136849, -0.257643, -0.320625,  0.114588,  0.252830,  0.156429, -0.206150,  0.386669,  0.212159, -0.269486, -0.239451],
    [-0.386627, -0.497916, -0.282296,  0.718828, -0.078079, -0.119808,  0.350765, -0.046233, -0.375166, -0.014669, -0.038969, -0.088825, -0.230513, -0.011073, -0.115391, -0.038309,  0.948101,  0.259572],
])

# Each name corresponds to the REFERENCE_CENTROIDS row with the same index.
# Dominant feature per row (verified from centroid values):
#   0 → lifetime_spend_vegetables high           → Vegetarians
#   1 → lifetime_spend_videogames high           → Gamers
#   2 → lifetime_spend_groceries high            → Grocery Guys
#   3 → kids_home / teens_home high              → Big Families
#   4 → lifetime_spend_hygiene high              → Body Conscious
#   5 → number_complaints high                   → Karens
#   6 → lifetime_spend_petfood / avg groceries   → Average Nice Joes
#   7 → percentage_of_products_bought_promotion  → Promo Hunters
CLUSTER_NAMES = [
    "Vegetarians",
    "Gamers",
    "Grocery Guys",
    "Big Families",
    "Body Conscious",
    "Karens",
    "Average Nice Joes",
    "Promo Hunters",
]


def get_cluster_names(centroids):
    """
    Match new centroids to reference centroids by Euclidean distance.

    Parameters
    ----------
    centroids : np.ndarray, shape (k, n_features)
        Centroids from a new K-Means run.

    Returns
    -------
    names : list[str]
        Descriptive names in the order of the input centroids.
        e.g. if centroids[0] is closest to the Gamers reference,
        names[0] = "Gamers".
    """
    from scipy.spatial.distance import cdist

    # Pairwise distances: (k_new x k_ref)
    distances = cdist(centroids, REFERENCE_CENTROIDS, metric="euclidean")

    names = []
    used = set()
    for i in range(len(centroids)):
        # Find closest unused reference centroid
        order = np.argsort(distances[i])
        for j in order:
            if j not in used:
                names.append(CLUSTER_NAMES[j])
                used.add(j)
                break

    return names
