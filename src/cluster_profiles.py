import numpy as np
from scipy.optimize import linear_sum_assignment



def assign_labels_by_heuristics(new_centroids, feature_names, min_score=0.8):
    """
    Assigns cluster names using a heuristic scoring matrix and 
    global bipartite matching to prevent duplicates.
    """
    k_clusters = len(new_centroids)
    
    # 1. Map features to their column indices for readability
    idx = {name: i for i, name in enumerate(feature_names)}
    

    # We expand to 14 labels to give the Hungarian algorithm breathing room.
    labels = [
        "Gamers", "Big Families", "Promo Hunters", "Vegetarians", 
        "Karens", "Body Conscious", "Average Nice Joes", "Grocery Guys",
        "Meat Lovers", "Party Animals", "New Parents", "Variety Seekers", 
        "Tech Enthusiasts", "Pet Owners"
    ]

    score_matrix = np.zeros((k_clusters, len(labels)))

    for i, c in enumerate(new_centroids):
        
        # 1. Gamers: High games, high electronics. (Weighted towards games to separate from pure tech)
        score_matrix[i, 0] = (c[idx["lifetime_spend_videogames"]] * 1.5) + c[idx["lifetime_spend_electronics"]]
        
        # 2. Big Families: High kids and teens, plus high general groceries.
        score_matrix[i, 1] = c[idx["kids_home"]] + c[idx["teens_home"]] + (c[idx["lifetime_spend_groceries"]] * 0.5)
        
        # 3. Promo Hunters: High promo percentage, penalized if they have huge total spend (usually budget-conscious).
        score_matrix[i, 2] = c[idx["percentage_of_products_bought_promotion"]] - (c[idx["lifetime_spend_groceries"]] * 0.5)
        
        # 4. Vegetarians: High veg, explicitly penalized for meat/fish spend.
        score_matrix[i, 3] = c[idx["lifetime_spend_vegetables"]] - c[idx["lifetime_spend_meat"]] - c[idx["lifetime_spend_fish"]]
        
        # 5. Karens: High complaints, high distinct stores (shopping around to complain).
        score_matrix[i, 4] = (c[idx["number_complaints"]] * 2.0) + c[idx["distinct_stores_visited"]]
        
        # 6. Body Conscious: Hygiene focus, penalize alcohol and high promos (brand loyal).
        score_matrix[i, 5] = c[idx["lifetime_spend_hygiene"]] - c[idx["lifetime_spend_alcohol_drinks"]]
        
        # 7. Average Nice Joes: No extremes. Normal groceries, zero/low complaints.
        score_matrix[i, 6] = c[idx["lifetime_spend_groceries"]] - (c[idx["number_complaints"]] * 2.0)
        
        # 8. Grocery Guys: Pure high volume grocery/meat/veg spenders, low on niche items like games.
        score_matrix[i, 7] = c[idx["lifetime_spend_groceries"]] + c[idx["lifetime_spend_meat"]] - c[idx["lifetime_spend_videogames"]]
        
        # 9. Meat Lovers: High meat/fish, low veg.
        score_matrix[i, 8] = c[idx["lifetime_spend_meat"]] + c[idx["lifetime_spend_fish"]] - c[idx["lifetime_spend_vegetables"]]
        
        # 10. Party Animals: High alcohol, high non-alcohol, penalize kids (usually younger/single demographics).
        score_matrix[i, 9] = c[idx["lifetime_spend_alcohol_drinks"]] + c[idx["lifetime_spend_nonalcohol_drinks"]] - c[idx["kids_home"]]
        
        # 11. New Parents: High kids, explicitly low teens.
        score_matrix[i, 10] = c[idx["kids_home"]] - c[idx["teens_home"]]
        
        # 12. Variety Seekers: Massive product diversity and store hopping.
        score_matrix[i, 11] = c[idx["lifetime_total_distinct_products"]] + c[idx["distinct_stores_visited"]]
        
        # 13. Tech Enthusiasts: High electronics, but average/low games (e.g., buying appliances/smart home gear).
        score_matrix[i, 12] = c[idx["lifetime_spend_electronics"]] - c[idx["lifetime_spend_videogames"]]
        
        # 14. Pet Owners: Pet food is the dominant defining trait.
        score_matrix[i, 13] = c[idx["lifetime_spend_petfood"]] * 2.0
            
            
    # 3. Global Resolution (The Magic Step)
    # linear_sum_assignment minimizes cost. Since we want to MAXIMIZE scores, 
    # we pass the negative of the score matrix.
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    
    final_assignments = [None] * k_clusters
    
    for cluster_idx, label_idx in zip(row_ind, col_ind):
        best_score = score_matrix[cluster_idx, label_idx]
        
        # The Rejection Gate
        if best_score < min_score:
            final_assignments[cluster_idx] = f"Undefined_Cluster_{cluster_idx}"
        else:
            final_assignments[cluster_idx] = labels[label_idx]

    return final_assignments


from kmeans import *


# TESTING
# if __name__ == "__main__":
#     data = pd.read_csv("./data/customer_info_cleaned.csv")
#     data["customer_birthdate"] = pd.to_datetime(data["customer_birthdate"])
#     data_for_clustering = data.iloc[:, 4:].copy()
#     algo = KmeansClustering(4, 12, data_for_clustering, 7)

#     kmeans_labels, inertia, centroids, cluster_avgs = algo.cluster(8, 20)

#     feature_names = list(data_for_clustering.columns)

#     final_assignments = assign_labels_by_heuristics(new_centroids=centroids, feature_names=feature_names)

#     print(final_assignments)
#     algo.plot_cluster_profiles(centroids=centroids, feature_names=feature_names)

