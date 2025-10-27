"""
fof_recommender.py
==================

Friends-of-Friends (FOF) Recommendation Algorithm

This algorithm recommends hobbies to users based on the structure 
of a weighted interest graph, where nodes represent hobbies and 
edges represent their relationships (similarity and co-occurrence).

The FOF approach propagates recommendations through direct and 
indirect connections — leveraging both first-degree (directly similar) 
and second-degree (friends-of-friends) relationships.

Core Idea:
If two hobbies are often liked together by the same users or appear 
in the same clusters, they reinforce each other in the graph.
Users who like one hobby are more likely to enjoy its “neighbors” 
or related hobbies via indirect connections.

Steps:
1. Build a weighted interest graph using cosine and cluster similarities.
2. Refine the graph using co-occurrence support and second-order connections.
3. For each user, identify liked hobbies (ratings ≥ threshold).
4. Propagate scores through direct and indirect links to find related hobbies.
5. Evaluate overall recommendation performance on test data.

Author: Daniella Leiba
"""

import numpy as np
import pandas as pd
from utils.helpers import select_sample_users, evaluate_recommender_generic
from feature_engineering import build_weighted_interest_graph
from exploratory_analysis import plot_interest_graph

def fof_recommendations(df_or_row, hobby_cols, G,k, sorted_neighbors=None, rating_threshold=4, fof_weight=0.5):
    """
    Generate Friends-of-Friends (FOF) hobby recommendations.
    Supports both a single user (pd.Series) and a full dataset (pd.DataFrame).
    """

    # Precompute sorted neighbors once (if not provided)
    if sorted_neighbors is None:
        sorted_neighbors = {
            node: sorted(G[node].items(), key=lambda x: x[1]['weight'], reverse=True)
            for node in G.nodes()
        }

    # Case 1: Single user (pd.Series)
    if isinstance(df_or_row, pd.Series):
        user_row = df_or_row
        liked = [h for h in hobby_cols if user_row[h] >= rating_threshold]
        if not liked:
            return []

        scores = {}
        # Direct neighbors
        for interest in liked:
            weight_factor = user_row[interest] / 5.0
            for neighbor, data in sorted_neighbors.get(interest, []):
                if neighbor in liked:
                    continue
                #  reliability weighting
                reliability = data.get('support', 1)
                reliability_factor = np.log1p(reliability)

                # Total contribution
                score_add = weight_factor * data['weight'] * reliability_factor
                scores[neighbor] = scores.get(neighbor, 0) + score_add

        # Friends-of-Friends
        if len(scores) < k:
            for rec, rec_score in list(scores.items()):
                for fof, data in sorted_neighbors.get(rec, []):
                    if fof not in liked:
                        continue
                    hop_weight = fof_weight * rec_score * data['weight']
                    scores[fof] = scores.get(fof, 0) + hop_weight

        return sorted(scores, key=scores.get, reverse=True)[:k]

    # Case 2: Full DataFrame
    recommendations = {}
    for user_id, row in df_or_row.iterrows():
        recommendations[user_id] = fof_recommendations(
            row, hobby_cols, G,k, sorted_neighbors,  rating_threshold, fof_weight
        )

    return recommendations

def refine_interest_graph(G, train_data, hobby_cols, min_shared_users=5, fof_reinforce=0.1, prune_threshold=0.02):
    """
    Refines an existing interest graph by reinforcing meaningful connections
    (based on co-occurrence) and pruning weak or noisy edges.
    """
     #  1. Compute a co-occurrence matrix (binary likes ≥ threshold) ---
    # This matrix counts how many users like both hobbies i and j.
    hobby_matrix = (train_data[hobby_cols] >= 4).astype(int)
    co_counts = pd.DataFrame(hobby_matrix.T @ hobby_matrix, 
                             index=hobby_cols, columns=hobby_cols)

    #  2. Reinforce strong edges and remove weak ones ---
    for i, j, data in list(G.edges(data=True)):
        shared = co_counts.loc[i, j] if i in co_counts.index and j in co_counts.columns else 0
        data["support"] = shared # NEW
        # Remove edges that have very low user overlap
        if shared < min_shared_users:
            G.remove_edge(i, j)
            continue
        # Reinforce existing edges proportionally to number of shared users
        G[i][j]['weight'] *= np.log1p(shared)  # log1p prevents excessive boosting

    #  3.  Add second-order (FOF) connections
    for node in list(G.nodes()):
        neighbors = list(G.neighbors(node))
        for a in neighbors:
            for b in neighbors:
                if a == b:
                    continue
                if not G.has_edge(a, b):
                    new_weight = fof_reinforce * (G[node][a]['weight'] * G[node][b]['weight'])
                    if new_weight > prune_threshold:
                        G.add_edge(a, b, weight=new_weight)

    # 4. Normalize edge weights to [0, 1] range 
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    max_w = max(weights) if weights else 1.0
    for i, j, d in G.edges(data=True):
        d['weight'] = d['weight'] / max_w

    #  5. prune very weak edges again 
    for i, j, d in list(G.edges(data=True)):
        if d['weight'] < prune_threshold:
            G.remove_edge(i, j)

    print(f"Graph refinement complete: {len(G.nodes())} nodes, {len(G.edges())} edges remaining.")
    return G

def run_pipeline_fof(train_data, test_data, clustered_features_train, hobby_cols, top_k, rating_threshold,alpha=0.75, threshold_percentile=85):
    """
    Full pipeline for the Friends-of-Friends (FOF) recommendation algorithm:
    1. Build a weighted hobby graph using the training set only.
    2. Refine the graph structure (support-based reinforcement & pruning).
    3. Generate recommendations for test users.
    4. Display sample recommendations for selected users.
    5. Evaluate algorithm performance (HitRate, Precision, Recall).
    """
    print("\n◆ Friend-of-Friend Algorithm (FOF):\n")
    #  Step 1: Build weighted interest graph (train only)
    G, cluster_corr = build_weighted_interest_graph(train_data, clustered_features_train, alpha, threshold_percentile)

    # Step 2: Refine the graph structure
    G = refine_interest_graph(G, train_data, hobby_cols)
    print(f"Refined graph: {len(G.nodes())} nodes, {len(G.edges())} edges.")

    # Step 3: Visualize and save the refined graph
    plot_interest_graph(G, clustered_features_train,alpha,threshold_percentile,
                        save=True, name_suffix="_FOF")

    # Step 4: Precompute sorted neighbor list
    sorted_neighbors = {
    node: sorted(G[node].items(), key=lambda x: x[1]['weight'], reverse=True)
        for node in G.nodes()
        }

    #Step 5: Generate recommendations for test users
    test_recs = fof_recommendations(test_data, hobby_cols, G, top_k, sorted_neighbors, rating_threshold=rating_threshold)

    # Step 6: Print sample users
    sample_users = select_sample_users(test_data, n_samples=10)
    print("\n~~~ FOF Recommendations (Example Users) ~~~")
    for user_id in sample_users:
        liked_with_ratings = {h: int(test_data.loc[user_id, h]) for h in hobby_cols if test_data.loc[user_id, h] >= rating_threshold}
        recs = test_recs[user_id] 
        print(f"User {user_id}")
        print(f"Preferred hobbies: {liked_with_ratings}")
        print(f"Recommended: {recs}")
        print("------")

    # Step 7: Evaluate performance
    eval_results = evaluate_recommender_generic(
        test_df=test_data,
        top_k=top_k,
        recommend_func=fof_recommendations, 
        algo_kwargs={
            "G": G,
            "k": top_k,
            "hobby_cols": hobby_cols,
            "sorted_neighbors": sorted_neighbors,
            "rating_threshold": rating_threshold
        },
        rating_threshold=rating_threshold
    )

    return eval_results