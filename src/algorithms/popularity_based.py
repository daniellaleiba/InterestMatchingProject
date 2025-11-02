"""
popularity_based.py
=========================

Popularity-Based (Cluster-Aware) Recommendation Algorithm

This algorithm recommends hobbies based on their global popularity 
among users, enhanced by correlation and cluster-level adjustments. 
It balances between individual item popularity, user preference patterns, 
and structural relationships between hobby clusters.

Core Idea:
----------
Rather than relying solely on item similarity, this method measures 
how “popular” each hobby is (fraction of users who rated it highly), 
then refines the recommendation by incorporating:
- Item-item correlations (for relevance within user context).
- Cluster-level boosts (to reinforce related hobby groups).
- Popularity smoothing (to handle sparse data).

Steps:
1. Normalize user ratings and compute the item-item correlation matrix.
2. Compute smoothed global popularity for each hobby.
3. Calculate cluster-level popularity and apply a dynamic boost.
4. Blend correlation and popularity into a unified score.
5. Recommend top-ranked hobbies not yet liked by the user.
6. Evaluate performance on the test set.

Advantages:
------------
- Interpretable: recommendations are explainable via popularity + cluster logic.
- Balanced: integrates both global trends and personalized affinity.
- Robust: smoothed scores mitigate overfitting to rare hobbies.

Author: Daniella Leiba
"""

import pandas as pd
import numpy as np
from utils.helpers import select_sample_users,  evaluate_recommender_generic

def normalize_user_ratings(df):
    # Normalize user ratings per user (row-wise z-score normalization).
    # Each user's ratings are centered and scaled to reduce individual bias
    df_norm = df.copy()
    df_norm = df_norm.sub(df_norm.mean(axis=1), axis=0)
    df_norm = df_norm.div(df_norm.std(axis=1).replace(0, 1), axis=0)
    return df_norm

def compute_hobby_popularity(train_data, hobby_cols, rating_threshold=4,alpha=5):
    """
    Compute smoothed global popularity of each hobby:
    Combines observed rating proportion with the global average
    to reduce sensitivity to small sample sizes.
    """
    counts = (train_data[hobby_cols] >= rating_threshold).sum()
    proportions = (train_data[hobby_cols] >= rating_threshold).mean()

    global_mean = proportions.mean()

    smoothed_popularity = (proportions * counts + global_mean * alpha) / (counts + alpha)

    return smoothed_popularity 


def popularity_method_clustered(user_row, corr_matrix, clustered_features, rating_threshold=4, popularity_series=None):
    """
    Generate hobby recommendations using a hybrid, cluster-aware popularity model.

    Combines three signals:
    - **Base similarity**: measures how related each hobby is to the user’s liked ones.
    - **Cluster context**: dynamically boosts hobbies from clusters the user tends to favor.
    - **Global popularity**: ensures stable relevance to broader user trends.

    Final scoring blends contextual relevance (similarity + cluster effects)
    with global popularity for a balanced recommendation.

    Steps:
        1. Compute base similarity (average correlation with liked hobbies).
        2. Normalize similarity and popularity scores to [0, 1].
        3. Compute cluster-level popularity and a dynamic boost.
        4. Apply cluster boost to similarity to obtain contextual relevance.
        5. Blend contextual relevance and popularity into the final score.
        6. Sort and return top-10 hobbies not already liked by the user.
    """

    liked_hobbies = [h for h in user_row.index if user_row[h] >= rating_threshold]
    if not liked_hobbies:
        return [], []

    cluster_map = clustered_features.set_index("Interest")["Cluster"].to_dict()
    cluster_ids = sorted(clustered_features["Cluster"].unique())

    # Step 1: Base similarity (average correlation with liked hobbies)
    base_similarity = corr_matrix[liked_hobbies].mean(axis=1).fillna(0)

    # Step 2: Global hobby popularity (scaled 0–1)
    hobby_popularity = popularity_series.reindex(base_similarity.index).fillna(0)
    hobby_popularity = (hobby_popularity - hobby_popularity.min()) / (
        hobby_popularity.max() - hobby_popularity.min() + 1e-6
    )

    # Step 3: Cluster popularity (mean popularity per cluster)
    cluster_popularity = pd.Series({
        c: hobby_popularity[[h for h, cl in cluster_map.items() if cl == c]].mean()
        for c in cluster_ids
    }).fillna(0)
    cluster_popularity = cluster_popularity.fillna(cluster_popularity.mean())

    # Step 4: Dynamic cluster boost
    cluster_boost_factors = (cluster_popularity / cluster_popularity.mean()) ** 1.2
    cluster_boost_factors = cluster_boost_factors.clip(lower=0.9, upper=1.6)

    # Apply cluster boost to obtain contextual relevance
    contextual_relevance = base_similarity.copy()
    for hobby in contextual_relevance.index:
        cluster_id = cluster_map[hobby]
        contextual_relevance[hobby] *= cluster_boost_factors[cluster_id]

    # Step 5: Normalize contextual relevance and compute final score
    contextual_scaled = (contextual_relevance - contextual_relevance.min()) / (
        contextual_relevance.max() - contextual_relevance.min() + 1e-6
    )
    final_score = 0.65 * contextual_scaled + 0.35 * hobby_popularity

    # Step 6: Filter out liked hobbies and rank
    recommended = final_score.sort_values(ascending=False).index.tolist()
    recommended = [r for r in recommended if r not in liked_hobbies]
    recommended_clusters = [cluster_map[r] for r in recommended]

    return recommended[:10], recommended_clusters[:10]

def run_popularity_pipeline(train_data, test_data, hobby_cols, clustered_features, top_k, rating_threshold, weight_5=1.25, alpha=0.7):
    """
    Full pipeline for the Popularity recommendation algorithm.

    - Normalize train rows, compute item correlation.
    - Compute global popularity per hobby.
    - Generate recommendations for test users.
    - Print a small sample.
    - Evaluate using the same algo+params (including popularity_series for consistency).
    """
    print("\n◆  Popularity-Based (Clustered) Algorithm:\n")
    # Step 1.  Compute correlation matrix on training data and  Normalize train data per user
    train_data_norm = normalize_user_ratings(train_data)
    corr_matrix = train_data_norm.corr()
    # 1.1: Compute Global popularity weights
    popularity_series = compute_hobby_popularity(train_data, hobby_cols)
    
    # Step 2.  Generate recommendations for for test users
    recommendations_dict = {}
    recommendation_clusters_dict = {}
    for user_id, row in test_data.iterrows():
        recs, rec_clusters = popularity_method_clustered(
            user_row=row,
            corr_matrix=corr_matrix,
            clustered_features=clustered_features,
            rating_threshold=rating_threshold,
            popularity_series=popularity_series
        )
        recommendations_dict[user_id] = recs
        recommendation_clusters_dict[user_id] = rec_clusters
        

    # Step 3. Print sample users
    print(f"\n~~~ Popularity-Based (Clustered) Recommendations (Example Users) ~~~")
    sample_users = select_sample_users(test_data, n_samples=10)
    for user_id in sample_users:
        preferred = [
            f"{h} (C{clustered_features.loc[clustered_features['Interest'] == h, 'Cluster'].values[0]})"
            for h in hobby_cols
            if test_data.loc[user_id, h] >= rating_threshold
        ]
        recs = recommendations_dict[user_id][:top_k]
        rec_clusters = list(dict.fromkeys(recommendation_clusters_dict[user_id][:top_k]))

        print(f"User {user_id}")
        print(f"Preferred hobbies (with clusters): {preferred}")
        print(f"Recommended: {recs}")
        print(f"Recommended clusters: {rec_clusters}")
        print("------")


    # Step 4. Evaluation 
    eval_results = evaluate_recommender_generic(
        test_df=test_data, top_k=top_k,
        recommend_func=lambda *args, **kwargs: popularity_method_clustered(*args, **kwargs)[0],
        algo_kwargs={
            "corr_matrix": corr_matrix,
            "clustered_features": clustered_features,
            "rating_threshold": rating_threshold,
            "popularity_series": popularity_series
        },
        rating_threshold=rating_threshold
        )

    return eval_results