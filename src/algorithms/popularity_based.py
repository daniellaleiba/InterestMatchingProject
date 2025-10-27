"""
popularity_recommender.py
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

def popularity_method_clustered( user_row, corr_matrix, clustered_features, rating_threshold=4, popularity_series=None ):
    """
    Generate hobby recommendations using a hybrid, cluster-aware popularity model.

    This method combines three key signals:
    - **Item-item correlation**: measures similarity between hobbies.
    - **Global popularity**: fraction of users rating each hobby above the threshold.
    - **Cluster boost**: reinforces hobbies belonging to popular clusters.

    Each user’s preferred hobbies (ratings ≥ threshold) guide the recommendation
    through correlation averaging, while global and cluster-level popularity adjust
    the final ranking to balance personalization and global trends.

    Logic:
        1. Compute mean correlation between user’s liked hobbies and all others.
        2. Normalize correlation and popularity scores to [0, 1].
        3. Compute cluster-level popularity and a dynamic boost:
               cluster_boost = (cluster_popularity / mean_cluster_popularity) ^ 1.2
               (bounded between 0.9 and 1.6)
        4. Apply the boost to the correlation values.
        5. Blend correlation and popularity:
               final_score = 0.65 * correlation + 0.35 * popularity
        6. Sort and return top-10 hobbies not already liked by the user.
    """

    liked_hobbies = [h for h in user_row.index if user_row[h] >= rating_threshold]
    if not liked_hobbies:
        return [], []

    cluster_map = clustered_features.set_index("Interest")["Cluster"].to_dict()
    cluster_ids = sorted(clustered_features["Cluster"].unique())

    #  Step 1: Base correlation score
    avg_corr = corr_matrix[liked_hobbies].mean(axis=1).fillna(0)

    #  Step 2: Hobby popularity (scaled 0–1)
    hobby_popularity = popularity_series.reindex(avg_corr.index).fillna(0)
    hobby_popularity = (hobby_popularity - hobby_popularity.min()) / (hobby_popularity.max() - hobby_popularity.min() + 1e-6)

    #  Step 3: Cluster popularity (mean popularity per cluster)
    cluster_popularity = pd.Series({
        c: hobby_popularity[[h for h, cl in cluster_map.items() if cl == c]].mean()
        for c in cluster_ids
    })
    cluster_popularity = cluster_popularity.fillna(cluster_popularity.mean())

    #  Step 4: Dynamic cluster boost (stronger impact)
    cluster_boost_factors = (cluster_popularity / cluster_popularity.mean()) ** 1.2
    cluster_boost_factors = cluster_boost_factors.clip(lower=0.9, upper=1.6)

    boosted_corr = avg_corr.copy()
    for hobby in boosted_corr.index:
        cluster_id = cluster_map[hobby]
        boosted_corr[hobby] *= cluster_boost_factors[cluster_id]

    #  Step 5: Final combination (normalized)
    corr_scaled = (boosted_corr - boosted_corr.min()) / (boosted_corr.max() - boosted_corr.min() + 1e-6)
    final_score = 0.65 * corr_scaled + 0.35 * hobby_popularity

    #  Step 6: Sort and filter liked hobbies
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
        preferred = {h: int(test_data.loc[user_id, h])
                     for h in hobby_cols
                     if test_data.loc[user_id, h] >= rating_threshold}
        recs = recommendations_dict[user_id][:top_k]
        rec_clusters = list(dict.fromkeys(recommendation_clusters_dict[user_id][:top_k]))
        print(f"User {user_id}")
        print(f"Preferred hobbies: {preferred}")
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