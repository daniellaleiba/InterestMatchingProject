"""
user_based_recommender.py
=========================

User-Based Collaborative Filtering Recommendation Algorithm

This algorithm recommends hobbies to a target user based on the preferences 
of other users with similar rating patterns. It computes user-user similarity 
using Pearson correlation (mean-centered) or cosine similarity (angle-based), 
and predicts unseen ratings from the user’s most similar neighbors.

Steps:
1. Build a sparse user–hobby matrix and split into train/test sets.
2. Compute similarity matrices (Pearson & Cosine).
3. Identify top-K nearest neighbors per user.
4. Predict missing ratings and generate recommendations.
5. Evaluate performance and compare both similarity methods.

Advantages:
-----------
- Captures behavioral similarity between users.
- Normalizes personal bias via Pearson correlation.
- Provides interpretable, data-driven user relationships.

Author: Daniella Leiba
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import (
    create_sparse_user_matrix,
    train_test_split_from_sparse,
    select_sample_users,
    evaluate_usersimilarity_recommender
)

def compute_user_similarity(train_df, method="pearson"):
    """
    Compute a user–user similarity matrix using Pearson or Cosine similarity.
    """
    train_df = train_df.astype(float)

    if method == "cosine":
        sim_matrix = cosine_similarity(np.nan_to_num(train_df, nan=0))

    elif method == "pearson":
        df_centered = train_df.sub(train_df.mean(axis=1), axis=0)
        sim_matrix = np.corrcoef(df_centered.fillna(0))
    else:
        raise ValueError("method must be either 'pearson' or 'cosine'")
 
    sim_df = pd.DataFrame(sim_matrix, index=train_df.index, columns=train_df.index)
    np.fill_diagonal(sim_df.values, 0)
    sim_df = sim_df.fillna(0)
    return sim_df

def get_top_k_neighbors(sim_df, k=10):
    """
    Return a dictionary of the top-K most similar neighbors for each user.
    """
    neighbors_dict = {}

    for user in sim_df.index:
        top_k = sim_df.loc[user][sim_df.loc[user] > 0].sort_values(ascending=False).head(k)
        if top_k.empty:
            top_k = sim_df.loc[user].sort_values(ascending=False).head(k)

        neighbors_dict[user] = top_k.to_dict()

    return neighbors_dict

def predict_user_ratings(train_df, neighbors_dict, sim_df, method="pearson"):
    """
    Predict hobby ratings for each user based on their most similar neighbors.

    Supports both:
    - 'pearson': mean-centered normalization.
    - 'cosine' : weighted average based on similarity strength.
    """
    predictions = pd.DataFrame(index=train_df.index, columns=train_df.columns, dtype=float)
    # Calculate average personal ratings for each user
    user_means = train_df.mean(axis=1)

    for user in train_df.index:
        neighbors = list(neighbors_dict.get(user, {}).keys())
        sims = np.array(list(neighbors_dict.get(user, {}).values()))

        # if no neighbors at all - > move to next
        if len(neighbors) == 0:
            continue

        # get neighbors ranking
        neighbor_ratings = train_df.loc[neighbors]

        # Cosine
        if method == "cosine":
            weighted_sum = np.nansum(neighbor_ratings.mul(sims, axis=0), axis=0)
            sim_sum = np.nansum(np.abs(sims))
            predictions.loc[user] = weighted_sum / (sim_sum + 1e-8)

        # Pearson
        elif method == "pearson":
            # Calculating averages of neighbors
            neighbor_means = neighbor_ratings.mean(axis=1)
            # Calculating deviation from the mean for each neighbor  (r_vi - mean_v)
            diff_from_mean = neighbor_ratings.sub(neighbor_means, axis=0)
            #Calculating a weighted sum based on similarity values: Σ(sim(u,v) * (r_vi - mean_v))
            weighted_sum = np.nansum(diff_from_mean.mul(sims, axis=0), axis=0)
            # Calculating the total similarity : Σ|sim(u,v)|
            sim_sum = np.nansum(np.abs(sims))
            # Adding our user average : mean_u + (weighted_sum / sim_sum)
            predictions.loc[user] = user_means[user] + (weighted_sum / (sim_sum + 1e-8))
        
        else:
            raise ValueError("method must be 'pearson' or 'cosine'")

    return predictions


def generate_recommendations_from_predictions(predictions, train_df, top_k=5):
    """
    Convert predicted rating matrix into ranked hobby recommendations for each user.
    """
    recommendations = {}

    for user in predictions.index:
        already_rated = train_df.loc[user].dropna().index.tolist()
        user_preds = predictions.loc[user].dropna()
        user_preds = user_preds.drop(index=already_rated, errors="ignore")
        top_recs = user_preds.sort_values(ascending=False).head(top_k).index.tolist()
        recommendations[user] = top_recs

    return recommendations


def us_cf_pipeline(df,top_k, rating_threshold, missing_ratio):
    """
    Full pipeline for User-Based Collaborative Filtering:
    1. Convert data to sparse matrix and split train/test.
    2. Compute similarity matrices (Pearson & Cosine).
    3. Identify top-K neighbors for each user.
    4. Predict ratings and generate recommendations.
    5. Evaluate and compare both similarity methods.
    """
    print("\n◆  User-Based Collaborative Filtering Algorithm:\n")
    # Step 1: Prepare sparse data
    df_sparse = create_sparse_user_matrix(df, missing_ratio)
    
    # Step 2: Split Train/Test
    train_data, test_data = train_test_split_from_sparse(df, df_sparse)

    # NOTE: Running algorithmn using : Pearson, Cosine (to compare)

   # Step 3: Compute similarity (train only)
    sim_cosine = compute_user_similarity(train_data, method="cosine")
    sim_pearson = compute_user_similarity(train_data, method="pearson")

    # Step 4: Retrieve top-K neighbors
    neighbors_dict_pearson = get_top_k_neighbors(sim_pearson, k=10)
    neighbors_dict_cosine = get_top_k_neighbors(sim_cosine, k=10)

    # Step 5: Predict ratings
    predicted_ratings_pearson = predict_user_ratings(train_data, neighbors_dict_pearson, sim_pearson, method="pearson")
    predicted_ratings_cosine = predict_user_ratings(train_data, neighbors_dict_cosine, sim_cosine, method="cosine")

    # Step 6: Generate recommendations
    recommendations_pearson = generate_recommendations_from_predictions(predicted_ratings_pearson, train_data, top_k=10) 
    recommendations_cosine = generate_recommendations_from_predictions(predicted_ratings_cosine, train_data, top_k=10) 

    # Step 7: Print sample users
    sample_users = select_sample_users(test_data, n_samples=10)
    print("\n~~~ User-Based Collaborative Filtering Recommendations (Example Users) ~~~")
    for user_id in sample_users:
        
        preferred_hobbies = {
            h: int(test_data.loc[user_id, h])
            for h in test_data.columns
            if not pd.isna(test_data.loc[user_id, h]) and test_data.loc[user_id, h] >= rating_threshold
        }
        # Algo`s recommendations`
        recs_pearson = recommendations_pearson.get(user_id, [])[:top_k]
        recs_cosine = recommendations_cosine.get(user_id, [])[:top_k]

        print(f"\nUser {user_id}")
        print(f"Preferred hobbies: {preferred_hobbies}")
        print(f"Recommended (using pearson method): {recs_pearson}")
        print(f"Recommended (using cosine method): {recs_cosine}")
        print("------")


    # checking of over leap :::: 
    train_overlap = (train_data.notna() & test_data.notna()).sum().sum()
    print("Overlap between train and test:", train_overlap)

    # Step 8: Evaluate performance
    print("\n=== Evaluation Comparison ===")
    eval_results_pearson = evaluate_usersimilarity_recommender(recommendations_pearson, test_data,'Pearson', top_k=top_k, rating_threshold=4)
    eval_results_cosine = evaluate_usersimilarity_recommender(recommendations_cosine, test_data,'Cosine', top_k=top_k, rating_threshold=4)
    # conclusion : pearson run better

    return eval_results_pearson