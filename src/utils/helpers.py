"""
utils/helpers.py
----------------
Utility functions shared across the Interest Matching Project.

This module centralizes helper utilities for:
1. Data preparation (creating sparse matrices, splitting train/test)
2. Sampling users for evaluation or display
3. Evaluation metrics for recommendation algorithms
   - Generic evaluator (for on-the-fly recommendations)
   - User-similarity evaluator (for precomputed recommendation lists)

Author: Daniella Leiba
"""

import numpy as np
import pandas as pd

# DATA PREPARATION UTILITIES
def create_sparse_user_matrix(df, missing_ratio, min_rated_items=5, random_state=42):
    """
    Create a sparse user-item rating matrix by randomly removing a percentage of ratings.
    This simulates real-world sparsity for recommender evaluation.
    """

    np.random.seed(random_state)
    df_sparse = df.copy()

    for user in df_sparse.index:
        rated_items = df_sparse.columns[df_sparse.loc[user].notna()]
        n_remove = int(len(rated_items) * missing_ratio)

        #  Ensure minimum rated items remain
        if len(rated_items) - n_remove < min_rated_items:
            n_remove = max(0, len(rated_items) - min_rated_items)

        if n_remove > 0:
            remove_items = np.random.choice(rated_items, size=n_remove, replace=False)
            df_sparse.loc[user, remove_items] = np.nan

    return df_sparse

def train_test_split_from_sparse(df_full, df_sparse):
    """
    Create train/test matrices from original and sparse datasets.

    Train = visible ratings (used by the model)
    Test  = hidden ratings (used for evaluation)
    """

    train_df = df_sparse.copy()
    test_df = df_full.copy()

    # Keep only hidden ratings in test
    test_df[~train_df.isna()] = np.nan
    return train_df, test_df


# 2. SAMPLING UTILITIES
def select_sample_users(train_data, n_samples=10, random_state=42):
    """
    Selects a fixed set of random users from the training set for consistent examples.
    Returns a numpy array of user IDs (index values).
    """
    np.random.seed(random_state)
    sample_users = np.random.choice(train_data.index, size=min(n_samples, len(train_data)), replace=False)
    return sample_users


# 3. EVALUATION FUNCTIONS
def evaluate_recommender_generic(
    test_df,top_k, recommend_func, algo_kwargs=None,
    rating_threshold=4, multi_hit_threshold=None
):
    """
    Generic evaluation function for any recommendation algorithm.

    - For each user, hide a subset of liked hobbies (>= rating_threshold).
    - Generate recommendations and compare with hidden hobbies.
    - Computes:
        * Hit Rate (≥1 hit)
        * Hit Rate (≥multi_hit_threshold hits)
        * Precision@K, Recall@K
    """
    import math
    np.random.seed(42)
    if algo_kwargs is None:
        algo_kwargs = {}
    if multi_hit_threshold is None:
        multi_hit_threshold = math.ceil(top_k / 2)

    hit_count_1 = 0
    hit_count_multi = 0
    precision_scores = []
    recall_scores = []
    total_users = 0

    for user_id, row in test_df.iterrows():
        liked_hobbies = [h for h, r in row.items() if r >= rating_threshold]
        if len(liked_hobbies) < 3:
            continue

        # Hide part of liked hobbies
        test_hobbies = np.random.choice(liked_hobbies, size=max(1, len(liked_hobbies)//3), replace=False)
        masked_row = row.copy()
        for h in test_hobbies:
            masked_row[h] = 0

        # Generate recommendations
        recs = recommend_func(masked_row, **algo_kwargs)
        top_recs = recs[:top_k]

        # Evaluate hits
        hits = len(set(top_recs) & set(test_hobbies))
        if hits >= 1:
            hit_count_1 += 1
        if hits >= multi_hit_threshold:
            hit_count_multi += 1

        precision_scores.append(hits / top_k if top_k > 0 else 0)
        recall_scores.append(hits / len(test_hobbies))
        total_users += 1

     #  Use shared metrics function
    return compute_eval_metrics(hit_count_1, hit_count_multi,
                                precision_scores, recall_scores,
                                total_users, top_k, multi_hit_threshold)
    
def evaluate_usersimilarity_recommender(recommendations, test_df, method,
                                        top_k=5, rating_threshold=4,
                                        multi_hit_threshold=3):
    """
    Evaluation function for algorithms that output precomputed user recommendations 
    (e.g., user-based or matrix factorization models).e
    """
    hit_count_1 = 0
    hit_count_multi = 0
    precision_scores = []
    recall_scores = []
    total_users = 0

    for user, recs in recommendations.items():
        true_likes = set(test_df.columns[(test_df.loc[user] >= rating_threshold)
                                         & (~test_df.loc[user].isna())])
        if len(true_likes) == 0:
            continue

        recs_topk = set(recs[:top_k])
        hits = len(recs_topk & true_likes)

        if hits >= 1:
            hit_count_1 += 1
        if hits >= multi_hit_threshold:
            hit_count_multi += 1

        precision_scores.append(hits / top_k)
        recall_scores.append(hits / len(true_likes))
        total_users += 1

    print(f"\n=== Evaluation Results: {method} ===")
    return compute_eval_metrics(hit_count_1, hit_count_multi,
                                precision_scores, recall_scores,
                                total_users, top_k, multi_hit_threshold)

def compute_eval_metrics(hit_count_1, hit_count_multi, precision_scores, recall_scores,
                         total_users, top_k, multi_hit_threshold):
    """
    Shared evaluation metric computation for all recommenders.
    Returns:
        dict: Summary metrics including hit rates, precision, recall.
    """
    hit_rate = hit_count_1 / total_users if total_users > 0 else 0
    hit_rate_multi = hit_count_multi / total_users if total_users > 0 else 0
    mean_precision = np.mean(precision_scores) if precision_scores else 0
    mean_recall = np.mean(recall_scores) if recall_scores else 0

    print("\n=== Evaluation Results ===")
    print(f"Users evaluated: {total_users}")
    print(f"Hit Rate @{top_k} (≥1 hit): {hit_rate:.3f}")
    print(f"Hit Rate @{top_k} (≥{multi_hit_threshold} hits): {hit_rate_multi:.3f}")
    print(f"Precision @{top_k}: {mean_precision:.3f}")
    print(f"Recall @{top_k}: {mean_recall:.3f}\n")

    return {
        "hit_rate": hit_rate,
        "hit_rate_multi": hit_rate_multi,
        "precision": mean_precision,
        "recall": mean_recall,
        "users": total_users,
        "recall_vector": recall_scores,
        "precision_vector": precision_scores
    }

def summarize_evaluations(results_dict):
    #  Build a summary DataFrame comparing evaluation metrics for all algorithms.
    summary_df = (
        pd.DataFrame(results_dict)
        .T[["hit_rate", "hit_rate_multi", "precision", "recall", "users"]]
        .rename(columns={
            "hit_rate": "HitRate@K (≥1)",
            "hit_rate_multi": "HitRate@K (≥multi)",
            "precision": "Precision@K",
            "recall": "Recall@K",
            "users": "Users evaluated"
        })
        .sort_values(by="Recall@K", ascending=False)
    )

    print("\n" + "~" * 50)
    print(" OVERALL ALGORITHM COMPARISON")
    print("~" * 50)
    print(summary_df.round(3))
    print("~" * 50 + "\n")

    return summary_df
