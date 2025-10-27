"""
item_based_recommender.py
=========================

Item-Based Similarity Recommendation Algorithm

This algorithm recommends hobbies (items) to a user based on
the similarity between items — hobbies that are similar to those
the user already likes (high ratings).

Steps:
1. Compute item-item similarity matrix (cosine similarity).
2. For each user, identify liked hobbies (ratings ≥ threshold).
3. Recommend the most similar hobbies not yet liked.
4. Evaluate performance on test data.

Author: Daniella Leiba
"""


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import evaluate_recommender_generic, select_sample_users

def compute_item_similarity_matrix(train_df, hobby_cols):
    """
    Compute the cosine similarity matrix between hobbies (item-item).
    """
    # Transpose to get hobbies as rows, users as columns
    item_vectors = train_df[hobby_cols].T  
    sim_matrix = pd.DataFrame(
        cosine_similarity(item_vectors),
        index=hobby_cols,
        columns=hobby_cols
    )
    return sim_matrix

def item_similarity_recommend(user_row, sim_matrix,top_k, rating_threshold=4 ):
    """
    Generate hobby recommendations for a single user using item-item similarity.

    """
    # Identify hobbies rated above threshold
    liked = [h for h, r in user_row.items() if r >= rating_threshold]
    if not liked:
        return []

    # Compute mean similarity of all other hobbies to the liked ones
    sim_scores = sim_matrix[liked].mean(axis=1)
    sim_scores = sim_scores.drop(liked, errors='ignore')
    
    # Sort by descending similarity
    recs = sim_scores.sort_values(ascending=False).head(top_k).index.tolist()
    return recs

def run_item_similarity_pipeline(train_data, test_data, hobby_cols, top_k, rating_threshold):
    """
     Full pipeline for the Item-Based Similarity recommendation algorithm.

    Steps:
    1. Compute item-item similarity matrix (train only).
    2. Generate recommendations for each user in test set.
    3. Print examples for sample users.
    4. Evaluate global performance.
    """
    print("\n◆ Item-Based Similarity Algorithm:\n")
    # Step 1: Compute item-item similarity matrix
    sim_matrix = compute_item_similarity_matrix(train_data, hobby_cols)

    # Step 2: Generate recommendations for all test set
    recommendations_dict = {
        user_id: item_similarity_recommend(row, sim_matrix,  top_k,rating_threshold)
        for user_id, row in test_data.iterrows()
    }
  
    # Step 3: Print sample users
    sample_users = select_sample_users(test_data, n_samples=10)
    print("\n~~~ Item-Based Similarity Recommendations (Example Users) ~~~")
    for user_id in sample_users:
        preferred_with_ratings = {
            h: int(test_data.loc[user_id, h])
            for h in hobby_cols
            if test_data.loc[user_id, h] >= rating_threshold
        }
        recs = recommendations_dict[user_id][:top_k]
        print(f"\nUser {user_id}")
        print(f"Preferred hobbies: {preferred_with_ratings}")
        print(f"Recommended : {recs}")
        print("------")

    # Step 4: Evaluate performance
    eval_results = evaluate_recommender_generic(
        test_df=test_data,top_k=top_k,
        recommend_func=item_similarity_recommend,
        algo_kwargs={
            "sim_matrix": sim_matrix,
            "top_k": top_k,
            "rating_threshold": rating_threshold
        },
        rating_threshold=rating_threshold
    )

    return eval_results
