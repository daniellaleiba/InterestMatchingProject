"""
matrix_factorization_recommender.py
===================================

Matrix Factorization Recommendation Algorithm (NMF & SVD)

This algorithm predicts missing user–hobby ratings by decomposing 
the rating matrix into latent user and hobby factors. It uses 
Non-negative Matrix Factorization (NMF) and Singular Value 
Decomposition (SVD) to learn hidden patterns that represent user 
preferences and item characteristics.

Steps:
1. Build a sparse user–hobby matrix and split into train/test sets.
2. Fill missing values using user-mean imputation.
3. Train NMF and SVD models and reconstruct predicted ratings.
4. Generate top-K recommendations per user.
5. Evaluate and compare model performance.

Advantages:
-----------
- Reveals hidden relationships between users and hobbies.
- Handles sparse, high-dimensional data effectively.
- Provides smooth and generalizable predictions.

Author: Daniella Leiba
"""

import pandas as pd
from surprise import SVD, NMF as SurpriseNMF, Dataset, Reader
from utils.helpers import (
    create_sparse_user_matrix,
    train_test_split_from_sparse,
    select_sample_users,
    evaluate_usersimilarity_recommender
)


def decompose_matrix(train_df, n_components=6, method="nmf"):
    """
    Train a matrix factorization model (NMF or SVD) using the Surprise library
    and reconstruct the predicted rating matrix.
    """
    # Convert wide user-item DataFrame into long format
    train_long = (
        train_df.reset_index()
        .melt(id_vars=train_df.index.name or "index", var_name="hobby", value_name="rating")
        .dropna()
    )
    train_long.columns = ["user", "hobby", "rating"]

    # Define rating scale (1–5)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_long[["user", "hobby", "rating"]], reader)
    trainset = data.build_full_trainset()

    # Choose model
    if method.lower() == "svd":
        print(f"Training Surprise SVD model with {n_components} factors...")
        model = SVD(n_factors=n_components, random_state=42)
    elif method.lower() == "nmf":
        print(f"Training Surprise NMF model with {n_components} factors...")
        model = SurpriseNMF(n_factors=n_components, random_state=42)
    else:
        raise ValueError("method must be either 'svd' or 'nmf'")

    # Train model on full training set
    model.fit(trainset)
    users = train_df.index
    hobbies = train_df.columns
    reconstructed = pd.DataFrame(index=users, columns=hobbies, dtype=float)

    for u in users:
        for h in hobbies:
            reconstructed.loc[u, h] = model.predict(u, h).est

    print(f"{method.upper()} training completed and matrix reconstructed")
    return reconstructed, model

def generate_recommendations_from_reconstructed(reconstructed_df, train_df, hobby_cols, top_k=5):
    """
    Generate top-K recommendations for each user from a reconstructed rating matrix.
    Avoids recommending hobbies already rated in the training data.
    """
    recommendations = {}

    for user in reconstructed_df.index:
        # Get predicted ratings for all hobbies
        user_pred = reconstructed_df.loc[user, hobby_cols]

        # Identify hobbies already rated in the training set
        already_rated = train_df.loc[user, hobby_cols].dropna().index

        # Remove already-rated hobbies from predictions
        user_pred_filtered = user_pred.drop(index=already_rated, errors="ignore")

        # Select the top-k highest predicted hobbies
        top_hobbies = user_pred_filtered.sort_values(ascending=False).head(top_k).index.tolist()

        # Final safeguard: remove any hobby that accidentally slipped through
        top_hobbies = [h for h in top_hobbies if h not in already_rated]

        recommendations[user] = top_hobbies

    return recommendations
  

def run_pipeline_matrix_factorization(df_full, hobby_cols,top_k, rating_threshold, missing_ratio, n_components):
    """
    Full pipeline for Matrix Factorization (NMF & SVD):
    1. Create sparse data (simulate missing ratings).
    2. Split into Train/Test sets. 
    3. Train NMF and SVD models.
    4. Reconstruct predicted ratings.
    5. Generate recommendations and visualize examples.
    6. Evaluate model performance (NMF vs SVD).
    """
    print("\n◆  Matrix Factorization Algorithm:\n")
    # Step 1: Create sparse data
    df_sparse = create_sparse_user_matrix(df_full, missing_ratio)

    # Step 2: Split Train/Test
    train_df, test_df = train_test_split_from_sparse(df_full, df_sparse)

    # Step 3: Train NMF and SVD models
    reconstructed_nmf_df, nmf_model = decompose_matrix(train_df, n_components=n_components, method='nmf')
    reconstructed_svd_df, svd_model = decompose_matrix(train_df, n_components=n_components, method='svd')

    # Step 4: Generate recommendations
    recs_nmf = generate_recommendations_from_reconstructed(reconstructed_nmf_df, train_df, hobby_cols, top_k=top_k)
    recs_svd = generate_recommendations_from_reconstructed(reconstructed_svd_df, train_df, hobby_cols, top_k=top_k)

    # Step 5: Print example users
    print(f"\n~~~ Matrix Factorization  (NMF vs SVD)  Recommendations (Example Users) ~~~")
    sample_users = select_sample_users(test_df, n_samples=5)
    for user in sample_users:
        liked = {h: int(test_df.loc[user, h]) for h in hobby_cols if test_df.loc[user, h] >= 4}
        print(f"\nUser {user}")
        print(f"Preferred hobbies: {liked}")
        print(f"Recommended (NMF): {recs_nmf[user]}")
        print(f"Recommended (SVD): {recs_svd[user]}")
        print("------")

    # checking of over leap :::: 
    train_overlap = (train_df.notna() & test_df.notna()).sum().sum()
    print("Overlap between train and test:", train_overlap)

    # Step 6. Evaluation 
    print("\n=== Evaluation Comparison ===")
    eval_nmf = evaluate_usersimilarity_recommender(recs_nmf, test_df, method="NMF", top_k=top_k, rating_threshold=rating_threshold)
    eval_svd = evaluate_usersimilarity_recommender(recs_svd, test_df, method="SVD", top_k=top_k, rating_threshold=rating_threshold)

    return eval_nmf
