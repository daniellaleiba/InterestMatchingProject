"""
main.py
-------
Main pipeline script for the Interest Matching Project.

This script orchestrates the full workflow:
1. Load and preprocess raw data
2. Run feature engineering (clustering + weighted graph)
3. Perform exploratory data analysis (EDA)
4. Split data into train/test sets for modeling
5. Run multiple recommendation algorithms
6. Evaluate and compare their performance

Author: Daniella Leiba
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Internal modules ---
from data_preprocessing import preprocess_data
from feature_engineering import run_feature_engineering, create_clusters
from exploratory_analysis import run_eda_pipeline
from utils.helpers import summarize_evaluations

# --- Algorithms ---
from algorithms.popularity_based import run_popularity_pipeline
from algorithms.FOF_algo import run_pipeline_fof
from algorithms.item_similarity_algo import run_item_similarity_pipeline
from algorithms.user_similarity_algo import us_cf_pipeline
from algorithms.matrix_factorization import run_pipeline_matrix_factorization


# Configuration
TOP_K = 5
rating_threshold = 4
n_components = 6
missing_ratio = 0.4

# MAIN EXECUTION PIPELINE

def main():
    """
    Full pipeline execution for the Interest Matching Project.
    Runs preprocessing, feature engineering, modeling, and evaluation sequentially.
    """

     # STEP 1: LOAD & PREPROCESS DATA
    pd.set_option('display.max_columns', None)
    filepath = "data/raw/interests.csv"

    print("\n" + "~" * 60)
    print("STEP 1: Loading and Cleaning Data\n")

    df_clean = preprocess_data(filepath)
    print("Data loaded and cleaned successfully.\n")
    print(df_clean.head(3))

    # STEP 2: FEATURE ENGINEERING (FULL DATASET)
    print("\n" + "~" * 60)
    print("STEP 2: Feature Engineering (Full Dataset)\n")
   
    clustered_features_EDA, cluster_corr_EDA, graph_EDA = run_feature_engineering(df_clean)

    # STEP 3: EXPLORATORY DATA ANALYSIS
    #run_eda_pipeline(df_clean, graph_EDA, clustered_features_EDA)

    # STEP 4: TRAIN/TEST SPLIT
    print("\n" + "~" * 60)
    print("STEP 3: Splitting Train/Test Data\n")

    hobby_cols = [c for c in df_clean.columns if c not in ['Gender', 'Village - town']]
    df_hobbies = df_clean[hobby_cols]

    train_data, test_data = train_test_split(df_hobbies, test_size=0.3, random_state=42)
    print(f"Training set shape: {train_data.shape}, Test set shape: {test_data.shape}\n")


    # STEP 5: FEATURE ENGINEERING (TRAINING DATA)
    print("\n" + "~" * 60)
    print("STEP 4: Feature Engineering on Training Set\n")
    clustered_features_train, cluster_corr_train = create_clusters(train_data, name_suffix="_train") 


    # STEP 6: RUN RECOMMENDER ALGORITHMS
    # ----------------------------------- # 
    #  ALGORITHM 1: Popularity-Based Recommender
    print("\nRunning Popularity-Based Recommender...")
    pb_evaluation = run_popularity_pipeline(
        train_data,
        test_data,
        hobby_cols,
        clustered_features_train,
        TOP_K,
        rating_threshold
    )

    # ----------------------------------- # 
    # ALGORITHM 2: Friends-of-Friends (FOF)
    print("\nRunning Friends-of-Friends (FOF) Recommender...")
    fof_evaluation = run_pipeline_fof(
        train_data,
        test_data,
        clustered_features_train,
        hobby_cols,
        TOP_K,
        rating_threshold
    )

    # ----------------------------------- # 
    # ALGORITHM 3: Item-Similarity-Based Recommender
    print("\nRunning Item-Similarity-Based Recommender...")
    item_evaluation = run_item_similarity_pipeline(
        train_data,
        test_data,
        hobby_cols,
        TOP_K,
        rating_threshold
    )
   
    # ----------------------------------- # 
    #  ALGORITHM 4: User-Similarity-Based Recommender
    print("\nRunning User-Similarity-Based Recommender...")
    user_cf_evaluation = us_cf_pipeline(
        df_hobbies,
        TOP_K,
        rating_threshold, 
        missing_ratio
    )

    # ----------------------------------- # 
    #  ALGORITHM 5: Matrix Factorization
    print("\nRunning Matrix Factorization Recommender...")
    mf_evaluation = run_pipeline_matrix_factorization(
        df_hobbies,
        hobby_cols,
        TOP_K,
        rating_threshold,
        missing_ratio,
        n_components
    )


     # STEP 7: FINAL SUMMARY
    # -----------------------------------------------------
    print("\n" + "~" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("~" * 50 + "\n")

     # STEP 8: COMPARISON TABLE
    results_summary = {
        "Popularity": pb_evaluation,
        "FOF": fof_evaluation,
        "Item-Based": item_evaluation,
        "User-CF (Pearson)": user_cf_evaluation,
        "Matrix Factorization (NMF)": mf_evaluation
    }

    summary_df = summarize_evaluations(results_summary)
    summary_df.to_csv("assets/evaluation_summary.csv", index=True)


if __name__ == "__main__":
    main()