"""
feature_engineering.py
----------------------
Module for constructing feature relationships and clustering in the 
Interest Matching Project.

Pipeline:
1. Create hierarchical clusters of hobbies (based on Pearson correlation)
2. Compute inter-cluster correlations
3. Build a weighted interest graph combining local and cluster-level similarities
4. Return clustered features, correlation matrix, and graph for later use

Author: Daniella Leiba
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# 1. HIERARCHICAL CLUSTERING OF INTERESTS
def create_clusters(df, exclude_cols=None, n_clusters=6, save_figures=True, name_suffix=""):

    """
    Create hierarchical clusters of hobbies based on Pearson correlation.

    This function identifies groups of related hobbies by analyzing their 
    linear correlation patterns. Pearson correlation is used to capture
    global relationships between hobbies, emphasizing interpretability and 
    the formation of a clear hierarchical structure.
    """

    if exclude_cols is None:
        exclude_cols = ['Gender', 'Village - town']

    # Select only hobby-related columns
    interest_cols = [col for col in df.columns if col not in exclude_cols]
    df_interest = df[interest_cols]

    # Compute Pearson correlation matrix
    corr = df_interest.corr(method='pearson')

    # Compute distance matrix (1 - |correlation|)
    distance = 1 - corr.abs()

    # Perform hierarchical clustering using Ward’s method
    linkage_matrix = linkage(squareform(distance), method='ward')

    # --- Visualization: Dendrogram ---
    if save_figures:
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, labels=corr.columns, leaf_rotation=90)
        plt.title("Hierarchical Clustering of Interests (Pearson-based)")
        plt.ylabel("Distance (1 - |correlation|)")
        plt.savefig(os.path.join("assets/cluster_graphs/", f"hierarchical_clustering{name_suffix}.png"), 
                    bbox_inches="tight", dpi=300)
        plt.close()

    # Assign each hobby to a cluster
    clusters = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    clustered_features = pd.DataFrame({
        'Interest': corr.columns,
        'Cluster': clusters
    })

    print(f"Created {n_clusters} clusters successfully.")

    # --- Compute average correlation between clusters ---
    cluster_ids = sorted(clustered_features['Cluster'].unique())
    cluster_corr = pd.DataFrame(index=cluster_ids, columns=cluster_ids, dtype=float)

    for i in cluster_ids:
        for j in cluster_ids:
            cols_i = clustered_features.loc[clustered_features['Cluster'] == i, 'Interest']
            cols_j = clustered_features.loc[clustered_features['Cluster'] == j, 'Interest']
            avg_corr = corr.loc[cols_i, cols_j].mean().mean()
            cluster_corr.loc[i, j] = avg_corr

    # --- Visualization: Heatmap ---
    if save_figures:
        plt.figure(figsize=(7, 5))
        sns.heatmap(cluster_corr.astype(float), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Average Correlation Between Clusters of Interests (Pearson-based)")
        plt.xlabel("Cluster")
        plt.ylabel("Cluster")
        plt.savefig(os.path.join("assets/cluster_graphs/", f"heatmap_clusters{name_suffix}.png"), 
                    bbox_inches="tight", dpi=300)
        plt.close()

    return clustered_features, cluster_corr


# 2. BUILD WEIGHTED GRAPH OF INTERESTS
def build_weighted_interest_graph(df, clustered_features, alpha=0.85, threshold_percentile=75, top_k=8, save_figures=True):
    """
    Build a weighted graph of interests using cosine similarity and cluster similarity.

    This function is part of the Feature Engineering layer.
    It constructs a graph that captures both local (cosine-based) and cluster-level relationships
    between hobbies, used later by algorithms such as FOF and Popularity.

    Steps:
    1. Compute cosine similarity between all hobbies.
    2. Compute average similarity between clusters (cluster_corr).
    3. Combine both similarities into a weighted graph.
    4. Keep only strong and top-k connections per node.
    5. Optionally save a network visualization to assets/.
    """

    # Prepare data
    interest_cols = [col for col in df.columns if col not in ['Gender', 'Village - town']]
    df_interest = df[interest_cols]

    # Compute cosine similarity between hobbies
    similarity_matrix = pd.DataFrame(
        cosine_similarity(df_interest.T),
        index=df_interest.columns,
        columns=df_interest.columns
    )

    # Compute cluster-level similarity
    cluster_map = clustered_features.set_index("Interest")["Cluster"].to_dict()
    cluster_ids = sorted(clustered_features["Cluster"].unique())
    cluster_corr = pd.DataFrame(index=cluster_ids, columns=cluster_ids, dtype=float)

    for ci in cluster_ids:
        interests_ci = [i for i, c in cluster_map.items() if c == ci]
        for cj in cluster_ids:
            interests_cj = [i for i, c in cluster_map.items() if c == cj]
            sims = [similarity_matrix.loc[i, j] for i in interests_ci for j in interests_cj]
            cluster_corr.loc[ci, cj] = np.mean(sims)

    # --- Build weighted graph ---
    G = nx.Graph()
    weights = []
    for i in similarity_matrix.columns:
        for j in similarity_matrix.columns:
            if i != j:
                sim_val = similarity_matrix.loc[i, j]
                ci, cj = cluster_map[i], cluster_map[j]
                cluster_sim = cluster_corr.loc[ci, cj]
                weight = alpha * sim_val + (1 - alpha) * cluster_sim
                G.add_edge(i, j, weight=weight)
                weights.append(weight)

    # Normalize weights
    min_w, max_w = min(weights), max(weights)
    for _, _, d in G.edges(data=True):
        d['weight'] = (d['weight'] - min_w) / (max_w - min_w)

    # Keep only top-percentile edges
    threshold_value = np.percentile([d['weight'] for _, _, d in G.edges(data=True)], threshold_percentile)
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold_value]
    G.remove_edges_from(edges_to_remove)

    # Limit connections per node (keep strongest)
    for node in G.nodes():
        edges = [(node, nbr, d['weight']) for nbr, d in G[node].items()]
        edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
        for u, v, w in edges_sorted[top_k:]:
            if G.has_edge(u, v):
                G.remove_edge(u, v)

    return G, cluster_corr

# 3. FULL FEATURE ENGINEERING PIPELINE
def run_feature_engineering(df, alpha=0.85, threshold_percentile=75, top_k=8):
    """
    Unified pipeline for feature engineering.

    Steps:
    1. Create hobby clusters using hierarchical clustering.
    2. Build weighted graph of interests.
    3. Return outputs for further modeling or visualization.
    """

    os.makedirs("assets/cluster_graphs", exist_ok=True)
    # Step 1. Clustering 
    clustered_features, cluster_corr = create_clusters(df, name_suffix="_full")

    # Step 2. Build weighted graph
    graph, cluster_corr_graph = build_weighted_interest_graph(
        df, clustered_features,
        alpha=alpha,
        threshold_percentile=threshold_percentile,
        top_k=top_k
    )
    
    return clustered_features, cluster_corr, graph