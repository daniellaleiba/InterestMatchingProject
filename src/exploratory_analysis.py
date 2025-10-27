"""
exploratory_analysis.py
-----------------------
Module for performing exploratory data analysis (EDA) and visualization 
in the Interest Matching Project.

Pipeline:
1. Basic dataset overview (info, statistics, missing values)
2. Visualize categorical and numeric distributions
3. Compute correlations and top-related features
4. Perform group comparisons (T-tests by category)
5. Extract feature importance using PCA
6. Display cluster and graph visualizations
7. Run full integrated EDA pipeline

Author: Daniella Leiba
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
import matplotlib.image as mpimg


def save_plot(folder, filename):
    """Utility to save a plot to the appropriate assets subfolder."""
    path = os.path.join("assets", folder)
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, filename), bbox_inches="tight", dpi=300)

# 1. BASIC DATA OVERVIEW

def basic_statistics(df):
  #  Print dataset info, descriptive statistics, and missing values summary.

    print("~~ Basic Statistics : ~~\n")
    # genral info about data
    print("Info about dataset:")
    print(df.info())
    print(f"\nShape: {df.shape}\n")

    # Exclude binary (emographic + Gender) columns
    exclude_cols = ['Gender', 'Village - town']
    numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                    if col not in exclude_cols]
    # Simplified describe: count, mean, std
    desc = df[numeric_cols].describe()
    summary = desc.loc[['count', 'mean', 'std']]
    print("\nSimplified Describe (count, mean, std):\n")
    print(summary.round(3).T)  # transpose for easier reading

    # checking missing vals
    print("Missing values per column:")
    print(df.isna().sum())

    print("\n~~ End of Basic Statistics ~~\n")


# 2. DISTRIBUTIONS & CORRELATIONS

def plot_categorical_distribution(df, column, mapping=None, save=True):
    # Plot categorical distribution (counts + percentages) for binary/categorical columns.
    data = df[column].value_counts().sort_index()
    if mapping:
        data.index = data.index.map(mapping)

    plt.figure(figsize=(6, 5))
    plt.pie(
        data.values,
        labels=data.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("viridis", len(data))
    )
    plt.title(f"{column} Distribution", fontsize=13, weight='bold')
    plt.axis('equal')
    plt.tight_layout()
    
    # Optional save
    if save:
        save_plot("eda_visuals", f"{column.lower().replace(' ', '_')}_distribution.png")
 
    plt.show()

def plot_all_hobbies_boxplot(df, exclude_cols=None, save=True):
    """
    Boxplot for all hobbies with uniform color — clean and minimal style.
    """
    exclude_cols = exclude_cols or ['Gender', 'Village - town']
    hobby_cols = [col for col in df.columns if col not in exclude_cols]

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df[hobby_cols], color="#4B8BBE", width=0.6, fliersize=2)
    plt.title("Distribution of Ratings Across Hobbies", fontsize=14, weight="bold")
    plt.ylabel("Rating (1–5)")
    plt.xlabel("Hobbies")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(range(1, 6))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        save_plot("eda_visuals", "boxplot_all_hobbies_clean.png")
    plt.show()

def correlation_heatmap(df, method='spearman', exclude_cols=None):
  #  Plot correlation heatmap between numeric columns (default: Spearman).

    exclude_cols = exclude_cols or ['Gender', 'Village - town']
    numeric_cols = [c for c in df.select_dtypes(include=['int64', 'float64']).columns
                    if c not in exclude_cols]
    
    corr = df[numeric_cols].corr(method=method)
    plt.figure(figsize=(14,10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f'{method.capitalize()} Correlation Heatmap')
    plt.show()
    return corr

def top_correlated_pairs(df, top_n=10):
    corr = df.corr(method='spearman')
    corr_pairs = (
        corr.unstack()
        .drop_duplicates()
        .sort_values(ascending=False)
    )
    top_pairs = corr_pairs[corr_pairs < 0.999].head(top_n)
    print("\nTop correlated hobby pairs:")
    print(top_pairs)
    return top_pairs

def plot_top_hobbies_mean_and_count(df, top_n=12, rating_threshold=3, save=True):
    # Shows the hobbies with the highest average rating, along with the number of users who rated them >= rating_threshold.
    
    mean_ratings = df.mean().rename("mean_rating")
    count_high = (df >= rating_threshold).sum().rename(f"num_users_rating_≥{rating_threshold}")

    summary = pd.concat([mean_ratings, count_high], axis=1)
    summary = summary.sort_values("mean_rating", ascending=False).head(top_n)
    
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=summary,
        y=summary.index, x="mean_rating",
        palette="rocket", edgecolor="black"
    )

    for i, (mean, count) in enumerate(zip(summary["mean_rating"], summary.iloc[:,1])):
        plt.text(mean + 0.02, i, f"{int(count)} users", va='center', fontsize=9, color='dimgray')

    plt.title(f"Top {top_n} Hobbies by Average Rating and Number of Users (≥{rating_threshold})", fontsize=13, weight="bold")
    plt.xlabel("Average Rating")
    plt.ylabel("Hobby")
    plt.xlim(0, 5)
    plt.tight_layout()

    if save:
        save_plot("eda_visuals", "top_hobbies_mean_count.png")
    plt.show()

    return summary


# 3. GROUP COMPARISONS (T-TESTS) 

def t_test_categories(df, cat_col, numeric_cols=None, mapping=None):
   # Perform independent-sample T-tests for a categorical feature (binary) vs numeric features.

    numeric_cols = numeric_cols or [c for c in df.select_dtypes(include=['int64','float64']).columns if c != cat_col]
    groups = df[cat_col].unique()

    if len(groups) != 2:
        print(f"Skipping {cat_col}: requires exactly 2 groups.")
        return

    g1, g2 = df[df[cat_col] == groups[0]], df[df[cat_col] == groups[1]]
    label1, label2 = mapping.get(groups[0], groups[0]), mapping.get(groups[1], groups[1]) if mapping else (groups[0], groups[1])

    print(f"\n=== T-tests: {cat_col} ({label1} vs {label2}) ===\n")
    for col in numeric_cols:
        stat, p = ttest_ind(g1[col], g2[col], equal_var=False)
        print(f"{col}: t={stat:.3f}, p={p:.4f} → {'SIGNIFICANT' if p < 0.05 else 'Not significant'}")

def plot_mean_ratings_by_category(df, category_col, numeric_cols=None, category_labels=None, save=True):
  #  Plot average rating per feature for each category (explains T-test differences).
    numeric_cols = [c for c in df.select_dtypes(include=['int64','float64']).columns if c != category_col]
    df_plot = df.groupby(category_col)[numeric_cols].mean().T
    
    if category_labels:
        df_plot.columns = category_labels
    
    df_plot.plot(kind='bar', figsize=(15,6))
    plt.title(f"Average Ratings per Feature by {category_col}")
    plt.ylabel("Average Rating (1-5)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save:
        save_plot("eda_visuals", f"{category_col.lower().replace(' ', '_')}_meanrating.png")
    plt.show()
    plt.show()


# 4. FEATURE IMPORTANCE (PCA)

def compute_and_plot_feature_importance_pca( df, exclude_cols=None, n_components=5, top_n=15, save=True):
    # Compute and optionally visualize feature importance using PCA loadings. 
    # Highlights the most influential hobbies contributing to data variance.
    # Can be used both for analysis and as input weighting for algorithms.

    exclude_cols = exclude_cols or ['Gender', 'Village - town']
    df_interest = df[[c for c in df.columns if c not in exclude_cols]]
    df_scaled = (df_interest - df_interest.mean()) / df_interest.std()

    # PCA fitting
    pca = PCA(n_components=n_components)
    pca.fit(df_scaled)

    # Compute importance scores
    loadings = np.abs(pca.components_)
    feature_importance = pd.DataFrame({
        'Feature': df_interest.columns,
        'Importance': loadings.mean(axis=0)
    }).sort_values('Importance', ascending=False)

    # print summary
    print("\n PCA-Based Feature Importance :")
    print(feature_importance.head(top_n))

    # Vizualization
    top_features = feature_importance.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_features,
        x='Importance',
        y='Feature',
        color="#4B8BBE",
        edgecolor="black"
    )
    plt.title(f"Top {top_n} Most Influential Hobbies (PCA-based Importance)",
                fontsize=13, weight="bold")
    plt.xlabel("Average PCA Loading (Importance)")
    plt.ylabel("Hobby")
    plt.tight_layout()

    if save:
            save_plot("eda_visuals", "pca_feature_importance.png")

    plt.show()
    return feature_importance


# 5. VISUAL SUMMARIES (CLUSTERS & GRAPH)

def show_cluster_visuals():
    # Display precomputed cluster visuals (saved in assets/ by feature_engineering.py)

    img_files = [
        os.path.join("assets", "cluster_graphs", "hierarchical_clustering_full.png"),
        os.path.join("assets", "cluster_graphs", "heatmap_clusters_full.png")
    ]
    # make sure images exists 
    for img in img_files:
        if not os.path.exists(img):
            print(f" Image not found: {img}")
            return
        
    img1 = mpimg.imread(img_files[0])
    img2 = mpimg.imread(img_files[1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img1)
    axes[0].set_title("Hierarchical Clustering of Interests", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title("Cluster Correlation Heatmap", fontsize=13)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def plot_interest_graph(G, clustered_features, alpha=0.85, threshold_percentile=75,save=False, name_suffix="_full"):
    """
    Visualize a weighted interest graph with nodes colored by cluster and edges scaled by weight.
    """
    # Map clusters to colors
    cluster_map = clustered_features.set_index("Interest")["Cluster"].to_dict()
    cluster_ids = sorted(clustered_features["Cluster"].unique())

    color_map = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
    cluster_colors = {cid: color_map[idx] for idx, cid in enumerate(cluster_ids)}
    node_colors = [cluster_colors[cluster_map[node]] for node in G.nodes()]

    # Layout
    pos = nx.spring_layout(G, seed=42, k=1.5)
    plt.figure(figsize=(18, 14))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    # Edge
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, width=[3*d['weight'] for (_, _, d) in edges], alpha=0.6)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

    # Legend
    for cid, color in cluster_colors.items():
        plt.scatter([], [], color=color, label=f"Cluster {cid}", s=120)
    plt.legend(title="Clusters", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.title(f"Weighted Graph of Interests (α={alpha}, threshold={threshold_percentile}%)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    # Optional save
    if save:
        save_plot("cluster_graphs", f"weighted_graph{name_suffix}.png")

    plt.show()


# 6. EDA PIPELINE

def run_eda_pipeline(df, graph, clustered_features):
    """
    Full exploratory analysis pipeline.
    Combines statistical summaries, visualizations, and precomputed cluster/graph visuals.
    """
    print("~"*50)
    print("Starting Exploratory Analysis : \n")

    # 1. Basic overview
    basic_statistics(df)

    # 2. Category distributions
    plot_categorical_distribution(df, 'Gender', mapping={0:'Female', 1:'Male'})
    plot_categorical_distribution(df, 'Village - town', mapping={0:'City', 1:'Village'})
    
    # 3. Numeric distributions 
    plot_all_hobbies_boxplot(df)
    
    # 4. Top hobbies
    plot_top_hobbies_mean_and_count(df, top_n=12, rating_threshold=3)

    # 5. Correlation structure
    correlation_heatmap(df)
    top_correlated_pairs(df)

    # 6. Group comparisons (T-tests)
    t_test_categories(df, 'Gender', mapping={0:'Female',1:'Male'})
    plot_mean_ratings_by_category(df, 'Gender', category_labels=['Female', 'Male'])
    t_test_categories(df, 'Village - town', mapping={0:'City',1:'Village'})
    plot_mean_ratings_by_category(df, 'Village - town', category_labels=['Village', 'City'])

    # 7. Feature importance (PCA)
    compute_and_plot_feature_importance_pca(df)

    # 8. Visual summaries
    show_cluster_visuals()
    plot_interest_graph(graph,clustered_features,save=True)
    
    print("- End of Exploratory Analysis - \n")
    print("~"*50)

    return 
   