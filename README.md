# Interest Matching Project

A modular end-to-end recommendation engine that models user preferences to generate personalized hobby suggestions. 

The system implements and compares five distinct recommendation paradigms; popularity-based, graph-based, collaborative filtering, item-similarity, and latent factor models - within a unified evaluation framework, enabling controlled analysis of how different structural assumptions impact recommendation quality.

This project emphasizes:
- Structured preprocessing and feature engineering
- Modular algorithm design
- Reproducible evaluation (Hit Rate, Precision, Recall)
- Clear comparison between interpretability and predictive strength

## System Overview
The pipeline follows a layered architecture:
1. Data preprocessing
2. EDA: Visualize patterns, distributions, correlations, and cluster relationships.   
3. Feature engineering (correlation matrices, clustering, weighted graph construction)
4. Independent algorithm modules
5. Unified evaluation framework: Measure Hit Rate, Precision, and Recall for each model and summarize overall performance.
6. Performance benchmarking

Each algorithm runs independently while sharing the same preprocessing and evaluation logic - enabling fair comparison and scalability.

## Architecture
```bash
project/
├── src/               # Core pipeline & algorithm implementations
│   ├── algorithms/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── exploratory_analysis.py
│   └── main.py
├── data/                # Raw and processed datasets
├── assets/              # Visualizations and evaluation outputs
├── docs/                # Detailed system documentation
└── README.md
```
The system separates data preparation, modeling logic, and evaluation to ensure modularity and maintainability.



## Implemented Algorithms
| **Algorithm**                | **Type**                | **Description** |
|------------------------------|--------------------------|-----------------|
| **Popularity-Based (Cluster-Aware)**       | Hybrid                 | Balances global popularity with correlation and cluster-level affinity. |
| **Friends-of-Friends (FOF)** | Graph-based             | Propagates scores through a weighted hobby graph to capture indirect relationships. |
| **Item-Similarity**          | Content-based           | Recommends hobbies similar to those already liked by the user. |
| **User-Similarity CF**       | Collaborative Filtering | Finds users with similar profiles using Pearson and Cosine similarity. |
| **Matrix Factorization (NMF/SVD)** | Latent Factor Model | Learns hidden preference patterns through matrix decomposition. |

---

##  Evaluation Summary

All models were evaluated using Top-5 recommendation metrics.  
Note: two variants of Hit Rate were used : one for users with at least one correct recommendation (`≥1 hit`) and one for users with multiple hits (`≥3 hits`).

| Algorithm | Hit Rate @5 (≥1 hit) | Hit Rate @5 (≥3 hits) | Precision @5 | Recall @5 | Users evaluated |
|------------|----------------------|------------------------|---------------|------------|------------------|
| **Matrix Factorization (NMF)** | **0.978** | **0.597** | **0.556** | **0.675** | 993 |
| **User-CF (Pearson)** | **0.985** | **0.574** | **0.543** | **0.661** | 993 |
| **Popularity (Cluster-Aware)** | 0.917 | 0.281 | 0.383 | 0.559 | 302 |
| **Friends-of-Friends (FOF)** | 0.914 | 0.209 | 0.354 | 0.514 | 302 |
| **Item-Based Similarity** | 0.891 | 0.195 | 0.340 | 0.492 | 302 |

> A full technical breakdown of evaluation methodology and algorithm-level analysis is available in the /docs directory

##  How to Run 

1. Clone the repository
```bash
git clone https://github.com/daniellaleiba/InterestMatchingProject.git
cd InterestMatchingProject
```
2. Install dependencies  
Make sure you have Python ≥3.9 installed. Then run:
```bash
pip install -r requirements.txt
```
3. Run the main script  
Execute the full main pipeline from the command line:
```bash
python src/main.py
```


## Project Highlights & Key Learnings

- **Five recommendation strategies implemented from scratch** : combining graph-based, collaborative, and factorization-based approaches for transparent benchmarking.  
- **Unified evaluation pipeline for fair benchmarking** : that connects preprocessing, feature engineering, model training, and evaluation into a reproducible end-to-end workflow.  
- **Explored multiple similarity paradigms** : Combination of graph modeling, similarity-based CF, and latent factor methods.  
- **Built custom evaluation framework** : Clear trade-off analysis between interpretability and predictive performance.

Designed with modularity, reproducibility, and controlled benchmarking in mind.

## Documentation
For full technical breakdown, including:
- Detailed algorithm implementations
- Extended evaluation comparisons (Pearson vs Cosine, NMF vs SVD)
- Example recommendation outputs
- Architectural design decisions

See [System Documentation](docs/system_documentation.md)