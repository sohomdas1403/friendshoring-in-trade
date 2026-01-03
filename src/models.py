import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

# Import PCA-transformed matrices and country ordering from data loader
from .data_loader import feature_matrix_2015_pca, feature_matrix_2021_pca, countries_list


def cluster_and_evaluate(X: np.ndarray, countries: list, year_label: str):
    """Run hierarchical clustering (Ward) with k = 2-10 clusters as outputs, and compute the silhouette scores for.
     each k. Then, return the k that yields the highest silhouette score. This is the optimal k number of clusters
     that will be used for the final clustering analysis output, as a higher silhouette score indicates 
     better-defined clusters.

    The function also computes and returns the linkage matrix so that it can be called upon easily for dendogram
    plotting."""

    results = {}
    scores = {}
    # Loop hierarchical clustering for k=2-10
    for k in range(2, 11):
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X)
        # compute silhouette; requires at least 2 clusters and fewer clusters than samples
        try:
            score = silhouette_score(X, labels)
        except Exception as e: # in case silhouette score cannot be computed
            score = np.nan 
        scores[k] = score 

    # find best k (highest silhouette)
    best_k = max(scores, key=lambda kk: (scores[kk] if not np.isnan(scores[kk]) else -np.inf))
    best_score = scores[best_k]

    # Fit final model at best_k
    final_model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    final_labels = final_model.fit_predict(X)

    # Build DataFrame mapping country -> cluster
    clusters_df = pd.DataFrame({'country': countries, 'cluster': final_labels})

    # Print concise summary statistics
    print(f"\n==== Hierarchical clustering (Ward) — {year_label} ====\n")
    print(f"Best number of clusters: {best_k}  —  silhouette score = {best_score:.4f}\n")

    counts = clusters_df['cluster'].value_counts().sort_index()
    print("Cluster sizes:") # size of each cluster in each year
    for c, cnt in counts.items():
        print(f"  cluster {c}: {cnt} countries")

    # Print up to 6 sample countries per cluster (for inspection and validation)
    print("\nSample countries per cluster:")
    for c in counts.index:
        members = clusters_df[clusters_df['cluster'] == c]['country'].tolist()
        preview = ", ".join(members[:6])
        if len(members) > 6:
            preview += ", ..."
        print(f"  cluster {c}: {preview}")

    # Compute linkage matrix for dendrogram
    Z = linkage(X, method='ward')

    return {
        'best_k': best_k,
        'best_score': best_score,
        'labels': final_labels,
        'clusters_df': clusters_df,
        'all_scores': scores,
        'linkage_matrix': Z
    }


def run_all():
    """Run clustering (k=2-10) for 2015 and 2021 PCA matrices and print best results."""

    # Ensure that X arrays are 2D and have more rows than max clusters
    """Here we ensure that the post-PCA feature matrices are numpy arrays for sklearn, and create a safe, 
     indexable copy of countries_list. """
    X2015 = np.asarray(feature_matrix_2015_pca) 
    X2021 = np.asarray(feature_matrix_2021_pca)
    countries = list(countries_list)

    # Make sure that PCA Matrix rows match country count:
    if X2015.shape[0] != len(countries) or X2021.shape[0] != len(countries):
        print("Warning: number of rows in PCA matrices and countries list do not match.")

    # Run clustering for each year with k=2-10
    print("Running clustering for 2015...")
    res2015 = cluster_and_evaluate(X2015, countries, '2015')

    print("\nRunning clustering for 2021...")
    res2021 = cluster_and_evaluate(X2021, countries, '2021')

    return {'2015': res2015, '2021': res2021}


def save_clusters_to_csv(results: dict, output_dir: str = 'results'):
    """Save cluster assignments from clustering results with best_k to CSV files for 2015 and 2021. Each CSV 
    has cluster names as row labels and countries in format makes for easy readability."""

    # Extract cluster assignments with best_k and optimal k for 2015 and 2021
    for year in ['2015', '2021']:

        clusters_df = results[year]['clusters_df'] 
        best_k = results[year]['best_k']
            
        # Group countries by cluster number (0, 1, ..., best_k-1)
        """Store each cluster's members as dictionary to make wide format .csv"""
        cluster_data = {}
        for c in range(best_k):
            members = clusters_df[clusters_df['cluster'] == c]['country'].tolist() 
            cluster_data[f'Cluster {c}'] = members      # df -> list -> dictionary
            
        # Pad to same length and create DataFrame
        """Ensures that array is rectangular and not jagged, """
        max_len = max(len(v) for v in cluster_data.values())
        padded_data = {k: v + [None] * (max_len - len(v)) for k, v in cluster_data.items()}

        # Convert to DataFrame: columns = clusters, rows = countries
        output_df = pd.DataFrame(padded_data)
            
        # Save to CSV
        output_path = Path(f'{output_dir}/clusters_{year}.csv') # create Path object for output
        output_path.parent.mkdir(parents=True, exist_ok=True)   # create parent directories if not existing
        output_df.to_csv(f'{output_dir}/clusters_{year}.csv', index=False)
        print(f"Saved clusters for {year} to '{output_dir}/clusters_{year}.csv'")

