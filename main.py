from src.data_loader import countries_list
from src.data_loader import pca
from src.models import run_all
from src.models import save_clusters_to_csv
from src.evaluation import prepare_regression_data
from src.evaluation import run_fixed_effects_regression
from src.evaluation import create_regression_table
from src.evaluation import _plot_and_save_dendrogram
from src.evaluation import plot_scree
from src.evaluation import create_cluster_maps


# MAIN SCRIPT TO RUN FULL FRIENDSHORING CLUSTERING ANALYSIS, CREATE RELEVANT PLOTS, & SAVE RESULTS
"""
1. Calls on data pipeline outputs, performs hierarchical clustering (once for 2015, once for 2021)
2. Conducts Fixed Effects Regression for baseline comparison
3. Create Regression Table
4. Plot Dendrograms (one for 2015, one for 2021)
5. Plot Scree Plot (one for the combined PCA)
6. Create World Maps Coloured by Clusters (one for 2015, one for 2021)
"""

def main():
    print("=" * 60)
    print("FRIENDSHORING TRADE ANALYSIS")
    print("=" * 60)
    
    print("\n1. Running hierarchical clustering for 2015 and 2021...")
    print("-" * 50)
    
    # 1. CLUSTERING
    results = run_all()
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    
    r2015 = results['2015']
    r2021 = results['2021']
    
    print(f"\n2015 Results:")
    print(f"  • Optimal clusters: {r2015['best_k']}")
    print(f"  • Silhouette score: {r2015['best_score']:.4f}")
    print(f"  • Cluster sizes: {r2015['clusters_df']['cluster'].value_counts().sort_index().tolist()}")
    
    print(f"\n2021 Results:")
    print(f"  • Optimal clusters: {r2021['best_k']}")
    print(f"  • Silhouette score: {r2021['best_score']:.4f}")
    print(f"  • Cluster sizes: {r2021['clusters_df']['cluster'].value_counts().sort_index().tolist()}")

    # Save results as CSV files
    save_clusters_to_csv(results)

    # 2. FIXED-EFFECTS REGRESSION
    print("\n" + "=" * 60)
    print("FIXED-EFFECTS REGRESSION ANALYSIS")
    print("=" * 60)

    print("\nRunning first-difference regression...")
    print("-" * 50)   

    # Prepare regression dataframe
    regression_df = prepare_regression_data()

    # Run regression
    print("Regression results:")
    model = run_fixed_effects_regression(regression_df)
    
    # Print key coefficients
    coeff_friend = model.params['poldist_diff']
    pval_friend = model.pvalues['poldist_diff']

    coeff_cultural = model.params['cultdist_diff']  
    pval_cultural = model.pvalues['cultdist_diff']

    coeff_religious = model.params['reldist_diff']
    pval_religious = model.pvalues['reldist_diff']

    print(f"\nRegression Coefficients:")
    print(f"  β₁ (ΔGeopolitical Distance) = {coeff_friend:.4f} (p={pval_friend:.4f})")
    print(f"  β₂ (ΔCultural Distance) = {coeff_cultural:.4f} (p={pval_cultural:.4f})")
    print(f"  β₃ (ΔReligious Distance) = {coeff_religious:.4f} (p={pval_religious:.4f})")

    # Create regression table
    create_regression_table(model, output_path='results/regression_table.csv')
    print("✓ Regression table saved to results/regression_table.csv")

    # 3. CREATE AND SAVE DENDROGRAMS
    print("\n" + "=" * 60)
    print("CREATING AND SAVING DENDROGRAMS")
    print("=" * 60)

    # 2015 dendrogram
    _plot_and_save_dendrogram(
        Z=results['2015']['linkage_matrix'],
        labels=countries_list,
        output_path='results/dendrogram_2015.png',
        title='Dendrogram — 2015 (Ward)',
        best_k=results['2015']['best_k'],
        figsize=(16, 8),
        leaf_font_size=6
    )
    
    # 2021 dendrogram  
    _plot_and_save_dendrogram(
        Z=results['2021']['linkage_matrix'],
        labels=countries_list,
        output_path='results/dendrogram_2021.png',
        title='Dendrogram — 2021 (Ward)',
        best_k=results['2021']['best_k'],
        figsize=(16, 8),
        leaf_font_size=6
    )
    
    print("✓ Dendrograms saved to results/dendrogram_2015.png and results/dendrogram_2021.png")

    # 4. CREATE AND SAVE SCREE PLOT
    print("\n" + "=" * 60)
    print("CREATING AND SAVING SCREE PLOTS")
    print("=" * 60)
    plot_scree(
        pca_object=pca,
        output_path='results/scree_plot.png',
        max_components_to_show=80
    )

    # 5. CREATE AND SAVE CLUSTER MAPS
    print("\n" + "=" * 60)
    print("CREATING AND SAVING CLUSTER MAPS")
    print("=" * 60)

    create_cluster_maps()  # Uses defaults: cluster_csv_dir='results', output_dir='results'
    
    print("✓ Cluster maps saved to results/ directory")

    return {
        'clustering_results': results,
        'regression_model': model,
        'regression_data': regression_df
    }


if __name__ == '__main__':
    results = main()