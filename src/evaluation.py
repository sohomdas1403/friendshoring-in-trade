import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from matplotlib.colors import ListedColormap
from statsmodels.formula.api import ols
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram

#=======================================================================
# FIXED-EFFECTS REGRESSION FOR EVALUATION & ROBUSTNESS
#=======================================================================

def prepare_regression_data():
    """Prepares all necessary data for Fixed-Effects Regression (First-Differences)"""

    # IMPORT DATAFRAMES AND PAIR SETS FROM DATA LOADER
    from .data_loader import (
        df_tradevolume_sum,     # trade volume dataset before transformation to log of squared deviations
        df_agreement_balanced,
        df_culturaldistance_balanced,
        df_religiousdistance_balanced,
        common_pairs,
        get_balanced_pairs,
    )

    #-----------------------------------------------------------------------
    # Prepare Balanced Trade Dataset for Regression
    #-----------------------------------------------------------------------
    """The final balanced trade volume dataset in data_loader.py involves the log of squared deviations from the
    baseline calculated from 2010-2014 data. For the regression, however, we need the log of trade volumes for 2015 
    and 2021 only. This section prepares that dataset and ensures it is balanced with respect to the other datasets."""

    # COMPUTE LOG OF TRADE VOLUMES FOR 2015 AND 2021, DROP NAN

    df_tradevolume_sum['log_tradevolume_USD'] = (       # compute log of trade volumes
        np.log(df_tradevolume_sum['trade_volume_USD'])
    ).where(df_tradevolume_sum['year'].isin([2015, 2021]))

    df_tradevolume_sum = df_tradevolume_sum[df_tradevolume_sum['year'].isin([2015, 2021])].copy() 
    df_tradevolume_sum = df_tradevolume_sum.dropna(subset=['log_tradevolume_USD']) 


    # MATCH REGRESSION TRADE VOLUME DATASET WITH BALANCING PAIRS WITH OTHER DATASETS
    """Here, we apply to df_tradevolume_sum the same data pipeline transformations that were done in 
    data_loader.py to achieve the other 3 imported balanced datasets. """

    # 1) Balance Pairs Inside the Regression Trade Volume Dataset
    df_tradevolume_sum = df_tradevolume_sum[['countrycode_1', 'countrycode_2', 'year', 'log_tradevolume_USD', 'pair']]

    trade_sum_pairs = get_balanced_pairs(df_tradevolume_sum)      
    print(f"Trade: {len(trade_sum_pairs)} balanced pairs")

    df_tradevolume_sum = df_tradevolume_sum[df_tradevolume_sum['pair'].isin(trade_sum_pairs)]

    # 2) Balance Trade Volume Dataset Pairs Across Datasets & Drop Duplicates

    df_tradevolume_sum_balanced = df_tradevolume_sum[df_tradevolume_sum['pair'].isin(common_pairs)].copy()

    df_tradevolume_sum_balanced = df_tradevolume_sum_balanced.drop_duplicates(subset=['pair', 'year']).copy()

    balanced_datasets_regression = {
        'Trade': df_tradevolume_sum_balanced,    # the other balanced datasets already prepared in data_loader.py
        'Agreement': df_agreement_balanced,
        'Cultural': df_culturaldistance_balanced,
        'Religious': df_religiousdistance_balanced,
    }

    #-----------------------------------------------------------------------
    # Calculate First Differences for Fixed-Effects Regression
    #-----------------------------------------------------------------------

    # PIVOT DATAFRAMES TO HAVE YEARS AS COLUMNS
    """Transform the DataFrames wide format (rows=dyads, columns=years) to enable computation of first=differences"""

    df_tradevolume_sum_balanced_regression = df_tradevolume_sum_balanced.pivot_table(
        index=['pair'],
        columns='year',
        values='log_tradevolume_USD'
    ).rename(columns={2015: 'logtrade_2015', 2021: 'logtrade_2021'}).reset_index()

    df_agreement_balanced_regression = df_agreement_balanced.pivot_table(
        index=['pair'],
        columns='year',
        values='pol_distance'
    ).rename(columns={2015: 'poldist_2015', 2021: 'poldist_2021'}).reset_index()

    df_culturaldistance_balanced_regression = df_culturaldistance_balanced.pivot_table(
        index=['pair'],
        columns='year',
        values='cultdist_std'
    ).rename(columns={2015: 'cultdist_2015', 2021: 'cultdist_2021'}).reset_index()
    df_religiousdistance_balanced_regression = df_religiousdistance_balanced.pivot_table(
        index=['pair'],
        columns='year',
        values='reldist_weighted'
    ).rename(columns={2015: 'reldist_2015', 2021: 'reldist_2021'}).reset_index()

    # COMPUTE DIFFERENCES BETWEEN 2021 AND 2015 (FOR FIRST-DIFFERENCE FIXED-EFFECTS REGRESSION)

    df_tradevolume_sum_balanced_regression['logtrade_diff'] = df_tradevolume_sum_balanced_regression['logtrade_2021'] - df_tradevolume_sum_balanced_regression['logtrade_2015']
    df_agreement_balanced_regression['poldist_diff'] = df_agreement_balanced_regression['poldist_2021'] - df_agreement_balanced_regression['poldist_2015']
    df_culturaldistance_balanced_regression['cultdist_diff'] = df_culturaldistance_balanced_regression['cultdist_2021'] - df_culturaldistance_balanced_regression['cultdist_2015']
    df_religiousdistance_balanced_regression['reldist_diff'] = df_religiousdistance_balanced_regression['reldist_2021'] - df_religiousdistance_balanced_regression['reldist_2015']

    # DROP UNUSED COLUMNS & MERGE DATAFRAMES FOR REGRESSION
    """Merge the dataframes and convert to cross-sectional dataset, enabling standard regression."""

    trade_diff_regression = df_tradevolume_sum_balanced_regression[['pair', 'logtrade_diff']]
    pol_diff_regression = df_agreement_balanced_regression[['pair', 'poldist_diff']]
    cultural_diff_regression = df_culturaldistance_balanced_regression[['pair', 'cultdist_diff']]
    religious_diff_regression = df_religiousdistance_balanced_regression[['pair', 'reldist_diff']]

    regression_df = (trade_diff_regression
        .merge(pol_diff_regression, on='pair', how='inner')
        .merge(cultural_diff_regression, on='pair', how='inner')
        .merge(religious_diff_regression, on='pair', how='inner'))
    
    # PRINT FINAL REGRESSION DATASET INFO
    print(f"Final regression dataset shape: {regression_df.shape}")
    print(f"Number of dyads: {len(regression_df)}")
    print(f"Columns: {regression_df.columns.tolist()}")

    return regression_df



#-----------------------------------------------------------------------
# Carry Out Fixed-Effects Regression
#-----------------------------------------------------------------------

def run_fixed_effects_regression(df: pd.DataFrame):
    """Run fixed-effects regression with constant."""
    # Add constant term
    X = sm.add_constant(df[['poldist_diff', 'cultdist_diff', 'reldist_diff']])
    
    model = sm.OLS(endog=df['logtrade_diff'], exog=X).fit()
    
    print(model.summary())
    return model


#=======================================================================
# DATA VISUALISATION HELPERS
#=======================================================================

"""
1. Regression Table & Summary Statistics
2. Dendrograms
3. Scree Plots (Silhouette Scores vs. Number of Clusters)
4. Cluster World Maps

"""
#-----------------------------------------------------------------------
# Create and Export Regression Table
#-----------------------------------------------------------------------

def create_regression_table(model: sm.regression.linear_model.RegressionResultsWrapper, output_path: str = 'results/regression_table.csv') -> pd.DataFrame:
    """Create a regression table from a fitted statsmodels OLS model and save it as CSV.

    The table includes coefficients, standard errors, t-statistics, p-values, 95% CI,
    significance stars, and a small block of model-level statistics (N, R-squared, Adj. R-squared, F-stat, F p-value).
    """
    # Coefficient-level details
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues
    conf = model.conf_int(alpha=0.05)
    conf.columns = ['ci_lower', 'ci_upper']

    coef_table = pd.concat([params, bse, tvals, pvals, conf], axis=1)
    coef_table.columns = ['coef', 'std_err', 't', 'p_value', 'ci_lower', 'ci_upper']
    coef_table.index.name = 'term'

    # Add significance stars based on conventional thresholds
    def _stars(p: float) -> str:
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.1:
            return '*'
        else:
            return ''

    coef_table['signif'] = coef_table['p_value'].apply(_stars)

    # Create a formatted coef+stars column for easy display
    coef_table['coefficient'] = coef_table.apply(lambda r: f"{r['coef']:.4f}{r['signif']}", axis=1)

    # Tidy numeric presentation (rounding for readability)
    coef_table['coef'] = coef_table['coef'].round(4)
    coef_table['std_err'] = coef_table['std_err'].round(4)
    coef_table['t'] = coef_table['t'].round(4)
    coef_table['p_value'] = coef_table['p_value'].round(4)
    coef_table['ci_lower'] = coef_table['ci_lower'].round(4)
    coef_table['ci_upper'] = coef_table['ci_upper'].round(4)

    # Model-level statistics
    model_stats = pd.DataFrame({
        'N': [int(model.nobs)],
        'R_squared': [round(model.rsquared, 4)],
        'Adj_R_squared': [round(model.rsquared_adj, 4)],
        'F_stat': [round(model.fvalue, 4) if model.fvalue is not None else None],
        'F_pvalue': [round(model.f_pvalue, 6) if model.f_pvalue is not None else None]
    }, index=['model_stats'])

    # Combine tables with a blank separator row for readability when saving to CSV
    separator = pd.DataFrame([{}])

    # Reorder columns to show formatted coef with stars first
    coef_table = coef_table[['coefficient', 'std_err', 't', 'p_value', 'ci_lower', 'ci_upper']]

    combined = pd.concat([coef_table, separator, model_stats], sort=False)

    # Ensure results directory exists and save
    out_path = Path(output_path)                        # create Path object for output
    out_path.parent.mkdir(parents=True, exist_ok=True)  # create parent directories if not existing
    combined.to_csv(out_path)

    print(f"Regression table saved to: {out_path}")
    return combined

#-----------------------------------------------------------------------
# Create & Export Dendrograms
#-----------------------------------------------------------------------
"""Create and save dendrogram visualizations from hierarchical clustering results."""

def _plot_and_save_dendrogram(Z, labels, output_path: str, title: str = None,
                              best_k: int | None = None,
                              figsize: tuple = (16, 8), leaf_font_size: int = 6):
    """Function with following Parameters:
    1. Z: Linkage matrix from models.py
    2. labels: Country names list
    3. output_path: Path to save the dendrogram image
    4. title: Optional title for the dendrogram
    5. best_k: Optimal number of clusters from silouhette analysis in models.py
    6. figsize: Figure dimensions
    7. leaf_font_size: Font size for country labels
    """

    # Create Figure of Specified Size and Plot Dendrogram

    fig, ax = plt.subplots(figsize=figsize) 
 
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=leaf_font_size, color_threshold=None)

    # Draw horizontal line to indicate cluster cut if best_k provided, enough merges in tree, 
    # and k>1 (meaning more than one cluster)

    if best_k is not None and Z.shape[0] >= (best_k - 1) and best_k > 1:
        cut_idx = Z.shape[0] - (best_k - 1) # merges needed for best_k clusters = total merges - (best_k-1)
        cut_threshold = Z[cut_idx, 2] # draw cutline at distance of that merge
        ax.axhline(y=cut_threshold, c='k', ls='--', lw=0.8) # dashed line
        ax.text(0.99, 0.95, f'k={best_k}', transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7)) # add text label for best_k

    # Add title and adjust spacing

    if title:
        ax.set_title(title)

    fig.tight_layout()

    out_path = Path(output_path) 
    out_path.parent.mkdir(parents=True, exist_ok=True) 
    fig.savefig(out_path, dpi=300) # save figure as png
    plt.close(fig)

    return out_path

#-----------------------------------------------------------------------
# Create Scree Plots for PCA Results
#-----------------------------------------------------------------------

def plot_scree(pca_object, output_path: str = 'results/scree_plot.png', 
               variance_threshold: float = 0.95, figsize=(12, 5),
               max_components_to_show: int = None):
    """
    Plot scree plot for PCA results.
    
    Parameters
    ----------
    1. pca_object : Fitted PCA object (from data_loader.py)
    2. output_path : Path to save the figure
    3. variance_threshold : Variance threshold for highlighting components (0.95)
    4. figsize : Figure size
    5. max_components_to_show : Maximum number of components to show in the plot
    """
    # Get explained variance ratios
    explained_variance = pca_object.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find component count for threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Create figure, limit display to reasonable number of components
    show_components = min(max_components_to_show, len(explained_variance))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Individual explained variance (scree)
    ax1.bar(range(1, show_components + 1), explained_variance[:show_components], 
            alpha=0.6, color='steelblue')
    ax1.axvline(x=n_components, color='red', linestyle='--', alpha=0.7, 
                label=f'{variance_threshold*100:.0f}% variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'Scree Plot (First {show_components} Components)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative explained variance

    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             color='darkorange', linewidth=2)
    ax2.axhline(y=variance_threshold, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=n_components, color='red', linestyle='--', alpha=0.7)
    ax2.scatter([n_components], [cumulative_variance[n_components-1]],
                color='red', zorder=5, s=100)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Title
    fig.suptitle('PCA on Combined 2015+2021 Data\n'
                 f'({n_components} components explain ≥95% variance)', 
                 fontsize=14, y=1.02)
    
    # Text box
    textstr = f'Observations: 214\n(107 countries × 2 years)\n'
    textstr += f'Original features: 642\n'
    textstr += f'Reduced to {n_components} components\n'
    textstr += f'Variance Explained: {cumulative_variance[n_components-1]:.1%}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.6, 0.05, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    
    fig.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved scree plot to {output_path}")
    
    return fig, n_components


#-----------------------------------------------------------------------
# Cluster World Maps
#-----------------------------------------------------------------------
"""Here, we colour two world maps with the clusters formed in each by the hierarchical clustering — one for 2015
and one for 2021. This requires geopandas and matplotlib, and improves interpretability of the clustering results. """


# CONVERT WIDE-FORMAT CLUSTER CSV TO LONG FORMAT
def read_clusters_from_wide_csv(path: str) -> pd.DataFrame:
    """Read the cluster CSVs created by `models.save_clusters_to_csv` and return a long
    DataFrame with columns `country` (ISO3) and `cluster` (int).
    """

    # Convert country codes to str to avoid issues with mixed types
    df = pd.read_csv(path, dtype=str)
    df = df.fillna('')

    # Melt wide-format Clusters into long format
    long = df.melt(var_name='cluster_label', value_name='country') # melt to long format
    long = long[long['country'].str.strip() != ''].copy() # drop empty country entries
    long['country'] = long['country'].str.strip() # strip whitespace (may cause problems with merge)

    # Extract integer cluster index from column name 'Cluster N', e.g. 'Cluster 0' -> 0
    """Need numeric values as cluster labels for plotting"""

    long['cluster'] = long['cluster_label'].str.extract(r"Cluster\s*(\d+)", expand=False).astype(int)

    long = long[['country', 'cluster']].drop_duplicates().reset_index(drop=True) # extract relevant columns
    return long


def map_and_plot(clusters_df: pd.DataFrame, output_path: str, title: str = None) -> None:
    """Merge clusters with a world GeoDataFrame and save a PNG choropleth.

    The function is robust to GeoPandas >=1.0 where the sample datasets helper was removed:
    it will try local shapefiles/geojsons first, then fall back to a small public GeoJSON URL.
    It also normalizes common ISO3 column names to `iso_a3` before merging.
    """

    # 1. LOAD WORLD MAP FROM PUBLIC URL
    url = 'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson'
    try:
        world = gpd.read_file(url)
    except Exception as e:          # Raise Error if unable to load world map
        raise RuntimeError(f"Could not load world map. Please check internet connection. Error: {e}")


    # 2. RENAME ISO COLUMN FOR EASE OF MATCHING (we know it's 'ISO3166-1-Alpha-3')
    world = world.rename(columns={'ISO3166-1-Alpha-3': 'iso_a3'})

    # Ensure country codes are str and uppercase
    world['iso_a3'] = world['iso_a3'].astype(str).str.strip().str.upper()
    clusters_df['country'] = clusters_df['country'].astype(str).str.strip().str.upper()

    # 4. FIX PLACEHOLDER CODES (e.g., -99)
    """The ISO-3 Column from the downloaded world map has some entries with '-99' or '-999' as placeholders, 
    causing problems with matching for France and Norway. Here, we rename that so France and Norway are not
    excluded from the cluster world map (otherwise they would just be greyed out)."""

    placeholder_vals = set(['-99', '-999', ''])
    mask = world['iso_a3'].isin(placeholder_vals)
    if mask.any():
        print(f"Fixing placeholder country codes...")
        for idx, row in world[mask].iterrows():
            name = row.get('name') or row.get('NAME') or ''
            if name == 'France':
                world.at[idx, 'iso_a3'] = 'FRA'
            elif name == 'Norway':
                world.at[idx, 'iso_a3'] = 'NOR'
            # Can add other countries here if needed

    # Merge and warn about unmatched codes
    merged = world.merge(clusters_df, left_on='iso_a3', right_on='country', how='left')

    missing = set(clusters_df['country']) - set(world['iso_a3'])
    if missing:
        print(f"Warning: {len(missing)} countries not in map")
        print(f"Missing countries: {sorted(missing)}") 
        # Countries missing from MAP, meaning countries in the clustering results that are unable to be coloured

    # Prepare categorical cluster column for plotting
    merged['cluster_cat'] = merged['cluster'].astype(pd.Int64Dtype())

    # Determine number of clusters (use max+1 so cluster numbers like 0..K-1 map cleanly)
    if clusters_df['cluster'].size == 0:
        print("No clusters to plot.")
        return

    n_clusters = int(clusters_df['cluster'].max()) + 1

    # Choose an appropriate qualitative colormap
    if n_clusters <= 10:
        base_cmap = plt.get_cmap('tab10')
    else:
        base_cmap = plt.get_cmap('tab20')

    colors = base_cmap.colors if hasattr(base_cmap, 'colors') else [base_cmap(i) for i in range(n_clusters)]
    cmap = ListedColormap(colors[:n_clusters])

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Plot countries colored by cluster; missing countries (NaN) will be lightgrey
    merged.plot(column='cluster_cat', categorical=True, cmap=cmap, linewidth=0.3, ax=ax,
                missing_kwds={'color': 'lightgrey', 'edgecolor': 'white'}, legend=False)

    # Build manual legend (so cluster order is clear)
    handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(n_clusters)]
    ax.legend(handles=handles, title='Cluster', loc='lower left', framealpha=0.9)

    ax.set_title(title or 'Country Clusters')
    ax.set_axis_off()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved map to {output_path}")


def create_cluster_maps(cluster_csv_dir: str = 'results', output_dir: str = 'results') -> None:
    """Reads cluster CSVs for 2015 and 2021 and produce PNG maps. By default, draws on .csv files from results/
    and saves created maps' PNG files in results/
    """
    for year in ['2015', '2021']:
        csv_path = os.path.join(cluster_csv_dir, f'clusters_{year}.csv')
        out_path = os.path.join(output_dir, f'clusters_{year}_map.png')

        if not os.path.exists(csv_path):
            print(f"Cluster CSV not found: {csv_path}")
            continue

        clusters = read_clusters_from_wide_csv(csv_path)
        map_and_plot(clusters, out_path, title=f'Country clusters — {year}')
