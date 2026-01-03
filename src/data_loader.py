import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set random seed for reproducibility
SEED = 42

#=======================================================================
# READ-IN RAW DATA
#=======================================================================

df_agreement: pd.DataFrame = pd.read_csv('data/raw/AgreementScores.csv')
df_culturaldistance: pd.DataFrame = pd.read_csv('data/raw/cultural_distance_PSW2024.csv')
df_geodistance: pd.DataFrame = pd.read_stata('data/raw/geodist_PSW24.dta')
df_tradevolume: pd.DataFrame = pd.read_csv('data/raw/imf_trade_volume(2010-2021).csv')
df_linguisticdistance: pd.DataFrame = pd.read_csv('data/raw/linguistic_distance_PSW2024.csv')
df_religiousdistance: pd.DataFrame = pd.read_stata('data/raw/religious_distance_PSW2024.dta')
COW_ISO_conversion_df: pd.DataFrame = pd.read_csv('data/raw/cow2iso.csv')

# EXTRACT DATA FOR YEARS OF INTEREST
"""Filter Agreement Score, Cultural Distance, and Religious Distance Datasets to include only the years 
2015 and 2021. Geographical Distance and Linguistic Distance datasets are time-invariant, so no filtering is
necessary. Trade Volume dataset is from 2010-2021, so further filtering is not necessary at this stage. We will
need the 2010-2014 data to calculate the baseline trade volume later on, and filter to 2015 and 2021 afterwards.
The COW to ISO conversion dataset is a historical reference table, so no filtering is necessary."""

# 1) Filter Agreement Score Dataset to Years 2015 and 2021
df_agreement['year'] = df_agreement['year'].astype(int)  # ensure year is int type
df_agreement = df_agreement[df_agreement['year'].isin([2015, 2021])].copy()

# 2) Filter Cultural Distance Dataset to Years 2015 and 2021
"""The Cultural Distance Dataset contains data only for years 1984, 1993, 1998, 2002, 2009, 2014, and 2021.
To align with our analysis years (2015 and 2021), we extrapolate by carryin over the earliest or most recent data.
For 2015, we use the 2014 data as a proxy, and for 2021, we use the available 2021 data. Such extrapolation is
justified by the authors, so we adopt the same approach here."""

cultural_2015 = df_culturaldistance[df_culturaldistance['year'] == 2014].copy() # use 2014 data as proxy for 2015
cultural_2015['year'] = 2015  # set year to 2015
df_culturaldistance = pd.concat([df_culturaldistance, cultural_2015], ignore_index=True) # append to main df

df_culturaldistance['year'] = df_culturaldistance['year'].astype(int)  # ensure year is int type
df_culturaldistance = df_culturaldistance[df_culturaldistance['year'].isin([2015, 2021])].copy() 

# 3) Filter Religious Distance Dataset to Years 2015 and 2021
"""The Religious Distance Dataset contains data only for years 1900, 1970, 2000, 2015, and 2020.To align with 
our analysis years (2015 and 2021), we extrapolate by carryin over the earliest or most recent data. For 2021, we
use the 2020 data as a proxy, and for 2015, we use the available 2015 data."""

religious_2021 = df_religiousdistance[df_religiousdistance['year'] == 2020].copy() # use 2020 data as proxy for 2021
religious_2021['year'] = 2021  # set year to 2021
df_religiousdistance = pd.concat([df_religiousdistance, religious_2021], ignore_index=True) # append to main df

df_religiousdistance['year'] = df_religiousdistance['year'].astype(int)  # ensure year is int type
df_religiousdistance = df_religiousdistance[df_religiousdistance['year'].isin([2015, 2021])].copy() 


#=======================================================================
# DATA CLEANING
#=======================================================================
"""Here, we do the following:
1. Standardise column names in Trade Volume and Agreement Score datasets.
2. Clean and prepare Agreement Score dataset (convert COW codes to ISO codes, convert agreement scores to
   geopolitical distance).
3. Clean and prepare Trade Volume dataset (extract years of interest, verify dyad symmetry, remove missing
   values, compute trade volume deviations from baseline).
4. Address missing countries and dyads mismatches between datasets.
"""


# Standardise Trade Volume & Agreement Score Dataset Column Names to Match Other Datasets
"""Here, we standardise the column names in the Trade Volume and Agreement Score datasets. Otherwise, merging
and filtering operations later on would become cumbersome."""

df_agreement.columns = df_agreement.columns.str.replace('ccode', 'countrycode_')
df_tradevolume.columns = (
    df_tradevolume.columns
        .str.replace('COUNTERPART_COUNTRY.ID', 'countrycode_2')
        .str.replace('COUNTRY.ID', 'countrycode_1')
        .str.replace('TIME_PERIOD', 'year')
)

#-----------------------------------------------------------------------
# Clean and Prepare Agreement Score Dataset
# (standardise Agreement Score Dataset Country Names from Correlates of War (COW) Codes 
# to ISO 3166-1 alpha-3 Codes)
#-----------------------------------------------------------------------
"""The Agreement Score dataset uses COW country codes. Here, we convert these to ISO 3166-1 alpha-3 codes
to ensure consistency across datasets. We first filter out irrelevant or outdated COW codes, then create a mapping
dictionary from COW codes to ISO codes, and finally apply this mapping to the Agreement Score dataset.

Furthermore, the agreement scores measure UN voting similarity (1 = identical, 0 = oppisite). However, the other
datasets we use (cultural, religious, etc...) are distance measures (where higher = more different). Thus, for 
consistency with other distance metrics we transform the agreement scores to a distance metric by computing 
(1 - agree). This way, a score of 0 indicates identical voting patterns, while a score of 1 indicates diametrically 
opposed voting patterns."""

# 1) Remove irrelevant COW Codes (no longer existing countries or no ISO codes)
codes_to_remove = [221, 223, 345, 816]  # Monaco, Liechtenstein, Yugoslavia, North Vietnam
df_agreement = df_agreement[
    ~(df_agreement['countrycode_1'].isin(codes_to_remove)) &
    ~(df_agreement['countrycode_2'].isin(codes_to_remove))
].copy()

# 2) Filter out invalid or outdated entries before creating the mapping dictionary
COW_ISO_conversion_df['valid_until'] = pd.to_numeric(
    COW_ISO_conversion_df['valid_until'], 
    errors='coerce'  # convert 'valid_until' to numeric, non-numeric → NaN
)

filtered_conversion_df = COW_ISO_conversion_df[
    (COW_ISO_conversion_df['valid_until'].isna()) |
    (COW_ISO_conversion_df['valid_until'] >= 2021) # filter to only valid or currently valid entries
].copy()

# 3) Create mapping dictionary from COW codes to ISO codes
filtered_conversion_df['cow_id'] = (
    filtered_conversion_df['cow_id']
    .fillna(-1)             # temporarily fill NaN with -1 to allow conversion (All COW codes > 0, so -1 is safe)
    .astype(int)            # convert to int
    .astype(str)            # convert to str for mapping
    .replace('-1', pd.NA)   # revert temporary NaN fill
)
filtered_conversion_df = filtered_conversion_df.drop_duplicates(subset='cow_id', keep='last') # keep most recent valid entry

# DEBUG Prints to Verify Filtered Conversion Table
print("AGREEMENT SCORE DATASET COW TO ISO MAPPING DICTIONARY CREATION:")
print(f"DEBUG: Filtered conversion table has {len(filtered_conversion_df)} rows.") 
print(f"DEBUG: First few rows of filtered table:\n{filtered_conversion_df[['cow_id', 'iso3', 'valid_until']].head()}")

cow2iso = dict(zip(filtered_conversion_df['cow_id'], filtered_conversion_df['iso3'])) # create mapping dictionary

# DEBUG Prints to Verify Mapping Dictionary
print("\nDEBUG: Created COW to ISO mapping dictionary.")
print(f"Dictionary size: {len(cow2iso)}")
print(f"Sample keys: {list(cow2iso.items())[:5]}")

# 4) Map COW codes to ISO 3166-1 alpha-3 Codes in Agreement Score Dataset
orig1 = df_agreement['countrycode_1'].astype(str) # convert to str for mapping
orig2 = df_agreement['countrycode_2'].astype(str)

df_agreement['countrycode_1_iso3'] = orig1.map(cow2iso) # map using dictionary
df_agreement['countrycode_2_iso3'] = orig2.map(cow2iso)

# DEBUG: Identify Unmapped COW Codes
unmapped1 = orig1[df_agreement['countrycode_1_iso3'].isna()].unique()
unmapped2 = orig2[df_agreement['countrycode_2_iso3'].isna()].unique()
if len(unmapped1) or len(unmapped2):
    import logging
    logging.getLogger(__name__).warning(
        "Unmapped COW codes found. countrycode_1: %s, countrycode_2: %s",
        list(unmapped1)[:10], list(unmapped2)[:10]
    )

# 5) Re-Standardise Column Names After Mapping and Drop COW Code Columns
df_agreement['countrycode_1'] = df_agreement['countrycode_1_iso3']
df_agreement['countrycode_2'] = df_agreement['countrycode_2_iso3']
df_agreement = df_agreement.drop(columns=['countrycode_1_iso3', 'countrycode_2_iso3'])

# 6) Convert Agreement Scores to Geopolitical Distance (1 - agree)
df_agreement['pol_distance'] = 1 - df_agreement['agree']

# DEBUG: Print Summary Statistics of Converted Distance Metric
print(f"\nConverted agreement scores to distance (1 - agree) — Summary Statistics:") 
print(f"Range: [{df_agreement['pol_distance'].min():.3f}, {df_agreement['pol_distance'].max():.3f}]") # min and max disagreement
print(f"Mean distance: {df_agreement['pol_distance'].mean():.3f}") # average disagreement


#-----------------------------------------------------------------------
# Clean and Prepare Trade Volume Dataset
#-----------------------------------------------------------------------
"""Here, we clean and prep the Trade Volume Dataset for use by. First, we extract the data for the years needed
for our analysis (2010-2015, and 2021). Then, we conduct a diagnostic test to verify that all countries appearing
as Country 1 also appear as Country 2 in the dyadic data and vice versa. This ensures that adding the exports in
both directions yields trade volume, allowing us to overcome the obstacle of incomplete import data in the 
IMF Database. Furthermore, it makes the results for 2015 and 2021 comparable, while also making PCA viable. 
Then, we proceed to with cleaning the dataset by dropping missing values and computing trade volume deviations 
from the 2010-2014 baseline for each dyadic relationship."""

# EXTRACT DATA FOR YEARS OF INTEREST
required_years = list(range(2010, 2016)) + [2021] # years 2010-2015 and 2021
df_tradevolume = df_tradevolume[df_tradevolume['year'].isin(required_years)]
df_tradevolume['year'] = df_tradevolume['year'].astype(int)  # ensure year is int type

# DIAGNOSTIC TEST — VERIFY DYAD SYMMETRY:

# 1) Check unique values in both columns
reporters = set(df_tradevolume['countrycode_1'])
partners = set(df_tradevolume['countrycode_2'])

print("\nTRADE VOLUME DATASET DYAD SYMMETRY CHECK:")
print("Number of unique reporters:", len(reporters)) # should be 209
print("Number of unique partners:", len(partners)) # should be 209

# 2) Find countries that appear as reporter but NOT as partner
missing_as_reporter = partners - reporters
print("Countries that are reporters but never partners:", missing_as_reporter) # should be empty set

# 3) Find countries that appear as partner but NOT as reporter
missing_as_partner = reporters - partners
print("Countries that are partners but never reporters:", missing_as_partner) # should be empty set

# 4) Check if sets are identical
if reporters == partners:
    print("\n✓ Perfect! All reporters appear as partners and vice versa.")
else:
    print(f"\nX  Mismatch: {len(missing_as_partner)} reporters missing as partners, {len(missing_as_reporter)} partners missing as reporters")


# CHECK FOR AND REMOVE ANY ROWS WITH MISSING VALUES
"""Remove any rows with missing year or OBS_VALUE values to ensure data integrity and report the number of rows
removed and retained for analysis. Missing values in these columns would create incomplete dyadic observations,
cause NaN entries in the country-level feature matrix, and thus break PCA."""

initial_count = len(df_tradevolume)
df_tradevolume = df_tradevolume.dropna(subset=['year']).dropna(subset=['OBS_VALUE'])
removed_count = initial_count - len(df_tradevolume)

# DEBUG Print summary of missing values removal
print("\nTRADE VOLUME DATASET REMOVING MISSING VALUES:")
if removed_count > 0:
    print(f"Removed {removed_count} rows with NA values ({removed_count/initial_count*100:.2f}%)")
else:
    print("No NA values found in 'year' or 'OBS_VALUE' columns.") 

print(f"{len(df_tradevolume)} rows available for analysis")


# COMPUTE TRADE VOLUME FOR EACH DYADIC RELATIONSHIP
"""Convert OBS_VALUE to numeric, normalise country codes, create ordered pairs (country_A, country_B), 
and aggregate exports in both directions so each row is an undirected pair-year with total trade in USD."""

# 1) Ensure OBS_VALUE is numeric (treat non-numeric as 0)
df_tradevolume['OBS_VALUE'] = pd.to_numeric(df_tradevolume['OBS_VALUE'], errors='coerce').fillna(0)

# 2) Standardise country code strings and strip whitespace
df_tradevolume['countrycode_1'] = df_tradevolume['countrycode_1'].astype(str).str.strip()
df_tradevolume['countrycode_2'] = df_tradevolume['countrycode_2'].astype(str).str.strip()

# 3) Define ordered pairs in alphabetical order to treat dyads as undirected
"""Converts both (USA->CHN) and (CHN->USA) row labels to (CHN->USA). This allows us to later add the the exports
in both rows with (CHN->USA) to get total trade volume between the two countries."""

df_tradevolume['ordered_1'] = df_tradevolume[['countrycode_1', 'countrycode_2']].min(axis=1)
df_tradevolume['ordered_2'] = df_tradevolume[['countrycode_1', 'countrycode_2']].max(axis=1)

# 4) Remove self-pairs (same country on both sides) - these are not dyadic between two countries
df_tradevolume = df_tradevolume[df_tradevolume['ordered_1'] != df_tradevolume['ordered_2']].copy()

# 5) Aggregate exports for both directions to yield trade volume into a single pair-year observation
df_trade_pair = (
    df_tradevolume
    .groupby(['ordered_1', 'ordered_2', 'year'], as_index=False)['OBS_VALUE']
    .sum()
    .rename(columns={'OBS_VALUE': 'trade_volume_USD'})
)

df_trade_pair = df_trade_pair.rename(columns={
    'ordered_1': 'countrycode_1',
    'ordered_2': 'countrycode_2',
})

# 6) Add pair identifiers, reorder columns for clarity, & add to trade volume dataframe
""" Define a function that performs this task, as it will be needed add pair identifiers to the other datasets
later on as well. Note that pair identifiers use a comma between ISO/codes, e.g. 'USA,CHN' """

def add_pair_column(df):  # define function for adding'pair' column to any dataframe
    """Create 'pair' column as 'countrycode_1,countrycode_2' (sorted alphabetically)."""
    df['pair'] = df.apply(
        lambda r: ','.join(sorted([str(r['countrycode_1']).strip(), 
                                   str(r['countrycode_2']).strip()])), 
        axis=1
    )
    return df

df_trade_pair = add_pair_column(df_trade_pair)

df_tradevolume_sum = df_trade_pair[['pair', 'year', 'trade_volume_USD', 'countrycode_1', 'countrycode_2']].copy()

# DEBUG: Print summary of aggregated trade volume dataframe
print("\nAGGREGATED TRADE VOLUME DATAFRAME INFO:")
print(f"Aggregated trade volume to {len(df_tradevolume_sum)} pair-year rows (USD).")
print('Preview:\n', df_tradevolume_sum.head())
print('(Number of Rows, Number of Columns) = ', df_tradevolume_sum.shape)

# DEBUG: Check an example dyad (CAN-USA) across years
can_usa = df_tradevolume_sum[
    (df_tradevolume_sum['countrycode_1'] == 'CAN') & 
    (df_tradevolume_sum['countrycode_2'] == 'USA')
]
print('\nCanada-USA Dyad Example:\n', can_usa.sort_values('year'))

# COMPUTE TRADE VOLUME DEVIATIONS FOR EACH DYADIC RELATIONSHIP IN 2015 AND 2021 FROM 2010-2014 BASELINE
"""Compute the average trade volume for each dyadic pair from 2010-2014 to establish a baseline. Then, calculate
the log of the squared deviations of trade volume in 2015 and 2021 from this baseline. This helps to normalise 
trade volumes and account for variations in trade activity over time."""

# 1) Compute baseline mean (2010-2014) per pair
baseline = (
    df_tradevolume_sum[df_tradevolume_sum['year'].between(2010, 2014)]
    .groupby('pair', as_index=False)['trade_volume_USD']
    .mean()
    .rename(columns={'trade_volume_USD': 'baseline_mean'})
)

df_tradevolume = df_tradevolume_sum.merge(baseline, on='pair', how='left') # merge with main dataframe

# 2) Compute LOG of squared deviation only for 2015 and 2021 (NaN for baseline years 2010-2014)
"""Using np.log1p = log(1+x) to handle zero deviations gracefully. Then, extract only for years 2015 and 2021.
This both retrieves the year of interest and drops NaN values for baseline years (they are NaN by design, not 
due to defective data)."""

df_tradevolume['trade_volume_std'] = (      # compute log of squared deviations
    np.log1p((df_tradevolume['trade_volume_USD'] - df_tradevolume['baseline_mean']) ** 2)
).where(df_tradevolume['year'].isin([2015, 2021])) # keep only for years 2015 and 2021, else NaN

df_tradevolume = df_tradevolume[df_tradevolume['year'].isin([2015, 2021])].copy()
df_tradevolume = df_tradevolume.dropna(subset=['trade_volume_std']) # drop NaN rows (baseline years)

# DEBUG Print summary of baseline & log deviations computation
print("\nTRADE VOLUME DEVIATIONS COMPUTATION INFO:")

print(f"Computed baseline mean for {len(baseline)} pairs.") # number of unique dyads

print("Non-NaN trade_volume_std rows:", df_tradevolume['trade_volume_std'].notna().sum()) # see note below

""" NOTE: This should equal number of unique dyads * 2 (for 2015 and 2021) if perfect. If less, some dyads are 
missing data in either 2015 or 2021, meaning balancing is required. Since that is the case here, we proceed
to balance the dyads below (along with balancing the other datasets)."""


#-----------------------------------------------------------------------
# Address Missing Countries & Dyads Mismatches Between Datasets
#-----------------------------------------------------------------------
"""The 4 datasets constructed by Spolaore & Wacziarg (Cultural Distance, Geographical Distance, 
Linguistic Distance, and Religious Distance) are complete datasets with no NaN values, and all labels are
as desired. Thus, for these datasets, further cleaning is not necessary. However, to ensure consistency across 
all datasets in anticipation of PCA and clustering, we identify the set of common countries and dyads present 
in all datasets and filter each dataset accordingly."""

# PREPARE DATASETS FOR FILTERING TO COMMON COUNTRIES AND DYADS

# 1) Drop Irrelevant Columns
"""df_geodistance and df_linguisticdistance do not have 'year' columns as they are time-invariant."""

df_agreement = df_agreement[['countrycode_1', 'countrycode_2', 'year', 'pol_distance']]
df_tradevolume = df_tradevolume[['countrycode_1', 'countrycode_2', 'year', 'trade_volume_std']]
df_culturaldistance = df_culturaldistance[['countrycode_1', 'countrycode_2', 'year', 'cultdist_std']]
df_religiousdistance = df_religiousdistance[['countrycode_1', 'countrycode_2', 'year', 'reldist_weighted']]
df_geodistance = df_geodistance[['countrycode_1', 'countrycode_2', 'avg_distance_norm']]
df_linguisticdistance = df_linguisticdistance[['countrycode_1', 'countrycode_2', 'lingdist_tree_weighted']]


# 2) Create 'pair' Columns for Each Dataset
"""Creating 'pair' columns in each dataset allows for easier identification and filtering of dyads without
separately identifying unique countries and dyads (increases efficiency). Call earlier function."""

df_agreement = add_pair_column(df_agreement)
df_tradevolume = add_pair_column(df_tradevolume)
df_culturaldistance = add_pair_column(df_culturaldistance)
df_religiousdistance = add_pair_column(df_religiousdistance)
df_geodistance = add_pair_column(df_geodistance)
df_linguisticdistance = add_pair_column(df_linguisticdistance)


# BALANCE DYADS WITHIN EACH DATASET TO INCLUDE ONLY THOSE WITH DATA IN BOTH 2015 AND 2021
"""Here, we identify dyads that have data for both years 2015 and 2021 in the time-variant datasets. They must
match, otherwise PCA and clustering will be affected by missing data. For time-invariant datasets, all dyads are 
considered balanced as they do not vary by year."""

# 1) Define function to get intersection of balanced pairs
def get_balanced_pairs(df): 
    if 'year' in df.columns:
        pairs_2015 = set(df[df['year'] == 2015]['pair'])
        pairs_2021 = set(df[df['year'] == 2021]['pair'])
        return pairs_2015 & pairs_2021
    else:
        return set(df['pair'])    # return all pairs for time-invariant datasets

# 2) Check number of balanced dyads in each dataset
print("\nDATASETS DYADS BALANCING INFO:")

agreement_pairs = get_balanced_pairs(df_agreement)
print(f"Agreement: {len(agreement_pairs)} balanced pairs")

trade_pairs = get_balanced_pairs(df_tradevolume)      
print(f"Trade: {len(trade_pairs)} balanced pairs")

cultural_pairs = get_balanced_pairs(df_culturaldistance)
print(f"Cultural: {len(cultural_pairs)} balanced pairs")

religious_pairs = get_balanced_pairs(df_religiousdistance)
print(f"Religious: {len(religious_pairs)} balanced pairs")

geo_pairs = get_balanced_pairs(df_geodistance)                  # All pairs as time-invariant (no year)
print(f"Geographical: {len(geo_pairs)} balanced pairs")

linguistic_pairs = get_balanced_pairs(df_linguisticdistance)    # All pairs as time-invariant (no year)
print(f"Linguistic: {len(linguistic_pairs)} balanced pairs")

# DIAGNOSTIC: Check number of unique countries and mismatched pairs in Cultural Distance Dataset
"""The reason the numbers for Cultural Distance Dataset are shown in particular is because it has the lowest 
number of balanced pairs, meaning the other datasets will most likely be filtered down to its level. This code is
not necessary for the functioning of the data loader, and can be commented out if desired. However, it is helpful
in understanding and visualising what is happening during the balancing process."""

print("\nCULTURAL DISTANCE DATASET DIAGNOSTIC CHECKS:")

# Check 1: How many unique countries in each year?
cultural_2015_countries = set(df_culturaldistance[df_culturaldistance['year'] == 2015]['countrycode_1']).union(
                         set(df_culturaldistance[df_culturaldistance['year'] == 2015]['countrycode_2']))
cultural_2021_countries = set(df_culturaldistance[df_culturaldistance['year'] == 2021]['countrycode_1']).union(
                         set(df_culturaldistance[df_culturaldistance['year'] == 2021]['countrycode_2']))

print(f"Cultural 2015 unique countries: {len(cultural_2015_countries)}")                    
print(f"Cultural 2021 unique countries: {len(cultural_2021_countries)}")                    
print(f"Countries in BOTH years: {len(cultural_2015_countries & cultural_2021_countries)}")

# Check 2: Sample missing dyads
all_cultural_pairs_2015 = set(df_culturaldistance[df_culturaldistance['year'] == 2015]['pair'])
all_cultural_pairs_2021 = set(df_culturaldistance[df_culturaldistance['year'] == 2021]['pair'])
missing_in_2015 = all_cultural_pairs_2021 - all_cultural_pairs_2015
missing_in_2021 = all_cultural_pairs_2015 - all_cultural_pairs_2021

print(f"\nPairs in 2021 but NOT 2015: {len(missing_in_2015)}") # should be 0
print(f"Pairs in 2015 but NOT 2021: {len(missing_in_2021)}") # should be 0

if missing_in_2015:
    print(f"2015 sample missing: {list(missing_in_2015)[:5]}")

if missing_in_2021:
    print(f"2021 sample missing: {list(missing_in_2021)[:5]}")

# 4) Balance the dyads in each dataset to include only those present in both 2015 and 2021
df_agreement = df_agreement[df_agreement['pair'].isin(agreement_pairs)]
df_tradevolume = df_tradevolume[df_tradevolume['pair'].isin(trade_pairs)]
df_culturaldistance = df_culturaldistance[df_culturaldistance['pair'].isin(cultural_pairs)]
df_religiousdistance = df_religiousdistance[df_religiousdistance['pair'].isin(religious_pairs)]
df_geodistance = df_geodistance[df_geodistance['pair'].isin(geo_pairs)]
df_linguisticdistance = df_linguisticdistance[df_linguisticdistance['pair'].isin(linguistic_pairs)]

print("\nBALANCING FOR EACH DATASET COMPLETE.")

"""Note that this balances dyads only WITHIN each dataset. The dyads are not yet balanced ACROSS all datasets. 
This will be done next"""

# BALANCE PAIRS/DYADS ACROSS ALL 6 DATASETS TO COMMON SET
"""Identify the intersection of balanced dyads across all 6 datasets and filter each dataset accordingly. This is
needed that each dataset contributes the same number of features for each dyad in the final country-level feature
matrix, on which PCA will be applied."""

# 1) Find intersection of balanced pairs across all datasets
common_pairs = (
    agreement_pairs & 
    trade_pairs & 
    cultural_pairs & 
    religious_pairs & 
    geo_pairs & 
    linguistic_pairs
)

print("\nCOMMON DYADS ACROSS ALL DATASETS:")
print(f"Number of dyads present in ALL 6 Datasets: {len(common_pairs)}") 
# This is the final number of dyads that will be used in the analysis

# 2) Filter each dataset to common pairs
df_agreement_balanced = df_agreement[df_agreement['pair'].isin(common_pairs)].copy()
df_tradevolume_balanced = df_tradevolume[df_tradevolume['pair'].isin(common_pairs)].copy()
df_culturaldistance_balanced = df_culturaldistance[df_culturaldistance['pair'].isin(common_pairs)].copy()
df_religiousdistance_balanced = df_religiousdistance[df_religiousdistance['pair'].isin(common_pairs)].copy()
df_geodistance_balanced = df_geodistance[df_geodistance['pair'].isin(common_pairs)].copy()
df_linguisticdistance_balanced = df_linguisticdistance[df_linguisticdistance['pair'].isin(common_pairs)].copy()

# 3) Extract common countries from the filtered pairs
"""Each pair is a string of the form 'countryA,countryB' (two ISO3 codes). We split these strings to get the 
unique countries. This is needed for mapping to indices for the country-level feature matrix later on. It also 
helps in verifying that key countries are present in the final balanced datasets."""

common_countries = set()
for pair in common_pairs:
    c1, c2 = pair.split(',')
    common_countries.add(c1)
    common_countries.add(c2)

print(f"Number of unique countries across datasets: {len(common_countries)}")
# This is the final number of countries that will be used in the analysis

# DEBUG: Check that key countries are present (G20 members excluding regional blocs)
"""Given that the G20 countries account for an overwhelming proportion of international trade, global GDP, and
the world's population, it is important to note if any of them are missing as that would represent a serious gap 
in the analysis. Thus, we check for their presence in the common_countries set. We exclude regional blocs 
(European Union & African Union) as they do not represent individual countries."""

G20_countries = ['USA', 'CHN', 'IND', 'RUS', 'BRA', 'DEU', 'FRA', 'GBR', 'JPN', 'CAN', 
                 'AUS', 'ITA', 'MEX', 'KOR', 'TUR', 'SAU', 'ARG', 'ZAF', 'IDN']
missing_G20_countries = [c for c in G20_countries if c not in common_countries]

if missing_G20_countries: # prints warning if any G20 countries missing & lists them
    print(f"\nWARNING: {len(missing_G20_countries)} G20 countries missing from common set") 
    print(f"   Missing: {missing_G20_countries}")
else:
    print("\n✓ All G20 countries present in common country set.")
    print("FILTERING TO COMMON DYADS COMPLETE.")

# DEBUG: Final Missing Value Check in Balanced Datasets (Should be None)
"""Here, we perform a final check for any missing values in the key distance/volume columns of each 
balanced dataset."""
print("\n=== MISSING VALUE CHECK IN BALANCED DATASETS ===")

balanced_datasets = {
    'Trade': df_tradevolume_balanced,
    'Agreement': df_agreement_balanced,
    'Cultural': df_culturaldistance_balanced,
    'Religious': df_religiousdistance_balanced,
    'Geographic': df_geodistance_balanced,
    'Linguistic': df_linguisticdistance_balanced
}

all_clean = True
for name, df in balanced_datasets.items():
    # Check the distance column for each dataset
    if name == 'Trade':
        col = 'trade_volume_std'
    elif name == 'Agreement':
        col = 'pol_distance'
    elif name == 'Cultural':
        col = 'cultdist_std'
    elif name == 'Religious':
        col = 'reldist_weighted'
    elif name == 'Geographic':
        col = 'avg_distance_norm'
    elif name == 'Linguistic':
        col = 'lingdist_tree_weighted'
    
    missing = df[col].isna().sum()
    if missing > 0:
        print(f"❌ {name}: {missing} missing values in '{col}'")
        all_clean = False
    else:
        print(f"✅ {name}: No missing values") # should print this for all 6 (now balanced) datasets

# REMOVE DUPLICATES AND CHECK DIMENSIONS OF BALANCED DATASETS
"""Here, we ensure that each balanced dataset contains unique rows for each (pair, year) combination or unique
pairs for time-invariant datasets. Then, we check and print out the dimensions and key statistics of each 
balanced dataset for verification."""

#1) Remove duplicate rows that may exist after balancing if any

df_agreement_balanced = df_agreement_balanced.drop_duplicates(subset=['pair', 'year']).copy()
df_tradevolume_balanced = df_tradevolume_balanced.drop_duplicates(subset=['pair', 'year']).copy()
df_culturaldistance_balanced = df_culturaldistance_balanced.drop_duplicates(subset=['pair', 'year']).copy()
df_religiousdistance_balanced = df_religiousdistance_balanced.drop_duplicates(subset=['pair', 'year']).copy()
df_geodistance_balanced = df_geodistance_balanced.drop_duplicates(subset=['pair']).copy()
df_linguisticdistance_balanced = df_linguisticdistance_balanced.drop_duplicates(subset=['pair']).copy()

balanced_datasets = {
    'Trade': df_tradevolume_balanced,
    'Agreement': df_agreement_balanced,
    'Cultural': df_culturaldistance_balanced,
    'Religious': df_religiousdistance_balanced,
    'Geographic': df_geodistance_balanced,
    'Linguistic': df_linguisticdistance_balanced
}

#2) Print dimensions and key statistics of balanced datasets (to verify data cleaning worked as intended)
"""The number of pairs should be the same across all 6 datasets. The number of rows in the time-invariant datasets
should be exactly half that of the time-varying datasets, as they do not vary by year and here we use 2 years for 
our analysis (2015 and 2021). The time-invariant datasets should also have one less column than the time-varying
data as they lack the 'year' column."""

print("\n" + "="*60)
print("BALANCED DATASETS DIMENSIONS CHECK")
print("="*60)
print(f"{'Dataset':<12} | {'Rows':>8} | {'Columns':>8} | {'Years':>12} | {'Pairs':>8}")
print("-" * 65)

for name, df in balanced_datasets.items():
    rows, cols = df.shape
    
    # Get unique years (if column exists, as in time-variant datasets)
    if 'year' in df.columns:
        years = sorted(df['year'].unique())
        years_str = f"{years[0]},{years[-1]}" if len(years) == 2 else str(years)
    else:
        years_str = "Time-invariant"
    
    # Get number of unique pairs
    unique_pairs = df['pair'].nunique()
    
    print(f"{name:<12} | {rows:>8,} | {cols:>8} | {years_str:>12} | {unique_pairs:>8,}")

print(f"\nFinal dataset: {len(common_countries)} countries, {len(common_pairs)} pairs")
print(f"Feature Matrix dimensions will be: {len(common_countries)}×{6*len(common_countries)}")


#=======================================================================
# CREATE COUNTRY-LEVEL FEATURE MATRICES (N x (N x 6)) FOR PCA
#=======================================================================
"""Here, we create the country-level feature matrices (one for 2015, one for 2021) for PCA by transforming 
each balanced dataset into an N x (N x 6) matrix, where N is the number of countries and 6 represents the six 
attributes represented by the six balanced datasets (Trade, Agreement, Cultural, Religious, Geographic, Linguistic).

The reason for creating country-level feature matrices is to enable the application of PCA. Stacking the 6 balanced
datasets into a single matrix means PCA finds components that capture variance across all six attributes 
simultaneously. 

The structure of the country-level feature matrix is as follows:
1. Each row corresponds to a country e.g. 'USA', 'CHN', 'IND', etc.
2. Each column corresponds to a specific attribute for a specific country e.g. 'trade_volume_std_USA',
    'pol_distance_USA', etc.

This results in a matrix of shape (N, N*6) where each country has 6 attributes for each of the N countries.
 """

#-----------------------------------------------------------------------
# Prepare for Country-Level Feature Matrix Engineering
#-----------------------------------------------------------------------
"""Here, we create a mapping from country codes to matrix indices so that we can efficiently populate the feature
matrices. We also initialize empty matrices for 2015 and 2021."""

# 1) Create alphabetically sorted list of common countries and mapping to indices
countries_list = sorted(list(common_countries))
n_countries = len(countries_list)

country_to_idx = {country: idx for idx, country in enumerate(countries_list)}
""" Creates a dictionary mapping country codes to matrix indices:
    {'ALB': 0, 'ARG': 1, 'ARM': 2, 'AUS': 3, 'AUT': 4, ...} """

# 2) Initialize feature matrices for 2015 and 2021
"""Shape: (n_countries, n_countries * 6) where 6 = number of distance metrics"""

feature_matrix_2015 = np.zeros((n_countries, n_countries * 6))      # numpy array of zeros
feature_matrix_2021 = np.zeros((n_countries, n_countries * 6))      # numpy array of zeros

# 3) Define the datasets and their corresponding column names containing the distance/volume metrics
"""This allows for easier iteration when populating the feature matrices."""

datasets_info = [
    (df_agreement_balanced, 'pol_distance'),
    (df_tradevolume_balanced, 'trade_volume_std'),
    (df_culturaldistance_balanced, 'cultdist_std'),
    (df_religiousdistance_balanced, 'reldist_weighted'),
    (df_geodistance_balanced, 'avg_distance_norm'),
    (df_linguisticdistance_balanced, 'lingdist_tree_weighted')
]


#-----------------------------------------------------------------------
# Complete Feature Matrix Engineering
#-----------------------------------------------------------------------
"""Populate the feature matrices for 2015 and 2021 by iterating through each balanced dataset and filling in 
the appropriate entries based on the dyadic distances. Then, define proper attribute names and create clean 
column names for the final DataFrames."""

# POPULATE FEATURE MATRICES FOR 2015 AND 2021

for dataset_idx, (df, col_name) in enumerate(datasets_info): 
    col_offset = dataset_idx * n_countries # calculate column offset for current dataset
    
    # For time-variant datasets (first 4), filter by year
    if col_name in ['pol_distance', 'trade_volume_std', 'cultdist_std', 'reldist_weighted']:
        df_2015 = df[df['year'] == 2015]
        df_2021 = df[df['year'] == 2021]
    else:  # time-invariant datasets (geographic, linguistic)
        df_2015 = df
        df_2021 = df
    
    # Populate 2015 matrix
    for _, row in df_2015.iterrows():       # iterate through each row, ignoring row index as not needed
        c1, c2 = row['countrycode_1'], row['countrycode_2']
        if c1 in country_to_idx and c2 in country_to_idx: 
            idx1, idx2 = country_to_idx[c1], country_to_idx[c2]     # get matrix indices
            feature_matrix_2015[idx1, col_offset + idx2] = row[col_name]
            feature_matrix_2015[idx2, col_offset + idx1] = row[col_name]
            """Ensures Symmetric Filing to make matrix complete and symmetric"""
    
    # Populate 2021 matrix
    for _, row in df_2021.iterrows():       # symmetric to function for populating 2015 matrix
        c1, c2 = row['countrycode_1'], row['countrycode_2']
        if c1 in country_to_idx and c2 in country_to_idx:
            idx1, idx2 = country_to_idx[c1], country_to_idx[c2]
            feature_matrix_2021[idx1, col_offset + idx2] = row[col_name]
            feature_matrix_2021[idx2, col_offset + idx1] = row[col_name]

"""No need to fill diagonal entries (self-distances) explicitly, as they are already 0 by initialisation, and
we already removed self-pairs from the datasets earlier. This was done when creating the common_pairs set earlier.

This is because self-distances were explicitly removed from df_tradevolume earlier, meaning that trade_pairs do not
contain any self-pairs. Since common_pairs is the intersection of trade_pairs with the other datasets' pairs, it 
cannot contain any self-pairs. Thus, the feature matrices will have 0s on the diagonal as desired."""

# GENERATE PREVIEW OF POPULATED FEATURE MATRICES

# 1) Define proper attribute names
attr_names = ['pol_distance', 'trade_volume_std', 'cultdist_std', 
              'reldist_weighted', 'avg_distance_norm', 'lingdist_tree_weighted']

# 2) Create clean column names
"""Nested loop to create column names in the format 'attribute_countrycode', e.g. 'pol_distance_USA'."""

clean_columns = []
for attr in attr_names:
    for country in countries_list:
        clean_columns.append(f"{attr}_{country}")

# 3) Create DataFrames with clean names
"""Will apply Standard Scaler on these dataframes later before PCA. """
feature_df_2015 = pd.DataFrame(
    feature_matrix_2015,
    index=countries_list,
    columns=clean_columns
)

feature_df_2021 = pd.DataFrame(
    feature_matrix_2021,
    index=countries_list,
    columns=clean_columns
)

# 4) Print summary statistics and previews of created feature matrices

# DEBUG: Print summary of created feature matrices
print("\nCOUNTRY-LEVEL FEATURE MATRICES CREATED:")
print(f"2015 matrix: (rows, columns) = {feature_df_2015.shape}")
print(f"2021 matrix: (rows, columns) = {feature_df_2021.shape}")

# DEBUG: Preview first 5 countries and first 5 features of each matrix (2015 and 2021)
print("\n" + "="*60)
print("2015 MATRIX PREVIEW (first 5 countries × first 5 features)")
print("="*60)
print(feature_df_2015.iloc[:5, :5])

print("\n" + "="*60)
print("2021 MATRIX PREVIEW (first 5 countries × first 5 features)")
print("="*60)
print(feature_df_2021.iloc[:5, :5])

print("\nFEATURE MATRICES ENGINEERING COMPLETE.")

#=======================================================================
# STANDARDISATION & APPLY PCA FOR DIMENSIONALITY REDUCTION
#=======================================================================
"""Given that we have constructed high-dimensional feature matrices (N x (N x 6)), we proceed to standardise 
the data and apply PCA to mitigate the Curse of Dimensionality."""

# 1) Standardise the feature matrices
"""This transforms each feature to have mean 0 and standard deviation 1, ensuring that all features contribute
equally to the PCA. This is needed as PCA is sensitive to feature scales. Clustering is done twice — once for each
year — so we standardise each year's feature matrix separately."""

scaler = StandardScaler()
feature_matrix_2015_scaled = scaler.fit_transform(feature_matrix_2015)
feature_matrix_2021_scaled = scaler.fit_transform(feature_matrix_2021)

# 2) Apply PCA to reduce dimensionality
"""Here, we first vertically stack the feature matrices for both years to fit PCA on all data. This ensures that
the principal components are consistent across both years, allowing for comparable clustering results. We then
transform each year's feature matrix using the fitted PCA and retain only the components needed to explain
95% of the variance."""

# Vertically stack Feature Matrix Arrays for both years for fitting PCA on all data
combined_scaled = np.vstack([feature_matrix_2015_scaled, feature_matrix_2021_scaled])

# Fit PCA on combined data
"""Fit PCA on combined scaled data so that component selection is consistent across both years. This ensures
that clustering results for 2015 and 2021 are comparable."""

pca = PCA(random_state=SEED)  # set random state for reproducibility
pca.fit(combined_scaled)

# Transform both years using the fitted PCA
feature_matrix_2015_pca = pca.transform(feature_matrix_2015_scaled)
feature_matrix_2021_pca = pca.transform(feature_matrix_2021_scaled)

# Determine number of components to retain (e.g., 95% variance explained)
cumsum_var = np.cumsum(pca.explained_variance_ratio_) # sum of % of variance explained by each component

n_components = np.argmax(cumsum_var >= 0.95) + 1 
"""Returns index of first TRUE in boolean array, then adds 1 to convert from index to count. This 
reflects the number of components needed to reach at least 95% variance explained."""

feature_matrix_2015_pca = feature_matrix_2015_pca[:, :n_components]
feature_matrix_2021_pca = feature_matrix_2021_pca[:, :n_components]
"""Retain only the PCA components needed for 95% variance explained (n_components count)"""

# Print summary of PCA results
print("\nPCA RESULTS SUMMARY:")
print(f"Variance explained by first {n_components} components: {cumsum_var[n_components-1]:.1%}")
print(f"2015 PCA matrix shape: (Rows, Columns) = {feature_matrix_2015_pca.shape}")
print(f"2021 PCA matrix shape: (Rows, Columns) = {feature_matrix_2021_pca.shape}")
print("\nPCA DIMENSIONALITY REDUCTION COMPLETE")