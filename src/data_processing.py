import pandas as pd

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

# Extract Data for Years of Interest
"""Filter Agreement Score, Cultural Distance, and Religious Distance Datasets to include only the years 
2015 and 2021. Geographical Distance and Linguistic Distance datasets are time-invariant, so no filtering is
necessary. Trade Volume dataset is from 2010-2021, so further filtering is not necessary at this stage. 
The COW to ISO conversion dataset is a historical reference table, so no filtering is necessary."""

df_agreement = df_agreement[df_agreement['year'].isin([2015, 2021])].copy()
df_culturaldistance = df_culturaldistance[df_culturaldistance['year'].isin([2015, 2021])].copy()
df_religiousdistance = df_religiousdistance[df_religiousdistance['year'].isin([2015, 2021])].copy()



#=======================================================================
# DATA CLEANING
#=======================================================================

# Standardise Trade Volume & Agreement Score Dataset Column Names to Match Other Datasets

df_agreement.columns = df_agreement.columns.str.replace('ccode', 'countrycode_')
df_tradevolume.columns = (
    df_tradevolume.columns
        .str.replace('COUNTERPART_COUNTRY.ID', 'countrycode_2')
        .str.replace('COUNTRY.ID', 'countrycode_1')
        .str.replace('TIME_PERIOD', 'year')
)

# Standardise Agreement Score Dataset Country Names from COW Codes to ISO 3166-1 alpha-3 Codes
"""Remove irrelevant COW Codes (no longer existing countries or no ISO codes)."""

codes_to_remove = [221, 223, 345, 816]  # Monaco, Liechtenstein, Yugoslavia, North Vietnam
df_agreement = df_agreement[
    ~(df_agreement['countrycode_1'].isin(codes_to_remove)) &
    ~(df_agreement['countrycode_2'].isin(codes_to_remove))
].copy()

"""Filter out invalid or outdated entries before creating the mapping dictionary."""

COW_ISO_conversion_df['valid_until'] = pd.to_numeric(
    COW_ISO_conversion_df['valid_until'], 
    errors='coerce'  # convert 'valid_until' to numeric, non-numeric → NaN
)

filtered_conversion_df = COW_ISO_conversion_df[
    (COW_ISO_conversion_df['valid_until'].isna()) |
    (COW_ISO_conversion_df['valid_until'] >= 2021) # filter to only valid or currently valid entries
].copy()

"""Create mapping dictionary from COW codes to ISO codes."""

filtered_conversion_df['cow_id'] = (
    filtered_conversion_df['cow_id']
    .fillna(-1)             # temporarily fill NaN to allow conversion
    .astype(int)            # convert to int
    .astype(str)            # convert to str for mapping
    .replace('-1', pd.NA)   # revert temporary NaN fill
)
filtered_conversion_df = filtered_conversion_df.drop_duplicates(subset='cow_id', keep='last') # keep most recent valid entry

"""DEBUG Prints to Verify Filtered Conversion Table"""
print(f"DEBUG: Filtered conversion table has {len(filtered_conversion_df)} rows.") 
print(f"DEBUG: First few rows of filtered table:\n{filtered_conversion_df[['cow_id', 'iso3', 'valid_until']].head()}")

cow2iso = dict(zip(filtered_conversion_df['cow_id'], filtered_conversion_df['iso3'])) # create mapping dictionary

"""DEBUG Prints to Verify Mapping Dictionary"""
print(f"Dictionary size: {len(cow2iso)}")
print(f"Sample keys: {list(cow2iso.items())[:5]}")

"""Map COW codes to ISO 3166-1 alpha-3 Codes in Agreement Score Dataset"""

orig1 = df_agreement['countrycode_1'].astype(str) # convert to str for mapping
orig2 = df_agreement['countrycode_2'].astype(str)

df_agreement['countrycode_1_iso3'] = orig1.map(cow2iso) # map using dictionary
df_agreement['countrycode_2_iso3'] = orig2.map(cow2iso)

"""DEBUG: Identify Unmapped COW Codes"""
unmapped1 = orig1[df_agreement['countrycode_1_iso3'].isna()].unique()
unmapped2 = orig2[df_agreement['countrycode_2_iso3'].isna()].unique()
if len(unmapped1) or len(unmapped2):
    import logging
    logging.getLogger(__name__).warning(
        "Unmapped COW codes found. countrycode_1: %s, countrycode_2: %s",
        list(unmapped1)[:10], list(unmapped2)[:10]
    )

"""Re-Standardise Column Names After Mapping and Drop COW Code Columns"""
df_agreement['countrycode_1'] = df_agreement['countrycode_1_iso3']
df_agreement['countrycode_2'] = df_agreement['countrycode_2_iso3']
df_agreement = df_agreement.drop(columns=['countrycode_1_iso3', 'countrycode_2_iso3'])


# Standardise Trade Volume Dataset Country Names from Full Names to ISO 3166-1 alpha-3 Codes









# Data Cleaning (name standarisation, missing value handling, etc.)

# Create the 6 Attribute Variable Matrices

# Concatenate Matrices, Standard Scaler, PCA

# Save Processed Data as Cache for Clustering
