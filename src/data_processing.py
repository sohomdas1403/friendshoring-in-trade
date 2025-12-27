import pandas as pd

# Read-in Raw Data

df_agreement: pd.DataFrame = pd.read_csv('data/raw/AgreementScores.csv')
df_culturaldistance: pd.DataFrame = pd.read_csv('data/raw/cultural_distance_PSW2024.csv')
df_geodistance: pd.DataFrame = pd.read_stata('data/raw/geodist_PSW24.dta')
df_tradevolume: pd.DataFrame = pd.read_csv('data/raw/imf_trade_volume(2010-2021).csv')
df_linguisticdistance: pd.DataFrame = pd.read_csv('data/raw/linguistic_distance_PSW2024.csv')
df_religiousdistance: pd.DataFrame = pd.read_stata('data/raw/religious_distance_PSW2024.dta')


# Data Cleaning: Column Names Standardisation for Country Names and Years

df_agreement.columns = df_agreement.columns.str.replace('ccode', 'countrycode_')
df_tradevolume.columns = (
    df_tradevolume.columns
        .str.replace('COUNTERPART_COUNTRY', 'countrycode_2')
        .str.replace('COUNTRY', 'countrycode_1')
        .str.replace('TIME_PERIOD', 'year')
)


# Data Cleaning (name standarisation, missing value handling, etc.)

# Create the 6 Attribute Variable Matrices

# Concatenate Matrices, Standard Scaler, PCA

# Save Processed Data as Cache for Clustering
