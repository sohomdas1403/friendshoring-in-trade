# Create a test.py file
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

print("All imports successful!")
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")



"""Remove historical COW codes that are no longer valid as of 2021 from the conversion dataframe"""""
#filtered_conversion_df = COW_ISO_conversion_df[
 #   (COW_ISO_conversion_df['valid_until'].isna()) |
  #  (COW_ISO_conversion_df['valid_until'] >= 2021)
#].copy()

"""Merge ISO codes into Agreement Score dataframe based on COW codes"""

#df_agreement = df_agreement.merge(
 #   filtered_conversion_df[['cow_id', 'iso3']],
  #  left_on='countrycode_1',
   # right_on='cow_id',
    #how='left'
#)
#df_agreement = df_agreement.drop(columns=['countrycode_1', 'cow_id'])
#df_agreement = df_agreement.rename(columns={'iso3': 'countrycode_1'})

#df_agreement = df_agreement.merge(
 #   filtered_conversion_df[['cow_id', 'iso3']],
  #  left_on='countrycode_2',
   # right_on='cow_id',
    #how='left'
#)
#df_agreement = df_agreement.drop(columns=['countrycode_2', 'cow_id'])
#df_agreement = df_agreement.rename(columns={'iso3': 'countrycode_2'})
