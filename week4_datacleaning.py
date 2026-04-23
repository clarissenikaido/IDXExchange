# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
## Step 1: Load in MLS data
## Reading in concatenated and filtered for Residential data
listings_df = pd.read_csv("listings_res_concated.csv")
sold_df = pd.read_csv("sold_listings_res_concated.csv")

# %%
## Sanity check: Print row and column counts to confirm data loaded correctly
print(f"Listings DataFrame shape: {listings_df.shape}")
print(f"Sold DataFrame shape: {sold_df.shape}")

# %%
## Step 3: Converting date fields to datetime format
def convert_to_datetime(df, columns):
    """
    Converts specified columns in a DataFrame to datetime objects.
    """
    for col in columns:
        if col in df.columns:
            # errors='coerce' turns unparseable dates into NaT (Not a Time)
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

cols_to_reformat = [
    'CloseDate', 
    'PurchaseContractDate', 
    'ListingContractDate', 
    'ContractStatusChangeDate'
]

listings_df = convert_to_datetime(listings_df, cols_to_reformat)
sold_df = convert_to_datetime(sold_df, cols_to_reformat)

# %%
## Sanity Check: Print the data types of the date columns to confirm they are now datetime objects
def check_datetime_types(df, columns):
    for col in columns:
        if col in df.columns:
            print(f"{col}: {df[col].dtypes}")
print("Listings DataFrame Date Column Types:")
check_datetime_types(listings_df, cols_to_reformat)
print("\nSold DataFrame Date Column Types:")
check_datetime_types(sold_df, cols_to_reformat)

# %%
# Step 4: Identify columns with missing values and calculate the percentage of missing data
def missing_data_summary(df):
    missing_summary = df.isnull().sum().to_frame(name='MissingCount')
    missing_summary['MissingPercentage'] = (missing_summary['MissingCount'] / len(df)) * 100
    return missing_summary['MissingPercentage'][missing_summary['MissingPercentage'] >= 90]

# %%
print("Listings DataFrame Missing Data Summary:")
print(missing_data_summary(listings_df))

# %%
print("\nSold DataFrame Missing Data Summary:")
print(missing_data_summary(sold_df))

# %%
# Drop columns with 90% or more missing data
listings_df = listings_df.drop(columns=missing_data_summary(listings_df).index)
sold_df = sold_df.drop(columns=missing_data_summary(sold_df).index)
print("\nListings DataFrame shape after dropping high-missing columns:", listings_df.shape)
print("Sold DataFrame shape after dropping high-missing columns:", sold_df.shape)

# %%
print(sold_df.columns)

# %%
## Step 5: Remove or flag invalid numeric values: ClosePrice <= 0, LivingArea <= 0, DaysOnMarket < 0, negative Bedrooms or Bathrooms
print("\nSold DataFrame shape before removing invalid numeric values:", sold_df.shape)
if 'ClosePrice' in sold_df.columns:
    sold_df = sold_df[sold_df['ClosePrice'] > 0]
if 'LivingArea' in sold_df.columns:
    sold_df = sold_df[sold_df['LivingArea'] > 0]
if 'DaysOnMarket' in sold_df.columns:
    sold_df = sold_df[sold_df['DaysOnMarket'] >= 0]
if 'Bedrooms' in sold_df.columns:
    sold_df = sold_df[sold_df['Bedrooms'] >= 0]
if 'Bathrooms' in sold_df.columns:
    sold_df = sold_df[sold_df['Bathrooms'] >= 0]  
print("\nSold DataFrame shape after removing invalid numeric values:", sold_df.shape) 

print("\nListing DataFrame shape before removing invalid numeric values:", listings_df.shape)
if 'ClosePrice' in listings_df.columns:
    listings_df = listings_df[listings_df['ClosePrice'] > 0]
if 'LivingArea' in listings_df.columns:
    listings_df = listings_df[listings_df['LivingArea'] > 0]
if 'DaysOnMarket' in listings_df.columns:
    listings_df = listings_df[listings_df['DaysOnMarket'] >= 0]
if 'Bedrooms' in listings_df.columns:
    listings_df = listings_df[listings_df['Bedrooms'] >= 0]
if 'Bathrooms' in listings_df.columns:
    listings_df = listings_df[listings_df['Bathrooms'] >= 0]
print("\nListing DataFrame shape after removing invalid numeric values:", listings_df.shape)

# %%
# Step 6: Flag geographic errors in Latitude and Longitude
def flag_geographic_errors(df, lat_col='Latitude', lon_col='Longitude'):
    # 1. Flag missing coordinates
    df['Flag_Missing_Coords'] = df[lat_col].isnull() | df[lon_col].isnull()
    
    # 2. Flag zeros (Sentinel nulls)
    df['Flag_Zero_Coords'] = (df[lat_col] == 0) | (df[lon_col] == 0)
    
    # 3. Flag Longitude > 0 (California must be negative/West)
    df['Flag_Positive_Lon'] = df[lon_col] > 0
    
    # 4. Flag Out-of-State / Implausible (Broad California Bounding Box)
    # Approx CA limits: Lat 32 to 42, Lon -124 to -114
    lat_bounds = (32.5, 42.0)
    lon_bounds = (-124.5, -114.1)
    
    df['Flag_Out_Of_Bounds'] = ~(
        df[lat_col].between(*lat_bounds) & 
        df[lon_col].between(*lon_bounds)
    )
    
    # Note: We only flag Out_Of_Bounds if they aren't already null/zero
    df.loc[df['Flag_Missing_Coords'] | df['Flag_Zero_Coords'], 'Flag_Out_Of_Bounds'] = False
    
    return df

# Usage
print("\nListings DataFrame shape after flagging geographic errors:", listings_df.shape)
listings_df = flag_geographic_errors(listings_df)
print("\nSold DataFrame shape after flagging geographic errors:", sold_df.shape)
sold_df = flag_geographic_errors(sold_df)


