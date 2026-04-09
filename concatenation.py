import pandas as pd

## Loading in Listings CSV
years = [2024, 2025]
months = [f"{m:02d}" for m in range(1, 13)] # Creates ['01', '02', ..., '12']
listings_dfs = {
    f"{year}{month}": pd.read_csv(f"listing_data/CRMLSListing{year}{month}.csv")
    for year in years
    for month in months
}
listings_dfs["202601"] = pd.read_csv("listing_data/CRMLSListing202601.csv")
listings_dfs["202602"] = pd.read_csv("listing_data/CRMLSListing202602.csv")
listings_dfs["202603"] = pd.read_csv("listing_data/CRMLSListing202603.csv")

## Validating number of listings in each month
listing_total = 0
for name, df in listings_dfs.items():
    print(f"{name} : {df.shape[0]}")
    listing_total += df.shape[0]
print(listing_total)

## Loading in Sold Listings CSV
years = [2024, 2025]
months = [f"{m:02d}" for m in range(1, 13)] # Creates ['01', '02', ..., '12']
sold_dfs = {
    f"{year}{month}": pd.read_csv(f"sold_data/CRMLSSold{year}{month}.csv")
    for year in years
    for month in months
}
sold_dfs["202601"] = pd.read_csv("sold_data/CRMLSSold202601.csv")
sold_dfs["202602"] = pd.read_csv("sold_data/CRMLSSold202602.csv")
sold_dfs["202603"] = pd.read_csv("sold_data/CRMLSSold202603.csv")

## Validating number of sold listings in each month
sold_total = 0
for name, df in sold_dfs.items():
    print(f"{name} : {df.shape[0]}")
    sold_total += df.shape[0]
print(sold_total)

## Concatenating all listings and sold listings into their own dataset
all_listings = pd.concat(listings_dfs.values(), ignore_index=True)
all_sold = pd.concat(sold_dfs.values(), ignore_index=True)

## Row & Column Count After Appending & Before Filtering
print("Row & Column Count Before Filtering:")
print(f"Listings has {all_listings.shape[0]} rows.")
print(f"Sold Listings has {all_sold.shape[0]} rows.")

## Row & Column Count After Filtering
print("Row & Column Count After Filtering:")
listings_res_df = all_listings[all_listings["PropertyType"]=="Residential"]
print(f"Residential Listings has {listings_res_df.shape[0]} rows and {listings_res_df.shape[1]} columns")
sold_res_df = all_sold[all_sold["PropertyType"]=="Residential"]
print(f"Sold Residential Listings has {sold_res_df.shape[0]} rows and {sold_res_df.shape[1]} columns")

listings_res_df.to_csv("listings_res_concated.csv", index = False)
sold_res_df.to_csv("sold_listings_res_concated.csv", index = False)