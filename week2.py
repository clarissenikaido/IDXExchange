## Task: Submit a .py script documenting unique property types found, the filtering logic applied, and a null-count summary table.
## Include a missing value report flagging any columns above 90% null. Produce a numeric
## distribution summary for the following columns: ClosePrice, LivingArea, DaysOnMarket. Save the filtered dataset as a new CSV.

import pandas as pd

## Loading in Listings CSV
years = [2024, 2025]
months = [f"{m:02d}" for m in range(1, 13)] # Creates ['01', '02', ..., '12']
listings_df = {
    f"{year}{month}": pd.read_csv(f"listing_data/CRMLSListing{year}{month}.csv")
    for year in years
    for month in months
}
listings_df["202601"] = pd.read_csv("listing_data/CRMLSListing202601.csv")
## Using centralized datasets provided
listings_df["202602"] = pd.read_csv("listing_data/CRMLSListing202602.csv")
listings_df["202603"] = pd.read_csv("listing_data/CRMLSListing202603.csv")

## Validating number of rows in each month 
listing_total = 0
for name, df in listings_df.items():
    print(f"{name} : {df.shape[0]}")
    listing_total += df.shape[0]
print(listing_total)

## Loading in Sold Listings CSV
years = [2024, 2025]
months = [f"{m:02d}" for m in range(1, 13)] # Creates ['01', '02', ..., '12']
sold_df = {
    f"{year}{month}": pd.read_csv(f"sold_data/CRMLSSold{year}{month}.csv")
    for year in years
    for month in months
}
sold_df["202601"] = pd.read_csv("sold_data/CRMLSSold202601.csv")
## Using centralized datasets provided
sold_df["202602"] = pd.read_csv("sold_data/CRMLSSold202602.csv")
sold_df["202603"] = pd.read_csv("sold_data/CRMLSSold202603.csv")

## Validating number of sold properties
sold_total = 0
for name, df in sold_df.items():
    print(f"{name} : {df.shape[0]}")
    sold_total += df.shape[0]
print(sold_total)

all_listings = pd.concat(listings_df.values(), ignore_index=True)
all_sold = pd.concat(sold_df.values(), ignore_index=True)
## Row & Column Count After Appending & Before Filtering
print("Row & Column Count Before Filtering:")
print(f"Listings has {all_listings.shape[0]} rows.")
print(f"Sold Listings has {all_sold.shape[0]} rows.")

## Identifying unique property types
sold_df['202601']['PropertyType'].unique()

## Row & Column Count After Filtering for Residential
print("Row & Column Count After Filtering:")
listings_res_df = all_listings[all_listings["PropertyType"]=="Residential"]
print(f"Residential Listings has {listings_res_df.shape[0]} rows and {listings_res_df.shape[1]} columns")
sold_res_df = all_sold[all_sold["PropertyType"]=="Residential"]
print(f"Sold Residential Listings has {sold_res_df.shape[0]} rows and {sold_res_df.shape[1]} columns")

## Function to create a null value report for a given DataFrame

def create_null_report(df, name="Dataset"):
    # 1. Calculate nulls
    null_counts = df.isna().sum()
    null_percent = (null_counts / len(df)) * 100
    
    # 2. Build the report table
    report = pd.DataFrame({
        'Column': null_counts.index,
        'Null Count': null_counts.values,
        'Null Percentage': null_percent.values
    })
    
    # 3. Create the flag (Consistency is key here!)
    flag_name = 'Flag (>90% Missing)'
    report[flag_name] = report['Null Percentage'] > 90
    
    # 4. Sort by most empty
    report = report.sort_values(by='Null Percentage', ascending=False).reset_index(drop=True)
    
    # 5. Print Summary
    print(f"\n--- {name} Missing Value Report ---")
    
    # Identify flagged columns
    high_null_cols = report[report[flag_name] == True]['Column'].tolist()

    print(f"Columns with >90% Missing: {len(high_null_cols)}")
    print(high_null_cols)

        
    return report

## Running missing values report

listing_report = create_null_report(listings_res_df, name="Active Listings")
sold_report = create_null_report(sold_res_df, name="Sold Properties")

# 1. Select the specific columns for distribution analysis
target_cols = ['ClosePrice', 'LivingArea', 'DaysOnMarket']

# 2. Generate the distribution summary
distribution_summary = listings_res_df[target_cols].describe(percentiles=[.25, .50, .75, .90, .95])

print("--- Numeric Distribution Summary ---")
print(distribution_summary)

# 3. Create a Filtered Dataset by dropping columns from previous report (columns with > 90% null)
cols_to_drop = listing_report[listing_report['Flag (>90% Missing)'] == True]['Column'].tolist()

# Create the new cleaned dataframe by dropping the empty columns
filtered_listings_df = listings_res_df.drop(columns=cols_to_drop)
filtered_listings_df = listings_res_df.drop(columns=cols_to_drop)

# 4. Save the new CSV
filtered_listings_df.to_csv('Cleaned_Listings_Data.csv', index=False)

print(f"\n✅ Success!")
print(f"1. Summary generated for: {target_cols}")
print(f"2. Dropped {len(cols_to_drop)} columns with >90% missing data.")
print(f"3. Filtered dataset saved as: Cleaned_Listings_Data.csv")

