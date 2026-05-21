# %%
# Week 6: Feature Engineering
import pandas as pd

# %%
listings = pd.read_csv("data/listing_cleaned.csv", low_memory=False)
sold = pd.read_csv("data/sold_cleaned.csv", low_memory=False)

# %%
# Adding in New Metrics: Price Ratio (ClosePrice / OriginalListPrice)
listings['price ratio'] = listings['ClosePrice'] / listings['OriginalListPrice']
sold['price ratio'] = sold['ClosePrice'] / sold['OriginalListPrice']
print("Price Ratio added to both datasets.")

# %%
# Adding in New Metrics: Price Per Sq Ft (ClosePrice / LivingArea)
listings['Price Per Sq Ft'] = listings['ClosePrice'] / listings['LivingArea']
sold['Price Per Sq Ft'] = sold['ClosePrice'] / sold['LivingArea']
print("Price Per Sq Ft added to both datasets.")

# %%
# Adding in New Metrics: Days on Market (DaysOnMarket)
listings['Days on Market'] = listings['DaysOnMarket']
sold['Days on Market'] = sold['DaysOnMarket']
print("Days on Market added to both datasets.")

# %%
# Adding in New Metrics: Year / Month / YrMo (derived from CloseDate)
for df in (listings, sold):
    df["Year / Month / YrMo"] = pd.to_datetime(df["CloseDate"], errors="coerce").dt.strftime("%Y-%m")
print("Year / Month / YrMo added to both datasets.")

# %%
listings["Year / Month / YrMo"]

# %%
# Adding in New Metrics: Close to Original List Ratio (ClosePrice / OriginalListPrice)
# 1 = ClosePrice is the same as OriginalListPrice, 
# >1 = ClosePrice is higher than OriginalListPrice, 
# <1 = ClosePrice is lower than OriginalListPrice
for dataset in (listings, sold):
    dataset['Close to Original List Ratio'] = dataset['ClosePrice'] / dataset['OriginalListPrice']

# %%
# Adding in New Metrics: Listing to Contract Days (PurchaseContractDate - ListingContractDate)
# This metric represents the number of days between when a property was listed and when it went under contract.
for dataset in (listings, sold):
    dataset["Listing to Contract Days"] = (
        pd.to_datetime(dataset["PurchaseContractDate"], errors="coerce")
        - pd.to_datetime(dataset["ListingContractDate"], errors="coerce")
    ).dt.days

# %%
# Sanity Check
sold[sold["Listing to Contract Days"] < 0].count()

# %%
# Adding new metric: Contract to Close Days (CloseDate - PurchaseContractDate)
# This metric represents the number of days between when a property went under contract and when it closed.
# If 
for dataset in (listings, sold):
    dataset["Contract to Close Days"] = (
        pd.to_datetime(dataset["CloseDate"], errors="coerce")
        - pd.to_datetime(dataset["PurchaseContractDate"], errors="coerce")
    ).dt.days

# %%
# Sanity Check
sold[sold["Contract to Close Days"] < 0].count()

# %%
# Sanity Check
(sold["Contract to Close Days"] < 0).sum()

# %%
# ── Segment Analysis: group by key market dimensions ──────────────────────────
# Groupings:
#   1. PropertyType / PropertySubType  → property-mix patterns
#   2. CountyOrParish / MLSAreaMajor   → geographic market patterns
#   3. ListOfficeName / BuyerOfficeName → competitive office intelligence

def summarize_segments(df, min_count=5, top_n=15):
    """
    Group df by three key market dimensions and compute summary statistics.

    Parameters
    ----------
    df        : DataFrame (listings or sold)
    min_count : drop segments with fewer than this many closed transactions
    top_n     : for competitive groupings, keep only the top N segments by volume

    Returns
    -------
    dict of {grouping_label: summary_DataFrame}
    """

    # --- resolve column name variants -----------------------------------------
    prop_type_col = 'PropertyType' if 'PropertyType' in df.columns else 'PropertyType.1'

    # Define groupings: (label, [columns], is_competitive)
    # is_competitive=True → restrict display to top_n segments by transaction count
    groupings = [
        ("PropertyType / PropertySubType",   [prop_type_col, 'PropertySubType'],  False),
        ("CountyOrParish / MLSAreaMajor",    ['CountyOrParish', 'MLSAreaMajor'],  False),
        ("ListOfficeName / BuyerOfficeName", ['ListOfficeName', 'BuyerOfficeName'], True),
    ]

    # --- metrics aggregated for every grouping ---------------------------------
    agg_metrics = {
        'ClosePrice':            ['count', 'mean', 'median', 'std', 'min', 'max'],
        'OriginalListPrice':      ['mean', 'median'],
        'price ratio':           ['mean', 'median', 'std'],
        'Price Per Sq Ft':       ['mean', 'median', 'std'],
        'Days on Market':        ['mean', 'median', 'std'],
        'Listing to Contract Days': ['mean', 'median', 'std'],
        'Contract to Close Days':  ['mean', 'median', 'std'],
    }

    # --- only aggregate columns that actually exist in this df -----------------
    safe_agg = {col: fns for col, fns in agg_metrics.items() if col in df.columns}

    summaries = {}

    for label, cols, competitive in groupings:
        # skip grouping if any required column is absent
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[SKIP] {label} — missing columns: {missing}")
            continue

        summary = (
            df
            .groupby(cols, dropna=False)
            .agg(safe_agg)
        )

        # flatten MultiIndex columns: "ClosePrice_mean", etc.
        summary.columns = ["_".join(c).strip("_") for c in summary.columns]
        summary = summary.reset_index()

        # drop low-volume segments (noisy statistics)
        if 'ClosePrice_count' in summary.columns:
            summary = summary[summary['ClosePrice_count'] >= min_count]

        # sort by transaction volume descending
        if 'ClosePrice_count' in summary.columns:
            summary = summary.sort_values('ClosePrice_count', ascending=False)

        # for competitive groupings, restrict to top_n offices by volume
        if competitive:
            summary = summary.head(top_n)

        summary = summary.reset_index(drop=True)
        summaries[label] = summary

    return summaries


# ── Run segment analysis on both datasets ─────────────────────────────────────
sold_segment_summaries     = summarize_segments(sold)
listings_segment_summaries = summarize_segments(listings)

# ── Display results ───────────────────────────────────────────────────────────
for dataset_name, summaries in [
    ('SOLD',     sold_segment_summaries),
    ('LISTINGS', listings_segment_summaries),
]:
    for grouping, summary in summaries.items():
        top_label = " (top 15 by volume)" if "OfficeName" in grouping else ""
        print(f"\n{'='*70}")
        print(f"  {dataset_name} — grouped by {grouping}{top_label}")
        print(f"{'='*70}")

# %%
# Save DataFrames as CSV files
listings.to_csv("data/listings_with_features.csv", index=False)
sold.to_csv("data/sold_with_features.csv", index=False)
print("DataFrames with new features saved to CSV files.")


