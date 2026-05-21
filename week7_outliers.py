# %%
import pandas as pd

# %%
listings = pd.read_csv("data/listings_with_features.csv")
sold = pd.read_csv("data/sold_with_features.csv")

# %%
def add_outlier_flags(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[f"{column}_Outlier"] = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)
    return df

# %%
# Adding Flags for Outliers
for col in ["ClosePrice", "LivingArea", "DaysOnMarket"]:
    print(f"Outliers in {col} (Listings):")
    print(add_outlier_flags(listings, col))
    print(f"Outliers in {col} (Sold):")
    print(add_outlier_flags(sold, col))

# %%
# Save DataFrames as CSV files
listings.to_csv("data/listings_flaggedoutliers.csv", index=False)
sold.to_csv("data/sold_flaggedoutliers.csv", index=False)
print("DataFrames with flagged outliers saved to CSV files.")

# %%
# Removing entries with outliers
listings2 = listings.copy()
sold2 = sold.copy()
for col in ["ClosePrice", "LivingArea", "DaysOnMarket"]:
    listings2 = listings2[listings2[f"{col}_Outlier"] == 0].drop(columns=[f"{col}_Outlier"])
    sold2 = sold2[sold2[f"{col}_Outlier"] == 0].drop(columns=[f"{col}_Outlier"])

# %%
print(listings.shape, sold.shape)
print(listings2.shape, sold2.shape)

# %%
# Save Datasets without outliers
listings2.to_csv("data/listings_removedoutliers.csv", index=False)
sold2.to_csv("data/sold_removedoutliers.csv", index=False)
print("DataFrames with outliers removed saved to CSV files.")

# %%
cols = ["ClosePrice", "LivingArea", "DaysOnMarket"]

summary = []
for label, df in [
    ("listings", listings),
    ("listings2", listings2),
    ("sold", sold),
    ("sold2", sold2),
]:
    medians = df[cols].median()
    summary.append(
        {
            "dataset": label,
            "rows": len(df),
            "columns": df.shape[1],
            **{f"median_{col}": medians[col] for col in cols},
        }
    )

summary_df = pd.DataFrame(summary).set_index("dataset")
print(summary_df)

for original, filtered in [("listings", "listings2"), ("sold", "sold2")]:
    orig_rows = len(eval(original))
    filt_rows = len(eval(filtered))
    print(
        f"{original} -> {filtered}: {orig_rows} rows -> {filt_rows} rows "
        f"({100 * filt_rows / orig_rows:.2f}% retained)"
    )


