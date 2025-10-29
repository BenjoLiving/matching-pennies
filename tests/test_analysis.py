import polars as pl
import matching_pennies.io.csv_parser as csv_parser
import matching_pennies.compute.compute_performance_metrics as cpm 
import matplotlib.pyplot as plt 
import pyarrow
import pandas


directory = "C:/Users/benli/Documents/PaperManuscripts_andProjects/_2018_ACC_hallway/Matching Pennies/Testing/2018_01_24"
tmap = "C:/Users/benli/Documents/PaperManuscripts_andProjects/_2018_ACC_hallway/Matching Pennies/2018_acc_treatment_map.csv"

csvs = csv_parser.get_csv_files(directory, recursive=False)
df = csv_parser.build_trials(csvs, tmap, paradigm="normal")   

tdf, sdf = cpm.compute_metrics(df, keys=["animal_id", "session_idx"])

"""
TODO: 
- Make sure python computed metrics are the same as matlab computed metrics
- Add response time
- Add time in well
"""

# Test to make sure this version matches MATLAB version
test_csv = "C:/Users/benli/Documents/code/matching-pennies/test_data.csv"
test_df = pl.read_csv(test_csv)

# Helper function to find correct col names 
def search_cols(search_term, columns):
    return [col for col in columns if search_term.lower() in col.lower()]

# Test DF has column `Ses_id` with format: P4020:2018-01-24:12:04:34
# Extract date so we can join tables and compare differences. 
# tdf has `session_date` col with format: 2018-01-24

# Clip test_df `Ses_id` to YYYY-MM-DD 
test_df = test_df.with_columns(
    pl.col("Ses_id")
    .str.split(':')
    .list.get(1)
    .str.to_date("%Y-%m-%d")
    .alias("session_date") 
)

# Rename columns in test_df to match tdf 
test_df = test_df.rename({
    "Rat_id": "animal_id", 
    "Trial_indx": "trial_idx", 
    "SampledBothWellsBeforeTrial_flg": "EFS_before_flg", 
    "LoseSwitch_all_flg": "LoseSwitch_flg", 
    "WinStay_all_flg": "WinStay_flg"
})

# Drop `Ses_id` column
test_df = test_df.drop("Ses_id")

new_cols = tdf.columns
test_cols = test_df.columns

# All columns in test_cols are in new_cols
# We can now align the dataframes and compare

# Take a subset of the new dataframe - only relevant cols
sub = tdf[:, test_cols] 

# Normalize data types: 
keys = ["animal_id", "session_date", "trial_idx"]
cmp_flags  = ["LoseSwitch_flg", "LoseSwitch_NoMultWellResp_flg", "WinStay_flg", "WinStay_NoMultWellResp_flg", "EFS_before_flg"]
cmp_floats = ["InterTrialInterval", "ResponseTime"]
tol = 1e-6

def normalize(df: pl.DataFrame) -> pl.DataFrame:
    # ensure keys and comparison columns have matching dtypes
    exprs = [
        pl.col("animal_id").cast(pl.Utf8, strict=False),
        pl.col("trial_idx").cast(pl.Int64, strict=False),
    ]

    # if session_date is string, convert to Date
    if df.schema.get("session_date") == pl.Utf8:
        exprs.append(pl.col("session_date").str.to_date("%Y-%m-%d", strict=False))
    else:
        exprs.append(pl.col("session_date"))

    # cast flags and floats
    cmp_flags  = ["LoseSwitch_flg", "LoseSwitch_NoMultWellResp_flg", "WinStay_flg", "WinStay_NoMultWellResp_flg", "EFS_before_flg"]
    cmp_floats = ["InterTrialInterval", "ResponseTime"]

    exprs += [pl.col(c).cast(pl.Int8, strict=False) for c in cmp_flags if c in df.columns]
    exprs += [pl.col(c).cast(pl.Float64, strict=False) for c in cmp_floats if c in df.columns]

    return df.with_columns(exprs)


tdf_n = normalize(tdf) 
test_n = normalize(test_df) 


# 2) check duplicates (should be empty)
for name, df in [("tdf", tdf_n), ("test_df", test_n)]:
    dups = df.group_by(keys).len().filter(pl.col("len") > 1)
    if dups.height:
        print(f"WARNING: duplicate keys in {name}:")
        print(dups.head(10))


# 3) align with outer join and mark presence
left  = tdf_n.select(keys + cmp_flags + cmp_floats).with_columns(pl.lit(True).alias("_in_new"))
right = test_n.select(keys + cmp_flags + cmp_floats).with_columns(pl.lit(True).alias("_in_old"))

joined = left.join(right, on=keys, how="full", suffix="_old")

# 4) rows missing on either side
only_in_old = joined.filter(pl.col("_in_new").is_null()).select(keys)  # present only in MATLAB
only_in_new = joined.filter(pl.col("_in_old").is_null()).select(keys)  # present only in Python

# 5) per-column difference masks
flag_diffs = [
    (pl.col(c).cast(pl.Int64).eq_missing(pl.col(f"{c}_old").cast(pl.Int64)).not_()).alias(f"{c}_diff")
    for c in cmp_flags
]
float_diffs = [
    (
        ((pl.col(c) - pl.col(f"{c}_old")).abs() > tol) |
        (pl.col(c).is_nan() ^ pl.col(f"{c}_old").is_nan())
    ).alias(f"{c}_diff")
    for c in cmp_floats
]

common = joined.filter(pl.all_horizontal([pl.col("_in_new").is_not_null(), pl.col("_in_old").is_not_null()]))
common = common.with_columns(flag_diffs + float_diffs)

summary = common.select([
    pl.col(c).sum().alias(c) for c in common.columns if c.endswith("_diff")
])
print(summary)


# Diffs for wsls are due to weird matlab stuff 
# When the animal did not choose a well weird shit happens - matlab will sometimes calculate winstay = 0
# even though the previous trial was not a win (it was an empty string) 
# Could be due to 0-initializing the array? not sure, but my implementation is correct. 
# I know mine is correct because I went through the errors line by line and computed it myself.  