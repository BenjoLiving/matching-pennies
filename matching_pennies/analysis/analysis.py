import numpy as np
import matplotlib.pyplot as plt 
import polars as pl 
import matching_pennies.io.csv_parser as csv_parser
import matching_pennies.compute.compute_performance_metrics as cpm 

"""
TODO:
- Check `TimeInWell` to make sure it matches matlab 
    - There are negative and null values -> double check 
"""

# Load data and compute metrics 
directory = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/normal_mp"
tmap = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/lesion_study_ingestion_map.csv"

csvs = csv_parser.get_csv_files(directory, recursive=True)
df = csv_parser.build_trials(csvs, tmap, paradigm="normal")   

tdf, sdf = cpm.compute_metrics(df, keys=["animal_id", "session_idx"])

model_cols = [
    "EFS_before_flg",
    "treatment",
    "trial_idx",
    "session_idx",
    "animal_id",
]

m = tdf[:, model_cols]

# Remove nan columns
m = m.filter(pl.col("EFS_before_flg").is_not_null())
m= m.drop_nulls()

# Scale both columns in place
m = m.with_columns([
    ((pl.col("trial_idx") - pl.col("trial_idx").min()) / 
     (pl.col("trial_idx").max() - pl.col("trial_idx").min())).alias("trial_idx_scaled"),

    ((pl.col("session_idx") - pl.col("session_idx").min()) / 
     (pl.col("session_idx").max() - pl.col("session_idx").min())).alias("session_idx_scaled")
])

# Suppose df is a pandas DataFrame with columns:
df = m.to_pandas()

import bambi as bmb

# make sure df is a pandas DataFrame
# categorical vars can stay as string/object; bambi will encode them

model = bmb.Model(
    "EFS_before_flg ~ treatment*trial_idx_scaled + treatment*session_idx_scaled + (1|animal_id)",
    df,
    family="bernoulli",  # logistic link by default
)

idata = model.fit()  # runs MCMC (NUTS)
print(bmb.summary(idata))

import arviz as az 
ppc = model.predict(idata, kind="response")
az.plot_ppc(ppc)