import numpy as np
import polars as pl 
import pandas as pd 
import bambi as bmb
import arviz as az 
import pymc as pm
from matching_pennies.io.metrics_store import load_metrics

EXPERIMENT = "lesion_study"
PARADIGM = "normal_mp"

tdf, sdf, manifest = load_metrics(EXPERIMENT, PARADIGM)

# =========================================
# First session analysis
# =========================================

tdf_ses1 = tdf.filter(
    pl.col("session_idx") == 1
)

efs_model_cols = [
    "EFS_before_flg",
    "treatment",
    "trial_idx",
    "animal_id",
]

efs_ses1 = tdf_ses1[:, efs_model_cols]
efs_ses1 = efs_ses1.filter(pl.col("EFS_before_flg").is_not_null())
efs_ses1.drop_nulls()

# Scale both columns in place
efs_ses1 = efs_ses1.with_columns([
    ((pl.col("trial_idx") - pl.col("trial_idx").min()) / 
    (pl.col("trial_idx").max() - pl.col("trial_idx").min())).alias("trial_idx_scaled")
])

efs_ses1 = efs_ses1.to_pandas()

# Make sure treatment is categorical with 'sham' as the reference
efs_ses1["treatment"] = pd.Categorical(efs_ses1["treatment"],
                                categories=["sham", "ofc", "mpfc"],
                                ordered=True)

# Define model structures
efs_model_intercept = "EFS_before_flg ~ treatment*trial_idx_scaled + (1|animal_id)"
efs_model_slope = "EFS_before_flg ~ treatment*trial_idx_scaled + (1 + trial_idx_scaled|animal_id)"

# Define model and fit it
model_intercept = bmb.Model(
    efs_model_intercept,
    efs_ses1,
    family="bernoulli",  # logistic link by default
)
model_slope = bmb.Model(
    efs_model_slope,
    efs_ses1, 
    family="bernoulli"
)

idata_intercept = model_intercept.fit(chains=4, cores=1, log_likelihood=True)
idata_slope = model_slope.fit(chains=4, cores=1, log_likelihood=True)

print("===========================================")
print("           RANDOM INTERCEPT MODEL") 
print("===========================================")
az.summary(idata_intercept)

print("===========================================")
print("           RANDOM SLOPE MODEL") 
print("===========================================")
az.summary(idata_slope)

# az.compare will not work 
# How can we manually compare these models? 