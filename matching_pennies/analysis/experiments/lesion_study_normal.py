import numpy as np
import polars as pl 
import pandas as pd 
import bambi as bmb
import arviz as az 
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
    "session_idx",
    "animal_id",
]

efs_ses1 = tdf_ses1[:, efs_model_cols]
efs_ses1.drop_nulls()

# Scale both columns in place
efs_ses1 = efs_ses1.with_columns([
    ((pl.col("trial_idx") - pl.col("trial_idx").min()) / 
    (pl.col("trial_idx").max() - pl.col("trial_idx").min())).alias("trial_idx_scaled"),

    ((pl.col("session_idx") - pl.col("session_idx").min()) / 
    (pl.col("session_idx").max() - pl.col("session_idx").min())).alias("session_idx_scaled")
])

efs_ses1 = efs_ses1.to_pandas()

# Make sure treatment is categorical with 'sham' as the reference
efs_ses1["treatment"] = pd.Categorical(efs_ses1["treatment"],
                                categories=["sham", "ofc", "mpfc"],
                                ordered=True)

# Define model structure 
efs_model = "EFS_before_flg ~ treatment*trial_idx_scaled + treatment*session_idx_scaled + (1|animal_id)"

# Define model and fit it
model = bmb.Model(
    efs_model,
    efs_ses1,
    family="bernoulli",  # logistic link by default
)
idata = model.fit()