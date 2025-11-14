import numpy as np
import polars as pl 
import pandas as pd 
import bambi as bmb
import arviz as az 
import pymc as pm
import matplotlib.pyplot as plt 
import altair as alt
from scipy.ndimage import gaussian_filter1d
from IPython.display import display
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matching_pennies.io.metrics_store import load_metrics
from matching_pennies.analysis.bambi_plots import plot_trace

EXPERIMENT = "lesion_study"
PARADIGM = "normal_mp"

tdf, sdf, manifest = load_metrics(EXPERIMENT, PARADIGM)

# =========================================
# First session analysis
# =========================================
tdf_ses1 = tdf.filter(
    pl.col("session_idx") == 1
)

# =========================================
# Basic Plots 
# =========================================
# Get trial number per session
max_trials = (
    tdf_ses1
    .group_by(["animal_id", "session_idx"])
    .agg(pl.col("trial_idx").max().alias("max_trial"))
)

# Find minimum session length
min_trial = max_trials["max_trial"].min()

# Truncate df to min session length 
tdf_trunc = tdf_ses1.filter(pl.col("trial_idx") <= min_trial)

# Get mean EFS per treatment per trial
efs_by_trt_trial = (
    tdf_trunc
    .select("treatment", "trial_idx", "EFS_before_flg")
    .group_by(["treatment", "trial_idx"])
    .agg([
        pl.mean("EFS_before_flg").alias("efs_mean"),        # proportion if 0/1
        pl.len().alias("n"),                                # group size
        pl.col("EFS_before_flg").cast(pl.Int8).sum().alias("n_efs")  # # of 1s
    ])
    .sort(["treatment", "trial_idx"])
)

# Smooth efs_mean with Gaussian kernel 
SIGMA = 8
efs_smoothed = (
    efs_by_trt_trial
    .group_by("treatment")
    .map_groups(
        lambda df: df.sort("trial_idx").with_columns(
            pl.Series(
                "efs_smooth",
                gaussian_filter1d(df["efs_mean"].fill_null(0).to_numpy(), sigma=SIGMA)
            )
        )
    )
)

idk = alt.Chart(efs_smoothed, width=800, height=800).mark_line().encode(
    alt.X("trial_idx"),
    alt.Y("efs_smooth"),
    color="treatment"
)
idk.save("efs_by_trial_smoth.html")

# =========================================
# Bayesian Analysis
# =========================================

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

idata_intercept = model_intercept.fit(idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
idata_slope = model_slope.fit(idata_kwargs={"log_likelihood": True}, cores=1, chains=4)

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

# Trace plot 
fig_int, ax_int = plot_trace(idata_intercept, compact=True)
plt.show()
fig_slope, ax_slope = plot_trace(idata_slope, compact=True)
plt.show()


# Posterior predictive checks 
ppc_intercept = model_intercept.predict(
    idata_intercept,
    kind="pps",
    inplace=False
)
az.plot_ppc(ppc_intercept, kind="kde", show=True)

ppc_slope = model_slope.predict(
    idata_slope,
    kind="pps",
    inplace=False
)
az.plot_ppc(ppc_slope, kind="kde", show=True)


# Forest plot to compare the two models: 
_, ax = plt.subplots(figsize = (8, 8))

plot_vars = [
    "Intercept",
    "treatment",
    "trial_idx_scaled",
    "treatment:trial_idx_scaled",
    "1|animal_id_sigma"
]
az.plot_forest(
    [idata_intercept, idata_slope],
    var_names=plot_vars,
    combined=True,
    colors=["#666666", "red"],
    ax=ax
)
# Create legend
handles = [
    Patch(label="Random Intercept", facecolor="#666666"),
    Patch(label="Random Slope", facecolor="red")
]

legend = ax.legend(handles=handles, loc=4, fontsize=14, frameon=True, framealpha=0.8)
plt.show()

# Compare models based on information criterion 
int_waic = az.waic(idata_intercept)
slope_waic = az.waic(idata_slope)
int_loo = az.loo(idata_intercept)
slope_loo = az.loo(idata_slope)

"""
Comparison results: 
- WAIC gain from random slope model is small (delta = 7) 
- Random slope model LOO shows +10 effective parameters 
- Pareto K diagnostics are OK for both models 

Conclusions: Use random intercept model as baseline for inference. 
** Figure out what all of these mean and how they are calculated! 
"""

# Compute probability of direction P(beta > 0 | data) 
# How much of the OFC P density is above 0? 
p_gt0_ofc = (idata_intercept.posterior.treatment.sel(treatment_dim="ofc") > 0).mean(("chain", "draw")).item()
print(f"P(OFC mean > 0) = {p_gt0_ofc}")

# How much of the mPFC P density is above 0? 
p_gt0_mpfc = (idata_intercept.posterior.treatment.sel(treatment_dim="mpfc") > 0).mean(("chain", "draw")).item()
print(f"P(mPFC mean > 0) = {p_gt0_mpfc}")
"""
TODO: 
- Figure out how to calculate probability of direction.
    - Integral of posterior > 0
"""