import numpy as np
import polars as pl 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
import platform
from matching_pennies.io.metrics_store import load_metrics
from matching_pennies.analysis.bambi_plots import plot_trace
from matching_pennies.utils import preprocessing, plotting 

from matching_pennies.utils.search_cols import search_cols

# %% 
###################################
# LOAD DATA 
###################################

EXPERIMENT = "lesion_study" 
PARADIGM = "normal_mp" 

system_os = platform.system().lower()

# if system_os == "windows": 
#     data_path = "C:/Users/benli/Documents/code/matching-pennies/data"
#     cores=4
#     export_path = "C:/Users/benli/Documents/projects/lesion_study/results/paper_analysis/notebook_figures/normal_mp"
# elif system_os == "darwin":
#     data_path = "/Users/ben/Documents/source/matching-pennies/data"
#     cores=1
# elif system_os == "linux": 
#     data_path = "/home/benjo/projects/matching-pennies/data"

data_path = "/home/benjo/projects/matching-pennies/data/"

tdf, sdf, manifest = load_metrics(EXPERIMENT, PARADIGM, root=data_path)

#%%
###################################
# FIG 1. Session Measures  
###################################

"""
The point of this figure is to show that lesions did not impact the animals ability to perform 
the task. 

We look at 4 means of behavioural performance: 
    - Number of trials -> Were animals doing similar numbers of trials? 
    - Number of rewarded trials / wins -> Were the animals getting similar numbers of rewards? 
    - Mean reaction time -> How fast did the animals respond to the 'go' cue? 
    - Mean response entropy -> How random the animals were 
"""

# Number of trials
trial_df = (
        sdf
        .group_by(["treatment", "animal_id"])
        .agg(pl.mean("NumTrials"))
        )

trial_df = trial_df.with_columns(
    pl.col("treatment").cast(pl.Utf8)
)

trial_df = trial_df.to_pandas()

sns.barplot(
    data=trial_df,
    x="treatment",
    y="NumTrials"
)
# Number of wins 

# Mean reaction time 

# Mean response entropy 
