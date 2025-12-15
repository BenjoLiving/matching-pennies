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


EXPERIMENT = "lesion_study"
PARADIGM = "swapping_hallways" 

system_os = platform.system().lower()

if system_os == "windows":
    hallway_legend_path = "C:/Users/benli/Documents/projects/lesion_study/post_sx_testing/swapping_halls_legend.csv"
    data_path = "C:/Users/benli/Documents/code/matching-pennies/data"
    cores = 4  
    export_path = "C:/Users/benli/Documents/projects/lesion_study/results/paper_analysis/notebook_figures/switching_halls"

elif system_os == "darwin": 
    hallway_legend_path = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/swapping_halls_legend.csv"
    data_path = "/Users/ben/Documents/source/matching-pennies/data"
    cores = 1  # MacOS poops its pants when asked to multithread 

tdf, sdf, manifest = load_metrics(EXPERIMENT, PARADIGM, root=data_path)

halls = pl.read_csv(hallway_legend_path)

# Clean up data from csv
halls = halls.with_columns(
    # Convert integer Animal -> string animal_id "P####" 
    pl.format("P{}", pl.col("Animal").cast(pl.Utf8)).cast(pl.Categorical).alias("animal_id"),

    # Convert "Date" string to Date object 
    pl.col("Date").alias("session_date").str.strip_chars('"').str.to_date(format="%Y-%m-%d"),

    # Rename Hall to hall while we're here 
    pl.col("Hall").alias("hall")
).drop(["Animal", "Date", "Hall"])

tdf = tdf.join(halls, on=["animal_id", "session_date"], how="left")

# Data 
efs_df = (
    tdf 
    .with_columns(
        swap = pl.when(pl.col("trial_idx") < 60)
            .then(0)
            .otherwise(1)
    )
    .filter(pl.col("trial_idx") <= 120)
    .filter(pl.col("hall") != "C") 
    .group_by(["treatment", "animal_id", "session_idx", "hall", "swap"])
    .agg(pl.mean("EFS_before_flg").alias("mean_efs"))
).sort(["treatment", "animal_id", "session_idx", "swap", "hall"])

# Plot before/after swap for the full duration 60 before - 60 after 
# What does the plot look like with S and L together? Separate? 

df = efs_df.to_pandas()
df["treatment"] = df["treatment"].astype(str) 
palette = ["#FFFFFF", "#000000"]
grey = ["#000000", "#656565"]

fig, ax = plt.subplots(figsize=(2, 4))

sns.barplot(
    data=df,
    x="treatment",
    y="mean_efs",
    hue="swap", 
    order=["sham", "mpfc", "ofc"],
    ax=ax, 
    palette=sns.color_palette(palette),
    errorbar="sd"
)

# Add in black borders for bars 
for patch in ax.patches:
    patch.set_edgecolor("black") 
    patch.set_linewidth(1.5) 

# Stripplot overlay 
sns.stripplot(
    data=df, 
    x="treatment", 
    y="mean_efs", 
    hue="swap", 
    order=["sham", "mpfc", "ofc"], 
    ax=ax,
    palette=sns.color_palette(grey), 
    jitter=0.15, 
    alpha=0.7, 
    dodge=True
)

# Labels and Formatting 

handles, labels = ax.get_legend_handles_labels()
# keep only first two (swap 0/1)
ax.legend(handles[:2], labels[:2], title="swap")

ax.set_title("60 pre - 60 post: including S and L", fontsize=20)
ax.set_ylabel("Number of EFS trials", fontsize=16)
ax.set_xlabel(""), 
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16)

sns.despine()
plt.tight_layout()

plt.show()


# Make similar plots but: 
#   - Split up S and L 
#   - Try doing 30 trials on either side of the swap 

# Splitting 'S' and 'L' 
hallways = ['S', 'L'] 
fig, axes = plt.subplots(
    nrows=1, ncols=2, 
    figsize=(4, 4),
    sharey=True
)

for ax, hall in zip(axes, hallways): 
    df = efs_df.filter(pl.col("hall") == hall).to_pandas() 
    df["treatment"] = df["treatment"].astype(str) 

    sns.barplot(
        data=df,
        x="treatment", 
        y="mean_efs",
        hue="swap", 
        order=["sham", "mpfc", "ofc"],
        ax=ax,
        palette=sns.color_palette(palette),
        errorbar="sd"
    )

    # Add bar borders 
    for patch in ax.patches:
        patch.set_edgecolor("black") 
        patch.set_linewidth(1.5) 

    sns.stripplot(
        data=df, 
        x="treatment",
        y="mean_efs", 
        hue="swap",
        order=["sham", "mpfc", "ofc"], 
        palette=sns.color_palette(grey), 
        jitter=0.15,
        alpha=0.7, 
        dodge=True
    )

    # Remove duplicate legends in each subplot 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([], [], frameon=False) 

    # Title and formatting 
    ax.set_title(f"{hall} - Hall", fontsize=16) 
    ax.set_xlabel("") 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12) 

# One shared legend for the figure 
fig.legend(
    handles[:2], 
    labels[:2], 
    title="swap", 
    loc="upper right", 
    bbox_to_anchor=[1.08, 0.95]
)

fig.supylabel("Mean EFS trials", fontsize=16)

sns.despine()
plt.tight_layout() 
plt.show() 

"""
TODO
- Above plot is missing overlayed scatter points? 
"""


###################################
# 30 Trial on either side - 30-90
###################################

# Data 
efs_df = (
    tdf 
    .with_columns(
        swap = pl.when(pl.col("trial_idx") < 60)
            .then(0)
            .otherwise(1)
    )
    .filter(pl.col("trial_idx") >= 30)
    .filter(pl.col("trial_idx") <= 90)
    .filter(pl.col("hall") != "C") 
    .group_by(["treatment", "animal_id", "session_idx", "hall", "swap"])
    .agg(pl.mean("EFS_before_flg").alias("mean_efs"))
).sort(["treatment", "animal_id", "session_idx", "swap", "hall"])

# Splitting 'S' and 'L' 
hallways = ['S', 'L'] 
fig, axes = plt.subplots(
    nrows=1, ncols=2, 
    figsize=(4, 4),
    sharey=True
)

for ax, hall in zip(axes, hallways): 
    df = efs_df.filter(pl.col("hall") == hall).to_pandas() 
    df["treatment"] = df["treatment"].astype(str) 

    sns.barplot(
        data=df,
        x="treatment", 
        y="mean_efs",
        hue="swap", 
        order=["sham", "mpfc", "ofc"],
        ax=ax,
        palette=sns.color_palette(palette),
        errorbar="sd"
    )

    # Add bar borders 
    for patch in ax.patches:
        patch.set_edgecolor("black") 
        patch.set_linewidth(1.5) 

    sns.stripplot(
        data=df, 
        x="treatment",
        y="mean_efs", 
        hue="swap",
        order=["sham", "mpfc", "ofc"], 
        palette=sns.color_palette(grey), 
        jitter=0.15,
        alpha=0.7, 
        dodge=True
    )

    # Remove duplicate legends in each subplot 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([], [], frameon=False) 

    # Title and formatting 
    ax.set_title(f"{hall} - Hall", fontsize=16) 
    ax.set_xlabel("") 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12) 

# One shared legend for the figure 
fig.legend(
    handles[:2], 
    labels[:2], 
    title="swap", 
    loc="upper right", 
    bbox_to_anchor=[1.08, 0.95]
)

fig.supylabel("Mean EFS trials", fontsize=16)

sns.despine()
plt.tight_layout() 
plt.show() 


"""
TODO 
- Make a line plot of before and after swap for first 3 sessions, 
  averaged over each treatment group? Or with average more prominent
  and every animal overlayed 
"""

import polars as pl
import matplotlib.pyplot as plt

# --- settings ---
treatments = ["sham", "mpfc", "ofc"]
x_order = [0, 1]
x_labels = {0: "Before swap", 1: "After swap"}

# If efs_df is already polars, keep it; otherwise convert:
# efs_df = pl.from_pandas(efs_df)

# --- aggregate within animal/session/hall so each animal has 1 point per swap ---
# (This prevents multiple sessions/halls from creating >2 points per animal.)
animal_df = (
    efs_df
    .group_by(["treatment", "animal_id", "swap"])
    .agg(pl.col("mean_efs").mean().alias("mean_efs"))
)

# --- group mean & SEM across animals for the thick line ---
group_df = (
    animal_df
    .group_by(["treatment", "swap"])
    .agg([
        pl.col("mean_efs").mean().alias("mean"),
        pl.col("mean_efs").std().alias("sd"),
        pl.len().alias("n"),
    ])
    .with_columns(
        (pl.col("sd") / pl.col("n").sqrt()).alias("sem")
    )
)

# --- plotting ---
fig, axes = plt.subplots(
    nrows=3, ncols=1, sharex=True, sharey=True,
    figsize=(6.5, 9.0), constrained_layout=True
)

for i, trt in enumerate(treatments):
    ax = axes[i]

    # --- individual animals (faint paired lines + small markers) ---
    sub = animal_df.filter(pl.col("treatment") == trt)
    for aid, g in sub.group_by("animal_id"):
        # ensure swap 0 then 1 order (and skip if missing either)
        g2 = g.sort("swap")
        swaps = g2["swap"].to_list()
        if swaps != [0, 1]:
            continue
        ys = g2["mean_efs"].to_list()
        ax.plot(
            x_order, ys,
            marker="o", markersize=4,
            linewidth=1.2, alpha=0.25
        )

    # --- mean line (thick + bright) with optional SEM error bars ---
    m = (
        group_df
        .filter(pl.col("treatment") == trt)
        .sort("swap")
    )
    mx = m["swap"].to_list()
    my = m["mean"].to_list()
    mse = m["sem"].to_list()

    ax.errorbar(
        mx, my, yerr=mse,
        marker="o", markersize=8,
        linewidth=4.0, capsize=5,
        alpha=1.0
    )

    ax.set_title(trt.upper(), fontsize=14, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Only one y label (middle subplot), only one x label (bottom subplot)
axes[1].set_ylabel("Mean EFS", fontsize=12)
axes[-1].set_xlabel("Hallway swap", fontsize=12)

# Only bottom x ticks/labels
axes[-1].set_xticks(x_order)
axes[-1].set_xticklabels([x_labels[x] for x in x_order], fontsize=11)

# Hide x tick labels on upper panels explicitly (shared x still draws ticks sometimes)
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)

plt.show()
