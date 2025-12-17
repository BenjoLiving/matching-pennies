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

################################
# GPT'd line plot
################################

# --- settings ---
treatments = ["sham", "mpfc", "ofc"]
x_order = [0, 1]
x_labels = {0: "Before swap", 1: "After swap"}

# sessions present in the data (sorted)
sessions = (
    efs_df
    .select(pl.col("session_idx"))
    .unique()
    .sort("session_idx")
    .get_column("session_idx")
    .to_list()
)
n_sessions = len(sessions)

# --- aggregate within animal/session/swap so each animal has 1 point per swap per session ---
animal_df = (
    efs_df
    .group_by(["treatment", "animal_id", "session_idx", "swap"])
    .agg(pl.col("mean_efs").mean().alias("mean_efs"))
)

# --- per-animal slope (after - before) used for ordering + optional counts ---
animal_slope_df = (
    animal_df
    .pivot(
        values="mean_efs",
        index=["treatment", "animal_id", "session_idx"],
        columns="swap",
        aggregate_function="first",
    )
    # expects columns "0" and "1" after pivot; skip animals missing either
    .drop_nulls(["0", "1"])
    .with_columns((pl.col("1") - pl.col("0")).alias("delta"))
)

# --- group mean & SEM across animals for the thick line ---
group_df = (
    animal_df
    .group_by(["treatment", "session_idx", "swap"])
    .agg([
        pl.col("mean_efs").mean().alias("mean"),
        pl.col("mean_efs").std().alias("sd"),
        pl.len().alias("n"),
    ])
    .with_columns((pl.col("sd") / pl.col("n").sqrt()).alias("sem"))
)

# --- slope of the *mean line* per treatment/session: mean(after) - mean(before) ---
group_slope_df = (
    group_df
    .pivot(
        values="mean",
        index=["treatment", "session_idx"],
        columns="swap",
        aggregate_function="first",
    )
    .drop_nulls(["0", "1"])
    .with_columns((pl.col("1") - pl.col("0")).alias("mean_delta"))
    .select(["treatment", "session_idx", "mean_delta"])
)

# --- plotting: rows=treatments, cols=sessions ---
fig, axes = plt.subplots(
    nrows=len(treatments),
    ncols=n_sessions,
    sharex=True,
    sharey=True,
    figsize=(3.8 * n_sessions, 8.5),
    constrained_layout=True
)

# If n_sessions==1, axes comes back 1D; force 2D for consistent indexing
if n_sessions == 1:
    axes = axes.reshape(len(treatments), 1)

for i, trt in enumerate(treatments):
    for j, sess in enumerate(sessions):
        ax = axes[i, j]

        # --- individual animals (ordered by slope: lowest to highest) ---
        # get animal ordering for this panel
        order = (
            animal_slope_df
            .filter((pl.col("treatment") == trt) & (pl.col("session_idx") == sess))
            .sort("delta")  # lowest (most negative) to highest
            .select(["animal_id", "delta"])
        )

        # build a dict of animal_id -> rank for plotting order / zorder
        ordered_ids = order.get_column("animal_id").to_list()

        # plot each animal in that order (plot order matters when lines overlap)
        sub = animal_df.filter((pl.col("treatment") == trt) & (pl.col("session_idx") == sess))
        for rank, aid in enumerate(ordered_ids):
            g = sub.filter(pl.col("animal_id") == aid).sort("swap")
            swaps = g["swap"].to_list()
            if swaps != [0, 1]:
                continue
            ys = g["mean_efs"].to_list()

            ax.plot(
                x_order, ys,
                color="black",
                marker="o", markersize=4,
                linewidth=1.2, alpha=0.25,
                zorder=rank  # ensures low->high is literally drawn bottom->top
            )

        # --- mean line (thick) + SEM ---
        m = (
            group_df
            .filter((pl.col("treatment") == trt) & (pl.col("session_idx") == sess))
            .sort("swap")
        )
        mx = m["swap"].to_list()
        my = m["mean"].to_list()
        mse = m["sem"].to_list()

        ax.errorbar(
            mx, my, yerr=mse,
            color="black",
            marker="o", markersize=8,
            linewidth=4.0, capsize=5,
            alpha=1.0,
            zorder=10_000
        )

        # --- annotate mean-line slope ---
        mean_delta_row = (
            group_slope_df
            .filter((pl.col("treatment") == trt) & (pl.col("session_idx") == sess))
        )
        if mean_delta_row.height == 1:
            mean_delta = float(mean_delta_row["mean_delta"][0])
            ax.text(
                0.02, 0.98,
                f"Δ(mean) = {mean_delta:+.3f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=10
            )

        # cosmetics
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # column titles (session) on top row only
        if i == 0:
            ax.set_title(f"Session {sess}", fontsize=13, pad=8)

        # row labels (treatment) on leftmost column only
        if j == 0:
            ax.set_ylabel(trt.upper(), fontsize=12)

# x axis ticks/labels only on bottom row
for ax in axes[:-1, :].ravel():
    ax.tick_params(labelbottom=False)

for ax in axes[-1, :]:
    ax.set_xticks(x_order)
    ax.set_xticklabels([x_labels[x] for x in x_order], fontsize=11)
    ax.set_xlabel("Hallway swap", fontsize=12)

# optional: single shared y label
fig.supylabel("Mean EFS", fontsize=12)

plt.show()


############################################
# BAR PLOT WITH CONTROL, SHORT, AND LONG
############################################

# Data 
efs_df = (
    tdf 
    .with_columns(
        swap = pl.when(pl.col("trial_idx") < 60)
            .then(0)
            .otherwise(1)
    )
    .filter(pl.col("trial_idx") > 30) 
    .filter(pl.col("trial_idx") <= 90)
    .group_by(["treatment", "animal_id", "session_idx", "hall", "swap"])
    .agg(pl.mean("EFS_before_flg").alias("mean_efs"))
).sort(["treatment", "animal_id", "session_idx", "swap", "hall"])


# Splitting 'C', 'S' and 'L' 
hallways = ['C', 'S', 'L'] 
fig, axes = plt.subplots(
    nrows=1, ncols=3, 
    figsize=(6, 4),
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
        ax=ax,
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




(
    tdf
    .filter(pl.col("session_idx") ==1)
    .group_by(["treatment", "animal_id"])
    .agg(
        number_halls = pl.count("")
    )
)