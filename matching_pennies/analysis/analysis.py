# To run: `uv run -m matching_pennies.analysis.analysis`

import numpy as np
import matplotlib.pyplot as plt 
import polars as pl 
import bambi as bmb
import arviz as az 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matching_pennies.io.csv_parser as csv_parser
import matching_pennies.compute.compute_performance_metrics as cpm 

"""
TODO:
- Check `TimeInWell` to make sure it matches matlab 
    - There are negative and null values -> double check 
- How to speed up sampling with GPU? Numba? JAX? 
"""

# Load data and compute metrics 
# Mac paths 
# directory = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/normal_mp"
# tmap = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/lesion_study_ingestion_map.csv"

# Windows paths: 
directory = "C:/Users/benli/Documents/projects/lesion_study/post_sx_testing/reorganized/normal_mp"
tmap = "C:/Users/benli/Documents/projects/lesion_study/post_sx_testing/reorganized/lesion_study_ingestion_map.csv"

good_lesions = ['P4667', 'P4668', 'P4676', 'P4677', 'P4638', 'P4662', 'P4661', 'P4637', 
                'P4643', 'P4663', 'P4665', 'P4679', 'P4680', 'P4647', 'P4648', 'P4651',
                'P4666', 'P4669', 'P4671', 'P4672', 'P4650', 'P4640', 'P4657', 'P4659']

csvs = csv_parser.get_csv_files(directory, recursive=True)
df = csv_parser.build_trials(csvs, tmap, paradigm="normal")   

tdf, sdf = cpm.compute_metrics(df, keys=["animal_id", "session_idx"])

tdf.filter(
    pl.col("session_idx") == 1
)

def efs_model(tdf):
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

    # Remove Poor lesions
    m = m.filter(pl.col("animal_id").is_in(good_lesions))

    # Scale both columns in place
    m = m.with_columns([
        ((pl.col("trial_idx") - pl.col("trial_idx").min()) / 
        (pl.col("trial_idx").max() - pl.col("trial_idx").min())).alias("trial_idx_scaled"),

        ((pl.col("session_idx") - pl.col("session_idx").min()) / 
        (pl.col("session_idx").max() - pl.col("session_idx").min())).alias("session_idx_scaled")
    ])

    # Suppose df is a pandas DataFrame with columns:
    df = m.to_pandas()

    # Make sure treatment is categorical with 'sham' as the reference
    df["treatment"] = pd.Categorical(df["treatment"],
                                    categories=["sham", "ofc", "mpfc"],
                                    ordered=True)

    # make sure df is a pandas DataFrame
    # categorical vars can stay as string/object; bambi will encode them

    model = bmb.Model(
        "EFS_before_flg ~ treatment*trial_idx_scaled + treatment*session_idx_scaled + (1|animal_id)",
        df,
        family="bernoulli",  # logistic link by default
    )

    idata = model.fit()  # runs MCMC (NUTS)

    # Visualization: 
    var_names = ["treatment", "trial_idx_scaled", "session_idx_scaled", "treatment:trial_idx_scaled", "treatment:session_idx_scaled"]
    # var_names = ["treatment:trial_idx_scaled", "treatment:session_idx_scaled"]
    print(az.summary(idata))
    trace_out = az.plot_trace(idata, var_names=var_names, combined=True, compact=False, legend=True)    # Needs legend. Super messy 
 
    # --- Normalize to an array of axes ---
    # In some versions trace_out is a Figure, in others it's already an array of Axes.
    if hasattr(trace_out, "axes"):
        # trace_out is a Figure
        axes_arr = np.array(trace_out.axes)
    else:
        # trace_out is already an array-like of Axes
        axes_arr = np.array(trace_out).ravel()

    # plot_trace arranges axes in pairs: [posterior, trace, posterior, trace, ...]
    # So we grab every 2nd axis starting at 0 (the posterior/kde panels)
    post_axes = axes_arr[0::2]

    # --- Find global xlim across all posterior panels ---
    xmin_list = []
    xmax_list = []
    for ax in post_axes:
        xmin, xmax = ax.get_xlim()
        max_abs = max(abs(xmin), abs(xmax))
        xmin_list.append(-max_abs)

        xmax_list.append(max_abs)

    global_xlim = (min(xmin_list), max(xmax_list))

    # --- Apply the shared xlim to all posterior panels ---
    for ax in post_axes:
        ax.set_xlim(global_xlim)
        # Optional: add vertical line at 0 for visual reference
        ax.axvline(0, linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.show()
    # model.plot('treatment', idata=idata)
    # model.plot('trial_idx_scaled', idata=idata)

    # Posterior predictive check
    ppc_idata = model.predict(
        idata=idata,
        kind="pps",        # posterior predictive samples
        inplace=False
    )
    az.plot_ppc(ppc_idata, show=True)


    az.plot_forest(
        idata,
        var_names=var_names,
        combined=True,  # combines chains
        hdi_prob=0.94,
    )
    plt.title("Posterior means and 94% HDIs for fixed effects (log-odds scale)")
    plt.show()

    # ppc = model.predict(idata)
    # az.plot_ppc(ppc)
    return idata

# Try PCA on session data: 
def model_based_pca(idata, tdf):
    posterior = idata.posterior

    # 1. pull the random intercepts per animal from the posterior
    #    This should be shape (chain, draw, animal_id__factor_dim)
    #    and have the var name "1|animal_id"
    re_da = posterior["1|animal_id"].mean(dim=("chain", "draw"))

    # 2. make a tidy dataframe
    #    columns will be ['animal_id__factor_dim', 'rand_intercept']
    re_df = (
        re_da
        .to_dataframe(name="rand_intercept")
        .reset_index()
    )

    # 3. normalize column names:
    # if Bambi used "animal_id__factor_dim" as the coord dim, rename it to "animal_id"
    if "animal_id__factor_dim" in re_df.columns:
        re_df = re_df.rename(columns={"animal_id__factor_dim": "animal_id"})
    elif "animal_id" in re_df.columns:
        # already good
        pass
    else:
        # fallback: try to infer the coord name programmatically
        # grab any column that ends with "__factor_dim"
        factor_cols = [c for c in re_df.columns if c.endswith("__factor_dim")]
        if len(factor_cols) == 1:
            re_df = re_df.rename(columns={factor_cols[0]: "animal_id"})
        else:
            raise RuntimeError(
                f"Couldn't find the animal id factor column. Got columns: {re_df.columns}"
            )

    # At this point re_df should look like:
    # animal_id | rand_intercept

    # 4. get one row per animal with treatment labels from your trial dataframe
    meta = (
        tdf[:, ["animal_id", "treatment"]]
        .unique()          # Polars version of drop_duplicates
        .to_pandas()       # convert to pandas
    )

    animal_effects = (
        meta.merge(re_df, on="animal_id", how="inner")
        .reset_index(drop=True)
    )

    # 5. Build feature matrix (right now it's just the random intercept)
    X = animal_effects[["rand_intercept"]].to_numpy()

    # 6. PCA on 1D -> gives PC1
    pca = PCA(n_components=1)
    animal_effects["PC1"] = pca.fit_transform(X)

    # 7. LDA on 1D -> gives LDA1 (can still work with 1 feature)
    lda = LDA(n_components=1)
    animal_effects["LDA1"] = lda.fit_transform(X, animal_effects["treatment"])

    # 8. simple visualization: PC1 vs 0, colored by treatment
    fig, ax = plt.subplots()
    for tr in animal_effects["treatment"].unique():
        sub = animal_effects[animal_effects["treatment"] == tr]
        ax.scatter(
            sub["PC1"],
            np.zeros_like(sub["PC1"]),
            s=80,
            alpha=0.8,
            edgecolor="k",
            label=tr,
        )
        # optional: annotate animals so you know who's who
        for _, row in sub.iterrows():
            ax.text(
                row["PC1"],
                0.02,
                row["animal_id"],
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_xlabel("PC1 (random intercept from hierarchical model)")
    ax.set_yticks([])
    ax.set_title("Between-animal behavioural bias after removing within-session trend")
    ax.legend(title="treatment")
    plt.tight_layout()
    plt.show()

    return animal_effects



if __name__ == "__main__":
    idata = efs_model(tdf)
    animal_effects = model_based_pca(idata, tdf)