# To run: `uv run -m matching_pennies.analysis.analysis`

import numpy as np
import matplotlib.pyplot as plt 
import polars as pl 
import bambi as bmb
import arviz as az 
import pandas as pd
from sklearn.decomposition import PCA 
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
directory = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/normal_mp"
tmap = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/lesion_study_ingestion_map.csv"

# Windows paths: 
# directory = "C:/Users/benli/Documents/projects/lesion_study/post_sx_testing/reorganized/normal_mp"
# tmap = "C:/Users/benli/Documents/projects/lesion_study/post_sx_testing/reorganized/lesion_study_ingestion_map.csv"

good_lesions = ['P4667', 'P4668', 'P4676', 'P4677', 'P4638', 'P4662', 'P4661', 'P4637', 
                'P4643', 'P4663', 'P4665', 'P4679', 'P4680', 'P4647', 'P4648', 'P4651',
                'P4666', 'P4669', 'P4671', 'P4672', 'P4650', 'P4640', 'P4657', 'P4659']

csvs = csv_parser.get_csv_files(directory, recursive=True)
df = csv_parser.build_trials(csvs, tmap, paradigm="normal")   

tdf, sdf = cpm.compute_metrics(df, keys=["animal_id", "session_idx"])

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
    var_names = ["treatment:trial_idx_scaled", "treatment:session_idx_scaled"]
    print(az.summary(idata))
    az.plot_trace(idata, var_names=var_names, combined=True, compact=False, legend=True)    # Needs legend. Super messy 
    # plt.legend(bbox_to_anchor=(0.05, 1), loc="upper left")  # push legend outside right edge
    # plt.tight_layout()
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
        var_names=[
            "Intercept",
            "treatment",
            "trial_idx_scaled",
            "treatment:trial_idx_scaled",
            "session_idx_scaled",
            "treatment:session_idx_scaled",
        ],
        combined=True,  # combines chains
        hdi_prob=0.94,
    )
    plt.title("Posterior means and 94% HDIs for fixed effects (log-odds scale)")
    plt.show()

    # ppc = model.predict(idata)
    # az.plot_ppc(ppc)

# Try PCA on session data: 

