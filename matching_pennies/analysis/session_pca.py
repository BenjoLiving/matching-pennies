import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matching_pennies.io.csv_parser as csv_parser
import matching_pennies.compute.compute_performance_metrics as cpm 

# Load data and compute metrics 
# Mac paths 
directory = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/normal_mp"
tmap = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/lesion_study_ingestion_map.csv"

csvs = csv_parser.get_csv_files(directory, recursive=False)
df = csv_parser.build_trials(csvs, tmap, paradigm="normal")   

tdf, sdf = cpm.compute_metrics(df, keys=["animal_id", "session_idx"])


# 1. Columns we will *not* include in PCA
exclude_cols = ["animal_id", "session_idx", "treatment"]

# 2. Feature matrix for PCA (Polars -> NumPy)
feature_cols = [c for c in sdf.columns if c not in exclude_cols]

X = sdf.select(feature_cols).to_numpy()  # shape: (n_sessions, n_features)

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Run PCA
pca = PCA(n_components=2)  # or more if you want full spectrum; 2 is nice for plotting
X_pca = pca.fit_transform(X_scaled)  # shape: (n_sessions, 2)

# 5. Build a results dataframe so you can analyze / plot
pca_df = pl.DataFrame({
    "animal_id": sdf["animal_id"],
    "session_idx": sdf["session_idx"],
    "treatment": sdf["treatment"],
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
})

# 6. Optionally look at explained variance
explained = pca.explained_variance_ratio_
print("Explained var PC1, PC2:", explained)

import matplotlib.pyplot as plt

# convert to pandas just for plotting convenience
pdf = pca_df.to_pandas()

fig, ax = plt.subplots()

for trt, group in pdf.groupby("treatment"):
    ax.scatter(group["PC1"], group["PC2"], label=trt, alpha=0.7)

ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
# ax.legend(title="treatment")
ax.set_title("Session-level PCA")

plt.show()

loadings = pl.DataFrame({
    "feature": feature_cols,
    "PC1_loading": pca.components_[0, :],
    "PC2_loading": pca.components_[1, :],
}).sort("PC1_loading", descending=True)

print(loadings)


# LDA 
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. Get treatment labels from sdf as a NumPy array
treatment = sdf["treatment"].to_numpy()

# 2. Build a boolean mask for rows that have a non-missing treatment
valid_mask = np.array([t is not None for t in treatment])

# 3. Filter X_scaled and treatment using that mask
X_valid = X_scaled[valid_mask]
y_valid = treatment[valid_mask]

# 4. Fit LDA on just the valid rows
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_valid, y_valid)

print("X_lda shape:", X_lda.shape)
print("classes seen:", lda.classes_)

import matplotlib.pyplot as plt
import pandas as pd

lda_df = pd.DataFrame({
    "LD1": X_lda[:, 0],
    "LD2": X_lda[:, 1],
    "treatment": y_valid,
})

fig, ax = plt.subplots()

for trt, grp in lda_df.groupby("treatment"):
    ax.scatter(grp["LD1"], grp["LD2"], label=trt, alpha=0.7)

ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.set_title("LDA of session metrics by treatment")
ax.legend(title="treatment")
plt.show()

feature_cols = [c for c in sdf.columns if c not in ["animal_id", "session_idx", "treatment"]]
coefs = lda.coef_  # shape: (n_classes-1, n_features)
loadings = pd.DataFrame(coefs, columns=feature_cols, index=lda.classes_[1:])
print(loadings.sort(by=loadings.columns[0]).head(10))
