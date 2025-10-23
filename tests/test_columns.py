import polars as pl
import matching_pennies.io.csv_parser as csv_parser
import matching_pennies.compute.compute_performance_metrics as cpm

# -----------------------------
# CONFIG: point to your inputs
# -----------------------------
directory = "C:/Users/benli/Documents/PaperManuscripts_andProjects/_2018_ACC_hallway/Matching Pennies/Testing/2018_01_24"
tmap = "C:/Users/benli/Documents/PaperManuscripts_andProjects/_2018_ACC_hallway/Matching Pennies/2018_acc_treatment_map.csv"
old_csv = "C:/Users/benli/Documents/code/matching-pennies/test_data.csv"  # MATLAB-exported CSV

# Columns to compare. Add/remove as you like.
CHECK_FLAGS  = ["LoseSwitch_flg", "LoseSwitch_NoMultWellResp_flg",
                "WinStay_flg", "WinStay_NoMultWellResp_flg", "EFS_before_flg"]
CHECK_FLOATS = ["InterTrialInterval", "ResponseTime"]
CHECK_OTHER  = []  # e.g. ["TimeInWell"] (we’ll auto-treat numeric vs string)

TOL_DEFAULT = 1e-6
# Optional per-column custom tolerances (overrides TOL_DEFAULT when provided)
TOL_MAP = {
    # "TimeInWell": 1e-6,
}

# -----------------------------
# 1) Build "new" dataframe via Python pipeline
# -----------------------------
csvs = csv_parser.get_csv_files(directory, recursive=False)
raw_df = csv_parser.build_trials(csvs, tmap, paradigm="normal")
tdf, sdf = cpm.compute_metrics(raw_df, keys=["animal_id", "session_idx"])

# -----------------------------
# 2) Read "old" MATLAB dataframe and normalize naming
# -----------------------------
old = pl.read_csv(old_csv)

# Keep your rename to align schemas
old = old.rename({
    "Rat_id": "animal_id",
    "Trial_indx": "trial_idx",
    "SampledBothWellsBeforeTrial_flg": "EFS_before_flg",
    "LoseSwitch_all_flg": "LoseSwitch_flg",
    "WinStay_all_flg": "WinStay_flg",
})

# If MATLAB export has Ses_id like "P4020:2018-01-24:12:04:34", extract date for session ordering
if "Ses_id" in old.columns and "session_date" not in old.columns:
    old = old.with_columns(
        pl.col("Ses_id").str.split(":").list.get(1).str.to_date("%Y-%m-%d").alias("session_date")
    ).drop("Ses_id")

# If session_idx is missing on the old CSV, derive a stable dense rank per animal by session_date
if "session_idx" not in old.columns:
    if "session_date" not in old.columns:
        raise ValueError("Old CSV is missing session_idx and session_date; "
                         "cannot derive session_idx for alignment.")
    old = (
        old
        .with_columns(
            pl.categorical(pl.col("animal_id")).alias("__aid"),
            pl.col("session_date").alias("__sdate"),
        )
        .with_columns(
            # dense rank 1..N per animal ordered by session_date
            pl.col("__sdate").rank("dense").over("__aid").cast(pl.Int32).alias("session_idx")
        )
        .drop(["__aid", "__sdate"])
    )

# -----------------------------
# 3) Normalize types for alignment & comparison
# -----------------------------
KEYS = ["animal_id", "session_idx", "trial_idx"]
ALL_CHECKS = CHECK_FLAGS + CHECK_FLOATS + CHECK_OTHER

def _normalize_for_compare(df: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        pl.col("animal_id").cast(pl.Utf8, strict=False),
        pl.col("session_idx").cast(pl.Int64, strict=False),
        pl.col("trial_idx").cast(pl.Int64, strict=False),
    ]
    # Cast flags as Int8 when present
    for c in CHECK_FLAGS:
        if c in df.columns:
            exprs.append(pl.col(c).cast(pl.Int8, strict=False))
    # Cast floats as Float64
    for c in CHECK_FLOATS:
        if c in df.columns:
            exprs.append(pl.col(c).cast(pl.Float64, strict=False))
    # For CHECK_OTHER: keep numeric as Float64/Int64, strings as Utf8
    for c in CHECK_OTHER:
        if c in df.columns:
            dt = df.schema.get(c)
            if dt is None:
                continue
            if pl.datatypes.is_numeric(dt):
                exprs.append(pl.col(c).cast(pl.Float64, strict=False))  # numeric compare w/ tol
            else:
                exprs.append(pl.col(c).cast(pl.Utf8, strict=False))     # exact match
    return df.with_columns(exprs)

new_n = _normalize_for_compare(tdf)
old_n = _normalize_for_compare(old)

# -----------------------------
# 4) Sanity check: unique keys
# -----------------------------
for name, df in [("new", new_n), ("old", old_n)]:
    dups = df.group_by(KEYS).len().filter(pl.col("len") > 1)
    if dups.height:
        print(f"WARNING: duplicate rows for KEYS in {name}:")
        print(dups.head(10))

# -----------------------------
# 5) Align and compute differences
# -----------------------------
left  = new_n.select(KEYS + [c for c in ALL_CHECKS if c in new_n.columns]).with_columns(pl.lit(True).alias("_in_new"))
right = old_n.select(KEYS + [c for c in ALL_CHECKS if c in old_n.columns]).with_columns(pl.lit(True).alias("_in_old"))

joined = left.join(right, on=KEYS, how="full", suffix="_old")

only_in_old = joined.filter(pl.col("_in_new").is_null()).select(KEYS)  # exists only in MATLAB
only_in_new = joined.filter(pl.col("_in_old").is_null()).select(KEYS)  # exists only in Python

# Create per-column diff masks with type-aware logic
def _float_diff_expr(col: str) -> pl.Expr:
    tol = TOL_MAP.get(col, TOL_DEFAULT)
    a = pl.col(col)
    b = pl.col(f"{col}_old")
    return (
        ((a - b).abs() > tol) |
        (a.is_nan() ^ b.is_nan())
    ).alias(f"{col}_diff")

def _flag_or_int_diff_expr(col: str) -> pl.Expr:
    a = pl.col(col).cast(pl.Int64)
    b = pl.col(f"{col}_old").cast(pl.Int64)
    return a.eq_missing(b).not_().alias(f"{col}_diff")

def _string_diff_expr(col: str) -> pl.Expr:
    a = pl.col(col).cast(pl.Utf8)
    b = pl.col(f"{col}_old").cast(pl.Utf8)
    return a.eq_missing(b).not_().alias(f"{col}_diff")

diff_exprs = []
for c in ALL_CHECKS:
    if c not in joined.columns or f"{c}_old" not in joined.columns:
        continue
    # Decide comparator
    if c in CHECK_FLOATS:
        diff_exprs.append(_float_diff_expr(c))
    elif c in CHECK_FLAGS:
        diff_exprs.append(_flag_or_int_diff_expr(c))
    else:
        # CHECK_OTHER: pick comparator by dtype if present, fallback to string
        dt = new_n.schema.get(c) or old_n.schema.get(c)
        if dt and pl.datatypes.is_numeric(dt):
            diff_exprs.append(_float_diff_expr(c))
        else:
            diff_exprs.append(_string_diff_expr(c))

common = joined.filter(pl.all_horizontal([pl.col("_in_new").is_not_null(), pl.col("_in_old").is_not_null()]))
common = common.with_columns(diff_exprs)

# -----------------------------
# 6) Summary + where-are-the-diffs tables
# -----------------------------
summary = common.select([pl.col(c).sum().alias(c) for c in common.columns if c.endswith("_diff")])
print("\n=== Diff summary (count of mismatched rows per column) ===")
print(summary)

if only_in_old.height:
    print("\nRows only in OLD (MATLAB) – not present in NEW:\n", only_in_old)
if only_in_new.height:
    print("\nRows only in NEW (Python) – not present in OLD:\n", only_in_new)

# For each checked column, print the rows that differ with side-by-side values
for c in ALL_CHECKS:
    diff_col = f"{c}_diff"
    if diff_col not in common.columns:
        continue
    mism = (
        common
        .filter(pl.col(diff_col) == True)
        .select(KEYS + [c, f"{c}_old"])
        .sort(KEYS)
    )
    if mism.height:
        print(f"\n--- Differences for {c} ---")
        print(mism.head(30))  # show first N; bump if you want

# Optional: produce a long-format table of all diffs (handy for saving)
long_diffs = []
for c in ALL_CHECKS:
    diff_col = f"{c}_diff"
    if diff_col in common.columns:
        df_c = (
            common
            .filter(pl.col(diff_col) == True)
            .select(KEYS + [c, f"{c}_old"])
            .with_columns([
                pl.lit(c).alias("__column"),
            ])
        )
        long_diffs.append(df_c)

if long_diffs:
    all_diffs = pl.concat(long_diffs)
    print("\n=== Long diff table (first 50 rows) ===")
    print(all_diffs.head(50))
    # Example: save
    # all_diffs.write_csv("column_diffs.csv")
