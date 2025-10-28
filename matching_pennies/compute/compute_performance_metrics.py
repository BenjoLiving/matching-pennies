"""
Compute trial‑ and session‑level performance metrics for matching pennies sessions.

This module operates on dataframes produced by the CSV parser.  It uses
`polars` expressions throughout for efficiency and to preserve the native
Arrow dtypes (including list columns).  The high‑level API consists of
three functions:

    * `compute_trial_metrics(df, keys=...)` – returns a copy of ``df``
      with per‑trial metrics attached.  These include inter‑trial
      intervals, win/lose stay/switch flags, lick counts and early food
      sampling flags.  The input dataframe must already contain
      appropriately typed columns, including list‑typed lick and well
      off‑time arrays.

    * `compute_session_metrics(df, keys=...)` – aggregates the
      trial‑level metrics down to one row per session.  Metrics include
      the number of trials, percent correct choices, mean rewards per
      trial and win/lose stay/switch probabilities.

    * `compute_metrics(df, keys=..., attach=False)` – convenience
      wrapper that runs both computations.  When ``attach=True`` the
      session metrics are left‑joined back onto the trial dataframe.

The ``keys`` parameter must uniquely identify a session.  In practice
``("animal_id", "session_idx")`` is a robust choice because
``session_idx`` is a dense rank per animal computed at ingest time.
If you choose to use other keys, ensure that no column names in
``keys`` conflict with the internal helper names used herein.
"""

from __future__ import annotations
from typing import Sequence, Iterable
import polars as pl
import numpy as np

__all__ = [
    "compute_trial_metrics",
    "compute_session_metrics",
    "compute_metrics",
]

# -----------------------------------------------------------------------------
# Column name constants.  Adjust these if your schema uses different names.
#
# These names are used as defaults throughout the metrics calculations.  If
# your input dataframe uses alternative names you should either rename your
# columns prior to calling the compute functions or override the constants
# here.
#
OP_ON   = "odor_port_on_ts"
ERR     = "error_flg"        # 0 = correct, 1 = wrong
WELL    = "well_id"          # "L" / "R"
L_REW   = "L_reward_ts"
R_REW   = "R_reward_ts"
HL_ON   = "house_light_on_ts"
TRIAL   = "trial_idx"
WELL_ON = "well_on_ts"
TRIAL_START = "trial_start_ts"
OP_OFF  = "odor_port_off_ts"
TONE_ON = "tone_on_ts"

L_OFFS  = "L_well_all_off_ts"   # List[f64]
R_OFFS  = "R_well_all_off_ts"   # List[f64]
L_LICKS = "L_lick_ts"           # List[f64]
R_LICKS = "R_lick_ts"           # List[f64]

def _ensure_list_f64(df: pl.DataFrame, cols: Iterable[str]) -> pl.DataFrame:
    """
    Ensure each named column is a List[Float64].
    - If it's Utf8 (space-separated numbers), split → cast to floats.
    - If it's already a List[...], cast inner dtype to Float64 (non-strict).
    - If column missing, silently ignore.
    """
    exprs: list[pl.Expr] = []
    for c in cols:
        if c not in df.columns:
            continue

        dt = df.schema.get(c)
        if dt == pl.Utf8:
            exprs.append(
                pl.when(pl.col(c).is_null() | (pl.col(c).str.strip_chars() == ""))
                .then(pl.lit([]).cast(pl.List(pl.Float64)))
                .otherwise(
                    pl.col(c)
                    .str.replace_all(r"\s+", " ")     # collapse runs of whitespace
                    .str.split(" ")                   # -> List[Utf8]
                    .list.eval(pl.element().filter(pl.element() != "").cast(pl.Float64))
                )
                .alias(c)
            )
        else:
            # Already list? Make sure it's List[f64]; if scalar numeric, wrap in 1-length list
            if isinstance(dt, pl.datatypes.List):
                exprs.append(pl.col(c).cast(pl.List(pl.Float64), strict=False).alias(c))
            else:
                # Fallback: try to cast to Utf8 then split (covers weird scalar types)
                exprs.append(
                    pl.when(pl.col(c).is_null())
                    .then(pl.lit([]).cast(pl.List(pl.Float64)))
                    .otherwise(
                        pl.col(c).cast(pl.Utf8, strict=False)
                        .str.replace_all(r"\s+", " ")
                        .str.split(" ")
                        .list.eval(pl.element().filter(pl.element() != "").cast(pl.Float64))
                    ).alias(c)
                )
    return df.with_columns(exprs) if exprs else df


def _miller_madow_entropy_on_bin_seq(x_bits: np.ndarray, word_len: int) -> float:
    """
    Reproduce CompEntropyOnBinSeq(x, len) using the Miller-Madow correction,
    as implemented in the MATLAB you provided.

    x_bits: 1D array-like of binary values (0/1 or 'L'/'R' already mapped to 0/1)
    word_len: int, length of consecutive elements to group
    """

    # 1. ensure column vector of 0/1 ints
    x = np.asarray(x_bits).reshape(-1)
    # if values aren't 0/1 (e.g. "L"/"R"), map to categorical 0/1
    if x.dtype.kind in ("U", "S", "O"):  # strings/objects
        # make categories, like grp2idx()-1 in MATLAB
        _, inv = np.unique(x, return_inverse=True)
        x = inv  # starts at 0
    else:
        # force to ints and reindex to 0,1 if needed
        cats, inv = np.unique(x, return_inverse=True)
        # sanity check: MATLAB errors if >2 levels
        if len(cats) > 2:
            raise ValueError(
                "Input has more than 2 unique levels; expected a binary sequence."
            )
        x = inv  # map to 0..num_levels-1

    n = x.shape[0]

    # 2. truncate so length is divisible by word_len
    n_trunc = n - (n % word_len)
    if n_trunc == 0:
        return np.nan  # not enough data to form even one word

    x_trunc = x[:n_trunc]

    # 3. "deal into m x len vec" (MATLAB loop building xm)
    # effectively: xm[k, :] = x[k : n_trunc : word_len]
    # shape = (word_len, n_trunc/word_len)
    xm = np.vstack([x_trunc[k::word_len] for k in range(word_len)])

    # 4. each column of xm is a binary word; convert to decimal
    # e.g. column [1,0,1] -> "1 0 1" -> bin 101 -> dec 5
    # We'll interpret xm rows as successive time points k=0..len-1,
    # so the first row is the first bit. To match MATLAB num2str+bin2dec,
    # row 0 is most significant bit.
    powers = 2 ** np.arange(word_len - 1, -1, -1)  # e.g. [4,2,1] for len=3
    x_dec = (xm.T * powers).sum(axis=1)

    # 5. empirical (maximum likelihood) entropy of x_dec, in bits
    vals, counts = np.unique(x_dec, return_counts=True)
    p = counts / counts.sum()
    H_ml = -(p * np.log2(p)).sum()

    # 6. Miller-Madow style correction they used:
    # E = E + (k-1)/(1.3863 * length(x))
    # where k = 2^len, and length(x) is ORIGINAL n, not n_trunc.
    k = 2 ** word_len
    E_mm = H_ml + (k - 1) / (1.3863 * n)

    return float(E_mm)


def compute_trial_metrics(df: pl.DataFrame, *, keys: Sequence[str]) -> pl.DataFrame:
    """Compute per-trial metrics, tolerating missing optional columns.

    When a required source column is absent for a metric, that metric is
    created as nulls to keep downstream aggregation predictable.
    """
    # Normalize list-typed columns if present
    df = _ensure_list_f64(df, [L_OFFS, R_OFFS, L_LICKS, R_LICKS])

    keys = list(keys)

    # Presence flags
    has_L_offs = L_OFFS in df.columns
    has_R_offs = R_OFFS in df.columns
    has_L_lick = L_LICKS in df.columns
    has_R_lick = R_LICKS in df.columns
    has_well   = WELL in df.columns
    has_err    = ERR in df.columns
    has_well_on = WELL_ON in df.columns

    # 1) Cast numerics first in a separate projection so later expressions
    #    see numeric dtypes and avoid string<->float comparisons.
    cast_exprs: list[pl.Expr] = []
    for c in [OP_ON, HL_ON, L_REW, R_REW, TRIAL_START, OP_OFF, TONE_ON, WELL_ON]:
        if c in df.columns:
            cast_exprs.append(pl.col(c).cast(pl.Float64, strict=False))
    if has_err:
        cast_exprs.append(pl.col(ERR).cast(pl.Int8, strict=False))
    df_num = df.with_columns(cast_exprs) if cast_exprs else df

    # Do this once, before computing _prev_well and before the WS/LS block
    if WELL in df_num.columns:
        df_num = df_num.with_columns(
            pl.when(pl.col(WELL).cast(pl.Utf8, strict=False).str.strip_chars().eq("") | pl.col(WELL).is_null())
            .then(None)
            .otherwise(pl.col(WELL).cast(pl.Utf8, strict=False).str.strip_chars())
            .alias(WELL)
        )

    exprs: list[pl.Expr] = []

    # Previous helpers
    if all(c in df_num.columns for c in [HL_ON, L_REW, R_REW]):
        exprs.append(
            pl.max_horizontal(pl.col(HL_ON), pl.col(L_REW), pl.col(R_REW))
              .shift(1).over(keys).alias("_prev_end")
        )
    else:
        exprs.append(pl.lit(None).alias("_prev_end"))
    if has_err:
        exprs.append(pl.col(ERR).shift(1).over(keys).alias("_prev_err"))
    else:
        exprs.append(pl.lit(None).alias("_prev_err"))
    if has_well:
        exprs.append(pl.col(WELL).shift(1).over(keys).alias("_prev_well"))
    else:
        exprs.append(pl.lit(None).alias("_prev_well"))

    # EFS current-row
    if OP_ON in df_num.columns and has_L_offs:
        exprs.append(pl.col(L_OFFS).list.min().lt(pl.col(OP_ON)).fill_null(False).alias("_Lpre"))
    else:
        exprs.append(pl.lit(None).alias("_Lpre"))
    if OP_ON in df_num.columns and has_R_offs:
        exprs.append(pl.col(R_OFFS).list.min().lt(pl.col(OP_ON)).fill_null(False).alias("_Rpre"))
    else:
        exprs.append(pl.lit(None).alias("_Rpre"))

    # Lick counts
    exprs.append((pl.col(L_LICKS).list.len() if has_L_lick else pl.lit(None)).alias("n_L_licks"))
    exprs.append((pl.col(R_LICKS).list.len() if has_R_lick else pl.lit(None)).alias("n_R_licks"))

    # Min reward timestamp per trial for lick-before-reward calculation
    if all(c in df_num.columns for c in [L_REW, R_REW]):
        exprs.append(pl.min_horizontal(pl.col(L_REW), pl.col(R_REW)).alias("_min_rew_ts"))
    else:
        exprs.append(pl.lit(None).alias("_min_rew_ts"))

    out = df_num.with_columns(exprs)

    # Totals and ITIs
    out = out.with_columns((pl.col("n_L_licks") + pl.col("n_R_licks")).alias("LicksTotal"))
    # Match MATLAB: if zero licks, set LicksTotal to NaN (null here)
    out = out.with_columns(
        pl.when(pl.col("LicksTotal") == 0).then(None).otherwise(pl.col("LicksTotal")).alias("LicksTotal")
    )

    # Licks before first reward (count only when there are licks; if reward ts is null -> count 0)
    if has_L_lick or has_R_lick:
        out = out.with_row_index(name="_rid")
        combined = pl.concat_list([
            (pl.col(L_LICKS) if has_L_lick else pl.lit([]).cast(pl.List(pl.Float64))),
            (pl.col(R_LICKS) if has_R_lick else pl.lit([]).cast(pl.List(pl.Float64))),
        ]).alias("_licks")
        licks_df = out.select(["_rid", "_min_rew_ts", "LicksTotal", combined])
        llong = licks_df.explode("_licks")
        lbw = (
            llong
            .filter(pl.col("_licks").is_not_null() & (pl.col("_licks") < pl.col("_min_rew_ts")))
            .group_by("_rid")
            .agg(pl.count().alias("__lbw"))
        )
        out = out.join(lbw, on="_rid", how="left")
        # Set null for trials with zero total licks; otherwise use counted value (missing -> 0)
        out = out.with_columns(
            pl.when(pl.col("LicksTotal").is_null()).then(None).otherwise(pl.col("__lbw").fill_null(0)).alias("LicksBeforeRew")
        ).drop(["_rid", "_licks", "__lbw"], strict=False)
    else:
        out = out.with_columns(pl.lit(None).alias("LicksBeforeRew"))

    # Stuff for computing ITI
    iti_keys = keys  # e.g., ["animal_id", "session_idx"]

    # The three timestamps MATLAB uses for the previous trial's "end"
    prev_three = [
        pl.col("house_light_on_ts"),
        pl.col("L_reward_ts"),
        pl.col("R_reward_ts"),
    ]

    # Base max over those three
    prev_end_base = pl.max_horizontal(prev_three)

    # Propagate null if ANY of the three is null (MATLAB max([a,b,c]) -> NaN when any NaN)
    prev_end_expr = (
        pl.when(pl.any_horizontal([e.is_null() for e in prev_three]))
        .then(None)
        .otherwise(prev_end_base)
        .shift(1)                      # previous trial
        .over(iti_keys)                # within same animal+session
    )

    if OP_ON in df_num.columns:
        # ITI = OP_ON - prev_end(k-1 in same session)
        out = out.with_columns(
            (pl.col(OP_ON) - prev_end_expr).alias("InterTrialInterval")
        ).with_columns(
            pl.when(pl.col("trial_idx") == 2)
            .then(None)
            .otherwise(pl.col("InterTrialInterval"))
            .alias("InterTrialInterval")
        )

    else:
        out = out.with_columns(pl.lit(None).alias("InterTrialInterval"))
    out = out.with_columns([
        pl.when(pl.col("_prev_err") == 0).then(pl.col("InterTrialInterval")).otherwise(None)
          .alias("InterTrialInterval_afterWin"),
        pl.when(pl.col("_prev_err") == 1).then(pl.col("InterTrialInterval")).otherwise(None)
          .alias("InterTrialInterval_afterLose"),
    ])

    # Last reward flag
    out = out.with_columns(
        pl.when(pl.col("_prev_err").is_null()).then(None)
          .otherwise((1 - pl.col("_prev_err")).cast(pl.Float64))
          .alias("LastTrialRewarded_flg")
    )

    # --- WS/LS & WSLS (simple, null-safe) ---

    cur  = pl.col(WELL)          # current well ("L"/"R" or null)
    prev = pl.col("_prev_well")  # previous well (already shifted over keys)
    perr = pl.col("_prev_err")   # previous outcome: 0=win, 1=loss

    # treat "", "   ", or null as missing
    def _valid_str(e: pl.Expr) -> pl.Expr:
        return e.is_not_null() & e.cast(pl.Utf8, strict=False).str.strip_chars().ne("")

    cur_ok   = _valid_str(cur)
    prev_ok  = _valid_str(prev)
    perr_ok  = perr.is_not_null()
    have_prev = prev_ok & perr_ok

    not_t2   = (pl.col("trial_idx") != 2)
    win_guard  = have_prev & cur_ok & (perr == 0) & not_t2
    lose_guard = have_prev & cur_ok & (perr == 1)  # unchanged

    out = out.with_columns([
        # unchanged:
        pl.when(have_prev & cur_ok).then((cur != prev).cast(pl.Int8)).otherwise(None)
        .alias("ChoiceSwitchFromLastTrial"),
        pl.when(have_prev & cur_ok).then((cur == prev).cast(pl.Int8)).otherwise(None)
        .alias("ProbRepeat"),

        # WIN flags (now nulled on trial 2)
        pl.when(win_guard).then((cur == prev).cast(pl.Int8)).otherwise(None).alias("WinStay_flg"),
        pl.when(win_guard).then((cur != prev).cast(pl.Int8)).otherwise(None).alias("WinSwitch_flg"),

        # LOSE flags (unchanged)
        pl.when(lose_guard).then((cur != prev).cast(pl.Int8)).otherwise(None).alias("LoseSwitch_flg"),
        pl.when(lose_guard).then((cur == prev).cast(pl.Int8)).otherwise(None).alias("LoseStay_flg"),

        # WSLS parity (unchanged)
        pl.when(have_prev & cur_ok)
        .then((perr == (cur != prev).cast(pl.Int8)).cast(pl.Int8))
        .otherwise(None)
        .alias("WSLS_flg"),
    ])



    # EFS previous-row comparisons and flags
    if OP_ON in df_num.columns and has_L_offs:
        out = out.with_columns(
            pl.col(L_OFFS).shift(1).over(keys).list.max()
              .gt(pl.col(OP_ON).shift(1).over(keys)).fill_null(False).alias("_Lpost")
        )
    else:
        out = out.with_columns(pl.lit(None).alias("_Lpost"))
    if OP_ON in df_num.columns and has_R_offs:
        out = out.with_columns(
            pl.col(R_OFFS).shift(1).over(keys).list.max()
              .gt(pl.col(OP_ON).shift(1).over(keys)).fill_null(False).alias("_Rpost")
        )
    else:
        out = out.with_columns(pl.lit(None).alias("_Rpost"))

    out = out.with_columns(
        pl.when(pl.col("_prev_err").is_null() | (pl.col("trial_idx") == 2))
        .then(None)
        .otherwise(
            ((pl.col("_Lpre") | pl.col("_Lpost")) & (pl.col("_Rpre") | pl.col("_Rpost"))).cast(pl.Int8)
        )
        .alias("EFS_before_flg")
    )
    out = out.with_columns(pl.col("EFS_before_flg").shift(-1).over(keys).alias("EFS_after_flg"))

    # === No-multi-well-response variants ===
    # only defined when EFS_before_flg == 0 AND we have prev well/outcome AND a valid current well
    allow_nomult = (pl.col("EFS_before_flg") == 0) & have_prev & cur_ok

    out = out.with_columns([
        pl.when(allow_nomult & (perr == 0))
        .then((cur == prev).cast(pl.Int8))
        .otherwise(None)
        .alias("WinStay_NoMultWellResp_flg"),

        pl.when(allow_nomult & (perr == 1))
        .then((cur != prev).cast(pl.Int8))
        .otherwise(None)
        .alias("LoseSwitch_NoMultWellResp_flg"),
    ])


    # Did nose poke? Did poke + response? Cumulative sums per session
    out = out.with_columns([
        (pl.col(OP_ON).is_not_null().cast(pl.Int8) if OP_ON in out.columns else pl.lit(None)).alias("DidNP_flg"),
        (pl.col(WELL_ON).is_not_null().cast(pl.Int8) if has_well_on else pl.lit(None)).alias("DidNPandWellResp_flg"),
    ])

    # Number of rewards this trial (count boli)
    out = out.with_columns(
        (
            (pl.col(L_REW).is_not_null().cast(pl.Int8) + pl.col(R_REW).is_not_null().cast(pl.Int8))
            if all(c in out.columns for c in [L_REW, R_REW]) else pl.lit(None)
        ).alias("NumRewards")
    )

    # Cumulative sums across a session (guard to avoid cum_sum on null literal)
    cum_exprs: list[pl.Expr] = [
        pl.col("DidNPandWellResp_flg").cast(pl.Int64).cum_sum().over(keys).alias("DidNPandWellResp_cumsum"),
        pl.col("EFS_before_flg").cast(pl.Int64).cum_sum().over(keys).alias("IFA_cumsum"),
        pl.col("NumRewards").cast(pl.Int64).cum_sum().over(keys).alias("NumRewBoli_cumsum"),
    ]
    if has_err:
        cum_exprs.append((1 - pl.col(ERR).cast(pl.Int64)).cum_sum().over(keys).alias("NumRewTrials_cumsum"))
    else:
        cum_exprs.append(pl.lit(None).alias("NumRewTrials_cumsum"))
    out = out.with_columns(cum_exprs)

    # Reaction and response timing metrics (match MATLAB computeRespTimeAndLicks)
    # InitTime = odor_port_on_ts - trial_start_ts
    # Build safe maxima of well off-times for TimeInWell
    rmax = (pl.col(R_OFFS).list.max() if has_R_offs else pl.lit(None))
    lmax = (pl.col(L_OFFS).list.max() if has_L_offs else pl.lit(None))

    out = out.with_columns([
        (
            (pl.col(OP_ON) - pl.col(TRIAL_START))
            if all(c in out.columns for c in [OP_ON, TRIAL_START]) else pl.lit(None)
        ).alias("InitTime"),
        # ReactTime = odor_port_off_ts - tone_on_ts
        (
            (pl.col(OP_OFF) - pl.col(TONE_ON))
            if all(c in out.columns for c in [OP_OFF, TONE_ON]) else pl.lit(None)
        ).alias("ReactTime"),
        # ResponseTime = well_on_ts - odor_port_off_ts
        (
            (pl.col(WELL_ON) - pl.col(OP_OFF))
            if all(c in out.columns for c in [WELL_ON, OP_OFF]) else pl.lit(None)
        ).alias("ResponseTime"),
        # TimeInWell = max(R_well_all_off_ts, L_well_all_off_ts) - well_on_ts
        (
            (pl.max_horizontal(rmax, lmax) - pl.col(WELL_ON))
            if has_well_on and (has_R_offs or has_L_offs) else pl.lit(None)
        ).alias("TimeInWell"),
    ])

    # Drop helpers
    helpers = [c for c in out.columns if c.startswith("_")]
    return out.drop(helpers)



# def compute_session_metrics(
#     trials_with_metrics: pl.DataFrame,
#     *,
#     keys: Sequence[str],
# ) -> pl.DataFrame:
#     """
#     Aggregate trial‑level metrics down to one row per session.

#     Parameters
#     ----------
#     trials_with_metrics : polars.DataFrame
#         A dataframe containing trial‑level metrics produced by
#         `compute_trial_metrics`.
#     keys : Sequence[str]
#         Columns that uniquely identify a session (e.g. ``("animal_id",
#         "session_idx")``).  Aggregation is performed per group of
#         ``keys``.

#     Returns
#     -------
#     polars.DataFrame
#         One row per session with aggregated metrics.
#     """
#     # Determine a zero‑based index within each session.  We use
#     # ``pl.int_range`` combined with ``pl.count`` to generate a 0,1,2…
#     # sequence per group.  This avoids the `pl.cum_count()` call which
#     # requires a column argument in newer Polars versions.
#     row_idx = pl.int_range(0, pl.count()).over(keys).alias("_row")

#     # Indicator for choosing the right well ("R")
#     is_R = (pl.col(WELL) == pl.lit("R")).cast(pl.Float64).alias("_isR")

#     # Trial rewarded indicator – true if either reward timestamp is
#     # non‑null.  Cast to float for aggregation.
#     rewarded = (pl.col(L_REW).is_not_null() | pl.col(R_REW).is_not_null()).cast(pl.Float64).alias("_rew")

#     # Prepare the frame with helper columns.  The helpers are needed
#     # only for aggregation; they will not be present in the final frame.
#     data = trials_with_metrics.with_columns([
#         row_idx,
#         is_R,
#         rewarded,
#         # robust numeric versions for correctness flags
#         pl.col("wrong_choice_flg").cast(pl.Float64, strict=False).alias("_wcf"),
#         pl.col(ERR).cast(pl.Float64, strict=False).alias("_errf"),
#     ])

#     # Perform groupby aggregation.  Each metric is computed per group of
#     # ``keys``.  Note that metrics involving the first trial exclude
#     # the first row using the helper ``_row``.
#     # Prefer MATLAB semantics for percent-correct: use Wrong_choice_flg if available;
#     # otherwise fall back to 1 - error_flg.
#     pct_correct = (
#         pl.when(pl.col("_wcf").is_not_null())
#         .then(1 - pl.col("_wcf"))
#         .otherwise(1 - pl.col("_errf"))
#     )

#     agg = (
#         data.group_by(list(keys)).agg([
#             pl.count().alias("NumTrials"),
#             pct_correct.mean().alias("PercentCorrectChoice"),
#             pl.col("_rew").mean().alias("MeanRewardsPerTrial"),
#             # Probability of choosing R, excluding the first trial
#             pl.col("_isR").filter(pl.col("_row") > 0).mean().alias("ProbR"),
#             # Mean of win/lose stay/switch flags.  Cast to float to avoid
#             # null propagation issues; Polars will treat nulls correctly
#             pl.col("WSLS_flg").cast(pl.Float64).mean().alias("ProbWSLS"),
#             pl.col("WinStay_flg").cast(pl.Float64).mean().alias("ProbWinStay"),
#             pl.col("LoseSwitch_flg").cast(pl.Float64).mean().alias("ProbLoseSwitch"),
#             pl.col("LoseStay_flg").cast(pl.Float64).mean().alias("ProbLoseStay"),
#         ])
#     )
#     return agg

def compute_session_metrics(
    trials_with_metrics: pl.DataFrame,
    *,
    keys: Sequence[str],
    entropy_word_len: int = 3,
) -> pl.DataFrame:
    """
    Aggregate trial-level metrics down to one row per session.

    Parameters
    ----------
    trials_with_metrics : polars.DataFrame
        A dataframe containing trial-level metrics produced by
        `compute_trial_metrics`.
    keys : Sequence[str]
        Columns that uniquely identify a session (e.g. ("animal_id", "session_idx")).
        Aggregation is performed per group of ``keys``.
    entropy_word_len : int
        (currently unused; placeholder for future Miller-Madow entropy)

    Returns
    -------
    polars.DataFrame
        One row per session with aggregated metrics.
    """

    WELL = "well_id"
    L_REW = "L_reward_ts"
    R_REW = "R_reward_ts"
    ERR = "error_flg"

    # row index within session (0,1,2,...) for first-trial exclusions
    row_idx = pl.int_range(0, pl.count()).over(keys).alias("_row")

    # choose right well?
    is_R = (pl.col(WELL) == pl.lit("R")).cast(pl.Float64).alias("_isR")

    # did choice repeat from previous trial?
    same_as_last = (
        (pl.col("choices") == pl.col("choices").shift(1).over(keys))
        .cast(pl.Float64)
        .alias("_same_as_last")
    )

    # was trial rewarded?
    rewarded = (
        (pl.col(L_REW).is_not_null() | pl.col(R_REW).is_not_null())
        .cast(pl.Float64)
        .alias("_rew")
    )

    # did the animal go to a well (respond)?
    responded = pl.col("well_on_ts").is_not_null().cast(pl.Float64).alias("_responded")

    data = trials_with_metrics.with_columns([
        row_idx,
        is_R,
        same_as_last,
        rewarded,
        responded,
        pl.col("wrong_choice_flg").cast(pl.Float64, strict=False).alias("_wcf"),
        pl.col(ERR).cast(pl.Float64, strict=False).alias("_errf"),
    ])

    # MATLAB semantics for % correct
    pct_correct = (
        pl.when(pl.col("_wcf").is_not_null())
        .then(1 - pl.col("_wcf"))
        .otherwise(1 - pl.col("_errf"))
    )

    agg = (
        data.group_by(list(keys)).agg([
            # session metadata (assumed constant per session)
            pl.col("treatment").first().alias("treatment"),

            # size / performance
            pl.count().alias("NumTrials"),
            pct_correct.mean().alias("PercentCorrectChoice"),
            pl.col("_rew").mean().alias("MeanRewardsPerTrial"),

            # bias to right well, ignoring first trial
            pl.col("_isR")
              .filter(pl.col("_row") > 0)
              .mean()
              .alias("ProbR"),

            # stay/switch metrics
            pl.col("WSLS_flg").cast(pl.Float64).mean().alias("ProbWSLS"),
            pl.col("WinStay_flg").cast(pl.Float64).mean().alias("ProbWinStay"),
            pl.col("LoseSwitch_flg").cast(pl.Float64).mean().alias("ProbLoseSwitch"),
            pl.col("LoseStay_flg").cast(pl.Float64).mean().alias("ProbLoseStay"),

            # response metrics
            (1 - pl.col("_responded").mean()).alias("ProbNoResp"),
            pl.col("_same_as_last")
              .filter(pl.col("_row") > 0)
              .mean()
              .alias("ProbSame"),

            # NEW: licking metrics
            pl.col("LicksTotal").mean().alias("MeanLicksTotal"),
            pl.col("LicksBeforeRew").mean().alias("MeanLicksBeforeRew"),
        ])
    )

    return agg




def compute_metrics(
    trials_df: pl.DataFrame,
    *,
    keys: Sequence[str],
    attach: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame] | pl.DataFrame:
    """
    Compute both trial‑ and session‑level metrics.

    Parameters
    ----------
    trials_df : polars.DataFrame
        The trial‑level dataframe (raw data plus any metadata columns).
    keys : Sequence[str]
        Columns that uniquely identify a session.  These are passed
        through to both `compute_trial_metrics` and `compute_session_metrics`.
    attach : bool, optional
        If True, join the session metrics back onto the trial dataframe
        using a left join on ``keys``.  If False (default), return a
        tuple ``(trial_df, session_df)``.

    Returns
    -------
    tuple or polars.DataFrame
        Either a tuple ``(trial_df, session_df)`` when ``attach=False`` or
        a single dataframe with session metrics attached when
        ``attach=True``.
    """
    tdf = compute_trial_metrics(trials_df, keys=keys)
    sdf = compute_session_metrics(tdf, keys=keys)
    if attach:
        return tdf.join(sdf, on=list(keys), how="left")
    return tdf, sdf

