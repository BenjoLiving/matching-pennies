import polars as pl
from typing import Sequence

def clip_to_minimum_extent(
    df: pl.DataFrame,
    column: str,
    groupby: Sequence[str] = ("animal_id", "session_idx"),
) -> pl.DataFrame:
    """
    Clip a DataFrame so that all groups share a common maximum value 
    of `column`, equal to the smallest group-wise maximum.

    For each group defined by `groupby`, this function:
    1. Computes max(column) within the group.
    2. Finds the minimum of these maxima across all groups.
    3. Returns only rows where column <= that minimum.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe.
    column : str
        Name of the integer-like column to base the clipping on 
        (e.g., 'trial_idx').
    groupby : Sequence[str], default ("animal_id", "session")
        Columns defining groups (e.g., animal and session).

    Returns
    -------
    pl.DataFrame
        A clipped dataframe where all groups have the same maximum
        value of `column`.
    """
    # --- Validate columns ---
    required = list(groupby) + [column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Compute smallest group-wise max ---
    max_per_group = (
        df
        .group_by(list(groupby))
        .agg(
            pl.col(column).max().alias(f"max_{column}")
        )
    )
    min_of_group_max = max_per_group[f"max_{column}"].min()

    # --- Clip original df ---
    return df.filter(pl.col(column) <= min_of_group_max)


def bin_data(
    df: pl.DataFrame,
    bin_size: int,
    bin_on: str = "trial_idx",
    bin_col: str = "bin",
) -> pl.DataFrame:
    """
    Add an integer bin index column based on `bin_on` using fixed-width bins.

    Each row gets:
        bin = floor( bin_on / bin_size )

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe.
    bin_size : int
        Width of each bin (e.g., 10 trials per bin).
    bin_on : str, default "trial_idx"
        Column to bin on.
    bin_col : str, default "bin"
        Name of the output bin column.

    Returns
    -------
    pl.DataFrame
        DataFrame with an additional `bin_col` column.
    """
    if bin_on not in df.columns:
        raise ValueError(f"Column '{bin_on}' not found in DataFrame")
    if bin_size <= 0:
        raise ValueError(f"bin_size must be > 0, got {bin_size}")

    return df.with_columns(
        (pl.col(bin_on) / bin_size).floor().cast(pl.Int64).alias(bin_col)   
    )


def scale_col(df: pl.DataFrame, column: str, scaled_name: str = None):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame") 
    
    if scaled_name == None: 
        scaled_name = column + "_scaled"

    return df.with_columns(
        ((pl.col(column) - pl.col(column).min()) / 
        (pl.col(column).max() - pl.col(column).min()))
        .alias(scaled_name)
    )

