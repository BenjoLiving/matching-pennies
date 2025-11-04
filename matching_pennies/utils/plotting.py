"""
NOT FUNCTIONAL - Fix Later 
"""


# utils/plotting.py
import polars as pl
from typing import Sequence, Literal, Optional, Dict, Any 
import plotly.express as px

YMode = Literal["fraction", "sum", "count", "mean"]

def bin_and_aggregate_flag(
    df: pl.DataFrame,
    flag_col: str,
    *,
    bin_col: str = "bin",
    group_cols: Sequence[str] = ("treatment",),
    y_mode: YMode = "fraction",
    y_name: Optional[str] = None,
) -> pl.DataFrame:
    """
    Aggregate a binary flag per (group_cols × bin), returning a long-format
    DataFrame ready for plotting.

    Parameters
    ----------
    df : pl.DataFrame
        Input data (must contain flag_col, bin_col, and group_cols).
    flag_col : str
        Name of the 0/1 flag column to aggregate.
    bin_col : str, default "bin"
        Name of the bin column.
    group_cols : sequence of str, default ("treatment",)
        Grouping columns in addition to `bin_col`.
    y_mode : {"fraction", "sum", "count", "mean"}, default "fraction"
        - "fraction": sum(flag) / n_trials
        - "sum":      sum(flag)
        - "count":    n_trials
        - "mean":     mean(flag)
    y_name : str or None
        Name of the output y column. If None, a sensible default is used.

    Returns
    -------
    pl.DataFrame
        Columns: group_cols..., bin_col, y
    """
    required = list(group_cols) + [flag_col, bin_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    group_keys = list(group_cols) + [bin_col]

    agg_df = (
        df.group_by(group_keys)
          .agg([
              pl.col(flag_col).sum().alias("flag_sum"),
              pl.len().alias("n_trials"),
              pl.col(flag_col).mean().alias("flag_mean"),
          ])
          .sort(group_keys)
    )

    if y_mode == "fraction":
        y_expr = (pl.col("flag_sum") / pl.col("n_trials"))
        default_name = f"{flag_col}_fraction"
    elif y_mode == "sum":
        y_expr = pl.col("flag_sum")
        default_name = f"{flag_col}_sum"
    elif y_mode == "count":
        y_expr = pl.col("n_trials")
        default_name = "n_trials"
    elif y_mode == "mean":
        y_expr = pl.col("flag_mean")
        default_name = f"{flag_col}_mean"
    else:
        raise ValueError(f"Unsupported y_mode: {y_mode}")

    if y_name is None:
        y_name = default_name

    return agg_df.with_columns(
        y_expr.alias(y_name)
    )



def line_by_treatment_across_bins(
    df: pl.DataFrame,
    *,
    x: str = "bin",
    y: str,
    color: str = "treatment",
    category_orders: Optional[Dict[str, list[str]]] = None,
    title: Optional[str] = None,
    x_label: str = "Trial bin",
    y_label: Optional[str] = None,
    markers: bool = True,
    extra_layout: Optional[Dict[str, Any]] = None,
):
    """
    Quick line plot for y vs bin, colored by treatment.

    df is expected to have columns x, y, and color.
    """
    if y_label is None:
        y_label = y

    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        markers=markers,
        labels={x: x_label, y: y_label},
        category_orders=category_orders,
    )

    if title is not None:
        fig.update_layout(title=title)

    if extra_layout:
        fig.update_layout(**extra_layout)

    return fig
