import numpy as np 
import polars as pl 
import datetime as dt 
from pathlib import Path 
from typing import Iterable
import warnings
import re 


PAT = re.compile(r'(?P<animal>P\d{4})_(?P<date>\d{4}_\d{2}_\d{2})(?:_(?P<time>\d{2}_\d{2}_\d{2}))?\.csv$', re.I)

def parse_meta(p: Path) -> dict:
    m = PAT.search(p.name)
    if not m:
        return {}
    d = m.groupdict()
    session_date = d["date"].replace("_", "-")
    session_time = (d.get("time") or "00_00_00").replace("_", ":")
    return {
        "animal_id": d["animal"].upper(),
        "session_date": session_date,
        "session_datetime": f"{session_date} {session_time}",
        "source_file": str(p),
    }

def clean_headers(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({c: c.strip() for c in df.columns})

def to_array(
    df: pl.DataFrame,
    cols: Iterable[str],
    *,
    dtype: pl.DataType = pl.Float64,
) -> pl.DataFrame:
    """
    Convert space-separated string columns into List[dtype] columns.
    - Handles empty strings and nulls -> []
    - Collapses multiple spaces
    """
    exprs = []
    for c in cols:
        exprs.append(
            pl.when(pl.col(c).is_null())
            .then(pl.lit([]).cast(pl.List(dtype)))
            .otherwise(
                pl.col(c)
                .str.strip_chars()
                .str.replace_all(r"\s+", " ")   # collapse any whitespace runs
                .str.split(" ")                  # -> list[Utf8]
                .list.eval(pl.element().filter(pl.element() != "").cast(dtype))
            )
            .alias(c)
        )
    return df.with_columns(exprs)

def load_one_csv(p: Path) -> pl.DataFrame:
    # sniff separator quickly
    with p.open("r", encoding="utf-8", errors="replace") as f:
        first = ""
        for _ in range(10):
            ln = f.readline()
            if ln.strip():
                first = ln
                break
    sep = max([",","\t",";","|"], key=first.count)

    df = pl.read_csv(p, separator=sep)
    df = clean_headers(df)

    # trim string cells
    df = df.with_columns(pl.col(pl.Utf8).str.strip_chars())

    # add trial index (per file)
    df = df.with_row_index(name="trial_idx", offset=1)

    # add metadata from filename/path
    meta = parse_meta(p)
    for k, v in meta.items():
        df = df.with_columns(pl.lit(v).alias(k))
    return df

def build_trials(
    csv_paths: list[Path],
    treatment_map_path: Path,
    paradigm: str,
) -> pl.DataFrame:
    """
    Combine and annotate trial CSVs into a single Polars DataFrame.

    Parameters
    ----------
    csv_paths : list[Path]
        List of per-session CSV files.
    treatment_map_path : Path
        Path to treatment map CSV (must include 'experiment', 'animal_id', 'treatment').
    paradigm : str
        Name of the behavioral paradigm (applied uniformly to all rows).
    """
    # Load and stack CSVs
    dfs = []
    for p in csv_paths:
        if p.name.startswith("._"):
            continue
        try:
            df = load_one_csv(p)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"⚠️ Failed to load {p.name}: {e}")
            continue

    if not dfs:
        raise RuntimeError("No valid CSVs were loaded.")
    big = pl.concat(dfs, how="diagonal_relaxed").rechunk()

    # Parse date/time
    big = big.with_columns([
        pl.col("session_date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
        pl.col("session_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
    ])

    # Convert array-like columns
    array_like_cols = ["L_lick_ts", "R_lick_ts"]   # <-- expand here as needed
    existing = [c for c in array_like_cols if c in big.columns]
    if existing:
        big = to_array(big, existing, dtype=pl.Float64)

    # Read and normalize treatment map
    tm = pl.read_csv(treatment_map_path)
    tm = tm.rename({c: c.strip().lower() for c in tm.columns})
    tm = tm.with_columns([
        pl.col("experiment"),
        pl.col("animal_id"),
    ])

    # Normalize join keys on `big`
    big = big.with_columns([
        pl.col("animal_id"),
    ])

    # Join treatment info
    big = big.join(
        tm.select(["experiment", "animal_id", "treatment"]),
        on="animal_id",
        how="left"
    )

    # Add paradigm as a constant column
    big = big.with_columns(pl.lit(paradigm).alias("paradigm"))

    # Set certain columns as categorical for performance 
    big = big.with_columns([
        pl.col("animal_id").cast(pl.Categorical),
        pl.col("experiment").cast(pl.Categorical),
        pl.col("treatment").cast(pl.Categorical),
        pl.col("paradigm").cast(pl.Categorical),
    ])

    # Sort and compute per-animal session index
    big = big.sort(["animal_id", "session_datetime", "trial_idx"]).with_columns(
        pl.col("session_datetime")
          .rank("dense")
          .over("animal_id")
          .cast(pl.Int32)
          .alias("session_idx")
    )

    return big

def get_csv_files(directory: str | Path, recursive: bool = False) -> list[Path]:
    """
    Return all CSV files in a directory (and optionally subdirectories)
    that do not start with '.'.

    Parameters
    ----------
    directory : str | Path
        The directory to search.
    recursive : bool, default False
        If True, also search all subdirectories.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    NotADirectoryError
        If the path is not a directory.
    """
    directory = Path(directory)

    # Check directory existence
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # Choose search method
    pattern = "**/*.csv" if recursive else "*.csv"
    csv_files = [f for f in directory.glob(pattern) if not f.name.startswith(".")]

    return csv_files


if __name__ == "__main__":
    directory = "C:/Users/benli/Documents/PaperManuscripts_andProjects/_2018_ACC_hallway/Matching Pennies/Testing/"
    tmap = "C:/Users/benli/Documents/PaperManuscripts_andProjects/_2018_ACC_hallway/Matching Pennies/2018_acc_treatment_map.csv"

    csvs = get_csv_files(directory, recursive=True)

    df = build_trials(csvs, tmap, paradigm="normal")     
    
  

    