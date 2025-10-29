import os
import json
from datetime import datetime, timezone
import polars as pl
import subprocess

def _safe_name(x: str) -> str:
    """Make strings safe for folder names."""
    return str(x).replace(" ", "_").lower()

def _experiment_paradigm_dir(root: str, experiment: str, paradigm: str) -> str:
    return os.path.join(
        root,
        f"experiment_{_safe_name(experiment)}",
        _safe_name(paradigm),
    )

def save_metrics(
    tdf: pl.DataFrame,
    sdf: pl.DataFrame,
    experiment: str,
    paradigm: str,
    root: str = "data",
    filters: dict | None = None,
    raw_sources: list[str] | None = None,
    overwrite: bool = False,
):
    """
    Save trial-level (tdf) and session-level (sdf) metrics to disk.

    Args:
        tdf, sdf: Polars DataFrames to save.
        experiment: Experiment name string.
        paradigm: Paradigm name string.
        root: Root directory for saving (default 'data').
        filters: Optional dict describing any filters applied.
        raw_sources: Optional list of source paths.
        overwrite: If False (default), raises an error when target already exists.
                   If True, overwrites existing files.

    Returns:
        Dict with written file paths.
    """
    out_dir = _experiment_paradigm_dir(root, experiment, paradigm)

    # --- check if existing directory ---
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(
                f"Metrics already exist for experiment='{experiment}', "
                f"paradigm='{paradigm}'. Pass overwrite=True to replace."
            )
        else:
            print(f"[metrics_store] Overwriting existing metrics at {out_dir}...")

    os.makedirs(out_dir, exist_ok=True)

    trials_path   = os.path.join(out_dir, "trials.parquet")
    sessions_path = os.path.join(out_dir, "sessions.parquet")
    manifest_path = os.path.join(out_dir, "manifest.json")

    # Write parquet files
    tdf.write_parquet(trials_path)
    sdf.write_parquet(sessions_path)

    # Try to get current git commit (best effort)
    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_sha = None

    manifest = {
        "experiment": experiment,
        "paradigm": paradigm,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "tdf_n_rows": tdf.height,
        "sdf_n_rows": sdf.height,
        "tdf_columns": tdf.columns,
        "sdf_columns": sdf.columns,
        "filters": filters or {},
        "raw_sources": raw_sources or [],
        "overwrite": overwrite,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[metrics_store] Saved metrics to: {out_dir}")
    return {
        "trials_path": trials_path,
        "sessions_path": sessions_path,
        "manifest_path": manifest_path,
    }


def load_metrics(
    experiment: str,
    paradigm: str,
    root: str = "data",
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """
    Load cached trial-level (tdf) and session-level (sdf) metrics, plus the manifest.

    Args:
        experiment: Experiment name string. (e.g. "lesion_study")
        paradigm: Paradigm name string.    (e.g. "normal_mp")
        root: Root directory where cached data lives. Default "data".

    Returns:
        (tdf, sdf, manifest)
            tdf: pl.DataFrame of trial-level metrics
            sdf: pl.DataFrame of session-level metrics
            manifest: dict with metadata (git commit, timestamp, filters, etc.)
    """
    out_dir = _experiment_paradigm_dir(root, experiment, paradigm)

    trials_path   = os.path.join(out_dir, "trials.parquet")
    sessions_path = os.path.join(out_dir, "sessions.parquet")
    manifest_path = os.path.join(out_dir, "manifest.json")

    # read parquet back into Polars
    tdf = pl.read_parquet(trials_path)
    sdf = pl.read_parquet(sessions_path)

    # read manifest metadata
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return tdf, sdf, manifest
