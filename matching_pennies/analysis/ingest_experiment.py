# matching_pennies/analysis/ingest_experiment.py

import tomllib  # Python 3.11+; if you're on 3.10, use `tomli` package instead
import polars as pl

from matching_pennies.io import csv_parser, metrics_store
from matching_pennies.compute import compute_performance_metrics as cpm


def _load_config(toml_path: str) -> dict:
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg


def _process_paradigm(
    experiment_name: str,
    paradigm_cfg: dict,
    treatment_map_path: str,
    good_lesions: list[str],
    config_path: str,
    overwrite: bool,
):
    """
    For one paradigm:
    - read raw CSVs
    - compute tdf/sdf
    - filter lesions
    - persist to metrics_store
    """
    if not paradigm_cfg.get("enabled", True):
        print(f"[ingest] Skipping {paradigm_cfg['name']} (enabled=false)")
        return

    paradigm_name = paradigm_cfg["name"]
    raw_dir = paradigm_cfg["directory"]
    paradigm_label = paradigm_cfg["paradigm_label"]

    print(f"[ingest] Processing {experiment_name}/{paradigm_name} from {raw_dir}")

    # 1. gather raw CSVs
    csvs = csv_parser.get_csv_files(raw_dir, recursive=True)

    # 2. build trial dataframe(s) from raw
    raw_trials_df = csv_parser.build_trials(
        csvs,
        treatment_map_path,
        paradigm=paradigm_label,
    )

    # 3. compute metrics
    tdf, sdf = cpm.compute_metrics(raw_trials_df, keys=["animal_id", "session_idx"])

    # 4. lesion filter (if provided)
    if good_lesions:
        tdf = tdf.filter(pl.col("animal_id").is_in(good_lesions))
        sdf = sdf.filter(pl.col("animal_id").is_in(good_lesions))

    # 5. save to disk under data/experiment_<exp>/<paradigm_name>/
    metrics_store.save_metrics(
        tdf=tdf,
        sdf=sdf,
        experiment=experiment_name,
        paradigm=paradigm_name,
        root="data",
        filters={"good_lesions_only": bool(good_lesions)},
        raw_sources=[raw_dir, treatment_map_path, config_path],
        overwrite=overwrite,
    )

    print(f"[ingest] Done {experiment_name}/{paradigm_name}")


def ingest_experiment(config_path: str, overwrite: bool = False):
    """
    Public function you can call from analysis scripts or tests.
    """
    cfg = _load_config(config_path)

    experiment_name = cfg["name"]
    treatment_map_path = cfg["treatment_map"]
    good_lesions = cfg.get("good_lesions", [])

    for paradigm_cfg in cfg["paradigm"]:
        _process_paradigm(
            experiment_name=experiment_name,
            paradigm_cfg=paradigm_cfg,
            treatment_map_path=treatment_map_path,
            good_lesions=good_lesions,
            config_path=config_path,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    # Hardcode a default config path for convenience, or
    # parse sys.argv later if you want CLI flexibility.
    CONFIG = "experiments/lesion_study.toml"
    ingest_experiment(CONFIG, overwrite=True)
