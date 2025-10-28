"""
session dataframe must include: 

RespEntropy
ProbNoResp
ProbSame
"""

import numpy as np
import polars as pl 
import matching_pennies.io.csv_parser as csv_parser
import matching_pennies.compute.compute_performance_metrics as cpm 

# Load data and compute metrics 
# Mac paths 
directory = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/normal_mp"
tmap = "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/lesion_study_ingestion_map.csv"

csvs = csv_parser.get_csv_files(directory, recursive=False)
df = csv_parser.build_trials(csvs, tmap, paradigm="normal")   

tdf, sdf = cpm.compute_metrics(df, keys=["animal_id", "session_idx"])

"""
Add mean reaction time and intertrial interval? Maybe TimeInWell too?

"""