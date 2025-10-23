import numpy as np
import matplotlib.pyplot as plt 
import polars as pl 
import matching_pennies.io.csv_parser as csv_parser
import matching_pennies.compute.compute_performance_metrics as cpm 

"""
TODO:
- Check `TimeInWell` to make sure it matches matlab 
    - There are negative and null values -> double check 
"""

# Load data and compute metrics 
directory = "C:/Users/benli/Documents/PaperManuscripts_andProjects/_2018_ACC_hallway/Matching Pennies/Testing/"
tmap = "C:/Users/benli/Documents/PaperManuscripts_andProjects/_2018_ACC_hallway/Matching Pennies/2018_acc_treatment_map.csv"

csvs = csv_parser.get_csv_files(directory, recursive=True)
df = csv_parser.build_trials(csvs, tmap, paradigm="normal")   

tdf, sdf = cpm.compute_metrics(df, keys=["animal_id", "session_idx"])

print(tdf.head())