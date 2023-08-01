import pandas as pd
import os
import sys

if sys.platform == "linux":
    _BASE_PATH = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"

def load_file(filename: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(_BASE_PATH, filename), sep="\t", header=0)
    return df