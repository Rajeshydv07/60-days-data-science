import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

def load_raw():
    return pd.read_csv(config.DATA_RAW)

def load_cleaned():
    return pd.read_csv(config.DATA_CLEANED)

def split_groups(df):
    ctrl = df[df["group"] == "control"].copy()
    exp  = df[df["group"] == "experiment"].copy()
    return ctrl, exp


