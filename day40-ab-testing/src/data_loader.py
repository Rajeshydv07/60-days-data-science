import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

def load_raw():
    return pd.read_csv(PROJECT_ROOT / config.DATA_RAW)

def load_cleaned():
    return pd.read_csv(PROJECT_ROOT / config.DATA_CLEANED)

def split_groups(df):
    ctrl = df[df["group"] == "control"].copy()
    exp  = df[df["group"] == "experiment"].copy()
    return ctrl, exp


