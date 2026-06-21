# config.py  —  Global constants shared across all notebooks and scripts

RANDOM_SEED      = 42
ALPHA            = 0.05       # significance level
POWER_TARGET     = 0.80       # desired statistical power

N_CONTROL        = 4800
N_EXPERIMENT     = 5200

CONTROL_COLOR    = "#4C72B0"
EXPERIMENT_COLOR = "#DD8452"

BASELINE_CVR     = 0.118      # baseline conversion rate used for power calc
MDE              = 0.02       # minimum detectable effect

DATA_RAW     = "data/ab_test_raw.csv"
DATA_CLEANED = "data/ab_test_cleaned.csv"
IMAGES_DIR   = "images/"




