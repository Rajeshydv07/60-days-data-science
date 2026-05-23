import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def verify_setup():
    print("--- Day 4: Environment Setup Verification ---")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    print(f"Seaborn version: {sns.__version__}")
    print("\n[SUCCESS] All libraries imported successfully!")
    print("Domain Selected: Sales (Retail)")
    print("Dataset: Superstore Sales (Kaggle)")
    print("Problem Statement documented in README.md")

if __name__ == "__main__":
    verify_setup()
