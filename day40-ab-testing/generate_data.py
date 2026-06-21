import numpy as np
import pandas as pd

np.random.seed(42)

n_control = 4800
n_experiment = 5200
total = n_control + n_experiment

# Control group: old website design
control_converted = np.random.binomial(1, 0.118, n_control)
control_revenue = np.where(
    control_converted == 1,
    np.random.lognormal(mean=3.8, sigma=0.7, size=n_control),
    0.0
)
control_session_time = np.random.gamma(shape=4.0, scale=2.2, size=n_control)
control_pages_viewed = np.random.poisson(lam=3.1, size=n_control) + 1

# Experiment group: new website design with redesigned CTA
experiment_converted = np.random.binomial(1, 0.143, n_experiment)
experiment_revenue = np.where(
    experiment_converted == 1,
    np.random.lognormal(mean=3.95, sigma=0.65, size=n_experiment),
    0.0
)
experiment_session_time = np.random.gamma(shape=4.6, scale=2.3, size=n_experiment)
experiment_pages_viewed = np.random.poisson(lam=3.6, size=n_experiment) + 1

# Build control dataframe
df_control = pd.DataFrame({
    "user_id": range(1, n_control + 1),
    "group": "control",
    "converted": control_converted,
    "revenue": control_revenue.round(2),
    "session_time_min": control_session_time.round(2),
    "pages_viewed": control_pages_viewed,
    "device": np.random.choice(["desktop", "mobile", "tablet"], size=n_control, p=[0.52, 0.38, 0.10]),
    "country": np.random.choice(["US", "UK", "CA", "AU", "IN"], size=n_control, p=[0.55, 0.18, 0.12, 0.08, 0.07]),
})

# Build experiment dataframe
df_experiment = pd.DataFrame({
    "user_id": range(n_control + 1, total + 1),
    "group": "experiment",
    "converted": experiment_converted,
    "revenue": experiment_revenue.round(2),
    "session_time_min": experiment_session_time.round(2),
    "pages_viewed": experiment_pages_viewed,
    "device": np.random.choice(["desktop", "mobile", "tablet"], size=n_experiment, p=[0.52, 0.38, 0.10]),
    "country": np.random.choice(["US", "UK", "CA", "AU", "IN"], size=n_experiment, p=[0.55, 0.18, 0.12, 0.08, 0.07]),
})

df = pd.concat([df_control, df_experiment], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["user_id"] = df.index + 1001

df.to_csv("ab_test_data.csv", index=False)
print("Dataset saved: ab_test_data.csv")
print(f"Total rows: {len(df)}")
print(df.groupby("group")[["converted", "revenue", "session_time_min", "pages_viewed"]].mean().round(4))
