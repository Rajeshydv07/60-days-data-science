import numpy as np
from scipy import stats
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

def check_srm(n_ctrl, n_exp, expected_split=(0.48, 0.52)):
    total = n_ctrl + n_exp
    observed = np.array([n_ctrl, n_exp])
    expected = np.array([total * expected_split[0], total * expected_split[1]])
    chi2, p = stats.chisquare(observed, expected)
    return chi2, p

def two_prop_ztest(x_ctrl, n_ctrl, x_exp, n_exp):
    p_c = x_ctrl / n_ctrl
    p_e = x_exp / n_exp
    p_pool = (x_ctrl + x_exp) / (n_ctrl + n_exp)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_exp))
    z = (p_e - p_c) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    z_crit = stats.norm.ppf(1 - config.ALPHA / 2)
    ci_lo = (p_e - p_c) - z_crit * se
    ci_hi = (p_e - p_c) + z_crit * se
    return {"z": z, "p_value": p_val, "ci": (ci_lo, ci_hi),
            "p_ctrl": p_c, "p_exp": p_e, "lift_pct": (p_e - p_c) / p_c * 100}

def two_sample_ttest(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = b.mean() - a.mean()
    pct  = diff / a.mean() * 100
    return {"t": t, "p_value": p, "mean_ctrl": a.mean(),
            "mean_exp": b.mean(), "diff": diff, "pct_change": pct}

def power_analysis(baseline, mde, alpha=config.ALPHA, power=config.POWER_TARGET):
    p_alt = baseline + mde
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    num = (z_a * np.sqrt(2 * baseline * (1 - baseline)) +
           z_b * np.sqrt(baseline * (1 - baseline) + p_alt * (1 - p_alt))) ** 2
    return int(np.ceil(num / mde**2))


