import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["figure.dpi"]        = 130

CC = config.CONTROL_COLOR
EC = config.EXPERIMENT_COLOR

def bar_compare(ax, ctrl_val, exp_val, title, ylabel, fmt="{:.2f}"):
    bars = ax.bar(["Control", "Experiment"], [ctrl_val, exp_val],
                  color=[CC, EC], width=0.45, edgecolor="white")
    for b, v in zip(bars, [ctrl_val, exp_val]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                fmt.format(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(ctrl_val, exp_val) * 1.3)
    return ax

def hist_compare(ax, ctrl_vals, exp_vals, title, xlabel):
    ax.hist(ctrl_vals, bins=30, alpha=0.65, color=CC, label="Control",  density=True)
    ax.hist(exp_vals,  bins=30, alpha=0.65, color=EC, label="Experiment", density=True)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    return ax


