import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = [
    ("eda_distributions.png",   "EDA: Distributions"),
    ("segment_analysis.png",    "Segment Analysis"),
    ("cumulative_timeline.png", "Experiment Timeline"),
    ("results_dashboard.png",   "Results Dashboard"),
]

fig, axes = plt.subplots(4, 1, figsize=(14, 40))
fig.patch.set_facecolor('#0f1117')

for ax, (fname, title) in zip(axes, images):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', color='white', pad=8)

plt.suptitle(
    'Day 40 — A/B Testing Full Report',
    fontsize=17, fontweight='bold', color='white', y=1.002
)

plt.tight_layout(pad=1.5)
plt.savefig('ab_testing_all_charts.png', bbox_inches='tight',
            facecolor=fig.get_facecolor(), dpi=130)
plt.close()
print("Saved: ab_testing_all_charts.png")
