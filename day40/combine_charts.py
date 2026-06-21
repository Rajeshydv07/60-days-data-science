import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = [
    ("eda_distributions.png",   "EDA: Distributions"),
    ("segment_analysis.png",    "Segment Analysis"),
    ("cumulative_timeline.png", "Experiment Timeline"),
    ("results_dashboard.png",   "Results Dashboard"),
]

fig, axes = plt.subplots(2, 2, figsize=(22, 14))
fig.patch.set_facecolor('#0f1117')

for ax, (fname, title) in zip(axes.flatten(), images):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=6)

fig.suptitle(
    'Day 40  |  A/B Testing Full Report  |  60 Days of Data Science',
    fontsize=15, fontweight='bold', color='white', y=1.01
)

plt.tight_layout(pad=1.2)
plt.savefig('ab_testing_all_charts.png', bbox_inches='tight',
            facecolor=fig.get_facecolor(), dpi=130)
plt.close()
print("Saved: ab_testing_all_charts.png")
