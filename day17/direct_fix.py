"""
Direct approach: Replace only the oversized decision tree PNG in the notebook
with a freshly generated, much smaller version.
No kernel re-execution needed.
"""
import json
import base64
import io
import os
import uuid

path = 'day17_loan_prediction.ipynb'

# Generate a fresh small decision tree image
print("Step 1: Generating fresh small decision tree image...")
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('loan_data.csv')
X = df.drop(columns=['Approved'])
y = df['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_pruned.fit(X_train, y_train)

# Very small figure
fig, ax = plt.subplots(figsize=(8, 4), dpi=60)
plot_tree(
    clf_pruned,
    feature_names=list(X.columns),
    class_names=['Rejected', 'Approved'],
    filled=True,
    rounded=True,
    fontsize=6,
    precision=2,
    ax=ax
)
ax.set_title('Pruned Decision Tree Flowchart (Depth=4)', fontsize=11, fontweight='bold', pad=8)
plt.tight_layout()

# Save to bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=60, bbox_inches='tight')
buf.seek(0)
png_bytes = buf.read()
plt.close()

# Convert to base64 string (no line wrapping - match nbconvert format)
b64_str = base64.b64encode(png_bytes).decode('ascii')
print(f"  Generated PNG: {len(b64_str):,} chars ({len(b64_str)//1024} KB)")

# Also save to file
with open('decision_tree_structure.png', 'wb') as f:
    f.write(png_bytes)
print(f"  Saved decision_tree_structure.png: {os.path.getsize('decision_tree_structure.png'):,} bytes")

# Load notebook and replace the large PNG
print("\nStep 2: Loading notebook and replacing large PNG...")
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replaced = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if 'image/png' in out.get('data', {}):
                sz = len(out['data']['image/png'])
                if sz > 50000:  # Replace any PNG > 50KB
                    print(f"  Replacing large PNG in cell {i} ({sz:,} chars -> {len(b64_str):,} chars)")
                    out['data']['image/png'] = b64_str
                    # Also update text/plain
                    out['data']['text/plain'] = ['<Figure size 480x240 with 1 Axes>']
                    replaced = True

if not replaced:
    print("  WARNING: No large PNG found to replace!")

# Fix nbformat_minor and add cell IDs
print("\nStep 3: Upgrading nbformat and adding cell IDs...")
nb['nbformat_minor'] = 5
for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]

# Save
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

final_size = os.path.getsize(path)
print(f"\nDone! Final notebook: {final_size:,} bytes ({final_size//1024} KB)")

# Verify final PNG sizes
print("\nFinal PNG sizes:")
with open(path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

total = 0
for i, cell in enumerate(nb2['cells']):
    for out in cell.get('outputs', []):
        if 'image/png' in out.get('data', {}):
            sz = len(out['data']['image/png'])
            total += sz
            flag = " <-- STILL LARGE!" if sz > 50000 else ""
            print(f"  Cell {i}: {sz:,} chars ({sz//1024} KB){flag}")
print(f"  Total: {total:,} chars ({total//1024} KB)")
