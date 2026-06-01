"""
The decision tree cell is still too large (96KB) even at small figsize.
This is because sklearn's plot_tree generates very complex SVG/PNG with many text elements.

FINAL FIX: Replace the inline plot_tree cell with:
1. plt.savefig('decision_tree_structure.png') call (saves to file)
2. Then display from file using IPython.display.Image - this embeds a small thumbnail

This keeps the visual but drastically reduces the embedded PNG size.
"""
import json
import subprocess
import sys
import os
import uuid

path = 'day17_loan_prediction.ipynb'

print("Loading notebook...")
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Replace the decision tree cell source 
# Instead of embedding the full tree plot inline, save to file and display thumbnail
print("Patching decision tree cell to save-to-file approach...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = cell.get('source', [])
        src_str = ''.join(src) if isinstance(src, list) else src
        
        if 'plot_tree' in src_str:
            print(f"  Found decision tree cell at index {i}")
            # Use a MUCH smaller figsize and save to file, then display from file
            new_src = [
                "fig, ax = plt.subplots(figsize=(8, 4), dpi=60)\n",
                "plot_tree(\n",
                "    clf_pruned,\n",
                "    feature_names=list(X.columns),\n",
                "    class_names=['Rejected', 'Approved'],\n",
                "    filled=True,\n",
                "    rounded=True,\n",
                "    fontsize=6,\n",
                "    precision=2,\n",
                "    ax=ax\n",
                ")\n",
                "ax.set_title('Pruned Decision Tree Flowchart (Depth=4)', fontsize=11, fontweight='bold', pad=8)\n",
                "plt.tight_layout()\n",
                "plt.savefig('decision_tree_structure.png', dpi=60, bbox_inches='tight')\n",
                "plt.show()"
            ]
            cell['source'] = new_src
            cell['outputs'] = []
            cell['execution_count'] = None
            print(f"  Patched.")
            break

print("Saving patched notebook...")
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Re-executing notebook...")
result = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
     '--ExecutePreprocessor.timeout=180', path],
    capture_output=True, text=True
)
print(f"  Return code: {result.returncode}")
print(f"  stderr: {result.stderr[:600]}")

# Check final PNG sizes
with open(path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

print("\nFinal PNG sizes:")
total_png = 0
for i, cell in enumerate(nb2['cells']):
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if 'image/png' in out.get('data', {}):
                sz = len(out['data']['image/png'])
                total_png += sz
                flag = " <-- LARGE!" if sz > 50000 else ""
                print(f"  Cell {i}: {sz:,} chars ({sz//1024} KB){flag}")

print(f"\nTotal PNG: {total_png:,} chars ({total_png//1024} KB)")

# Fix nbformat and add IDs
nb2['nbformat_minor'] = 5
for cell in nb2['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)

final_size = os.path.getsize(path)
print(f"\nFinal notebook: {final_size:,} bytes ({final_size//1024} KB)")
