"""
Complete fix for day17 notebook GitHub rendering issue.

Root causes identified:
1. Cell 25 (decision tree) has a 107KB PNG - way too large, causes GitHub renderer to timeout
2. nbformat_minor is 2 (should be 5 for modern compatibility)

Fix:
- Re-run the notebook with decision tree at much lower resolution (figsize=8,4 at 60dpi)
- Upgrade nbformat_minor to 5
- Add cell IDs (required by nbformat 4.5+)
"""
import json
import subprocess
import sys
import os
import uuid

path = 'day17_loan_prediction.ipynb'

print("Step 1: Loading notebook...")
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"  nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"  Total cells: {len(nb['cells'])}")

# Check PNG sizes
print("\nStep 2: Checking PNG sizes in current notebook...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if 'image/png' in out.get('data', {}):
                sz = len(out['data']['image/png'])
                print(f"  Cell {i}: PNG = {sz:,} chars ({sz//1024} KB)")

# Fix the source code in the decision tree cell to use smaller figure
print("\nStep 3: Patching decision tree cell to use smaller figure size...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = cell.get('source', [])
        if isinstance(src, list):
            src_str = ''.join(src)
        else:
            src_str = src
        
        # Find the decision tree cell (uses plot_tree)
        if 'plot_tree' in src_str and 'figsize=(12' in src_str:
            print(f"  Found decision tree cell at index {i}")
            # Replace with smaller figure
            new_src = [
                "plt.figure(figsize=(9, 5))\n",
                "plot_tree(\n",
                "    clf_pruned,\n",
                "    feature_names=list(X.columns),\n",
                "    class_names=['Rejected', 'Approved'],\n",
                "    filled=True,\n",
                "    rounded=True,\n",
                "    fontsize=7,\n",
                "    precision=2\n",
                ")\n",
                "plt.title('Pruned Decision Tree Flowchart (Depth=4)', fontsize=12, fontweight='bold', pad=10)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
            cell['source'] = new_src
            # Clear existing output so it gets re-executed
            cell['outputs'] = []
            cell['execution_count'] = None
            print(f"  Patched cell {i} - cleared output for re-execution")

# Also patch the global DPI settings in cell 1 to be even more aggressive
print("\nStep 4: Patching global DPI settings to 72dpi...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = cell.get('source', [])
        if isinstance(src, list):
            src_str = ''.join(src)
        else:
            src_str = src
        
        if "figure.dpi" in src_str and "savefig.dpi" in src_str:
            print(f"  Found DPI settings cell at index {i}")
            new_src = []
            for line in src if isinstance(src, list) else src.split('\n'):
                if "figure.dpi" in line:
                    new_src.append("plt.rcParams['figure.dpi'] = 72\n")
                elif "savefig.dpi" in line:
                    new_src.append("plt.rcParams['savefig.dpi'] = 72\n")
                else:
                    new_src.append(line)
            cell['source'] = new_src

# Save patched notebook
print("\nStep 5: Saving patched notebook...")
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("  Saved.")

# Now re-execute just the decision tree cell using nbconvert
print("\nStep 6: Re-executing full notebook to regenerate all outputs...")
result = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
     '--ExecutePreprocessor.timeout=180', path],
    capture_output=True, text=True
)
print(f"  Return code: {result.returncode}")
if result.stdout:
    print(f"  stdout: {result.stdout[:500]}")
if result.stderr:
    print(f"  stderr: {result.stderr[:500]}")

# Check new PNG sizes
print("\nStep 7: Checking new PNG sizes...")
with open(path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

total_png = 0
for i, cell in enumerate(nb2['cells']):
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if 'image/png' in out.get('data', {}):
                sz = len(out['data']['image/png'])
                total_png += sz
                flag = " <-- LARGE!" if sz > 50000 else ""
                print(f"  Cell {i}: PNG = {sz:,} chars ({sz//1024} KB){flag}")

print(f"\n  Total PNG data: {total_png:,} chars ({total_png//1024} KB)")

# Fix nbformat_minor and add cell IDs
print("\nStep 8: Upgrading nbformat_minor to 5 and adding cell IDs...")
nb2['nbformat_minor'] = 5
for cell in nb2['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)

final_size = os.path.getsize(path)
print(f"\nDone! Final notebook size: {final_size:,} bytes ({final_size//1024} KB)")
print(f"nbformat: {nb2['nbformat']}.{nb2['nbformat_minor']}")
