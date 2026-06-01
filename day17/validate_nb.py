"""
The working notebooks all use nbformat 4.2.
Day17 was upgraded to 4.5 which REQUIRES unique 8-char alphanumeric cell IDs.
Let me check day17's cell IDs for any issues, and also compare with 
what GitHub's nbconvert version expects.

Also check: maybe the real fix is to just revert to nbformat 4.2 like all other working notebooks.
"""
import json
import os

path = 'day17_loan_prediction.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Total cells: {len(nb['cells'])}")
print()

# Check all cell IDs
ids_seen = set()
print("Cell IDs:")
for i, cell in enumerate(nb['cells']):
    cell_id = cell.get('id', 'MISSING')
    dup = " <-- DUPLICATE!" if cell_id in ids_seen else ""
    invalid = ""
    if cell_id != 'MISSING':
        # ID must be 1-64 chars, alphanumeric + hyphen/underscore
        import re
        if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', cell_id):
            invalid = f" <-- INVALID FORMAT: '{cell_id}'"
        ids_seen.add(cell_id)
    print(f"  Cell {i} ({cell['cell_type']}): id='{cell_id}'{dup}{invalid}")

print()
# Try to validate notebook with nbformat
try:
    import nbformat
    nb_obj = nbformat.read(path, as_version=4)
    nbformat.validate(nb_obj)
    print("nbformat validation: PASSED")
except Exception as e:
    print(f"nbformat validation: FAILED - {e}")
